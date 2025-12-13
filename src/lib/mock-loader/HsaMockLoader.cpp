////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "luthier/mock-loader/MockAMDGPULoader.h"
#include <hsa/amd_hsa_kernel_code.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/common/GenericLuthierError.h>

namespace luthier {

MockLoadedCodeObject::MockLoadedCodeObject(MockAMDGPULoader &Owner,
                                           llvm::ArrayRef<std::byte> Elf,
                                           llvm::Error &Err)
    : Parent(Owner) {
  llvm::ErrorAsOutParameter EAO(Err);

  /// Parse the code object
  Err = object::AMDGCNObjectFile::createAMDGCNObjectFile(
            llvm::StringRef(reinterpret_cast<const char *>(Elf.data()),
                            Elf.size()))
            .moveInto(this->Elf);
  if (Err)
    return;

  /// Cast to object::ELFObjectFileBase since for some reason methods for
  /// querying the ELF EMachine and the ABI versions are private in the
  /// little endian 64-bit sub-class version
  auto &ElfBase = llvm::cast<llvm::object::ELFObjectFileBase>(*this->Elf);

  /// We don't support HSA code object V2 and earlier
  if (ElfBase.getOS() == llvm::Triple::AMDHSA) {
    uint8_t CodeObjectVersion = ElfBase.getEIdentABIVersion();

    if (CodeObjectVersion < llvm::ELF::ELFABIVERSION_AMDGPU_HSA_V3 ||
        CodeObjectVersion > llvm::ELF::ELFABIVERSION_AMDGPU_HSA_V6) {
      Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Unsupported code object version {0}", CodeObjectVersion + 2));
      return;
    }
  }

  /// Before doing any loading, check if there are any symbols in the dynsym
  /// section of this code object that are already defined by other code
  /// objects or an external symbol; If so, return an error
  for (llvm::object::ELFSymbolRef SymIter :
       llvm::make_range(this->Elf->dynamic_symbol_begin(),
                        this->Elf->dynamic_symbol_end())) {
    llvm::Expected<llvm::StringRef> SymNameOrErr = SymIter.getName();
    if (SymNameOrErr.takeError()) {
      return;
    }
    Err = Parent.iterateLoadedCodeObjects(
        [&](const MockLoadedCodeObject &LCO) -> llvm::Error {
          auto SymbolIfExists = LCO.getCodeObject().lookupSymbol(*SymNameOrErr);
          LUTHIER_RETURN_ON_ERROR(SymbolIfExists.takeError());
          if (*SymbolIfExists != std::nullopt &&
              (*SymbolIfExists)->getBinding() == llvm::ELF::STB_GLOBAL) {
            return LUTHIER_MAKE_GENERIC_ERROR(
                llvm::formatv("Code object defines symbol named {0} already "
                              "defined by an earlier code object"));
          }
          return llvm::Error::success();
        });
    if (Err)
      return;
    if (auto It = Parent.findExternalSymbol(*SymNameOrErr);
        It != Parent.external_symbol_end()) {
      Err = LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Code object defines symbol {0} already defined as an "
                        "external symbol in its parent executable"));
      return;
    }
  }

  const auto &CodeObjectELFFile = this->Elf->getELFFile();

  /// Get the PT_LOAD segments of the ELF
  auto ProgramHeadersOrErr = CodeObjectELFFile.program_headers();
  Err = ProgramHeadersOrErr.takeError();
  if (Err) {
    return;
  }

  for (const auto Phdr : *ProgramHeadersOrErr) {
    if (Phdr.p_type == llvm::ELF::PT_LOAD) {
      PTLoadSegments.push_back(Phdr);
    }
  }

  if (PTLoadSegments.empty()) {
    Err = LUTHIER_MAKE_GENERIC_ERROR("The code object has no PT_LOAD sections");
    return;
  }

  /// Even though the load segments should be  pre-sorted w.r.t their virtual
  /// address, we take a precaution and sort it anyway
  llvm::sort(PTLoadSegments, [](const auto &Lhs, const auto &Rhs) {
    return Lhs.get().p_vaddr < Rhs.get().p_vaddr;
  });

  uint64_t Size =
      PTLoadSegments.back().get().p_vaddr + PTLoadSegments.back().get().p_memsz;

  /// Allocate the region and zero its memory
  Segment = {new (std::align_val_t{AMD_ISA_ALIGN_BYTES}, std::nothrow)
                 std::byte[Size],
             Size};
  if (!Segment.data()) {
    Err = LUTHIER_MAKE_GENERIC_ERROR(
        "Failed to allocate segment memory for the loaded code object");
    return;
  }

  std::memset(Segment.data(), 0, Size);

  /// If region allocation was successful, load the PT_LOAD segments

  for (auto PTLoadSegment : PTLoadSegments) {
    std::memcpy(&Segment[PTLoadSegment.get().p_vaddr],
                &Elf[PTLoadSegment.get().p_offset],
                PTLoadSegment.get().p_filesz);
  }

  /// Apply static relocations
  for (const llvm::object::SectionRef Section : this->Elf->sections()) {
    for (const llvm::object::ELFRelocationRef Reloc : Section.relocations()) {
      Err = applyStaticRelocation(Reloc);
      if (Err)
        return;
    }
  }
}

llvm::Error MockLoadedCodeObject::applyStaticRelocation(
    const llvm::object::ELFRelocationRef Rel) {
  uint64_t RelOffset = Rel.getOffset();

  llvm::Expected<uint64_t> AddendOrErr = Rel.getAddend();
  LUTHIER_RETURN_ON_ERROR(AddendOrErr.takeError());

  switch (Rel.getType()) {
  case llvm::ELF::R_AMDGPU_ABS32_LO:
  case llvm::ELF::R_AMDGPU_ABS32_HI:
  case llvm::ELF::R_AMDGPU_ABS64:
  case llvm::ELF::R_AMDGPU_REL64:
  case llvm::ELF::R_AMDGPU_REL32_LO:
  case llvm::ELF::R_AMDGPU_REL32_HI:
  case llvm::ELF::R_AMDGPU_REL16: {
    llvm::object::elf_symbol_iterator Sym = Rel.getSymbol();
    if (Sym == Rel.getObject()->symbol_end()) {
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation section doesn't have a symbol");
    }
    uint64_t Addr;
    switch (Sym->getELFType()) {
    default:
      LUTHIER_RETURN_ON_ERROR(Sym->getAddress().moveInto(Addr));
      break;
    case llvm::ELF::STT_NOTYPE:
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Static relocation symbol is undefined");
    }

    Addr += *AddendOrErr;
    uint32_t Addr32 = 0;
    uint16_t Addr16 = 0;
    switch (Rel.getType()) {
    case llvm::ELF::R_AMDGPU_ABS32_LO:
      Addr += reinterpret_cast<uint64_t>(Segment.data());
      Addr32 = static_cast<uint32_t>(Addr & 0xFFFFFFFF);
      std::memcpy(&Segment[RelOffset], &Addr32, sizeof(Addr32));
      break;
    case llvm::ELF::R_AMDGPU_ABS32_HI:
      Addr += reinterpret_cast<uint64_t>(Segment.data());
      Addr32 = static_cast<uint32_t>((Addr >> 32) & 0xFFFFFFFF);
      std::memcpy(&Segment[RelOffset], &Addr32, sizeof(Addr32));
      break;
    case llvm::ELF::R_AMDGPU_ABS64:
      Addr += reinterpret_cast<uint64_t>(Segment.data());
      std::memcpy(&Segment[RelOffset], &Addr, sizeof(Addr));
      break;
    case llvm::ELF::R_AMDGPU_REL32:
      Addr32 = static_cast<uint32_t>(Addr - RelOffset);
      std::memcpy(&Segment[RelOffset], &Addr32, sizeof(Addr32));
      break;
    case llvm::ELF::R_AMDGPU_REL64:
      Addr -= RelOffset;
      std::memcpy(&Segment[RelOffset], &Addr, sizeof(Addr));
      break;
    case llvm::ELF::R_AMDGPU_REL32_LO:
      Addr = Addr - RelOffset;
      Addr32 = static_cast<uint32_t>(Addr & 0xFFFFFFFF);
      std::memcpy(&Segment[RelOffset], &Addr32, sizeof(Addr32));
      break;
    case llvm::ELF::R_AMDGPU_REL32_HI:
      Addr -= RelOffset;
      Addr32 = static_cast<uint32_t>((Addr >> 32) & 0xFFFFFFFF);
      std::memcpy(&Segment[RelOffset], &Addr32, sizeof(Addr32));
    case llvm::ELF::R_AMDGPU_REL16:
      Addr16 = static_cast<uint16_t>((Addr - RelOffset - 4) / 4);
      std::memcpy(&Segment[RelOffset], &Addr16, sizeof(Addr16));
    default:
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Encountered invalid relocation type {0}", Rel.getType()));
    }
    break;
  }
    /// TODO: implement GOT relocations

  default:
    /// Ignore 32-bit relocations and anything else not supported
    break;
  }
  return llvm::Error::success();
}

llvm::Error MockLoadedCodeObject::finalize() {
  /// Apply dynamic relocations
  for (const llvm::object::SectionRef DynRelocSection :
       llvm::cast<llvm::object::ObjectFile>(Elf.get())
           ->dynamic_relocation_sections()) {
    for (const llvm::object::ELFRelocationRef Reloc :
         DynRelocSection.relocations()) {
      LUTHIER_RETURN_ON_ERROR(applyDynamicRelocation(Reloc));
    }
  }
  return llvm::Error::success();
}

llvm::Error MockLoadedCodeObject::applyDynamicRelocation(
    llvm::object::ELFRelocationRef Rel) {

  switch (Rel.getType()) {
  case llvm::ELF::R_AMDGPU_ABS32_LO:
  case llvm::ELF::R_AMDGPU_ABS32_HI:
  case llvm::ELF::R_AMDGPU_ABS64:
  case llvm::ELF::R_AMDGPU_REL64:
  case llvm::ELF::R_AMDGPU_REL32_LO:
  case llvm::ELF::R_AMDGPU_REL32_HI:
  case llvm::ELF::R_AMDGPU_REL16: {
    llvm::object::elf_symbol_iterator Sym = Rel.getSymbol();
    if (Sym == Rel.getObject()->symbol_end()) {
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation section doesn't have a symbol");
    }
    uint64_t Addr;
    switch (Sym->getELFType()) {
    default:
      LUTHIER_RETURN_ON_ERROR(Sym->getAddress().moveInto(Addr));
      break;
    case llvm::ELF::STT_NOTYPE:
      llvm::Expected<llvm::StringRef> SymNameOrErr = Sym->getName();
      LUTHIER_RETURN_ON_ERROR(SymNameOrErr.takeError());

      /// First query the parent executable and other loaded code objects in the
      /// parent executable if the symbol is defined there
      if (auto It = Parent.findExternalSymbol(*SymNameOrErr);
          It != Parent.external_symbol_end()) {
      }
    }

    Addr += *AddendOrErr;
    uint32_t Addr32 = 0;
    uint16_t Addr16 = 0;
    switch (Rel.getType()) {
    case llvm::ELF::R_AMDGPU_ABS32_LO:
      Addr += reinterpret_cast<uint64_t>(Segment.data());
      Addr32 = static_cast<uint32_t>(Addr & 0xFFFFFFFF);
      std::memcpy(&Segment[RelOffset], &Addr32, sizeof(Addr32));
      break;
    case llvm::ELF::R_AMDGPU_ABS32_HI:
      Addr += reinterpret_cast<uint64_t>(Segment.data());
      Addr32 = static_cast<uint32_t>((Addr >> 32) & 0xFFFFFFFF);
      std::memcpy(&Segment[RelOffset], &Addr32, sizeof(Addr32));
      break;
    case llvm::ELF::R_AMDGPU_ABS64:
      Addr += reinterpret_cast<uint64_t>(Segment.data());
      std::memcpy(&Segment[RelOffset], &Addr, sizeof(Addr));
      break;
    case llvm::ELF::R_AMDGPU_REL32:
      Addr32 = static_cast<uint32_t>(Addr - RelOffset);
      std::memcpy(&Segment[RelOffset], &Addr32, sizeof(Addr32));
      break;
    case llvm::ELF::R_AMDGPU_REL64:
      Addr -= RelOffset;
      std::memcpy(&Segment[RelOffset], &Addr, sizeof(Addr));
      break;
    case llvm::ELF::R_AMDGPU_REL32_LO:
      Addr = Addr - RelOffset;
      Addr32 = static_cast<uint32_t>(Addr & 0xFFFFFFFF);
      std::memcpy(&Segment[RelOffset], &Addr32, sizeof(Addr32));
      break;
    case llvm::ELF::R_AMDGPU_REL32_HI:
      Addr -= RelOffset;
      Addr32 = static_cast<uint32_t>((Addr >> 32) & 0xFFFFFFFF);
      std::memcpy(&Segment[RelOffset], &Addr32, sizeof(Addr32));
    case llvm::ELF::R_AMDGPU_REL16:
      Addr16 = static_cast<uint16_t>((Addr - RelOffset - 4) / 4);
      std::memcpy(&Segment[RelOffset], &Addr16, sizeof(Addr16));
    default:
      return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "Encountered invalid relocation type {0}", Rel.getType()));
    }
    break;
  }
    /// TODO: implement GOT relocations

  default:
    /// Ignore 32-bit relocations and anything else not supported
    break;
  }

  switch (Rel.getType()) {
  default: {
    llvm::object::elf_symbol_iterator Sym = Rel.getSymbol();
    if (Sym == Rel.getObject()->symbol_end()) {
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation section doesn't have a symbol");
    }
    LUTHIER_RETURN_ON_ERROR(Sym->getAddress().moveInto(SymAddr));
    SymAddr += reinterpret_cast<uint64_t>(Segment.data());
    break;
  }

    // External symbols, they must be defined prior loading.
  case llvm::ELF::STT_NOTYPE: {
    // TODO: Only agent allocation variables are supported in v2.1. How will
    // we distinguish between program allocation and agent allocation
    // variables?
    auto agent_symbol =
        agent_symbols_.find(std::make_pair(rel->symbol()->name(), agent));
    if (agent_symbol != agent_symbols_.end())
      SymAddr = agent_symbol->second->address;
    break;
  }
  }

  llvm::Expected<uint64_t> AddendOrErr = Rel.getAddend();
  LUTHIER_RETURN_ON_ERROR(AddendOrErr.takeError());
  SymAddr += *AddendOrErr;

  switch (Rel.getType()) {
  case llvm::ELF::R_AMDGPU_ABS32_HI: {
    auto SymAddr32 = static_cast<uint32_t>((SymAddr >> 32) & 0xFFFFFFFF);
    std::memcpy(&Segment[Rel.getOffset()], &SymAddr32, sizeof(SymAddr32));
    break;
  }

  case llvm::ELF::R_AMDGPU_ABS32_LO: {
    auto SymAddr32 = static_cast<uint32_t>(SymAddr & 0xFFFFFFFF);
    std::memcpy(&Segment[Rel.getOffset()], &SymAddr32, sizeof(SymAddr32));
    break;
  }

  case llvm::ELF::R_AMDGPU_ABS32: {

    auto symAddr32 = static_cast<uint32_t>(SymAddr);
    std::memcpy(&Segment[Rel.getOffset()], &symAddr32, sizeof(symAddr32));
    break;
  }

  case llvm::ELF::R_AMDGPU_ABS64: {
    std::memcpy(&Segment[Rel.getOffset()], &SymAddr, sizeof(SymAddr));
    break;
  }

  case llvm::ELF::R_AMDGPU_RELATIVE64: {
    int64_t baseDelta =
        reinterpret_cast<uint64_t>(relSeg->Address(0)) - relSeg->VAddr();
    uint64_t relocatedAddr = baseDelta + rel->addend();
    relSeg->Copy(rel->offset(), &relocatedAddr, sizeof(relocatedAddr));
    break;
  }

  default:
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Invalid relocation type {0}", Rel.getType()));
  }
  return llvm::Error::success();
}

llvm::Error MockAMDGPULoader::defineExternalSymbol(llvm::StringRef Name,
                                                   void *Address) {
  if (IsFinalized)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Cannot define a new external variable after the loader is finalized");

  if (ExternalSymbols.contains(Name))
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Symbol {0} is already defined in the loader", Name));

  if (auto [_, InsertionStatus] = ExternalSymbols.insert({Name, Address});
      !InsertionStatus)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "Failed to insert the new symbol definition into the loader's symbol "
        "map");

  return llvm::Error::success();
}

llvm::Expected<const MockLoadedCodeObject &>
MockAMDGPULoader::loadCodeObject(llvm::ArrayRef<std::byte> CodeObject) {

  if (isFinalized())
    return LUTHIER_MAKE_GENERIC_ERROR("The loader is already finalized");

  llvm::Error Err = llvm::Error::success();

  LoadedCodeObjects.emplace_back(std::unique_ptr<MockLoadedCodeObject>(
      new MockLoadedCodeObject(*this, CodeObject, Err)));
  LUTHIER_RETURN_ON_ERROR(Err);

  return *LoadedCodeObjects.back();
}

llvm::Error MockAMDGPULoader::finalize() {
  if (isFinalized()) {
    return LUTHIER_MAKE_GENERIC_ERROR(
        "The loader has already finalized the loaded code objects");
  }

  for (auto &LCO : LoadedCodeObjects) {
    LUTHIER_RETURN_ON_ERROR(LCO->finalize());
  }

  IsFinalized = true;
  return llvm::Error::success();
}

} // namespace luthier
