//===-- MockAMDGPULoader.cpp ------------------------------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
/// \file Implements the \c MockAMDGPULoader and \c MockLoadedCodeObject
/// classes.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/MockAMDGPULoader.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include <hsa/amd_hsa_kernel_code.h>

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

    for (const auto &LCO : Parent.loaded_code_objects()) {
      auto SymbolIfExists = LCO.getCodeObject().lookupSymbol(*SymNameOrErr);
      Err = SymbolIfExists.takeError();
      if (Err)
        return;
      if (*SymbolIfExists != std::nullopt &&
          (*SymbolIfExists)->getBinding() == llvm::ELF::STB_GLOBAL) {
        Err = LUTHIER_MAKE_GENERIC_ERROR(
            llvm::formatv("Code object defines symbol named {0} already "
                          "defined by an earlier code object"));
        return;
      }
    }
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

  /// Even though the load segments should be  pre-sorted w.r.t their
  /// virtual address, we take a precaution and sort it anyway
  llvm::sort(PTLoadSegments, [](const auto &Lhs, const auto &Rhs) {
    return Lhs.get().p_vaddr < Rhs.get().p_vaddr;
  });

  uint64_t Size =
      PTLoadSegments.back().get().p_vaddr + PTLoadSegments.back().get().p_memsz;

  /// Allocate the region and zero its memory
  LoadedRegion = {new (std::align_val_t{AMD_ISA_ALIGN_BYTES}, std::nothrow)
                      std::byte[Size],
                  Size};
  if (!LoadedRegion.data()) {
    Err = LUTHIER_MAKE_GENERIC_ERROR(
        "Failed to allocate segment memory for the loaded code object");
    return;
  }

  std::memset(LoadedRegion.data(), 0, Size);

  /// If region allocation was successful, load the PT_LOAD segments

  for (auto PTLoadSegment : PTLoadSegments) {
    std::memcpy(&LoadedRegion[PTLoadSegment.get().p_vaddr],
                &Elf[PTLoadSegment.get().p_offset],
                PTLoadSegment.get().p_filesz);
  }
}

llvm::Error MockLoadedCodeObject::finalize() {

  /// Apply static relocations
  for (const llvm::object::SectionRef Section : this->Elf->sections()) {
    for (const llvm::object::ELFRelocationRef Reloc : Section.relocations()) {
      LUTHIER_RETURN_ON_ERROR(applyRelocation(Reloc));
    }
  }

  /// Apply dynamic relocations
  for (const llvm::object::SectionRef DynRelocSection :
       llvm::cast<llvm::object::ObjectFile>(Elf.get())
           ->dynamic_relocation_sections()) {
    for (const llvm::object::ELFRelocationRef Reloc :
         DynRelocSection.relocations()) {
      LUTHIER_RETURN_ON_ERROR(applyRelocation(Reloc));
    }
  }
  return llvm::Error::success();
}

llvm::Error MockLoadedCodeObject::applyRelocation(
    const llvm::object::ELFRelocationRef Rel) {
  uint64_t RelOffset = Rel.getOffset();
  uint64_t RelType = Rel.getType();
  auto LoadBase = reinterpret_cast<uint64_t>(LoadedRegion.data());

  /// Resolve and calculate symbol address if exists
  uint64_t SymAddr = 0;
  llvm::object::elf_symbol_iterator Sym = Rel.getSymbol();
  if (Sym != Rel.getObject()->symbol_end()) {
    /// Resolve the external symbol by looking it up in other loaded code
    /// objects in the loader or in the external symbols defined in the
    /// loader
    if (Sym->getELFType() == llvm::ELF::STT_NOTYPE) {
      llvm::Expected<llvm::StringRef> SymNameOrErr = Sym->getName();
      LUTHIER_RETURN_ON_ERROR(SymNameOrErr.takeError());
      /// Use in-place called lambda for easier termination of the symbol
      /// lookup
      LUTHIER_RETURN_ON_ERROR([&]() -> llvm::Error {
        for (const auto &LCO : Parent.loaded_code_objects()) {
          auto SymbolIfExists = LCO.getCodeObject().lookupSymbol(*SymNameOrErr);
          if (SymbolIfExists)
            return SymbolIfExists.takeError();

          if (*SymbolIfExists != std::nullopt &&
              (*SymbolIfExists)->getBinding() == llvm::ELF::STB_GLOBAL) {
            if (auto Err = (*SymbolIfExists)->getAddress().moveInto(SymAddr)) {
              return std::move(Err);
            }
            SymAddr += reinterpret_cast<uint64_t>(LCO.getLoadedRegion().data());
            return llvm::Error::success();
          }
        }
        if (auto It = Parent.findExternalSymbol(*SymNameOrErr);
            It != Parent.external_symbol_end()) {
          SymAddr = reinterpret_cast<uint64_t>(It->second);
        }
        return llvm::Error::success();
      }());
    } else {
      LUTHIER_RETURN_ON_ERROR(Sym->getAddress().moveInto(SymAddr));
      SymAddr += LoadBase;
    }
  }

  /// Calculate the addend
  uint64_t Addend = 0;

  llvm::Expected<uint64_t> AddendOrErr = Rel.getAddend();
  /// If there is an error it means that we are dealing with the a REL
  /// section, not a RELA. Typically REL is emitted in shader code (e.g.
  /// Mesa) while RELA is emitted in compute code (e.g. HSA)
  if (auto Err = AddendOrErr.takeError()) {
    /// It is not a fatal error so we consume it first
    llvm::consumeError(std::move(Err));
    /// RELs store their relocation info in the offset location of the
    /// loaded region (and the ELF section)
    switch (RelType) {
    case llvm::ELF::R_AMDGPU_REL16:
      Addend = static_cast<uint64_t>(
          llvm::support::endian::read16le(&LoadedRegion[RelOffset]));
      break;
    case llvm::ELF::R_AMDGPU_ABS32:
    case llvm::ELF::R_AMDGPU_ABS32_LO:
    case llvm::ELF::R_AMDGPU_ABS32_HI:
    case llvm::ELF::R_AMDGPU_REL32:
    case llvm::ELF::R_AMDGPU_REL32_LO:
    case llvm::ELF::R_AMDGPU_REL32_HI:
      Addend = static_cast<uint64_t>(
          llvm::support::endian::read32le(&LoadedRegion[RelOffset]));
      break;
    case llvm::ELF::R_AMDGPU_ABS64:
    case llvm::ELF::R_AMDGPU_REL64:
      Addend = llvm::support::endian::read64le(&LoadedRegion[RelOffset]);
      break;
    default:
      /// Skip GOT and any other unsupported relocations
      break;
    }
  } else {
    Addend = *AddendOrErr;
  }

  switch (RelType) {
  case llvm::ELF::R_AMDGPU_ABS32:
  case llvm::ELF::R_AMDGPU_ABS32_LO:
    if (SymAddr == 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation symbol address is zero; Likely it was not defined");
    llvm::support::endian::write32le(
        &LoadedRegion[RelOffset],
        static_cast<uint32_t>((SymAddr + Addend) & 0xFFFFFFFF));
    break;
  case llvm::ELF::R_AMDGPU_ABS32_HI:
    if (SymAddr == 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation symbol address is zero; Likely it was not defined");
    llvm::support::endian::write32le(
        &LoadedRegion[RelOffset],
        static_cast<uint32_t>((SymAddr + Addend) >> 32));
    break;
  case llvm::ELF::R_AMDGPU_ABS64:
    if (SymAddr == 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation symbol address is zero; Likely it was not defined");
    llvm::support::endian::write64le(&LoadedRegion[RelOffset],
                                     SymAddr + Addend);
    break;
  case llvm::ELF::R_AMDGPU_REL32:
    if (SymAddr == 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation symbol address is zero; Likely it was not defined");
    llvm::support::endian::write32le(&LoadedRegion[RelOffset],
                                     SymAddr + Addend - RelOffset - LoadBase);
    break;
  case llvm::ELF::R_AMDGPU_REL64:
    if (SymAddr == 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation symbol address is zero; Likely it was not defined");
    llvm::support::endian::write64le(&LoadedRegion[RelOffset],
                                     SymAddr + Addend - RelOffset - LoadBase);
    break;
  case llvm::ELF::R_AMDGPU_REL32_LO:
    if (SymAddr == 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation symbol address is zero; Likely it was not defined");
    llvm::support::endian::write32le(
        &LoadedRegion[RelOffset],
        static_cast<uint32_t>((SymAddr + Addend - RelOffset - LoadBase) &
                              0xFFFFFFFF));
    break;
  case llvm::ELF::R_AMDGPU_REL32_HI:
    if (SymAddr == 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation symbol address is zero; Likely it was not defined");
    llvm::support::endian::write32le(
        &LoadedRegion[RelOffset],
        static_cast<uint32_t>((SymAddr + Addend - RelOffset - LoadBase) >> 32));
    break;
  case llvm::ELF::R_AMDGPU_REL16:
    if (SymAddr == 0)
      return LUTHIER_MAKE_GENERIC_ERROR(
          "Relocation symbol address is zero; Likely it was not defined");
    llvm::support::endian::write16le(
        &LoadedRegion[RelOffset],
        static_cast<uint16_t>(((SymAddr + Addend - RelOffset - LoadBase) - 4) /
                              4));
    break;
  case llvm::ELF::R_AMDGPU_RELATIVE64:
    llvm::support::endian::write64le(&LoadedRegion[RelOffset],
                                     Addend + LoadBase);
    break;
  default:
    /// skip any other relocation type
    break;
  }

  return llvm::Error::success();
}

llvm::Error MockAMDGPULoader::defineExternalSymbol(llvm::StringRef Name,
                                                   void *Address) {
  if (IsFinalized)
    return LUTHIER_MAKE_GENERIC_ERROR("Cannot define a new external variable "
                                      "after the loader is finalized");

  if (ExternalSymbols.contains(Name))
    return LUTHIER_MAKE_GENERIC_ERROR(
        llvm::formatv("Symbol {0} is already defined in the loader", Name));

  if (auto [_, InsertionStatus] = ExternalSymbols.insert({Name, Address});
      !InsertionStatus)
    return LUTHIER_MAKE_GENERIC_ERROR("Failed to insert the new symbol "
                                      "definition into the loader's symbol "
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
