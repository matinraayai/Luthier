//===-- MockAMDGPULoader.cpp ------------------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file
/// Implements the \c MockAMDGPULoader and \c MockLoadedCodeObject
/// classes.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/MockAMDGPULoader.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Object/ObjectFileUtils.h"

#include <hsa/amd_hsa_kernel_code.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/BinaryFormat/ELF.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCTargetOptions.h>

namespace luthier {

MockLoadedCodeObject::MockLoadedCodeObject(MockAMDGPULoader &Owner,
                                           const llvm::MemoryBuffer &Elf,
                                           llvm::Error &Err)
    : Parent(Owner) {
  llvm::ErrorAsOutParameter EAO(Err);

  /// Parse the code object
  Err =
      object::AMDGCNObjectFile::createAMDGCNObjectFile(Elf).moveInto(this->Elf);
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
                          "defined by an earlier code object",
                          *SymNameOrErr));
        return;
      }
    }
    if (auto It = Parent.findExternalSymbol(*SymNameOrErr);
        It != Parent.external_symbol_end()) {
      Err = LUTHIER_MAKE_GENERIC_ERROR(
          llvm::formatv("Code object defines symbol {0} already defined as an "
                        "external symbol in its parent executable",
                        *SymNameOrErr));
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

  for (const auto &Phdr : *ProgramHeadersOrErr) {
    if (Phdr.p_type == llvm::ELF::PT_LOAD) {
      PTLoadSegments.push_back(&Phdr);
    }
  }

  if (PTLoadSegments.empty()) {
    Err = LUTHIER_MAKE_GENERIC_ERROR("The code object has no PT_LOAD sections");
    return;
  }

  /// Even though the load segments should be  pre-sorted w.r.t their
  /// virtual address, we take a precaution and sort it anyway
  llvm::sort(PTLoadSegments, [](const auto *Lhs, const auto *Rhs) {
    return Lhs->p_vaddr < Rhs->p_vaddr;
  });

  uint64_t Size =
      PTLoadSegments.back()->p_vaddr + PTLoadSegments.back()->p_memsz;

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
    std::memcpy(&LoadedRegion[PTLoadSegment->p_vaddr],
                &Elf.getBufferStart()[PTLoadSegment->p_offset],
                PTLoadSegment->p_filesz);
  }
}

MockLoadedCodeObject::~MockLoadedCodeObject() {
  if (LoadedRegion.data())
    ::operator delete[](LoadedRegion.data(),
                        std::align_val_t{AMD_ISA_ALIGN_BYTES});
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
          if (auto E = SymbolIfExists.takeError())
            return E;

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
MockAMDGPULoader::loadCodeObject(const llvm::MemoryBuffer &CodeObject) {

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

llvm::AnalysisKey MockAMDGPULoaderAnalysis::Key;


static void printBytes(llvm::ArrayRef<uint8_t> Bytes, llvm::raw_ostream &OS) {
  llvm::interleave(
      Bytes, [&](uint8_t Byte) { OS << llvm::format("%02X", Byte); },
      [&]() { OS << " "; });
}

static void printELFType(const llvm::object::ELF64LE::Phdr &Phdr,
                         llvm::raw_ostream &OS) {
  switch (Phdr.p_type) {
  case llvm::ELF::PT_DYNAMIC:
    OS << "DYNAMIC";
    break;
  case llvm::ELF::PT_GNU_EH_FRAME:
    OS << "EH_FRAME";
    break;
  case llvm::ELF::PT_GNU_RELRO:
    OS << "RELRO";
    break;
  case llvm::ELF::PT_GNU_PROPERTY:
    OS << "PROPERTY";
    break;
  case llvm::ELF::PT_GNU_STACK:
    OS << "STACK";
    break;
  case llvm::ELF::PT_GNU_SFRAME:
    OS << "SFRAME";
    break;
  case llvm::ELF::PT_INTERP:
    OS << "INTERP";
    break;
  case llvm::ELF::PT_LOAD:
    OS << "LOAD";
    break;
  case llvm::ELF::PT_NOTE:
    OS << "NOTE";
    break;
  case llvm::ELF::PT_OPENBSD_BOOTDATA:
    OS << "OPENBSD_BOOTDATA";
    break;
  case llvm::ELF::PT_OPENBSD_MUTABLE:
    OS << "OPENBSD_MUTABLE";
    break;
  case llvm::ELF::PT_OPENBSD_NOBTCFI:
    OS << "OPENBSD_NOBTCFI";
    break;
  case llvm::ELF::PT_OPENBSD_RANDOMIZE:
    OS << "OPENBSD_RANDOMIZE";
    break;
  case llvm::ELF::PT_OPENBSD_SYSCALLS:
    OS << "OPENBSD_SYSCALLS";
    break;
  case llvm::ELF::PT_OPENBSD_WXNEEDED:
    OS << "OPENBSD_WXNEEDED";
    break;
  case llvm::ELF::PT_PHDR:
    OS << "PHDR";
    break;
  case llvm::ELF::PT_TLS:
    OS << "TLS";
    break;
  default:
    OS << "UNKNOWN";
  }
}

AMDGPUMockLoaderPrinter::AMDGPUMockLoaderPrinter(llvm::raw_ostream &OS)
    : OS(OS) {}

llvm::PreservedAnalyses
AMDGPUMockLoaderPrinter::run(llvm::Module &M,
                             llvm::ModuleAnalysisManager &MAM) {
  llvm::LLVMContext &Ctx = M.getContext();
  /// Get the mock loader analysis for printing
  MockAMDGPULoader &Loader =
      MAM.getResult<MockAMDGPULoaderAnalysis>(M).getLoader();

  OS << "Num Code Objects: " << Loader.loaded_code_objects_size() << "\n";
  OS << "Loaded Code Object Contents:\n";

  for (const auto &[CodeObjectIdx, LCO] :
       llvm::enumerate(Loader.loaded_code_objects())) {
    auto TargetTripleOrErr =
        object::getObjectFileTargetTuple(LCO.getCodeObject());

    if (auto Err = TargetTripleOrErr.takeError()) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    auto [TT, CPU, FS] = *TargetTripleOrErr;

    std::string Error;
    const llvm::Target *Target = llvm::TargetRegistry::lookupTarget(TT, Error);
    if (auto Err = LUTHIER_GENERIC_ERROR_CHECK(
            Target, llvm::formatv("Failed to lookup target {0} in LLVM. Reason "
                                  "according to LLVM: {1}.",
                                  TT.normalize(), Error))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    auto MRI =
        std::unique_ptr<llvm::MCRegisterInfo>(Target->createMCRegInfo(TT));
    if (auto Err = LUTHIER_GENERIC_ERROR_CHECK(
            MRI.get(),
            llvm::formatv("Failed to create machine register info for {0}.",
                          TT.getTriple()))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    llvm::MCTargetOptions Options{};

    auto MAI = std::unique_ptr<llvm::MCAsmInfo>(
        Target->createMCAsmInfo(*MRI, TT, Options));
    if (auto Err = LUTHIER_GENERIC_ERROR_CHECK(
            MAI, llvm::formatv("Failed to create MCAsmInfo from target "
                               "{0} for Target Triple {1}.",
                               Target, TT.getTriple()))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    auto MII = std::unique_ptr<llvm::MCInstrInfo>(Target->createMCInstrInfo());

    if (auto Err = LUTHIER_GENERIC_ERROR_CHECK(
            MII, llvm::formatv("Failed to create MCInstrInfo from target {0}",
                               Target))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    auto MIA = std::unique_ptr<llvm::MCInstrAnalysis>(
        Target->createMCInstrAnalysis(MII.get()));
    if (auto Err = LUTHIER_GENERIC_ERROR_CHECK(
            MIA,
            llvm::formatv("Failed to create MCInstrAnalysis for target {0}.",
                          Target))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    auto STI = std::unique_ptr<llvm::MCSubtargetInfo>(
        Target->createMCSubtargetInfo(TT, CPU, FS.getString()));
    if (auto Err = LUTHIER_GENERIC_ERROR_CHECK(
            STI,
            llvm::formatv("Failed to create MCSubTargetInfo from target {0} "
                          "for triple {1}, CPU {2}, with feature string {3}",
                          Target, TT.getTriple(), CPU, FS.getString()))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    auto IP = std::unique_ptr<llvm::MCInstPrinter>(Target->createMCInstPrinter(
        TT, MAI->getAssemblerDialect(), *MAI, *MII, *MRI));
    if (auto Err = LUTHIER_GENERIC_ERROR_CHECK(
            IP, llvm::formatv("Failed to create MCInstPrinter from Target "
                              "{0} for Triple {1}.",
                              Target, TT.normalize()))) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    llvm::MCContext MCCtx(TT, *MAI, *MRI, *STI);

    auto DisAsm = std::unique_ptr<llvm::MCDisassembler>(
        Target->createMCDisassembler(*STI, MCCtx));

    if (auto Err = LUTHIER_GENERIC_ERROR_CHECK(
            DisAsm.get(), "Failed to create a disassembler")) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }

    uint64_t LoadBase =
        reinterpret_cast<uint64_t>(LCO.getLoadedRegion().data());

    OS << "Loaded Code Object #" << CodeObjectIdx << "\n";
    OS << llvm::formatv("- Load Base Address: {0:x}\n", LoadBase);
    OS << llvm::formatv("- Load size: {0:x}\n", LCO.getLoadedRegion().size());
    OS << "- Segment Program Headers: \n";
    for (const auto &[PHIdx, PH] : llvm::enumerate(LCO.getLoadSegments())) {
      OS.indent(2) << "- Idx: " << PHIdx << "\n";
      OS.indent(2) << "- Type: ";
      printELFType(*PH, OS);
      OS << "\n";
      OS.indent(2) << llvm::formatv("- Offset: {0:x}\n",
                                    static_cast<uint64_t>(PH->p_offset));
      OS.indent(2) << llvm::formatv("- VAddr: {0:x}\n",
                                    static_cast<uint64_t>(PH->p_vaddr));
      OS.indent(2) << llvm::formatv("- PAddr: {0:x}\n",
                                    static_cast<uint64_t>(PH->p_paddr));
      OS.indent(2) << llvm::format("- Alignment: 2**%u",
                                   llvm::countr_zero<uint64_t>(PH->p_align))
                   << "\n";
      OS.indent(2) << llvm::formatv("- Filesz: {0:x}\n",
                                    static_cast<uint64_t>(PH->p_filesz));
      OS.indent(2) << llvm::formatv("- Memsz: {0:x}\n",
                                    static_cast<uint64_t>(PH->p_memsz));
      OS.indent(2) << "- Flags: "
                   << ((PH->p_flags & llvm::ELF::PF_R) ? "r" : "-")
                   << ((PH->p_flags & llvm::ELF::PF_W) ? "w" : "-")
                   << ((PH->p_flags & llvm::ELF::PF_X) ? "x" : "-") << "\n";
      OS.indent(2) << "----------\n";
    }

    OS << "- Kernels:\n";

    llvm::Error Err = llvm::Error::success();
    for (object::AMDGCNKernelDescSymbolRef KDSymbol :
         LCO.getCodeObject().kernel_descriptors(Err)) {
      llvm::Expected<llvm::StringRef> KernelNameOrErr = KDSymbol.getName();
      if (auto NameErr = KernelNameOrErr.takeError()) {
        Ctx.emitError(llvm::toString(std::move(NameErr)));
        return llvm::PreservedAnalyses::all();
      }
      llvm::Expected<uint64_t> AddrOrErr = KDSymbol.getAddress();
      if (auto AddrErr = AddrOrErr.takeError()) {
        Ctx.emitError(llvm::toString(std::move(AddrErr)));
        return llvm::PreservedAnalyses::all();
      }
      OS.indent(2) << llvm::formatv("- {0}, {1:x}\n", *KernelNameOrErr,
                                    *AddrOrErr);
    }
    if (Err) {
      Ctx.emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }
    OS << "----------\n";
    OS << "- Loaded Contents:\n";
    for (const auto &[PHIdx, PH] : llvm::enumerate(LCO.getLoadSegments())) {
      OS << "Segment #" << PHIdx << ":\n";
      uint64_t SegmentCurrAddr =
          reinterpret_cast<uint64_t>(LCO.getLoadedRegion().data()) +
          PH->p_vaddr;
      uint64_t SegmentEndAddr = SegmentCurrAddr + PH->p_filesz;
      if ((PH->p_flags & llvm::ELF::PF_X)) {
        uint64_t MaxReadSize = MAI->getMaxInstLength();
        while (SegmentCurrAddr < SegmentEndAddr) {
          OS.indent(2) << llvm::formatv("// {0:x}: ",
                                        SegmentCurrAddr - LoadBase);
          size_t ReadSize = (SegmentCurrAddr + MaxReadSize) < SegmentEndAddr
                                ? MaxReadSize
                                : SegmentEndAddr - SegmentCurrAddr;
          llvm::MCInst Inst;
          size_t InstSize{0};
          llvm::ArrayRef ReadBytes = {
              reinterpret_cast<uint8_t *>(SegmentCurrAddr), ReadSize};

          auto DecodeResult = DisAsm->getInstruction(
              Inst, InstSize, ReadBytes, SegmentCurrAddr, llvm::nulls());
          if (DecodeResult != llvm::MCDisassembler::Success) {
            Ctx.emitError(
                llvm::toString(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                    "Failed to disassemble instruction at address {0:x}",
                    SegmentCurrAddr))));
            break;
          }
          printBytes(llvm::ArrayRef(ReadBytes.data(), InstSize), OS);
          OS << "\n  |->";
          IP->printInst(&Inst, SegmentCurrAddr, "", *STI, OS);
          OS << "\n";
          SegmentCurrAddr += InstSize;
        }

      } else {
        while (SegmentCurrAddr < SegmentEndAddr) {
          OS.indent(2);
          OS << llvm::formatv("// {0:x}: ", SegmentCurrAddr - LoadBase);
          printBytes(
              llvm::ArrayRef(reinterpret_cast<uint8_t *>(SegmentCurrAddr), 8),
              OS);
          OS << "\n";
          SegmentCurrAddr += 8;
        }
      }
    }
  }
  return llvm::PreservedAnalyses::all();
};


} // namespace luthier
