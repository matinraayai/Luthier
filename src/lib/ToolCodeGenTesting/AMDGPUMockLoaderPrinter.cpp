//===-- AMDGPUMockLoaderPrinter.cpp ---------------------------------------===//
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
/// \file AMDGPUMockLoaderPrinter.cpp
/// Implements the \c AMDGPUMockLoaderPrinter class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGenTesting/AMDGPUMockLoaderPrinter.h"
#include "luthier/Object/ObjectFileUtils.h"
#include "luthier/ToolCodeGenTesting/MockAMDGPULoader.h"
#include <llvm/BinaryFormat/ELF.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCTargetOptions.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>

namespace luthier {

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