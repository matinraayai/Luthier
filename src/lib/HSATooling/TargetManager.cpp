//===-- TargetManager.cpp - Luthier's LLVM Target Management  -------------===//
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
///
/// \file
/// This file implements Luthier's Target Manager Singleton.
//===----------------------------------------------------------------------===//
#include "../../../include/luthier/HSATooling/TargetManager.h"
#include "luthier/HSA/Agent.h"
#include <AMDGPUTargetMachine.h>
#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCDisassembler/MCDisassembler.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCParser/MCAsmParser.h>
#include <llvm/MC/MCParser/MCTargetAsmParser.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>

namespace luthier {

template <> TargetManager *Singleton<TargetManager>::Instance{nullptr};

TargetManager::TargetManager(
    const rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
        &CoreApiTableSnapshot)
    : Singleton<TargetManager>(), CoreApiTableSnapshot(CoreApiTableSnapshot) {
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUDisassembler();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUTargetMCA();
}

TargetManager::~TargetManager() {
  for (auto &It : LLVMTargetInfo) {
    delete It.second.MRI;
    delete It.second.MAI;
    delete It.second.MII;
    delete It.second.MIA;
    delete It.second.STI;
    delete It.second.IP;
    delete It.second.TargetOptions;
  }
  LLVMTargetInfo.clear();
  llvm::llvm_shutdown();
  Singleton<TargetManager>::~Singleton();
}

llvm::Expected<const TargetInfo &>
TargetManager::getTargetInfo(hsa_isa_t Isa) const {
  if (!LLVMTargetInfo.contains(Isa)) {
    auto Info = LLVMTargetInfo.insert({Isa, TargetInfo()}).first;

    const auto HsaApiTableSnapshot = CoreApiTableSnapshot.getTable();

    auto TT = hsa::isaGetTargetTriple(HsaApiTableSnapshot, Isa);
    LUTHIER_RETURN_ON_ERROR(TT.takeError());

    std::string Error;

    auto Target = llvm::TargetRegistry::lookupTarget(*TT, Error);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        Target, llvm::formatv("Failed to lookup target {0} in LLVM. Reason "
                              "according to LLVM: {1}.",
                              TT->normalize(), Error)));

    auto MRI = Target->createMCRegInfo(*TT);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        MRI, llvm::formatv("Failed to create machine register info for {0}.",
                           TT->getTriple())));

    auto TargetOptions = new llvm::TargetOptions();

    TargetOptions->MCOptions.AsmVerbose = true;

    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        TargetOptions, "Failed to create target options."));

    auto MAI = Target->createMCAsmInfo(*MRI, *TT, TargetOptions->MCOptions);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        MAI,
        llvm::formatv(
            "Failed to create MCAsmInfo from target {0} for Target Triple {1}.",
            Target, TT->getTriple())));

    auto MII = Target->createMCInstrInfo();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        MII,
        llvm::formatv("Failed to create MCInstrInfo from target {0}", Target)));

    auto MIA = Target->createMCInstrAnalysis(MII);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        MIA, llvm::formatv("Failed to create MCInstrAnalysis for target {0}.",
                           Target)));

    auto CPU = hsa::isaGetGPUName(HsaApiTableSnapshot, Isa);
    LUTHIER_RETURN_ON_ERROR(CPU.takeError());

    auto FeatureString = hsa::isaGetSubTargetFeatures(HsaApiTableSnapshot, Isa);
    LUTHIER_RETURN_ON_ERROR(FeatureString.takeError());

    auto STI =
        Target->createMCSubtargetInfo(*TT, *CPU, FeatureString->getString());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        STI, llvm::formatv("Failed to create MCSubTargetInfo from target {0} "
                           "for triple {1}, CPU {2}, with feature string {3}",
                           Target, TT->getTriple(), *CPU,
                           FeatureString->getString())));

    auto IP = Target->createMCInstPrinter(
        llvm::Triple(*TT), MAI->getAssemblerDialect(), *MAI, *MII, *MRI);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        IP,
        llvm::formatv(
            "Failed to create MCInstPrinter from Target {0} for Triple {1}.",
            Target, TT->getTriple())));

    Info->second.Target = Target;
    Info->second.MRI = MRI;
    Info->second.MAI = MAI;
    Info->second.MII = MII;
    Info->second.MIA = MIA;
    Info->second.STI = STI;
    Info->second.IP = IP;
    Info->second.TargetOptions = TargetOptions;
  }
  return LLVMTargetInfo[Isa];
}

llvm::Expected<std::unique_ptr<llvm::GCNTargetMachine>>
TargetManager::createTargetMachine(
    hsa_isa_t ISA, const llvm::TargetOptions &TargetOptions) const {
  const auto HsaApiTable = CoreApiTableSnapshot.getTable();
  auto TT = hsa::isaGetTargetTriple(HsaApiTable, ISA);
  LUTHIER_RETURN_ON_ERROR(TT.takeError());
  std::string Error;
  auto Target = llvm::TargetRegistry::lookupTarget(*TT, Error);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      Target,
      llvm::formatv(
          "Failed to get target {0} from LLVM. Error according to LLVM: {1}.",
          TT->normalize(), Error)));
  auto CPU = hsa::isaGetGPUName(HsaApiTable, ISA);
  LUTHIER_RETURN_ON_ERROR(CPU.takeError());

  auto FeatureString = hsa::isaGetSubTargetFeatures(HsaApiTable, ISA);
  LUTHIER_RETURN_ON_ERROR(FeatureString.takeError());
  return std::unique_ptr<llvm::GCNTargetMachine>(
      reinterpret_cast<llvm::GCNTargetMachine *>(Target->createTargetMachine(
          llvm::Triple(*TT), *CPU, FeatureString->getString(), TargetOptions,
          llvm::Reloc::PIC_)));
}

} // namespace luthier
