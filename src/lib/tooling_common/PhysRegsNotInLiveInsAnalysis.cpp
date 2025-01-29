//===-- PhysRegsNotInLiveInsAnalysis.cpp ----------------------------------===//
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
/// This file implements the analysis pass which aggregates the accessed
/// physical registers in injected payloads that are not preserved by the
/// live-in set.
//===----------------------------------------------------------------------===//
#include "tooling_common/PhysRegsNotInLiveInsAnalysis.hpp"
#include "luthier/tooling/LRRegisterLiveness.h"
#include "tooling_common/IModuleIRGeneratorPass.hpp"
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/Support/FormatVariadic.h>
#include <luthier/Consts.h>

#include "tooling_common/WrapperAnalysisPasses.hpp"

namespace luthier {

llvm::AnalysisKey PhysRegsNotInLiveInsAnalysis::Key;

PhysRegsNotInLiveInsAnalysis::Result
PhysRegsNotInLiveInsAnalysis::run(llvm::Module &IModule,
                                  llvm::ModuleAnalysisManager &IMAM) {
  auto &IntrinsicIRLoweringInfoMap =
      IMAM.getCachedResult<IntrinsicIRLoweringInfoMapAnalysis>(IModule)
          ->getLoweringInfo();
  const auto &IPIP =
      *IMAM.getCachedResult<InjectedPayloadAndInstPointAnalysis>(IModule);

  auto &MAM =
      IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule).getTargetAppMAM();
  auto &TargetModule = IMAM.getResult<TargetAppModuleAndMAMAnalysis>(IModule)
                           .getTargetAppModule();
  const auto &TM = MAM.getResult<llvm::MachineModuleAnalysis>(TargetModule)
                       .getMMI()
                       .getTarget();

  auto Out = std::make_unique<llvm::LivePhysRegs>(
      *TM.getSubtargetImpl(*IModule.functions().begin())->getRegisterInfo());

  const auto &LRRegLiveness =
      MAM.getResult<LRRegLivenessAnalysis>(TargetModule);

  for (const auto &LoweringInfo : IntrinsicIRLoweringInfoMap) {
    auto &PlaceHolderInlineAsm = LoweringInfo.getPlaceHolderInlineAsm();
    // Check if the Placeholder inline assembly has only one user
    if (!PlaceHolderInlineAsm.hasOneUser()) {
      IModule.getContext().emitError(
          "Expected a single user for a Luthier intrinsic "
          "place holder inline assembly.");
    }

    for (const auto &User : PlaceHolderInlineAsm.users()) {
      if (auto *InlineAsmCallInst = llvm::dyn_cast<llvm::CallInst>(User)) {
        auto IntrinsicUserFunction = InlineAsmCallInst->getFunction();
        if (IntrinsicUserFunction->hasFnAttribute(InjectedPayloadAttribute)) {
          for (const auto &UsedPhysReg : LoweringInfo.accessed_phys_regs()) {
            const auto *MIInsertionPoint = IPIP.at(*IntrinsicUserFunction);
            if (!LRRegLiveness
                     .getLiveInPhysRegsOfMachineInstr(*MIInsertionPoint)
                     ->contains(UsedPhysReg)) {
              if (Out->empty())
                Out->init(*TM.getSubtargetImpl(*IntrinsicUserFunction)
                               ->getRegisterInfo());
              Out->addReg(UsedPhysReg);
            }
          }
        }
      } else
        IModule.getContext().emitError(
            llvm::formatv("Found user of intrinsic inline assembly "
                          "place holder that's not a call function; "
                          "Place holder: {0}, User: {1}",
                          PlaceHolderInlineAsm, User));
    }
  }
  return Result(std::move(Out));
}

} // namespace luthier
