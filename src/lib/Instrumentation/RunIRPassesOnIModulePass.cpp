//===-- RunIRPassesOnIModulePass.cpp --------------------------------------===//
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
/// Implements the \c RunIRPassesOnIModulePass class.
//===----------------------------------------------------------------------===//
#include "luthier/Instrumentation/RunIRPassesOnIModulePass.h"
#include "luthier/Instrumentation/IModuleGenerationPass.h"
#include "luthier/Instrumentation/IntrinsicIRLoweringInfoAnalysis.h"
#include "luthier/Instrumentation/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/Instrumentation/WrapperAnalysisPasses.h"

#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TimeProfiler.h>
#include <luthier/Instrumentation/IntrinsicProcessorsAnalysis.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-imodule-ir-passes"

namespace luthier {

llvm::PreservedAnalyses
RunIRPassesOnIModulePass::run(llvm::Module &TargetAppM,
                              llvm::ModuleAnalysisManager &TargetAppMAM) {
  /// Get the target machine of the target application
  auto &TM = const_cast<llvm::TargetMachine &>(
      TargetAppMAM.getResult<llvm::MachineModuleAnalysis>(TargetAppM)
          .getMMI()
          .getTarget());

  auto &IModulePMRes = TargetAppMAM.getResult<IModulePMAnalysis>(TargetAppM);
  // Get the instrumentation analysis managers
  auto &LAM = IModulePMRes.getLAM();
  auto &FAM = IModulePMRes.getFAM();
  auto &CGAM = IModulePMRes.getCGAM();
  auto &IMAM = IModulePMRes.getMAM();
  // Get the instrumentation pass manager
  auto &IMPM = IModulePMRes.getPM();

  // Create a PM Builder for the IR pipeline
  llvm::PassBuilder PB(&TM);
  {
    llvm::TimeTraceScope Scope("Instrumentation Module IR Optimization");
    // Add the Intrinsic Lowering Info analysis pass
    IMAM.registerPass([&]() { return IntrinsicIRLoweringInfoAnalysis(); });
    // Add the Intrinsic processors Map analysis pass
    IMAM.registerPass([&]() { return IntrinsicsProcessorsAnalysis(); });
    // Add the Target app's MAM as an analysis pass
    IMAM.registerPass([&]() {
      return TargetAppModuleAndMAMAnalysis(TargetAppMAM, TargetAppM);
    });
    // Add the analysis for holding the instrumentation point to injected
    // payload mapping
    IMAM.registerPass([&]() { return InjectedPayloadAndInstPointAnalysis(); });

    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(IMAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, IMAM);
    // Add the pass that generates the IR for the instrumentation module
    IMPM.addPass(IModuleGenerationPass());
    // Add the IR optimization pipeline
    IMPM.addPass(PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3));
    // Add the Intrinsic Processing IR stage pass
    IMPM.addPass(ProcessIntrinsicsAtIRLevelPass());
    // Run the scheduled IR passes
    IMPM.run(IModule, IMAM);
  }

  LLVM_DEBUG(

      llvm::dbgs() << "Instrumentation Module after IR optimization:\n";
      IModule.print(llvm::dbgs(), nullptr)

  );

  return llvm::PreservedAnalyses::none();
}
} // namespace luthier