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
/// This file implements the <tt>RunIRPassesOnIModulePass</tt>.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/RunIRPassesOnIModulePass.h"
#include "luthier/Tooling/IModuleIRGeneratorPass.h"
#include "luthier/Tooling/IntrinsicProcessorsAnalysis.h"
#include "luthier/Tooling/PhysRegsNotInLiveInsAnalysis.h"
#include "luthier/Tooling/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/Tooling/WrapperAnalysisPasses.h"
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
#include <llvm/Support/TimeProfiler.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-imodule-ir-passes"

namespace luthier {

RunIRPassesOnIModulePass::RunIRPassesOnIModulePass(
    const InstrumentationTask &Task,
    const llvm::StringMap<IntrinsicProcessor> &IntrinsicProcessors,
    llvm::GCNTargetMachine &TM, llvm::Module &IModule)
    : TM(TM), Task(Task), IModule(IModule),
      IntrinsicProcessors(IntrinsicProcessors) {}

llvm::PreservedAnalyses
RunIRPassesOnIModulePass::run(llvm::Module &TargetAppM,
                              llvm::ModuleAnalysisManager &TargetAppMAM) {
  auto &IModulePMRes = TargetAppMAM.getResult<IModulePMAnalysis>(TargetAppM);
  // Get the instrumentation analysis managers
  auto &LAM = IModulePMRes.getLAM();
  auto &FAM = IModulePMRes.getFAM();
  auto &CGAM = IModulePMRes.getCGAM();
  auto &IMAM = IModulePMRes.getMAM();
  // Get the instrumentation pass manager
  auto &IMPM = IModulePMRes.getPM();

  llvm::PassInstrumentationCallbacks PIC;
  llvm::StandardInstrumentations SI(IModule.getContext(), true);

  // Create a PM Builder for the IR pipeline
  llvm::PassBuilder PB(&TM);
  {
    llvm::TimeTraceScope Scope("Instrumentation Module IR Optimization");
    SI.registerCallbacks(PIC, &IMAM);
    // Add the Intrinsic Lowering Info analysis pass
    IMAM.registerPass([&]() { return IntrinsicIRLoweringInfoMapAnalysis(); });
    // Add the Intrinsic processors Map analysis pass
    IMAM.registerPass([&]() { return IntrinsicsProcessorsAnalysis(); });
    // Add the analysis that accumulates the physical registers accessed that
    // are not in live-ins sets
    IMAM.registerPass([&]() { return PhysRegsNotInLiveInsAnalysis(); });
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
    IMPM.addPass(IModuleIRGeneratorPass(Task));
    // Add the IR optimization pipeline
    IMPM.addPass(PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3));
    // Add the Intrinsic Processing IR stage pass
    IMPM.addPass(ProcessIntrinsicsAtIRLevelPass(TM));
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