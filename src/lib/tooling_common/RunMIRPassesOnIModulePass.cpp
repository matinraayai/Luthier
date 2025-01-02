//===-- RunMIRPassesOnIModulePass.cpp -------------------------------------===//
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
/// This file implements the <tt>RunMIRPassesOnIModulePass</tt>.
//===----------------------------------------------------------------------===//
#include "tooling_common/RunMIRPassesOnIModulePass.hpp"
#include "tooling_common/InjectedPayloadPEIPass.hpp"
#include "tooling_common/IntrinsicMIRLoweringPass.hpp"
#include "tooling_common/PhysicalRegAccessVirtualizationPass.hpp"
#include "tooling_common/WrapperAnalysisPasses.hpp"
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Support/TimeProfiler.h>

namespace luthier {

llvm::PreservedAnalyses
RunMIRPassesOnIModulePass::run(llvm::Module &TargetAppM,
                               llvm::ModuleAnalysisManager &TargetMAM) {
  auto &IMAM =
      TargetMAM.getCachedResult<IModulePMAnalysis>(TargetAppM)->getMAM();
  // Trace for profiling
  llvm::TimeTraceScope Scope("Instrumentation Module MIR CodeGen Optimization");

  // Target library info pass, required by the code gen pipeline
  llvm::TargetLibraryInfoImpl TLII(llvm::Triple(IModule.getTargetTriple()));

  ILegacyPM.add(new llvm::TargetLibraryInfoWrapperPass(TLII));

  ILegacyPM.add(new IModuleMAMWrapperPass(&IMAM));

  auto *TPC = TM.createPassConfig(ILegacyPM);

  TPC->setDisableVerify(true);

  ILegacyPM.add(TPC);

  ILegacyPM.add(&IMMIWP);

  TPC->addISelPasses();

  auto PhysRegPass = new PhysicalRegAccessVirtualizationPass();

  ILegacyPM.add(PhysRegPass);
  ILegacyPM.add(new IntrinsicMIRLoweringPass());
  TPC->insertPass(&llvm::PrologEpilogCodeInserterID,
                  new InjectedPayloadPEIPass(*PhysRegPass));
  TPC->addMachinePasses();

  TPC->setInitialized();

  ILegacyPM.run(IModule);

  return llvm::PreservedAnalyses::all();
}
} // namespace luthier