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
/// Implements the \c RunMIRPassesOnIModulePass class.
//===----------------------------------------------------------------------===//
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Support/TimeProfiler.h>
#include <luthier/Instrumentation/InjectedPayloadPEIPass.h>
#include <luthier/Instrumentation/IntrinsicMIRLoweringPass.h>
#include <luthier/Instrumentation/PhysicalRegAccessVirtualizationPass.h>
#include <luthier/Instrumentation/RunMIRPassesOnIModulePass.h>
#include <luthier/Instrumentation/WrapperAnalysisPasses.h>

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

  ILegacyPM.add(new IModuleAnalysisWrapperPass(&IMAM));

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