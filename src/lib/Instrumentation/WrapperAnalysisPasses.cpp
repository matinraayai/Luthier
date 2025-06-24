//===-- WrapperAnalysisPasses.cpp -----------------------------------------===//
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
/// Implements a set of analysis passes that provide commonly used data
/// structures by the instrumentation passes in Luthier.
//===----------------------------------------------------------------------===//
#include <luthier/Instrumentation/WrapperAnalysisPasses.h>

namespace luthier {

llvm::AnalysisKey IntrinsicIRLoweringInfoMapAnalysis::Key;

llvm::AnalysisKey TargetAppModuleAndMAMAnalysis::Key;

llvm::AnalysisKey IModulePMAnalysis::Key;

static void *
initializeIModuleMAMWrapperPassPassOnce(llvm::PassRegistry &Registry) {
  auto *PI = new llvm::PassInfo("imam-wrapper-pass",
                                "Instrumentation Module MAM Wrapper Pass",
                                (void *)&IModulePMAnalysis::ID,
                                static_cast<llvm::PassInfo::NormalCtor_t>(
                                    llvm::callDefaultCtor<IModulePMAnalysis>),
                                false, true);
  Registry.registerPass(*PI, true);
  return PI;
}

static std::once_flag InitializeIModulePMAnalysisPassFlag;
void initializeIModulePMAnalysisPass(llvm::PassRegistry &Registry) {
  std::call_once(InitializeIModulePMAnalysisPassFlag,
                 initializeIModuleMAMWrapperPassPassOnce, std::ref(Registry));
}

char IModuleMAMWrapperPass::ID;

IModuleMAMWrapperPass::IModuleMAMWrapperPass(llvm::ModuleAnalysisManager *IMAM)
    : llvm::ImmutablePass(ID), IMAM(*IMAM) {
  initializeIModulePMAnalysisPass(*llvm::PassRegistry::getPassRegistry());
}

} // namespace luthier