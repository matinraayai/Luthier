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
/// This file implements a set of analysis passes that wrap around data
/// structures commonly used by the instrumentation passes in Luthier.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/WrapperAnalysisPasses.h"

namespace luthier {

llvm::AnalysisKey IntrinsicIRLoweringInfoMapAnalysis::Key;

llvm::AnalysisKey TargetAppModuleAndMAMAnalysis::Key;

llvm::AnalysisKey IModulePMAnalysis::Key;

LUTHIER_INITIALIZE_LEGACY_PASS_BODY(IModuleMAMWrapperPass,
                                    "Instrumentation Module MAM Wrapper Pass",
                                    "imam-wrapper-pass", false, true);

char IModuleMAMWrapperPass::ID;

IModuleMAMWrapperPass::IModuleMAMWrapperPass(llvm::ModuleAnalysisManager *IMAM)
    : llvm::ImmutablePass(ID), IMAM(*IMAM) {}

} // namespace luthier