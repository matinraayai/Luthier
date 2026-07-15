//===-- TargetAndInstrumentationModule.cpp ----------------------*- C++ -*-===//
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
/// Implements out-of-line definitions for \c TargetAndInstrumentationModule.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TargetAndInstrumentationModule.h"

// The invalidate specializations must be defined in namespace llvm (the
// namespace enclosing InnerAnalysisManagerProxy); see the header for details.
namespace llvm {

using luthier::FunctionAnalysisManagerTAIModuleProxy;
using luthier::MachineFunctionAnalysisManagerTAIModuleProxy;
using luthier::ModuleAnalysisManagerTAIModuleProxy;
using luthier::TargetAndInstrumentationAnalysisManager;
using luthier::TargetAndInstrumentationModule;

template <>
bool ModuleAnalysisManagerTAIModuleProxy::Result::invalidate(
    TargetAndInstrumentationModule &B, const llvm::PreservedAnalyses &PA,
    TargetAndInstrumentationAnalysisManager::Invalidator &Inv) {
  auto PAC = PA.getChecker<ModuleAnalysisManagerTAIModuleProxy>();
  if (!PAC.preserved() &&
      !PAC.preservedSet<
          llvm::AllAnalysesOn<TargetAndInstrumentationModule>>()) {
    InnerAM->clear(B.getInstrumentationModule(),
                   B.getInstrumentationModule().getName());
    InnerAM->clear(B.getTargetModule(), B.getTargetModule().getName());
    return true; // the proxy result itself is now invalid
  }
  return false;
}

template <>
bool FunctionAnalysisManagerTAIModuleProxy::Result::invalidate(
    TargetAndInstrumentationModule &B, const llvm::PreservedAnalyses &PA,
    TargetAndInstrumentationAnalysisManager::Invalidator &Inv) {
  auto PAC = PA.getChecker<FunctionAnalysisManagerTAIModuleProxy>();
  if (!PAC.preserved() &&
      !PAC.preservedSet<
          llvm::AllAnalysesOn<TargetAndInstrumentationModule>>()) {
    InnerAM->clear();
    return true;
  }
  return false;
}

template <>
bool MachineFunctionAnalysisManagerTAIModuleProxy::Result::invalidate(
    TargetAndInstrumentationModule &B, const llvm::PreservedAnalyses &PA,
    TargetAndInstrumentationAnalysisManager::Invalidator &Inv) {
  auto PAC = PA.getChecker<MachineFunctionAnalysisManagerTAIModuleProxy>();
  if (!PAC.preserved() &&
      !PAC.preservedSet<
          llvm::AllAnalysesOn<TargetAndInstrumentationModule>>()) {
    InnerAM->clear();
    return true;
  }
  return false;
}

} // namespace llvm
