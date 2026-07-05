//===-- TargetAndInstrumentationModule.h ------------------------*- C++ -*-===//
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
/// Defines \c TargetAndInstrumentationModule, an IR unit that pairs
/// the instrumentation module with the target-app module, together with the
/// pass/analysis-manager machinery needed to run passes over them under
/// LLVM's Pass Manager.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_INSTRUMENTATION_BUNDLE_H
#define LUTHIER_TOOL_CODE_GEN_INSTRUMENTATION_BUNDLE_H
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <utility>

namespace luthier {

/// IR Unit that encompasses both the instrumentation modules and target modules
class TargetAndInstrumentationModule {
  llvm::Module &TargetModule;

  llvm::Module &IModule;

public:
  TargetAndInstrumentationModule(llvm::Module &IModule,
                                 llvm::Module &TargetModule)
      : TargetModule(TargetModule), IModule(IModule) {}

  TargetAndInstrumentationModule(const TargetAndInstrumentationModule &) =
      delete;
  TargetAndInstrumentationModule &
  operator=(const TargetAndInstrumentationModule &) = delete;

  [[nodiscard]] llvm::Module &getInstrumentationModule() const {
    return IModule;
  }

  [[nodiscard]] llvm::Module &getTargetModule() const { return TargetModule; }

  [[nodiscard]] llvm::StringRef getName() const {
    return TargetModule.getName();
  }
};

using TargetAndInstrumentationAnalysisManager =
    llvm::AnalysisManager<TargetAndInstrumentationModule>;

using TargetAndInstrumentationPassManager =
    llvm::PassManager<TargetAndInstrumentationModule>;

//===----------------------------------------------------------------------===//
// Cross-level analysis-manager proxies
//===----------------------------------------------------------------------===//

using ModuleAnalysisManagerTAIModuleProxy =
    llvm::InnerAnalysisManagerProxy<llvm::ModuleAnalysisManager,
                                    TargetAndInstrumentationModule>;

using TAIModuleAnalysisManagerModuleProxy =
    llvm::OuterAnalysisManagerProxy<TargetAndInstrumentationAnalysisManager,
                                    llvm::Module>;

using FunctionAnalysisManagerTAIModuleProxy =
    llvm::InnerAnalysisManagerProxy<llvm::FunctionAnalysisManager,
                                    TargetAndInstrumentationModule>;

using MachineFunctionAnalysisManagerTAIModuleProxy =
    llvm::InnerAnalysisManagerProxy<llvm::MachineFunctionAnalysisManager,
                                    TargetAndInstrumentationModule>;

using TAIModuleAnalysisManagerFunctionProxy =
    llvm::OuterAnalysisManagerProxy<TargetAndInstrumentationAnalysisManager,
                                    llvm::Function>;

using TAIModuleAnalysisManagerMachineFunctionProxy =
    llvm::OuterAnalysisManagerProxy<TargetAndInstrumentationAnalysisManager,
                                    llvm::MachineFunction>;

} // namespace luthier

//===----------------------------------------------------------------------===//
// Explicit specializations of the inner proxies' invalidation hooks.
//===----------------------------------------------------------------------===//
//
// These MUST live in \c namespace llvm: an explicit specialization of a member
// has to be declared in a namespace enclosing the specialized template.
namespace llvm {

template <>
bool luthier::ModuleAnalysisManagerTAIModuleProxy::Result::invalidate(
    luthier::TargetAndInstrumentationModule &B, const PreservedAnalyses &PA,
    luthier::TargetAndInstrumentationAnalysisManager::Invalidator &Inv);

template <>
bool luthier::FunctionAnalysisManagerTAIModuleProxy::Result::invalidate(
    luthier::TargetAndInstrumentationModule &B, const PreservedAnalyses &PA,
    luthier::TargetAndInstrumentationAnalysisManager::Invalidator &Inv);

template <>
bool luthier::MachineFunctionAnalysisManagerTAIModuleProxy::Result::invalidate(
    luthier::TargetAndInstrumentationModule &B, const PreservedAnalyses &PA,
    luthier::TargetAndInstrumentationAnalysisManager::Invalidator &Inv);

} // namespace llvm

#endif
