//===-- WrapperAnalysisPasses.h ---------------------------------*- C++ -*-===//
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
/// \file WrapperAnalysisPasses.h
/// Describes a set of analysis passes that wrap around pass managers and
/// analysis managers in order for instrumentation passes to access them.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_WRAPPER_ANALYSIS_PASSES_H
#define LUTHIER_TOOL_CODE_GEN_WRAPPER_ANALYSIS_PASSES_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "luthier/ToolCodeGen/LegacyPassSupport.h"
#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>

namespace luthier {

/// \brief an analysis used by the instrumentation module passes that gives
/// access to the Target App's \c llvm::Module and
/// \c llvm::ModuleAnalysisManager
class TargetAppModuleAndMAMAnalysis
    : public llvm::AnalysisInfoMixin<TargetAppModuleAndMAMAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<TargetAppModuleAndMAMAnalysis>;

  static llvm::AnalysisKey Key;

  llvm::ModuleAnalysisManager &TargetAppMAM;

  llvm::Module &TargetAppModule;

public:
  class Result {
    friend class TargetAppModuleAndMAMAnalysis;

    llvm::ModuleAnalysisManager &MAM;

    llvm::Module &M;

    Result(llvm::ModuleAnalysisManager &MAM, llvm::Module &M)
        : MAM(MAM), M(M) {};

  public:
    [[nodiscard]] llvm::ModuleAnalysisManager &getTargetAppMAM() const {
      return MAM;
    }

    [[nodiscard]] llvm::Module &getTargetAppModule() const { return M; }

    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  explicit TargetAppModuleAndMAMAnalysis(
      llvm::ModuleAnalysisManager &TargetAppMAM, llvm::Module &TargetAppModule)
      : TargetAppMAM(TargetAppMAM), TargetAppModule(TargetAppModule) {};

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return {TargetAppMAM, TargetAppModule};
  }
};

} // namespace luthier

#endif