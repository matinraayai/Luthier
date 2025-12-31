//===-- IntrinsicProcessorsAnalysis.h ---------------------------*- C++ -*-===//
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
/// Describes the intrinsics processors analysis which provides the intrinsic
/// lowering functions for Luthier intrinsics.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INTRINSIC_PROCESSORS_ANALYSIS_H
#define LUTHIER_TOOLING_INTRINSIC_PROCESSORS_ANALYSIS_H
#include <llvm/IR/Analysis.h>
#include <llvm/IR/PassManager.h>
#include <luthier/Intrinsic/IntrinsicProcessor.h>

namespace luthier {

/// \brief Produces the map which holds the processors for all intrinsics
class IntrinsicsProcessorsAnalysis
    : public llvm::AnalysisInfoMixin<IntrinsicsProcessorsAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<IntrinsicsProcessorsAnalysis>;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend class IntrinsicsProcessorsAnalysis;

    Result() = default;

  public:
    [[nodiscard]] IntrinsicProcessor getProcessor(llvm::StringRef Name) const;

    [[nodiscard]] bool isProcessorRegistered(llvm::StringRef Name) const;

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  IntrinsicsProcessorsAnalysis() = default;

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) { return Result{}; }
};

} // namespace luthier

#endif