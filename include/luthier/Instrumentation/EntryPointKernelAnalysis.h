//===-- EntryPointKernelAnalysis.h ------------------------------*- C++ -*-===//
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
/// \file
/// Defines the \c EntryPointKernelAnalysis class, which provide the symbol
/// of the entry kernel executed in the current instrumentation task.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_ENTRY_POINT_KERNEL_ANALYSIS_H
#define LUTHIER_INSTRUMENTATION_ENTRY_POINT_KERNEL_ANALYSIS_H
#include <llvm/IR/PassManager.h>
#include <luthier/Object/AMDGCNObjectFile.h>

namespace luthier {
class EntryPointKernelAnalysis
    : public llvm::AnalysisInfoMixin<EntryPointKernelAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {

    object::AMDGCNKernelDescSymbolRef KernelSymbol;

    explicit Result(object::AMDGCNKernelDescSymbolRef KernelSymbol)
        : KernelSymbol(KernelSymbol) {};

  public:
    object::AMDGCNKernelDescSymbolRef getSymbol() const { return KernelSymbol; }

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  Result KernelSymbolRes;

  explicit EntryPointKernelAnalysis(
      object::AMDGCNKernelDescSymbolRef KernelSymbol)
      : KernelSymbolRes(KernelSymbol) {};

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return KernelSymbolRes;
  }
};
} // namespace luthier

#endif
