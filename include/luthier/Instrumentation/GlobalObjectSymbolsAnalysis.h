//===-- GlobalObjectSymbolsAnalysis.h ---------------------------*- C++ -*-===//
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
/// Describes the \c GlobalValueSymbolsAnalysis which provides a
/// mapping between the global objects inside a module and object symbols
/// they correspond to. It is used for both instrumentation and target
/// modules.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_GLOBAL_OBJECT_SYMBOLS_ANALYSIS_H
#define LUTHIER_INSTRUMENTATION_GLOBAL_OBJECT_SYMBOLS_ANALYSIS_H
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Object/ObjectFile.h>

namespace luthier {

class GlobalObjectSymbolsAnalysis
    : public llvm::AnalysisInfoMixin<GlobalObjectSymbolsAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend class GlobalObjectSymbolsAnalysis;

    llvm::DenseMap<const llvm::GlobalObject &, llvm::object::SymbolRef>
        GlobalObjectSymbols{};

    Result() = default;

  public:
    std::optional<llvm::object::SymbolRef>
    getSymbolRef(const llvm::GlobalObject &GO) const;

    void insertSymbol(const llvm::GlobalObject &GO,
                      llvm::object::SymbolRef Sym) {
      GlobalObjectSymbols.insert({GO, Sym});
    }

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
               llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

private:
  Result GlobalValueSymbols;

public:
  GlobalObjectSymbolsAnalysis() = default;

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return GlobalValueSymbols;
  }
};

} // namespace luthier

#endif