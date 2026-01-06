//===-- InitialEntryPointAnalysis.h -------------------------------*-C++-*-===//
// Copyright 2026 @ Northeastern University Computer Architecture Lab
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
/// \file InitialEntryPointAnalysis.h
/// Describes the \c InitialEntryPointAnalysis class which provides access to
/// the initial entrypoint of the lifting process.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INITIAL_ENTRY_POINT_ANALYSIS_H
#define LUTHIER_TOOLING_INITIAL_ENTRY_POINT_ANALYSIS_H
#include "luthier/Tooling/EntryPoint.h"
#include <llvm/IR/PassManager.h>

namespace luthier {

class InitialEntryPointAnalysis
    : public llvm::AnalysisInfoMixin<InitialEntryPointAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

  std::function<EntryPoint(llvm::Module &, llvm::ModuleAnalysisManager &)>
      EntryPointResolver;

public:
  class Result {
    friend InitialEntryPointAnalysis;

    EntryPoint InitialEP;

    explicit Result(EntryPoint EP) : InitialEP(EP) {};

  public:
    EntryPoint getInitialEntryPoint() const { return InitialEP; }

    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  explicit InitialEntryPointAnalysis(
      std::function<EntryPoint(llvm::Module &, llvm::ModuleAnalysisManager &)>
          EntryPointResolver)
      : EntryPointResolver(std::move(EntryPointResolver)) {};

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    return Result{EntryPointResolver(M, MAM)};
  }
};

} // namespace luthier

#endif