//===-- IndirectBranchResolverAnalysis.h --------------------------*-C++-*-===//
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
/// \file IndirectBranchResolverAnalysis.h
/// Describes the \c IndirectBranchResolverAnalysis class, a module analysis
/// used to try to resolve the target of indirect branches and all call
/// instructions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_INDIRECT_BRANCH_RESOLVER_ANALYSIS_H
#define LUTHIER_TOOLING_INDIRECT_BRANCH_RESOLVER_ANALYSIS_H
#include <llvm/IR/PassManager.h>

namespace luthier {

class IndirectBranchResolverAnalysis
    : public llvm::AnalysisInfoMixin<IndirectBranchResolverAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend IndirectBranchResolverAnalysis;

    explicit Result() = default;

  public:
    bool invalidate(llvm::Module &M, const llvm::PreservedAnalyses &PA,
                    llvm::ModuleAnalysisManager::Invalidator &Inv);
  };

  IndirectBranchResolverAnalysis() = default;

  Result run(llvm::Module &TargetModule,
             llvm::ModuleAnalysisManager &TargetMAM);
};

} // namespace luthier

#endif