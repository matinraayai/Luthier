//===-- BranchTargetAnalysis.h ----------------------------------*- C++ -*-===//
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
/// Describes \c BranchTargetOffsetsAnalysis class. It provides the load offsets
/// of the branch instruction targets inside the symbol of a
/// <tt>llvm::MachineFunction</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_BRANCH_TARGET_OFFSETS_ANALYSIS_H
#define LUTHIER_INSTRUMENTATION_BRANCH_TARGET_OFFSETS_ANALYSIS_H
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class BranchTargetOffsetsAnalysis
    : public llvm::AnalysisInfoMixin<BranchTargetOffsetsAnalysis> {
  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend class BranchTargetOffsetsAnalysis;

    llvm::DenseSet<uint64_t> BranchTargetOffsets{};

    Result() = default;

  public:
    using const_iterator = decltype(BranchTargetOffsets)::const_iterator;

    const_iterator begin() { return BranchTargetOffsets.begin(); }

    const_iterator end() { return BranchTargetOffsets.end(); }

    [[nodiscard]] bool empty() const { return BranchTargetOffsets.empty(); }

    [[nodiscard]] unsigned size() const { return BranchTargetOffsets.size(); }

    bool contains(uint64_t Offset) const {
      return BranchTargetOffsets.contains(Offset);
    }

    /// Prevents invalidation of the analysis result
    __attribute__((used)) bool
    invalidate(llvm::MachineFunction &, const llvm::PreservedAnalyses &,
               llvm::MachineFunctionAnalysisManager::Invalidator &) {
      return false;
    }
  };

  /// Default constructor
  BranchTargetOffsetsAnalysis() = default;

  Result run(llvm::MachineFunction &MF,
             llvm::MachineFunctionAnalysisManager &MFAM);
};

} // namespace luthier

#endif
