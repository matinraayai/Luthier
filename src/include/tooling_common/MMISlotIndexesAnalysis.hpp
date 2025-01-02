//===-- MMISlotIndexesAnalysis.hpp ----------------------------------------===//
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
/// This file describes the <tt>MMISlotIndexesAnalysis</tt> pass, an analysis
/// which provides the \c llvm::SlotIndexes for the entire Instrumentation
/// <tt>llvm::MachineModuleInfo</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_COMMON_MMI_SLOT_INDEXES_ANALYSIS_HPP
#define LUTHIER_TOOLING_COMMON_MMI_SLOT_INDEXES_ANALYSIS_HPP
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/SlotIndexes.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class MMISlotIndexesAnalysis
    : public llvm::AnalysisInfoMixin<MMISlotIndexesAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<MMISlotIndexesAnalysis>;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend MMISlotIndexesAnalysis;
    llvm::DenseMap<llvm::MachineFunction *, llvm::SlotIndexes> Res;
    Result() = default;

  public:
    typedef llvm::DenseMap<llvm::MachineFunction *,
                           llvm::SlotIndexes>::const_iterator const_iterator;

    [[nodiscard]] const_iterator begin() const { return Res.begin(); }

    [[nodiscard]] const_iterator end() const { return Res.end(); }

    const llvm::SlotIndexes &at(llvm::MachineFunction &MF) const {
      return Res.at(&MF);
    }

    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  MMISlotIndexesAnalysis() = default;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif