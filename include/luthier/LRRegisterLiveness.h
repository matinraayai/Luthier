//===-- LRRegisterLiveness.h ------------------------------------*- C++ -*-===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file describes the \c LRRegisterLiveness class, which calculates the
/// register live-in sets for each \c llvm::MachineInstr of a
/// \c LiftedRepresentation at a machine function level.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LR_REGISTER_LIVENESS_H
#define LUTHIER_LR_REGISTER_LIVENESS_H
#include <hsa/hsa.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/PassManager.h>
#include <luthier/VectorCFG.h>

namespace luthier {

class LiftedRepresentation;

class LRRegisterLiveness {
private:
  llvm::DenseMap<const llvm::MachineFunction *, std::unique_ptr<VectorCFG>>
      VecCFG;

  /// A mapping between an \c llvm::MachineInstr and the set of physical
  /// registers that are live right before it\n
  /// This mapping is only valid before any LLVM pass is run over the Modules
  /// and MMIs of the \c LR
  llvm::DenseMap<const llvm::MachineInstr *,
                 std::unique_ptr<llvm::LivePhysRegs>>
      MachineInstrLivenessMap{};

public:
  LRRegisterLiveness() = default;

  /// \returns the set of physical registers that are live before executing
  /// the instruction \p MI
  /// \note This
  [[nodiscard]] const llvm::LivePhysRegs *
  getLiveInPhysRegsOfMachineInstr(const llvm::MachineInstr &MI) const {
    auto It = MachineInstrLivenessMap.find(&MI);
    if (It == MachineInstrLivenessMap.end())
      return nullptr;
    else
      return It->second.get();
  }

  /// Recomputes the Live-ins map for each \c llvm::MachineInstr in the Lifted
  /// Representation
  /// This includes both the instructions lifted from the code objects, and
  /// the ones manually injected by the tool writer
  void recomputeLiveIns(const llvm::Module &M,
                        const llvm::MachineModuleInfo &MMI);
};

class LRRegLivenessAnalysis
    : public llvm::AnalysisInfoMixin<LRRegLivenessAnalysis> {
private:
  friend AnalysisInfoMixin<LRRegLivenessAnalysis>;

  static llvm::AnalysisKey Key;

  LRRegisterLiveness RegLiveness{};

public:
  using Result = LRRegisterLiveness &;

  LRRegLivenessAnalysis() = default;

  /// Run the analysis pass that would
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  /// Never invalidate the results
  bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                  llvm::ModuleAnalysisManager::Invalidator &) {
    return false;
  }
};

} // namespace luthier

#endif