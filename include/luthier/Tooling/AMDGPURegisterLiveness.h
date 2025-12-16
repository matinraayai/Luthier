//===-- AMDGPURegisterLiveness.h --------------------------------*- C++ -*-===//
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
/// This file describes the \c AMDGPURegisterLiveness class and its pass,
/// which calculates the register live-in sets for each \c llvm::MachineInstr
/// of all <tt>llvm::MachineFunction</tt>s inside a <tt>llvm::MachineModuleInfo.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_AMDGPU_REGISTER_LIVENESS_H
#define LUTHIER_TOOLING_AMDGPU_REGISTER_LIVENESS_H
#include "luthier/Tooling/LRCallgraph.h"
#include "luthier/Tooling/VectorCFG.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class AMDGPURegisterLiveness {
private:
  /// The callgraph analysis result of the MMI
  /// TODO: Use call graph to calculate liveness at a global level
  const LRCallGraph &CG;

  /// A mapping between an \c llvm::MachineInstr of the MMI and the set of
  /// physical registers that are live right before it is executed.
  /// The live registers here only includes ones obtained using data-flow
  /// at the \c llvm::MachineFunction level. It does not consider
  /// the registers that are live at the call sites of the function
  /// the \c llvm::MachineInstr belongs to
  llvm::DenseMap<const llvm::MachineInstr *,
                 std::unique_ptr<llvm::LivePhysRegs>>
      MachineInstrLivenessMap{};

public:
  AMDGPURegisterLiveness(const llvm::Module &M,
                         const llvm::MachineModuleInfo &MMI,
                         const LRCallGraph &CG);

  /// \returns the set of physical registers that are live before executing
  /// the instruction \p MI at the function level, or nullptr if the
  /// live register set of \p MI was not found
  [[nodiscard]] const llvm::LivePhysRegs *
  getMFLevelInstrLiveIns(const llvm::MachineInstr &MI) const {
    auto It = MachineInstrLivenessMap.find(&MI);
    if (It == MachineInstrLivenessMap.end())
      return nullptr;
    else
      return It->second.get();
  }

  /// Never invalidate the results
  bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                  llvm::ModuleAnalysisManager::Invalidator &) {
    return false;
  }
};

/// \brief the analysis pass used to obtain the \c AMDGPURegisterLiveness
class AMDGPURegLivenessAnalysis
    : public llvm::AnalysisInfoMixin<AMDGPURegLivenessAnalysis> {
private:
  friend AnalysisInfoMixin<AMDGPURegLivenessAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = AMDGPURegisterLiveness;

  AMDGPURegLivenessAnalysis() = default;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif