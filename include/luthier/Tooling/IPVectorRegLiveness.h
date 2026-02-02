//===-- IPVectorRegLiveness.h ------------------------------------*- C++-*-===//
// Copyright 2022-2026 @ Northeastern University Computer Architecture Lab
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
/// \file IPVectorRegLiveness.h
/// Describes the \c IPVectorRegLiveness and its corresponding analysis,
/// which calculates the register live-in sets for each \c llvm::MachineInstr
/// of all <tt>llvm::MachineFunction</tt>s inside a <tt>llvm::MachineModuleInfo.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_IP_VECTOR_REG_LIVENESS_H
#define LUTHIER_TOOLING_IP_VECTOR_REG_LIVENESS_H
#include "luthier/Tooling/IPVectorCFG.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

class IPVectorRegLiveness {
private:
  llvm::DenseMap<const PredicatedMachineBasicBlock *,
                 std::vector<llvm::MachineBasicBlock::RegisterMaskPair>>
      VectorMBBLivenessMap{};

public:
  llvm::ArrayRef<llvm::MachineBasicBlock::RegisterMaskPair>
  getVectorMBBLiveIns(const PredicatedMachineBasicBlock &VecMBB) const {
    return VectorMBBLivenessMap.at(&VecMBB);
  }

  IPVectorRegLiveness(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  bool invalidate(llvm::Module &M, const llvm::PreservedAnalyses &PA,
                  llvm::ModuleAnalysisManager::Invalidator &Inv);
};

/// \brief the analysis pass used to obtain the \c IPVectorRegLiveness
class IPVectorRegLivenessAnalysis
    : public llvm::AnalysisInfoMixin<IPVectorRegLivenessAnalysis> {
private:
  friend AnalysisInfoMixin<IPVectorRegLivenessAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = IPVectorRegLiveness;

  IPVectorRegLivenessAnalysis() = default;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif