//===-- VectorCFG.h ---------------------------------------------*- C++ -*-===//
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
/// \file This file describes the \c VectorMBB and \c VectorCFG classes, used
/// to represent control flow done with exec mask manipulation.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_VECTOR_CFG_H
#define LUTHIER_VECTOR_CFG_H
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/MachineBasicBlock.h>

namespace llvm {

class MachineInstr;

class MachineFunction;
} // namespace llvm

namespace luthier {

class VectorCFG;

/// \brief A basic block inside the vector control flow graph; Used primarily
/// in flow analysis involving vector GPRs and instructions
/// \details In addition to normal scalar control flow operations that end a
/// basic block, any operation that ends up modifying an execute mask will
/// result in the termination of a vector basic block. A single
/// \c llvm::MachineBasicBlock will have multiple vector basic blocks inside it
class VectorMBB {
private:
  /// CFG that manages the memory of this \c VectorMBB
  const VectorCFG &CFG;

  /// The range of instructions in the block
  llvm::iterator_range<llvm::MachineBasicBlock::const_iterator> Instructions{
      {}, {}};

  /// Set of predecessor blocks
  llvm::SmallDenseSet<const VectorMBB *, 4> Predecessors{};

  /// Set of successor blocks
  llvm::SmallDenseSet<const VectorMBB *, 2> Successors{};

public:
  /// Disallowed copy construction
  VectorMBB(const VectorMBB &) = delete;

  /// Disallowed assignment operation
  VectorMBB &operator=(const VectorMBB &) = delete;

  explicit VectorMBB(const VectorCFG &CFG) : CFG(CFG) {};

  VectorMBB(const VectorCFG &CFG, const llvm::MachineInstr &BeginMI)
      : CFG(CFG), Instructions({BeginMI, *BeginMI.getNextNode()}) {};

  VectorMBB(const VectorCFG &CFG, const llvm::MachineInstr &BeginMI,
            const llvm::MachineInstr &EndMI)
      : CFG(CFG), Instructions({BeginMI, *EndMI.getNextNode()}) {};

  const VectorCFG &getParent() { return CFG; };

  [[nodiscard]] llvm::MachineBasicBlock::const_iterator begin() const {
    return Instructions.begin();
  }

  [[nodiscard]] llvm::MachineBasicBlock::const_iterator end() const {
    return Instructions.end();
  }

  /// Makes the MBB point to a new range of instructions in a single MBB
  void setInstRange(llvm::MachineBasicBlock::const_iterator Begin,
                    llvm::MachineBasicBlock::const_iterator End);

  [[nodiscard]] bool empty() const { return Instructions.empty(); }

  void addPredecessorBlock(VectorMBB &MBB) {
    Predecessors.insert(&MBB);
    MBB.Successors.insert(this);
  }

  void addSuccessorBlock(VectorMBB &MBB) {
    Successors.insert(&MBB);
    MBB.Predecessors.insert(this);
  }
};

class VectorCFG {
private:
  llvm::SmallVector<std::unique_ptr<VectorMBB>, 0> MBBs;

  VectorCFG() = default;

  VectorMBB &createVectorMBB();

public:
  static std::unique_ptr<VectorCFG>
  getVectorCFG(const llvm::MachineFunction &MF);
};

} // namespace luthier

#endif