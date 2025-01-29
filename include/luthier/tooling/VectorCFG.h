//===-- VectorCFG.h ---------------------------------------------*- C++ -*-===//
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

class LivePhysRegs;
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

  /// Basic block name
  std::string Name;

  /// The range of instructions in the block
  llvm::iterator_range<llvm::MachineBasicBlock::const_iterator> Instructions{
      {}, {}};

  /// Set of predecessor blocks
  llvm::SmallDenseSet<const VectorMBB *, 4> Predecessors{};

  /// Set of successor blocks
  llvm::SmallDenseSet<const VectorMBB *, 2> Successors{};

  typedef std::vector<llvm::MachineBasicBlock::RegisterMaskPair> LiveInVector;

  std::vector<llvm::MachineBasicBlock::RegisterMaskPair> LiveIns;

public:
  /// Disallowed copy construction
  VectorMBB(const VectorMBB &) = delete;

  /// Disallowed assignment operation
  VectorMBB &operator=(const VectorMBB &) = delete;

  VectorMBB(const VectorCFG &CFG, llvm::StringRef Name)
      : CFG(CFG), Name(Name) {};

  VectorMBB(const VectorCFG &CFG, const llvm::MachineInstr &BeginMI,
            llvm::StringRef Name)
      : CFG(CFG), Instructions({BeginMI, *BeginMI.getNextNode()}),
        Name(Name) {};

  VectorMBB(const VectorCFG &CFG, const llvm::MachineInstr &BeginMI,
            const llvm::MachineInstr &EndMI, llvm::StringRef Name)
      : CFG(CFG), Instructions({BeginMI, *EndMI.getNextNode()}), Name(Name) {};

  [[nodiscard]] const VectorCFG &getParent() const { return CFG; };

  [[nodiscard]] llvm::MachineBasicBlock::const_iterator begin() const {
    return Instructions.begin();
  }

  [[nodiscard]] llvm::MachineBasicBlock::const_iterator end() const {
    return Instructions.end();
  }

  [[nodiscard]] llvm::StringRef getNum() const { return Name; }

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

  [[nodiscard]] auto predecessors() const {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  [[nodiscard]] auto successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  void addLiveIn(llvm::MCRegister PhysReg,
                 llvm::LaneBitmask LaneMask = llvm::LaneBitmask::getAll());

  void clearLiveIns(
      std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &OldLiveIns);

  void sortUniqueLiveIns();

  [[nodiscard]] const std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &
  getLiveIns() const {
    return LiveIns;
  }

  llvm::iterator_range<LiveInVector::iterator> liveins() {
    return llvm::make_range(LiveIns.begin(), LiveIns.end());
  }

  [[nodiscard]] llvm::iterator_range<LiveInVector::const_iterator>
  liveins() const {
    return llvm::make_range(LiveIns.begin(), LiveIns.end());
  }

  void print(llvm::raw_ostream &OS) const;
};

//class ScalarMBB {
//private:
//  /// The MIR basic block this wraps around
//  const llvm::MachineBasicBlock &ParentMBB;
//  /// The CFG this scalar MBB belongs to
//  VectorCFG &ParentCFG;
//  /// MBB Vector typedef
//  typedef llvm::SmallVector<std::unique_ptr<VectorMBB>, 0> MBBVector;
//  MBBVector MBBs;
//
//  /// The entry taken Vector MBB
//  VectorMBB *EntryTakenMBB;
//  /// The entry not taken Vector MBB
//  VectorMBB *EntryNotTakenMBB;
//  /// The exit taken Vector MBB
//  VectorMBB *ExitTakenMBB;
//  /// The exit not taken Vector MBB
//  VectorMBB *ExitNotTakenMBB;
//
//public:
//  ScalarMBB(const llvm::MachineBasicBlock &ParentMBB, VectorCFG &ParentCFG);
//};

class VectorCFG {
private:
  typedef llvm::SmallVector<std::unique_ptr<VectorMBB>, 0> MBBVector;

  MBBVector MBBs;

  const llvm::MachineFunction &MF;

  explicit VectorCFG(const llvm::MachineFunction &MF) : MF(MF) {};

  VectorMBB &createVectorMBB();

public:
  using iterator = MBBVector::iterator;

  using const_iterator = MBBVector::const_iterator;

  iterator begin() { return MBBs.begin(); }

  [[nodiscard]] const_iterator begin() const { return MBBs.begin(); }

  iterator end() { return MBBs.end(); }

  [[nodiscard]] const_iterator end() const { return MBBs.end(); }

  [[nodiscard]] const llvm::MachineFunction &getMF() const { return MF; }

  void print(llvm::raw_ostream &OS) const;

  static std::unique_ptr<VectorCFG>
  getVectorCFG(const llvm::MachineFunction &MF);
};

/// Re-implementation of \c llvm::LivePhysRegs::addLiveOutsNoPristines for
/// \c VectorMBB
/// \param VecMBB
void addBlockLiveIns(llvm::LivePhysRegs &LPR, const VectorMBB &VecMBB);

void addLiveIns(VectorMBB &MBB, const llvm::LivePhysRegs &LiveRegs);

void addLiveOutsNoPristines(llvm::LivePhysRegs &LPR, const VectorMBB &MBB);

} // namespace luthier

#endif