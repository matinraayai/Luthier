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
/// \file This file describes the \c VectorCFG class and its basic blocks.
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

class ScalarMBB;

/// \brief A basic block inside the vector control flow graph; Used primarily
/// in flow analysis involving vector GPRs and instructions
/// \details In addition to normal scalar control flow operations that end a
/// basic block in LLVM MIR, any operation that modifies the execute
/// mask will result in the termination of the vector basic block
class VectorMBB {
private:
  /// CFG that this MBB belongs to
  const VectorCFG &CFG;

  /// The scalar MBB this vector MBB belongs to
  /// If nullptr, this vector MBB is a dummy entry/exit block of the vector CFG
  const ScalarMBB *ParentScalarMBB;

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

  VectorMBB(const ScalarMBB &SMBB, llvm::StringRef Name);

  VectorMBB(const VectorCFG &CFG, llvm::StringRef Name)
      : CFG(CFG), Name(Name), ParentScalarMBB(nullptr) {};

  [[nodiscard]] const VectorCFG &getParent() const { return CFG; };

  [[nodiscard]] const ScalarMBB *getScalarMBB() const {
    return ParentScalarMBB;
  };

  [[nodiscard]] llvm::MachineBasicBlock::const_iterator begin() const {
    return Instructions.begin();
  }

  [[nodiscard]] llvm::MachineBasicBlock::const_iterator end() const {
    return Instructions.end();
  }

  [[nodiscard]] bool empty() const { return Instructions.empty(); }

  [[nodiscard]] llvm::StringRef getName() const { return Name; }

  /// Makes the MBB point to a new range of instructions in a single MBB
  void setInstRange(llvm::MachineBasicBlock::const_iterator Begin,
                    llvm::MachineBasicBlock::const_iterator End);

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

  void print(llvm::raw_ostream &OS, unsigned int Indent) const;

  LLVM_DUMP_METHOD void dump() const;
};

/// \brief a wrapper around a group of <tt>VectorMBB</tt>s inside a single
/// \c llvm::MachineBasicBlock for easier construction of the \c VectorCFG
class ScalarMBB {
private:
  friend VectorCFG;
  /// The CFG this scalar MBB belongs to
  VectorCFG &ParentCFG;
  /// The MIR basic block this scalar MBB wraps around
  const llvm::MachineBasicBlock &ParentMBB;
  /// All Vector MBBs in the scalar block
  typedef llvm::SmallVector<std::unique_ptr<VectorMBB>, 6> VectorMBBs;
  VectorMBBs MBBs{};

  /// A struct of two <tt>VectorMBB</tt>s representing the entry and exit of
  /// scalar block
  typedef struct {
    VectorMBB &TakenBlock;
    VectorMBB &NotTakenBlock;
  } ScalarEntryOrExitBlocks;

  ScalarEntryOrExitBlocks Entry;

  ScalarEntryOrExitBlocks Exit;

  ScalarMBB(const llvm::MachineBasicBlock &ParentMBB, VectorCFG &ParentCFG);

  VectorMBB &createVectorMBB() {
    MBBs.emplace_back(std::make_unique<VectorMBB>(
        *this,
        (ParentMBB.getFullName() + "." + llvm::Twine(MBBs.size() - 4)).str()));
    return *MBBs.back();
  }

public:
  /// Disallowed copy construction
  ScalarMBB(const ScalarMBB &) = delete;

  /// Disallowed assignment operation
  ScalarMBB &operator=(const ScalarMBB &) = delete;

  [[nodiscard]] VectorCFG &getCFG() const { return ParentCFG; }

  using iterator = VectorMBBs::iterator;

  using const_iterator = VectorMBBs::const_iterator;

  iterator begin() { return MBBs.begin(); }

  [[nodiscard]] const_iterator begin() const { return MBBs.begin(); }

  iterator end() { return MBBs.end(); }

  [[nodiscard]] const_iterator end() const { return MBBs.end(); }

  void print(llvm::raw_ostream &OS, unsigned int Indent) const;
};

/// \brief A control-flow graph representation for
/// <tt>llvm::MachineFunction</tt>s of the AMDGPU backend that, in addition to
/// scalar branches, regards instructions that manipulate the execute mask as a
/// terminator for its basic blocks
/// \details The vector CFG can be used to perform data flow analysis
/// involving vector registers (e.g. register liveness) which cannot be done
/// with the CFG of LLVM MIR for the AMD GPU backend
class VectorCFG {
private:
  /// The machine function being analyzed
  const llvm::MachineFunction &MF;
  /// A dummy vector block that marks the start of the function
  std::unique_ptr<VectorMBB> EntryBlock;
  /// A dummy vector block that marks the end of the function
  std::unique_ptr<VectorMBB> ExitBlock;
  /// A mapping between the <tt>llvm::MachineBasicBlock</tt>s of the \c MF and
  /// their \c ScalarMBB
  typedef llvm::DenseMap<const llvm::MachineBasicBlock *,
                         std::unique_ptr<ScalarMBB>>
      MBBMap;
  MBBMap MBBs;

  explicit VectorCFG(const llvm::MachineFunction &MF)
      : MF(MF), EntryBlock(std::make_unique<VectorMBB>(
                    *this, (MF.getName() + ".entry").str())),
        ExitBlock(std::make_unique<VectorMBB>(
            *this, (MF.getName() + ".exit").str())) {};

public:
  using iterator = MBBMap::iterator;

  using const_iterator = MBBMap::const_iterator;

  iterator begin() { return MBBs.begin(); }

  [[nodiscard]] const_iterator begin() const { return MBBs.begin(); }

  iterator end() { return MBBs.end(); }

  [[nodiscard]] const_iterator end() const { return MBBs.end(); }

  [[nodiscard]] const llvm::MachineFunction &getMF() const { return MF; }

  VectorMBB &getEntryBlock() { return *EntryBlock; }

  VectorMBB &getExitBlock() { return *ExitBlock; }

  // TODO: Implement an iterator for directly going over VectorMBBs

  void print(llvm::raw_ostream &OS) const;

  LLVM_DUMP_METHOD void dump() const;

  static std::unique_ptr<VectorCFG>
  getVectorCFG(const llvm::MachineFunction &MF);
};

void addBlockLiveIns(llvm::LivePhysRegs &LPR, const VectorMBB &VecMBB);

void addLiveIns(VectorMBB &MBB, const llvm::LivePhysRegs &LiveRegs);

void addLiveOutsNoPristines(llvm::LivePhysRegs &LPR, const VectorMBB &MBB);

} // namespace luthier

#endif