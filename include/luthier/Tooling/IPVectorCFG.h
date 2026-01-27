//===-- IPVectorCFG.h --------------------------------------------*- C++-*-===//
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
/// \file IPVectorCFG.h
/// Describes the \c IPVectorCFG class and its basic blocks.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_IP_VECTOR_CFG_H
#define LUTHIER_TOOLING_IP_VECTOR_CFG_H
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Twine.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>

namespace llvm {

class MachineInstr;

class MachineFunction;

class LivePhysRegs;
} // namespace llvm

namespace luthier {

class IPVectorCFG;

class VectorCFG;

class ScalarMBB;

/// \brief A basic block inside the inter-procedural vector control flow graph;
/// Used primarily in flow analysis involving vector GPRs and instructions
/// \details In addition to normal scalar control flow operations that end a
/// basic block in LLVM MIR, any operation that modifies the execute
/// mask or a call instruction will result in the termination of the vector
/// basic block
class VectorMBB {
private:
  /// Index of the vector MBB in the parent Scalar MBB
  unsigned Idx;

  /// The scalar MBB this vector MBB belongs to
  const ScalarMBB &Parent;

  /// The range of instructions in the block
  llvm::iterator_range<llvm::MachineBasicBlock::const_instr_iterator>
      Instructions{{}, {}};

  /// Set of predecessor blocks
  llvm::SmallDenseSet<const VectorMBB *> Predecessors{};

  /// Set of successor blocks
  llvm::SmallDenseSet<const VectorMBB *> Successors{};

public:
  /// \note Do not use; Use Scalar MBB to create VectorMBBs instead
  VectorMBB(const ScalarMBB &Parent, unsigned Idx) : Idx(Idx), Parent(Parent) {}

  /// Makes the MBB point to a new range of instructions in a single MBB
  void setInstRange(llvm::MachineBasicBlock::const_instr_iterator Begin,
                    llvm::MachineBasicBlock::const_instr_iterator End);

  /// Disallowed copy construction
  VectorMBB(const VectorMBB &) = delete;

  /// Disallowed assignment operation
  VectorMBB &operator=(const VectorMBB &) = delete;

  [[nodiscard]] const ScalarMBB &getParent() const { return Parent; };

  [[nodiscard]] llvm::MachineBasicBlock::const_instr_iterator begin() const {
    return Instructions.begin();
  }

  [[nodiscard]] const llvm::MachineInstr &front() const { return *begin(); }

  [[nodiscard]] llvm::MachineBasicBlock::const_instr_iterator end() const {
    return Instructions.end();
  }

  [[nodiscard]] const llvm::MachineInstr &back() const {
    return *(--Instructions.end());
  }

  [[nodiscard]] bool empty() const { return Instructions.empty(); }

  [[nodiscard]] unsigned getIndex() const { return Idx; }

  void setIndex(unsigned I) { Idx = I; }

  [[nodiscard]] llvm::StringRef getName() const;

  [[nodiscard]] auto predecessors() const {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  auto preds_size() const { return Predecessors.size(); }

  bool preds_empty() const { return Predecessors.empty(); }

  [[nodiscard]] auto successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  auto succs_size() const { return Successors.size(); }

  bool succs_empty() const { return Successors.empty(); }

  void addPredecessorBlock(VectorMBB &MBB) {
    Predecessors.insert(&MBB);
    MBB.Successors.insert(this);
  }

  void addSuccessorBlock(VectorMBB &MBB) {
    Successors.insert(&MBB);
    MBB.Predecessors.insert(this);
  }

  void print(llvm::raw_ostream &OS, unsigned int Indent) const;

  LLVM_DUMP_METHOD void dump() const;
};

/// \brief a wrapper around a group of <tt>VectorMBB</tt>s inside a single
/// \c llvm::MachineBasicBlock for easier construction of the \c VectorCFG
class ScalarMBB {
  /// The CFG this scalar MBB belongs to
  VectorCFG &ParentCFG;
  /// The MIR MBB this scalar block wraps around; If \c nullptr then this
  /// is an entry/exit block of the \c IPVectorCFG
  const llvm::MachineBasicBlock *ParentMBB{nullptr};

  llvm::SmallVector<std::unique_ptr<VectorMBB>> VectorMBBs{};

  struct ScalarEntryOrExitBlocks {
    VectorMBB *TakenBlock{nullptr};
    VectorMBB *NotTakenBlock{nullptr};
  };

  ScalarEntryOrExitBlocks Entry{};

  ScalarEntryOrExitBlocks Exit{};

  ScalarMBB(const llvm::MachineBasicBlock &ParentMBB, VectorCFG &ParentCFG)
      : ParentCFG(ParentCFG), ParentMBB(&ParentMBB) {};

  explicit ScalarMBB(VectorCFG &ParentCFG) : ParentCFG(ParentCFG) {};

public:
  static llvm::Expected<std::unique_ptr<ScalarMBB>>
  createScalarMBB(const llvm::MachineBasicBlock &ParentMBB,
                  VectorCFG &ParentCFG);

  static llvm::Expected<std::unique_ptr<ScalarMBB>>
  createEntryScalarMBB(VectorCFG &ParentCFG);

  /// Disallowed copy construction
  ScalarMBB(const ScalarMBB &) = delete;

  /// Disallowed assignment operation
  ScalarMBB &operator=(const ScalarMBB &) = delete;

  class iterator {
    decltype(VectorMBBs)::iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = VectorMBB;
    using reference = value_type &;
    using pointer = value_type *;

    explicit iterator(const decltype(VectorMBBs)::iterator &It) : It(It) {}

    reference operator*() const { return **It; }

    pointer operator->() const { return It->get(); }

    iterator operator++() {
      ++It;
      return *this;
    }

    iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const iterator &Other) const { return It == Other.It; }

    bool operator!=(const iterator &Other) const { return !(*this == Other); }
  };

  class const_iterator {
    decltype(VectorMBBs)::const_iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = const VectorMBB;
    using reference = value_type &;
    using pointer = value_type *;

    explicit const_iterator(const decltype(VectorMBBs)::const_iterator &It)
        : It(It) {}

    reference operator*() const { return **It; }

    pointer operator->() const { return It->get(); }

    const_iterator operator++() {
      ++It;
      return *this;
    }

    const_iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const const_iterator &Other) const {
      return It == Other.It;
    }

    bool operator!=(const const_iterator &Other) const {
      return !(*this == Other);
    }
  };

  [[nodiscard]] const VectorCFG &getParent() const { return ParentCFG; }

  [[nodiscard]] const llvm::MachineBasicBlock *getMBB() const {
    return ParentMBB;
  }

  iterator begin() { return iterator(VectorMBBs.begin()); }

  [[nodiscard]] const_iterator begin() const {
    return const_iterator(VectorMBBs.begin());
  }

  iterator end() { return iterator(VectorMBBs.end()); }

  [[nodiscard]] const_iterator end() const {
    return const_iterator(VectorMBBs.end());
  }

  void addSuccessor(ScalarMBB &SMBB);

  void addPredecessor(ScalarMBB &SMBB);

  void print(llvm::raw_ostream &OS, unsigned int Indent) const;
};

class VectorCFG {
  IPVectorCFG &ParentCFG;

  const llvm::MachineFunction &MF;

  llvm::SmallVector<std::unique_ptr<ScalarMBB>> ScalarMBBs{};

  llvm::SmallDenseMap<const llvm::MachineBasicBlock *, ScalarMBB *>
      MBBToScalarMBBs{};

  VectorCFG(IPVectorCFG &ParentCFG, const llvm::MachineFunction &MF)
      : ParentCFG(ParentCFG), MF(MF) {};

public:
  /// Disallowed copy construction
  VectorCFG(const VectorCFG &) = delete;

  /// Disallowed assignment operation
  VectorCFG &operator=(const VectorCFG &) = delete;

  static llvm::Expected<std::unique_ptr<VectorCFG>>
  createVectorCFG(IPVectorCFG &ParentCFG, const llvm::MachineFunction &MF);

  class iterator {
    decltype(ScalarMBBs)::iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = ScalarMBB;
    using reference = value_type &;
    using pointer = value_type *;

    explicit iterator(const decltype(ScalarMBBs)::iterator &It) : It(It) {}

    reference operator*() const { return **It; }

    pointer operator->() const { return It->get(); }

    iterator operator++() {
      ++It;
      return *this;
    }

    iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const iterator &Other) const { return It == Other.It; }

    bool operator!=(const iterator &Other) const { return !(*this == Other); }
  };

  class const_iterator {
    decltype(ScalarMBBs)::const_iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = const ScalarMBB;
    using reference = value_type &;
    using pointer = value_type *;

    explicit const_iterator(const decltype(ScalarMBBs)::const_iterator &It)
        : It(It) {}

    reference operator*() const { return **It; }

    pointer operator->() const { return It->get(); }

    const_iterator operator++() {
      ++It;
      return *this;
    }

    const_iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const const_iterator &Other) const {
      return It == Other.It;
    }

    bool operator!=(const const_iterator &Other) const {
      return !(*this == Other);
    }
  };

  iterator begin() { return iterator(ScalarMBBs.begin()); }

  [[nodiscard]] const_iterator begin() const {
    return const_iterator(ScalarMBBs.begin());
  }

  iterator end() { return iterator(ScalarMBBs.end()); }

  [[nodiscard]] const_iterator end() const {
    return const_iterator(ScalarMBBs.end());
  }

  [[nodiscard]] const ScalarMBB &
  getScalarMBB(const llvm::MachineBasicBlock &MBB) const {
    return *MBBToScalarMBBs.at(&MBB);
  }

  [[nodiscard]] const llvm::MachineFunction &getMF() const { return MF; }

  void print(llvm::raw_ostream &OS) const;

  LLVM_DUMP_METHOD void dump() const;
};

/// \brief An inter-procedural control-flow graph representation for the MIR of
/// the AMDGPU backend that, in addition to scalar branches, regards
/// instructions that manipulate the execute mask as well as call instructions
/// as a terminator for its basic blocks
/// \details The vector CFG can be used to perform data flow analysis
/// involving vector registers (e.g. register liveness) which cannot be done
/// with the CFG of LLVM MIR for the AMD GPU backend
class IPVectorCFG {
private:
  llvm::SmallDenseMap<const llvm::MachineFunction *, std::unique_ptr<VectorCFG>>
      VectorCFGs{};

  unsigned NumVecMBBs{0};

  IPVectorCFG() = default;

public:
  class iterator {
    decltype(VectorCFGs)::iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = VectorCFG;
    using reference = value_type &;
    using pointer = value_type *;

    explicit iterator(const decltype(VectorCFGs)::iterator &It) : It(It) {}

    reference operator*() const { return *It->second; }

    pointer operator->() const { return It->second.get(); }

    iterator operator++() {
      ++It;
      return *this;
    }

    iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const iterator &Other) const { return It == Other.It; }

    bool operator!=(const iterator &Other) const { return !(*this == Other); }
  };

  class const_iterator {
    decltype(VectorCFGs)::const_iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = VectorCFG;
    using reference = value_type &;
    using pointer = value_type *;

    explicit const_iterator(const decltype(VectorCFGs)::const_iterator &It)
        : It(It) {}

    reference operator*() const { return *It->second; }

    pointer operator->() const { return It->second.get(); }

    const_iterator operator++() {
      ++It;
      return *this;
    }

    const_iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const const_iterator &Other) const {
      return It == Other.It;
    }

    bool operator!=(const const_iterator &Other) const {
      return !(*this == Other);
    }
  };

  iterator begin() { return iterator(VectorCFGs.begin()); }

  [[nodiscard]] const_iterator begin() const {
    return const_iterator(VectorCFGs.begin());
  }

  iterator end() { return iterator(VectorCFGs.end()); }

  [[nodiscard]] const_iterator end() const {
    return const_iterator(VectorCFGs.end());
  }

  VectorCFG &operator[](const llvm::MachineFunction &MF) {
    assert(VectorCFGs.contains(&MF) && "Entry is not in the map");
    return *VectorCFGs[&MF];
  }

  [[nodiscard]] const VectorCFG &at(const llvm::MachineFunction &MF) const {
    assert(VectorCFGs.contains(&MF) && "Entry is not in the map");
    return *VectorCFGs.at(&MF);
  }

  unsigned getNumVecMBBs() const { return NumVecMBBs; }

  void print(llvm::raw_ostream &OS) const;

  LLVM_DUMP_METHOD void dump() const;

  static llvm::Expected<std::unique_ptr<IPVectorCFG>>
  calculateIPVectorCFG(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

class IPVectorCFGAnalysis
    : public llvm::AnalysisInfoMixin<IPVectorCFGAnalysis> {
  friend llvm::AnalysisInfoMixin<IPVectorCFGAnalysis>;

private:
  static llvm::AnalysisKey Key;

public:
  class Result {
    friend IPVectorCFGAnalysis;

    std::unique_ptr<IPVectorCFG> IPCFG;

    Result(std::unique_ptr<IPVectorCFG> IPCFG) : IPCFG(std::move(IPCFG)) {};

  public:
    bool invalidate(llvm::Module &M, const llvm::PreservedAnalyses &PA,
                    llvm::ModuleAnalysisManager::Invalidator &Inv);

    const IPVectorCFG &getVecCFG() const { return *IPCFG; }
  };

  IPVectorCFGAnalysis() = default;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

namespace llvm {
template <> struct GraphTraits<const luthier::VectorMBB *> {
  using NodeRef = const luthier::VectorMBB *;
  using ChildIteratorType =
      SmallDenseSet<const luthier::VectorMBB *>::const_iterator;

  static NodeRef getEntryNode(const luthier::VectorMBB *BB) { return BB; }

  static ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }

  static unsigned getNumber(const luthier::VectorMBB *BB) {
    return BB->getIndex();
  }
};

static_assert(GraphHasNodeNumbers<const MachineBasicBlock *>,
              "GraphTraits getNumber() not detected");

template <> struct GraphTraits<Inverse<const luthier::VectorMBB *>> {
  using NodeRef = const luthier::VectorMBB *;
  using ChildIteratorType =
      SmallDenseSet<const luthier::VectorMBB *>::const_iterator;

  static NodeRef getEntryNode(Inverse<luthier::VectorMBB *> G) {
    return G.Graph;
  }

  static ChildIteratorType child_begin(NodeRef N) {
    return N->predecessors().begin();
  }
  static ChildIteratorType child_end(NodeRef N) {
    return N->predecessors().end();
  }

  static unsigned getNumber(const luthier::VectorMBB *BB) {
    return BB->getIndex();
  }
};

static_assert(GraphHasNodeNumbers<Inverse<const luthier::VectorMBB *>>,
              "GraphTraits getNumber() not detected");
} // namespace llvm

#endif