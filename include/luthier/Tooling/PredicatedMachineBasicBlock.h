//===-- PredicatedMachineBasicBlock.h ---------------------------*- C++ -*-===//
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
///
/// \file PredicatedMachineBasicBlock.h
/// Defines the \c PredicatedMachineBasicBlock class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_PREDICATED_MACHINE_BASIC_BLOCK_H
#define LUTHIER_TOOLING_PREDICATED_MACHINE_BASIC_BLOCK_H
#include "luthier/Common/DenseMapInfo.h"
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstrBundle.h>
#include <llvm/Support/GenericDomTree.h>

namespace luthier {

class LinearMachineBasicBlock;

class IPPredicatedCFG;

class PredMBBBuilder;

/// \brief A basic block inside the inter-procedural predicated control flow
/// graph
class PredicatedMachineBasicBlock {
  friend class PredMBBBuilder;

public:
  /// \brief Indicates the predicate value (execute mask value) of the
  /// instructions in the current predicated machine basic block
  enum PredicateValue {
    /// Indicates that all instructions in the current block are scalar (i.e.
    /// execute regardless of the predicate value)
    ZeroOrOne = 0,
    /// Indicates that all instructions in the current block are vector (i.e.
    /// only execute when predicate value is one)
    One = 1
  };

  using PredSuccSetType =
      llvm::SmallDenseSet<std::reference_wrapper<PredMBBBuilder>>;

private:
  /// Indicates the value of the execute mask inside this vector MBB
  PredicateValue EMV;

  /// Index of the vector MBB inside the parent IP predicated CFG
  unsigned GlobalIdx{0};

  /// Index of the vector MBB inside its parent Scalar MBB
  unsigned LinearMBBIdx{0};

  /// The scalar MBB this vector MBB belongs to
  LinearMachineBasicBlock &Parent;

  /// The range of instructions in the block
  llvm::iterator_range<llvm::MachineBasicBlock::iterator> Instructions{{}, {}};

  llvm::SmallPtrSet<const llvm::MachineInstr *, 32> MIsSet{};

  /// Set of predecessor blocks
  PredSuccSetType Predecessors{};

  /// Set of successor blocks
  PredSuccSetType Successors{};

  PredicatedMachineBasicBlock(LinearMachineBasicBlock &Parent,
                              PredicateValue EMV)
      : EMV(EMV), Parent(Parent) {}

public:
  /// Disallowed copy construction
  PredicatedMachineBasicBlock(const PredicatedMachineBasicBlock &) = delete;

  /// Disallowed assignment operation
  PredicatedMachineBasicBlock &
  operator=(const PredicatedMachineBasicBlock &) = delete;

  [[nodiscard]] const LinearMachineBasicBlock &getParent() const {
    return Parent;
  }

  [[nodiscard]] LinearMachineBasicBlock &getParent() { return Parent; }

  using instr_iterator = llvm::MachineBasicBlock::instr_iterator;
  using const_instr_iterator = llvm::MachineBasicBlock::const_instr_iterator;

  using iterator = llvm::MachineBasicBlock::iterator;
  using const_iterator = llvm::MachineBasicBlock::const_iterator;

  using reverse_instr_iterator = std::reverse_iterator<instr_iterator>;
  using const_reverse_instr_iterator =
      std::reverse_iterator<const_instr_iterator>;

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() { return Instructions.begin(); }

  [[nodiscard]] const_iterator begin() const { return Instructions.begin(); }

  instr_iterator instr_begin() {
    return Instructions.begin().getInstrIterator();
  }

  const_instr_iterator instr_begin() const {
    return Instructions.begin().getInstrIterator();
  }

  [[nodiscard]] bool contains(const llvm::MachineInstr &MI) const {
    return MIsSet.contains(&MI);
  }

  llvm::MachineInstr &front() { return *begin(); }

  [[nodiscard]] const llvm::MachineInstr &front() const { return *begin(); }

  llvm::MachineInstr &instr_front() { return *instr_begin(); }

  [[nodiscard]] const llvm::MachineInstr &instr_front() const {
    return *instr_begin();
  }

  [[nodiscard]] iterator end() { return Instructions.end(); }

  [[nodiscard]] const_iterator end() const { return Instructions.end(); }

  [[nodiscard]] instr_iterator instr_end() {
    return llvm::getBundleEnd(Instructions.end().getInstrIterator());
  }

  [[nodiscard]] const_instr_iterator instr_end() const {
    return llvm::getBundleEnd(Instructions.end().getInstrIterator());
  }

  reverse_iterator rbegin() { return std::make_reverse_iterator(end()); }

  [[nodiscard]] const_reverse_iterator rbegin() const {
    return std::make_reverse_iterator(end());
  }

  reverse_instr_iterator instr_rbegin() {
    return std::make_reverse_iterator(instr_end());
  }

  [[nodiscard]] const_reverse_instr_iterator instr_rbegin() const {
    return std::make_reverse_iterator(instr_end());
  }

  reverse_iterator rend() { return std::make_reverse_iterator(begin()); }

  [[nodiscard]] const_reverse_iterator rend() const {
    return std::make_reverse_iterator(begin());
  }

  reverse_instr_iterator instr_rend() {
    return std::make_reverse_iterator(instr_begin());
  }

  [[nodiscard]] const_reverse_instr_iterator instr_rend() const {
    return std::make_reverse_iterator(instr_begin());
  }

  [[nodiscard]] const llvm::MachineInstr &back() const {
    return *(--Instructions.end());
  }

  [[nodiscard]] const llvm::MachineInstr &instr_back() const {
    return *(--instr_end());
  }

  [[nodiscard]] bool empty() const { return Instructions.empty(); }

  [[nodiscard]] unsigned size() const {
    return std::distance(Instructions.begin(), Instructions.end());
  }

  [[nodiscard]] unsigned instr_size() const {
    return std::distance(instr_begin(), instr_end());
  }

  [[nodiscard]] PredicateValue getExecMaskValue() const { return EMV; }

  [[nodiscard]] unsigned getGlobalNumber() const { return GlobalIdx; }

  [[nodiscard]] unsigned getLocalNumber() const { return LinearMBBIdx; }

  [[nodiscard]] std::string getName() const;

  class pred_succ_iterator {
    PredSuccSetType::iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = PredicatedMachineBasicBlock;
    using reference = value_type &;
    using pointer = value_type *;

    explicit pred_succ_iterator(const PredSuccSetType::iterator &It) : It(It) {}

    reference operator*() const;

    pointer operator->() const;

    pred_succ_iterator operator++() {
      ++It;
      return *this;
    }

    pred_succ_iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const pred_succ_iterator &Other) const {
      return It == Other.It;
    }

    bool operator!=(const pred_succ_iterator &Other) const {
      return !(*this == Other);
    }
  };

  class const_pred_succ_iterator {
    PredSuccSetType::const_iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = const PredicatedMachineBasicBlock;
    using reference = value_type &;
    using pointer = value_type *;

    explicit const_pred_succ_iterator(const PredSuccSetType::const_iterator &It)
        : It(It) {}

    reference operator*() const;

    pointer operator->() const;

    [[nodiscard]] decltype(It) getIterator() const { return It; }

    const_pred_succ_iterator operator++() {
      ++It;
      return *this;
    }

    const_pred_succ_iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const const_pred_succ_iterator &Other) const {
      return It == Other.It;
    }

    bool operator!=(const const_pred_succ_iterator &Other) const {
      return !(*this == Other);
    }
  };

  pred_succ_iterator preds_begin() {
    return pred_succ_iterator(Predecessors.begin());
  }

  [[nodiscard]] const_pred_succ_iterator preds_begin() const {
    return const_pred_succ_iterator(Predecessors.begin());
  }

  pred_succ_iterator preds_end() {
    return pred_succ_iterator(Predecessors.end());
  }

  [[nodiscard]] const_pred_succ_iterator preds_end() const {
    return const_pred_succ_iterator(Predecessors.end());
  }

  [[nodiscard]] llvm::iterator_range<const_pred_succ_iterator>
  predecessors() const {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  [[nodiscard]] llvm::iterator_range<pred_succ_iterator> predecessors() {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  [[nodiscard]] unsigned preds_size() const { return Predecessors.size(); }

  [[nodiscard]] bool preds_empty() const { return Predecessors.empty(); }

  pred_succ_iterator succs_begin() {
    return pred_succ_iterator(Successors.begin());
  }

  [[nodiscard]] const_pred_succ_iterator succs_begin() const {
    return const_pred_succ_iterator(Successors.begin());
  }

  [[nodiscard]] const_pred_succ_iterator succs_end() const {
    return const_pred_succ_iterator(Successors.end());
  }

  pred_succ_iterator succs_end() {
    return pred_succ_iterator(Successors.end());
  }

  [[nodiscard]] llvm::iterator_range<const_pred_succ_iterator>
  successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  [[nodiscard]] llvm::iterator_range<pred_succ_iterator> successors() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  [[nodiscard]] unsigned succs_size() const { return Successors.size(); }

  [[nodiscard]] bool succs_empty() const { return Successors.empty(); }

  iterator getFirstNonDebugInstr(bool SkipPseudoOp = true);

  [[nodiscard]] const_iterator
  getFirstNonDebugInstr(bool SkipPseudoOp = true) const {
    return const_cast<PredicatedMachineBasicBlock *>(this)
        ->getFirstNonDebugInstr(SkipPseudoOp);
  }

  iterator getLastNonDebugInstr(bool SkipPseudoOp = true);

  [[nodiscard]] const_iterator
  getLastNonDebugInstr(bool SkipPseudoOp = true) const {
    return const_cast<PredicatedMachineBasicBlock *>(this)
        ->getLastNonDebugInstr(SkipPseudoOp);
  }

  [[nodiscard]] bool doesLastInstrModifyPredicate() const;

  void print(llvm::raw_ostream &OS, unsigned int Indent) const;

  LLVM_DUMP_METHOD void dump() const;
};

/// \brief Class used to construct \c PredicatedMachineBasicBlock instances
class PredMBBBuilder {
  PredicatedMachineBasicBlock Out;

public:
  explicit PredMBBBuilder(LinearMachineBasicBlock &Parent,
                          PredicatedMachineBasicBlock::PredicateValue EMV)
      : Out(Parent, EMV) {};

  PredMBBBuilder(const PredMBBBuilder &Other) = delete;

  PredMBBBuilder &operator=(const PredMBBBuilder &Other) = delete;

  /// Breaks down the \p MBB into a list of linked predicated machine basic
  /// block builders (i.e., their predecessors and successors are initialized)
  /// The list will contain two empty predicated blocks (the first one with \c
  /// One and the second one with \c ZeroOrOne predicate value) both before and
  /// after the "actual" blocks (a total of 4 blocks). These empty blocks can
  /// then be used to link the incoming and outgoing edges of other broken down
  /// \c llvm::MachineBasicBlocks together.
  /// \pre \p MBB must be attached to a \c llvm::MachineFunction
  static llvm::SmallVector<std::unique_ptr<PredMBBBuilder>, 6>
  BreakDownToPredicatedMBBs(LinearMachineBasicBlock &Parent,
                            llvm::MachineBasicBlock &MBB);

  void addPredecessorBlock(PredMBBBuilder &MBB) {
    Out.Predecessors.insert(MBB);
    MBB.Out.Successors.insert(*this);
  }

  void removePredecessorBlock(PredMBBBuilder &MBB) {
    Out.Predecessors.erase(MBB);
    MBB.Out.Successors.erase(*this);
  }

  void addSuccessorBlock(PredMBBBuilder &MBB) {
    Out.Successors.insert(MBB);
    MBB.Out.Predecessors.insert(*this);
  }

  void removeSuccessorBlock(PredMBBBuilder &MBB) {
    Out.Successors.erase(MBB);
    MBB.Out.Predecessors.erase(*this);
  }

  bool unlinkIfTrivialEmptyBlock();

  void setGlobalIndex(unsigned int Idx) { Out.GlobalIdx = Idx; };

  void setScalarIndex(unsigned int Idx) { Out.LinearMBBIdx = Idx; }

  PredicatedMachineBasicBlock &getPredMBB() { return Out; }

  [[nodiscard]] const PredicatedMachineBasicBlock &getPredMBB() const {
    return Out;
  }
};

inline PredicatedMachineBasicBlock::pred_succ_iterator::reference
PredicatedMachineBasicBlock::pred_succ_iterator::operator*() const {
  return It->get().getPredMBB();
}

inline PredicatedMachineBasicBlock::pred_succ_iterator::pointer
PredicatedMachineBasicBlock::pred_succ_iterator::operator->() const {
  return &It->get().getPredMBB();
}

inline PredicatedMachineBasicBlock::const_pred_succ_iterator::reference
PredicatedMachineBasicBlock::const_pred_succ_iterator::operator*() const {
  return It->get().getPredMBB();
}

inline PredicatedMachineBasicBlock::const_pred_succ_iterator::pointer
PredicatedMachineBasicBlock::const_pred_succ_iterator::operator->() const {
  return &It->get().getPredMBB();
}

} // namespace luthier

namespace llvm {
template <> struct GraphTraits<luthier::PredicatedMachineBasicBlock *> {
  using NodeRef = luthier::PredicatedMachineBasicBlock *;
  using ChildIteratorType = llvm::pointer_iterator<
      luthier::PredicatedMachineBasicBlock::pred_succ_iterator>;

  static NodeRef getEntryNode(luthier::PredicatedMachineBasicBlock *BB) {
    return BB;
  }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->succs_begin());
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->succs_end());
  }

  static unsigned getNumber(luthier::PredicatedMachineBasicBlock *BB) {
    return BB->getGlobalNumber();
  }
};

static_assert(GraphHasNodeNumbers<luthier::PredicatedMachineBasicBlock *>,
              "GraphTraits getNumber() not detected");

template <> struct GraphTraits<const luthier::PredicatedMachineBasicBlock *> {
  using NodeRef = const luthier::PredicatedMachineBasicBlock *;
  using ChildIteratorType = llvm::pointer_iterator<
      luthier::PredicatedMachineBasicBlock::const_pred_succ_iterator>;

  static NodeRef getEntryNode(const luthier::PredicatedMachineBasicBlock *BB) {
    return BB;
  }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->succs_begin());
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->succs_end());
  }

  static unsigned getNumber(const luthier::PredicatedMachineBasicBlock *BB) {
    return BB->getGlobalNumber();
  }
};

static_assert(GraphHasNodeNumbers<const luthier::PredicatedMachineBasicBlock *>,
              "GraphTraits getNumber() not detected");

template <>
struct GraphTraits<Inverse<luthier::PredicatedMachineBasicBlock *>> {
  using NodeRef = luthier::PredicatedMachineBasicBlock *;
  using ChildIteratorType = llvm::pointer_iterator<
      luthier::PredicatedMachineBasicBlock::pred_succ_iterator>;

  static NodeRef
  getEntryNode(Inverse<luthier::PredicatedMachineBasicBlock *> G) {
    return G.Graph;
  }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->preds_begin());
  }
  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->preds_end());
  }

  static unsigned getNumber(luthier::PredicatedMachineBasicBlock *BB) {
    return BB->getGlobalNumber();
  }
};

template <>
struct GraphTraits<Inverse<const luthier::PredicatedMachineBasicBlock *>> {
  using NodeRef = const luthier::PredicatedMachineBasicBlock *;
  using ChildIteratorType = llvm::pointer_iterator<
      luthier::PredicatedMachineBasicBlock::const_pred_succ_iterator>;

  static NodeRef
  getEntryNode(Inverse<const luthier::PredicatedMachineBasicBlock *> G) {
    return G.Graph;
  }

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->preds_begin());
  }
  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->preds_end());
  }

  static unsigned getNumber(const luthier::PredicatedMachineBasicBlock *BB) {
    return BB->getGlobalNumber();
  }
};

static_assert(
    GraphHasNodeNumbers<Inverse<const luthier::PredicatedMachineBasicBlock *>>,
    "GraphTraits getNumber() not detected");
} // namespace llvm

#endif