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

  /// The range of instructions in the block; bundles and debug instructions
  /// are ignored
  llvm::iterator_range<llvm::MachineBasicBlock::instr_iterator> Instructions{
      {}, {}};

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

  using iterator = llvm::MachineBasicBlock::instr_iterator;
  using const_iterator = llvm::MachineBasicBlock::const_instr_iterator;

  using reverse_iterator = std::reverse_iterator<iterator>;

  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  iterator begin() { return Instructions.begin(); }

  [[nodiscard]] const_iterator begin() const {
    return Instructions.begin();
  }

  [[nodiscard]] bool contains(const llvm::MachineInstr &MI) const {
    return MIsSet.contains(&MI);
  }

  llvm::MachineInstr &front() { return *begin(); }

  [[nodiscard]] const llvm::MachineInstr &front() const { return *begin(); }

  [[nodiscard]] iterator end() { return Instructions.end(); }

  [[nodiscard]] const_iterator end() const {
    return Instructions.end();
  }

  reverse_iterator rbegin() { return std::make_reverse_iterator(end()); }

  [[nodiscard]] const_reverse_iterator rbegin() const {
    return std::make_reverse_iterator(end());
  }

  reverse_iterator rend() { return std::make_reverse_iterator(begin()); }

  [[nodiscard]] const_reverse_iterator rend() const {
    return std::make_reverse_iterator(begin());
  }

  [[nodiscard]] const llvm::MachineInstr &back() const {
    return *(--Instructions.end());
  }

  [[nodiscard]] bool empty() const { return Instructions.empty(); }

  [[nodiscard]] unsigned size() const {
    return std::distance(Instructions.begin(), Instructions.end());
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

  PredSuccSetType::iterator preds_begin() { return Predecessors.begin(); }

  [[nodiscard]] PredSuccSetType::const_iterator preds_begin() const {
    return Predecessors.begin();
  }

  PredSuccSetType::iterator preds_end() { return Predecessors.end(); }

  [[nodiscard]] PredSuccSetType::const_iterator preds_end() const {
    return Predecessors.end();
  }

  [[nodiscard]] llvm::iterator_range<PredSuccSetType::const_iterator>
  predecessors() const {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  [[nodiscard]] llvm::iterator_range<PredSuccSetType::iterator> predecessors() {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  [[nodiscard]] unsigned preds_size() const { return Predecessors.size(); }

  [[nodiscard]] bool preds_empty() const { return Predecessors.empty(); }

  PredSuccSetType::iterator succs_begin() { return Successors.begin(); }

  [[nodiscard]] PredSuccSetType::const_iterator succs_begin() const {
    return Successors.begin();
  }

  [[nodiscard]] PredSuccSetType::const_iterator succs_end() const {
    return Successors.end();
  }

  PredSuccSetType::iterator succs_end() { return Successors.end(); }

  [[nodiscard]] llvm::iterator_range<PredSuccSetType::const_iterator>
  successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  [[nodiscard]] llvm::iterator_range<PredSuccSetType::iterator> successors() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  [[nodiscard]] unsigned succs_size() const { return Successors.size(); }

  [[nodiscard]] bool succs_empty() const { return Successors.empty(); }

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

} // namespace luthier

// namespace llvm {
// template <> struct GraphTraits<luthier::PredicatedMachineBasicBlock *> {
//   using NodeRef = luthier::PredicatedMachineBasicBlock *;
//   using ChildIteratorType = SmallDenseSet<luthier::PredMBBBuilder
//   *>::iterator;
//
//   static NodeRef getEntryNode(luthier::PredicatedMachineBasicBlock *BB) {
//     return BB;
//   }
//
//   static ChildIteratorType child_begin(NodeRef N) { return N->succs_begin();
//   }
//
//   static ChildIteratorType child_end(NodeRef N) {
//     return N->successors().end();
//   }
//
//   static unsigned getNumber(luthier::PredicatedMachineBasicBlock *BB) {
//     return BB->getGlobalNumber();
//   }
// };
//
// static_assert(GraphHasNodeNumbers<luthier::PredicatedMachineBasicBlock *>,
//               "GraphTraits getNumber() not detected");
//
// template <> struct GraphTraits<const luthier::PredicatedMachineBasicBlock *>
// {
//   using NodeRef = const luthier::PredicatedMachineBasicBlock *;
//   using ChildIteratorType =
//       SmallDenseSet<luthier::PredMBBBuilder *>::const_iterator;
//
//   static NodeRef getEntryNode(const luthier::PredicatedMachineBasicBlock *BB)
//   {
//     return BB;
//   }
//
//   static ChildIteratorType child_begin(NodeRef N) {
//     return N->successors().begin();
//   }
//
//   static ChildIteratorType child_end(NodeRef N) {
//     return N->successors().end();
//   }
//
//   static unsigned getNumber(const luthier::PredicatedMachineBasicBlock *BB) {
//     return BB->getGlobalNumber();
//   }
// };
//
// static_assert(GraphHasNodeNumbers<const luthier::PredicatedMachineBasicBlock
// *>,
//               "GraphTraits getNumber() not detected");
//
// template <>
// struct GraphTraits<Inverse<luthier::PredicatedMachineBasicBlock *>> {
//   using NodeRef = luthier::PredicatedMachineBasicBlock *;
//   using ChildIteratorType = SmallDenseSet<luthier::PredMBBBuilder
//   *>::iterator;
//
//   static NodeRef
//   getEntryNode(Inverse<luthier::PredicatedMachineBasicBlock *> G) {
//     return G.Graph;
//   }
//
//   static ChildIteratorType child_begin(NodeRef N) { return N->preds_begin();
//   } static ChildIteratorType child_end(NodeRef N) { return N->preds_end(); }
//
//   static unsigned getNumber(luthier::PredicatedMachineBasicBlock *BB) {
//     return BB->getGlobalNumber();
//   }
// };
//
// template <>
// struct GraphTraits<Inverse<const luthier::PredicatedMachineBasicBlock *>> {
//   using NodeRef = const luthier::PredicatedMachineBasicBlock *;
//   using ChildIteratorType =
//       SmallDenseSet<luthier::PredMBBBuilder *>::const_iterator;
//
//   static NodeRef
//   getEntryNode(Inverse<const luthier::PredicatedMachineBasicBlock *> G) {
//     return G.Graph;
//   }
//
//   static ChildIteratorType child_begin(NodeRef N) { return N->preds_begin();
//   } static ChildIteratorType child_end(NodeRef N) { return N->preds_end(); }
//
//   static unsigned getNumber(const luthier::PredicatedMachineBasicBlock *BB) {
//     return BB->getGlobalNumber();
//   }
// };
//
// static_assert(
//     GraphHasNodeNumbers<Inverse<const luthier::PredicatedMachineBasicBlock
//     *>>, "GraphTraits getNumber() not detected");
//
// template <> struct DomTreeNodeTraits<luthier::PredicatedMachineBasicBlock> {
//   using NodeType = luthier::PredicatedMachineBasicBlock;
//   using NodePtr = luthier::PredicatedMachineBasicBlock *;
//   using ParentPtr = luthier::IPPredicatedCFG *;
//   using ParentType = std::remove_pointer_t<ParentPtr>;
//
//   static luthier::PredicatedMachineBasicBlock *getEntryNode(ParentPtr Parent)
//   {
//     auto MFPredIt = Parent->getEntry();
//     return MFPredIt != Parent->end() && !MFPredIt->empty()
//                ? &*MFPredIt->begin()->begin()
//                : nullptr;
//   }
//
//   static ParentPtr getParent(NodePtr BB) {
//     return &BB->getParent().getParent().getParent();
//   }
// };
//
// template <>
// struct DomTreeNodeTraits<const luthier::PredicatedMachineBasicBlock> {
//   using NodeType = const luthier::PredicatedMachineBasicBlock;
//   using NodePtr = const luthier::PredicatedMachineBasicBlock *;
//   using ParentPtr = const luthier::IPPredicatedCFG *;
//   using ParentType = std::remove_pointer_t<ParentPtr>;
//
//   static const luthier::PredicatedMachineBasicBlock *
//   getEntryNode(ParentPtr Parent) {
//     auto MFPredIt = Parent->getEntry();
//     return MFPredIt != Parent->end() && !MFPredIt->empty()
//                ? &*MFPredIt->begin()->begin()
//                : nullptr;
//   }
//
//   static ParentPtr getParent(NodePtr BB) {
//     return &BB->getParent().getParent().getParent();
//   }
// };
// // } // namespace llvm

#endif