//===-- IPPredicatedCFG.h ----------------------------------------*- C++-*-===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file IPPredicatedCFG.h
/// Describes the \c IPPredicatedCFG class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_IP_PREDICATED_CFG_H
#define LUTHIER_TOOL_CODE_GEN_IP_PREDICATED_CFG_H
#include "luthier/Common/DenseMapInfo.h"
#include "luthier/ToolCodeGen/InstrumentPrototype.h"
#include "luthier/ToolCodeGen/PredicatedMachineBasicBlock.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/iterator.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/GenericDomTree.h>

namespace luthier {

class IPPredicatedCFG;
class PredicatedMachineBasicBlock;

/// \brief Inter-procedural predicated control-flow graph for target modules
///
/// \details Each node is a \c PredicatedMachineBasicBlock — a single MBB broken
/// down to have either scalar or vector instructions. Intra-procedural edges
/// come from MBB successor links; inter-procedural call edges come from
/// \c TraceCallGraph.  MBBs whose call targets could not be fully resolved
/// are flagged with \c hasUnresolvedEdges().
class IPPredicatedCFG {
private:
  llvm::SmallVector<std::unique_ptr<PredMBBBuilder>> AllPredMBBs{};

  llvm::DenseMap<std::reference_wrapper<const llvm::MachineBasicBlock>,
                 PredMBBBuilder *>
      MBBToPredMBB{};

  /// Maps each MachineFunction to the PredMBBBuilder of its entry MBB,
  /// used when wiring inter-procedural call edges.
  llvm::DenseMap<const llvm::MachineFunction *, PredMBBBuilder *>
      MFToEntryPredMBB{};

  unsigned NumPredMBBs{0};

  PredMBBBuilder *EntryPredMBB{nullptr};

  IPPredicatedCFG() = default;

public:
  class iterator {
    decltype(AllPredMBBs)::iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = PredicatedMachineBasicBlock;
    using reference = value_type &;
    using pointer = value_type *;

    explicit iterator(const decltype(AllPredMBBs)::iterator &It) : It(It) {}

    reference operator*() const { return (*It)->getPredMBB(); }
    pointer operator->() const { return &(*It)->getPredMBB(); }

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
    decltype(AllPredMBBs)::const_iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = const PredicatedMachineBasicBlock;
    using reference = value_type &;
    using pointer = value_type *;

    explicit const_iterator(const decltype(AllPredMBBs)::const_iterator &It)
        : It(It) {}

    reference operator*() const { return (*It)->getPredMBB(); }
    pointer operator->() const { return &(*It)->getPredMBB(); }

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

  iterator begin() { return iterator(AllPredMBBs.begin()); }
  [[nodiscard]] const_iterator begin() const {
    return const_iterator(AllPredMBBs.begin());
  }
  iterator end() { return iterator(AllPredMBBs.end()); }
  [[nodiscard]] const_iterator end() const {
    return const_iterator(AllPredMBBs.end());
  }

  [[nodiscard]] unsigned size() const { return AllPredMBBs.size(); }
  [[nodiscard]] bool empty() const { return AllPredMBBs.empty(); }
  [[nodiscard]] unsigned getNumPredMBBs() const { return NumPredMBBs; }

  [[nodiscard]] const PredicatedMachineBasicBlock &front() const {
    return AllPredMBBs.front()->getPredMBB();
  }
  [[nodiscard]] const PredicatedMachineBasicBlock &back() const {
    return AllPredMBBs.back()->getPredMBB();
  }
  [[nodiscard]] PredicatedMachineBasicBlock &back() {
    return AllPredMBBs.back()->getPredMBB();
  }

  [[nodiscard]] bool contains(const llvm::MachineBasicBlock &MBB) const {
    return MBBToPredMBB.contains(MBB);
  }

  PredicatedMachineBasicBlock &operator[](const llvm::MachineBasicBlock &MBB) {
    assert(MBBToPredMBB.contains(MBB) && "MBB not in IPPredicatedCFG");
    return MBBToPredMBB.at(MBB)->getPredMBB();
  }

  [[nodiscard]] const PredicatedMachineBasicBlock &
  at(const llvm::MachineBasicBlock &MBB) const {
    assert(MBBToPredMBB.contains(MBB) && "MBB not in IPPredicatedCFG");
    return MBBToPredMBB.at(MBB)->getPredMBB();
  }

  /// Returns the entry \c PredicatedMachineBasicBlock (first MBB of the entry
  /// function).
  PredicatedMachineBasicBlock &getEntry() {
    assert(EntryPredMBB && "Entry MBB must be set");
    return EntryPredMBB->getPredMBB();
  }
  [[nodiscard]] const PredicatedMachineBasicBlock &getEntry() const {
    return const_cast<IPPredicatedCFG *>(this)->getEntry();
  }

  PredicatedMachineBasicBlock &getPredMBB(const llvm::MachineInstr &MI);
  [[nodiscard]] const PredicatedMachineBasicBlock &
  getPredMBB(const llvm::MachineInstr &MI) const {
    return const_cast<IPPredicatedCFG *>(this)->getPredMBB(MI);
  }

  void print(llvm::raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;

  static llvm::Expected<std::unique_ptr<IPPredicatedCFG>>
  getIPPredCFG(InstrumentPrototype &IP,
               InstrumentPrototypeAnalysisManager &IPAM);
};

class IPPredCFGAnalysis : public llvm::AnalysisInfoMixin<IPPredCFGAnalysis> {
  friend llvm::AnalysisInfoMixin<IPPredCFGAnalysis>;

private:
  static llvm::AnalysisKey Key;

public:
  class Result {
    friend IPPredCFGAnalysis;

    std::unique_ptr<IPPredicatedCFG> IPPredCFG;

    explicit Result(std::unique_ptr<IPPredicatedCFG> IPCFG)
        : IPPredCFG(std::move(IPCFG)) {}

  public:
    bool invalidate(InstrumentPrototype &IP, const llvm::PreservedAnalyses &PA,
                    InstrumentPrototypeAnalysisManager::Invalidator &Inv);

    [[nodiscard]] const IPPredicatedCFG &getVecCFG() const {
      return *IPPredCFG;
    }
    IPPredicatedCFG &getVecCFG() { return *IPPredCFG; }
  };

  IPPredCFGAnalysis() = default;

  Result run(InstrumentPrototype &IP,
             InstrumentPrototypeAnalysisManager &IPAM);
};

class IPPredCFGPrinter : public llvm::PassInfoMixin<IPPredCFGAnalysis> {
  llvm::raw_ostream &OS;

public:
  explicit IPPredCFGPrinter(llvm::raw_ostream &OS) : OS(OS) {}

  llvm::PreservedAnalyses run(InstrumentPrototype &IP,
                              InstrumentPrototypeAnalysisManager &IPAM);
};

} // namespace luthier

namespace llvm {

template <>
struct GraphTraits<luthier::IPPredicatedCFG *>
    : public GraphTraits<luthier::PredicatedMachineBasicBlock *> {
  static NodeRef getEntryNode(luthier::IPPredicatedCFG *F) {
    return F->empty() ? nullptr : &F->getEntry();
  }

  using nodes_iterator =
      llvm::pointer_iterator<luthier::IPPredicatedCFG::iterator>;

  static nodes_iterator nodes_begin(luthier::IPPredicatedCFG *F) {
    return nodes_iterator(F->begin());
  }
  static nodes_iterator nodes_end(luthier::IPPredicatedCFG *F) {
    return nodes_iterator(F->end());
  }
  static unsigned size(luthier::IPPredicatedCFG *F) {
    return F->getNumPredMBBs();
  }
  static unsigned getMaxNumber(luthier::IPPredicatedCFG *F) {
    return F->getNumPredMBBs();
  }
  static unsigned getNumberEpoch(luthier::IPPredicatedCFG *F) {
    return F->getNumPredMBBs();
  }
};

template <>
struct GraphTraits<const luthier::IPPredicatedCFG *>
    : public GraphTraits<const luthier::PredicatedMachineBasicBlock *> {
  static NodeRef getEntryNode(const luthier::IPPredicatedCFG *F) {
    return F->empty() ? nullptr : &F->getEntry();
  }

  using nodes_iterator =
      llvm::pointer_iterator<luthier::IPPredicatedCFG::const_iterator>;

  static nodes_iterator nodes_begin(const luthier::IPPredicatedCFG *F) {
    return nodes_iterator(F->begin());
  }
  static nodes_iterator nodes_end(const luthier::IPPredicatedCFG *F) {
    return nodes_iterator(F->end());
  }
  static unsigned size(const luthier::IPPredicatedCFG *F) {
    return F->getNumPredMBBs();
  }
  static unsigned getMaxNumber(const luthier::IPPredicatedCFG *F) {
    return F->getNumPredMBBs();
  }
  static unsigned getNumberEpoch(const luthier::IPPredicatedCFG *F) {
    return F->getNumPredMBBs();
  }
};

template <> struct DomTreeNodeTraits<luthier::PredicatedMachineBasicBlock> {
  using NodeType = luthier::PredicatedMachineBasicBlock;
  using NodePtr = luthier::PredicatedMachineBasicBlock *;
  using ParentPtr = luthier::IPPredicatedCFG *;
  using ParentType = std::remove_pointer_t<ParentPtr>;

  static luthier::PredicatedMachineBasicBlock *getEntryNode(ParentPtr Parent) {
    return GraphTraits<luthier::IPPredicatedCFG *>::getEntryNode(Parent);
  }
  static ParentPtr getParent(NodePtr BB) { return &BB->getParent(); }
};

template <>
struct DomTreeNodeTraits<const luthier::PredicatedMachineBasicBlock> {
  using NodeType = const luthier::PredicatedMachineBasicBlock;
  using NodePtr = const luthier::PredicatedMachineBasicBlock *;
  using ParentPtr = const luthier::IPPredicatedCFG *;
  using ParentType = std::remove_pointer_t<ParentPtr>;

  static const luthier::PredicatedMachineBasicBlock *
  getEntryNode(ParentPtr Parent) {
    return GraphTraits<const luthier::IPPredicatedCFG *>::getEntryNode(Parent);
  }
  static ParentPtr getParent(NodePtr BB) { return &BB->getParent(); }
};

} // namespace llvm

#endif
