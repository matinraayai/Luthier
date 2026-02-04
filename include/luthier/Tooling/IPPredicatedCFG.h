//===-- IPPredicatedCFG.h ----------------------------------------*- C++-*-===//
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
/// \file IPPredicatedCFG.h
/// Describes the \c IPPredicatedCFG class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_IP_PREDICATED_CFG_H
#define LUTHIER_TOOLING_IP_PREDICATED_CFG_H
#include "luthier/Common/DenseMapInfo.h"
#include "luthier/Tooling/PredicatedMachineBasicBlock.h"
#include "luthier/Tooling/PredicatedMachineFunction.h"
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Support/GenericDomTree.h>

namespace luthier {

class IPPredicatedCFG;

class PredicatedMachineFunction;

class PredicatedMachineBasicBlock;

/// \brief An inter-procedural control-flow graph representation for the MIR of
/// the AMDGPU backend that, in addition to scalar branches, regards
/// instructions that manipulate the execute mask as well as call instructions
/// as a terminator for its basic blocks
/// \details The vector CFG can be used to perform data flow analysis
/// involving vector registers (e.g. register liveness) which cannot be done
/// with the CFG of LLVM MIR for the AMD GPU backend
class IPPredicatedCFG {
private:
  /// Where we store all the predicated MFs
  llvm::SmallVector<std::unique_ptr<PredMFBuilder>> PredMFs{};

  /// Mapping between all machine functions in the module and their predicated
  /// machine function
  llvm::SmallDenseMap<std::reference_wrapper<const llvm::MachineFunction>,
                      std::reference_wrapper<PredMFBuilder>>
      MFToPredMF{};

  unsigned NumVecMBBs{0};

  PredMFBuilder *EntryPredMF;

  IPPredicatedCFG() = default;

public:
  class iterator {
    decltype(PredMFs)::iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = PredicatedMachineFunction;
    using reference = value_type &;
    using pointer = value_type *;

    explicit iterator(const decltype(PredMFs)::iterator &It) : It(It) {}

    reference operator*() const { return (*It)->getPredMF(); }

    pointer operator->() const { return &(*It)->getPredMF(); }

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
    decltype(PredMFs)::const_iterator It;

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = const PredicatedMachineFunction;
    using reference = value_type &;
    using pointer = value_type *;

    explicit const_iterator(const decltype(PredMFs)::const_iterator &It)
        : It(It) {}

    reference operator*() const { return (*It)->getPredMF(); }

    pointer operator->() const { return &(*It)->getPredMF(); }

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

  iterator begin() { return iterator(PredMFs.begin()); }

  [[nodiscard]] const_iterator begin() const {
    return const_iterator(PredMFs.begin());
  }

  iterator end() { return iterator(PredMFs.end()); }

  [[nodiscard]] const_iterator end() const {
    return const_iterator(PredMFs.end());
  }

  [[nodiscard]] unsigned size() const { return PredMFs.size(); }

  [[nodiscard]] bool empty() const { return PredMFs.empty(); }

  [[nodiscard]] const PredicatedMachineFunction &front() const {
    return PredMFs.front()->getPredMF();
  }

  [[nodiscard]] const PredicatedMachineFunction &back() const {
    return PredMFs.back()->getPredMF();
  }

  [[nodiscard]] PredicatedMachineFunction &back() {
    return PredMFs.back()->getPredMF();
  }

  [[nodiscard]] bool contains(const llvm::MachineFunction &MF) const {
    return MFToPredMF.contains(MF);
  }

  PredicatedMachineFunction &operator[](const llvm::MachineFunction &MF) {
    assert(MFToPredMF.contains(MF) && "Entry is not in the map");
    return MFToPredMF.at(MF).get().getPredMF();
  }

  [[nodiscard]] const PredicatedMachineFunction &
  at(const llvm::MachineFunction &MF) const {
    assert(MFToPredMF.contains(MF) && "Entry is not in the map");
    return MFToPredMF.at(MF).get().getPredMF();
  }

  [[nodiscard]] unsigned getNumVecMBBs() const { return NumVecMBBs; }

  void print(llvm::raw_ostream &OS) const;

  LLVM_DUMP_METHOD void dump() const;

  PredicatedMachineFunction &getEntry() {
    assert(EntryPredMF && "Entry function must be set");
    return EntryPredMF->getPredMF();
  }

  [[nodiscard]] const PredicatedMachineFunction &getEntry() const {
    return const_cast<IPPredicatedCFG *>(this)->getEntry();
  }

  PredicatedMachineBasicBlock &getPredMBB(const llvm::MachineInstr &MI);

  [[nodiscard]] const PredicatedMachineBasicBlock &
  getPredMBB(const llvm::MachineInstr &MI) const {
    return const_cast<IPPredicatedCFG *>(this)->getPredMBB(MI);
  }

  /// get entry block
  /// Iterate over all predicated mbbs

  static llvm::Expected<std::unique_ptr<IPPredicatedCFG>>
  getIPPredCFG(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
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
        : IPPredCFG(std::move(IPCFG)) {};

  public:
    bool invalidate(llvm::Module &M, const llvm::PreservedAnalyses &PA,
                    llvm::ModuleAnalysisManager::Invalidator &Inv);

    [[nodiscard]] const IPPredicatedCFG &getVecCFG() const {
      return *IPPredCFG;
    }

    IPPredicatedCFG &getVecCFG() { return *IPPredCFG; }
  };

  IPPredCFGAnalysis() = default;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

class IPPredCFGPrinter : public llvm::PassInfoMixin<IPPredCFGAnalysis> {

private:
  llvm::raw_ostream &OS;

public:
  explicit IPPredCFGPrinter(llvm::raw_ostream &OS) : OS(OS) {};

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

namespace llvm {

template <>
struct GraphTraits<luthier::IPPredicatedCFG *>
    : public GraphTraits<luthier::PredicatedMachineBasicBlock *> {
  static NodeRef getEntryNode(luthier::IPPredicatedCFG *F) {
    luthier::PredicatedMachineFunction &EntryPredMF = F->getEntry();
    return F->empty() ? nullptr : &*EntryPredMF.begin()->begin();
  }

  class node_iterator {
    luthier::IPPredicatedCFG::iterator PredMFIter;
    luthier::PredicatedMachineFunction::iterator LinearMBBIter;
    luthier::LinearMachineBasicBlock::iterator PredMBBIter;

  public:
    explicit node_iterator(
        luthier::IPPredicatedCFG::iterator PredMFIter,
        luthier::PredicatedMachineFunction::iterator LinearMBBIter = {},
        luthier::LinearMachineBasicBlock::iterator PredMBBIter = {})
        : PredMFIter(PredMFIter), LinearMBBIter(LinearMBBIter),
          PredMBBIter(PredMBBIter) {}

    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = luthier::PredicatedMachineBasicBlock;
    using pointer = value_type *;
    using reference = value_type &;

    reference operator*() const { return *PredMBBIter; }

    pointer operator->() const { return &*PredMBBIter; }

    node_iterator operator++() {
      ++PredMBBIter;
      if (PredMBBIter == LinearMBBIter->end()) {
        assert(LinearMBBIter != PredMFIter->end() && "went over the end");
        ++LinearMBBIter;
        if (LinearMBBIter == PredMFIter->end()) {
          assert(PredMFIter != PredMFIter->getParent().end() &&
                 "went over the end");
          ++PredMFIter;
          LinearMBBIter = PredMFIter->begin();
        }
        PredMBBIter = LinearMBBIter->begin();
      }
      return *this;
    }

    node_iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const node_iterator &Other) const {
      return PredMFIter == Other.PredMFIter &&
             PredMBBIter == Other.PredMBBIter &&
             LinearMBBIter == Other.LinearMBBIter;
    }

    bool operator!=(const node_iterator &Other) const {
      return !(*this == Other);
    }
  };

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph

  static node_iterator nodes_begin(luthier::IPPredicatedCFG *F) {
    auto MFPredIt = F->end();
    auto LinearMBBIt = MFPredIt != F->end()
                           ? MFPredIt->begin()
                           : luthier::PredicatedMachineFunction::iterator{};
    auto PredMBBIt =
        LinearMBBIt != luthier::PredicatedMachineFunction::iterator{}
            ? LinearMBBIt->begin()
            : luthier::LinearMachineBasicBlock::iterator{};
    return node_iterator{MFPredIt, LinearMBBIt, PredMBBIt};
  }

  static node_iterator nodes_end(luthier::IPPredicatedCFG *F) {
    auto MFPredIt = F->begin();
    auto LinearMBBIt = F->empty()
                           ? luthier::PredicatedMachineFunction::iterator{}
                           : F->back().end();
    auto PredMBBIt =
        LinearMBBIt != luthier::PredicatedMachineFunction::iterator{}
            ? F->back().back().end()
            : luthier::LinearMachineBasicBlock::iterator{};
    return node_iterator{MFPredIt, LinearMBBIt, PredMBBIt};
  }

  static unsigned size(luthier::IPPredicatedCFG *F) {
    return F->getNumVecMBBs();
  }

  static unsigned getMaxNumber(luthier::IPPredicatedCFG *F) {
    return F->getNumVecMBBs();
  }
  static unsigned getNumberEpoch(luthier::IPPredicatedCFG *F) {
    return F->getNumVecMBBs();
  }
};

template <>
struct GraphTraits<const luthier::IPPredicatedCFG *>
    : public GraphTraits<const luthier::PredicatedMachineBasicBlock *> {
  static NodeRef getEntryNode(const luthier::IPPredicatedCFG *F) {
    const luthier::PredicatedMachineFunction &EntryPredMF = F->getEntry();
    return F->empty() ? nullptr : &*EntryPredMF.begin()->begin();
  }

  class node_iterator {
    luthier::IPPredicatedCFG::const_iterator PredMFIter;
    luthier::PredicatedMachineFunction::const_iterator LinearMBBIter;
    luthier::LinearMachineBasicBlock::const_iterator PredMBBIter;

  public:
    node_iterator(
        luthier::IPPredicatedCFG::const_iterator PredMFIter,
        luthier::PredicatedMachineFunction::const_iterator LinearMBBIter,
        luthier::LinearMachineBasicBlock::const_iterator PredMBBIter)
        : PredMFIter(PredMFIter), LinearMBBIter(LinearMBBIter),
          PredMBBIter(PredMBBIter) {}

    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = const luthier::PredicatedMachineBasicBlock;
    using pointer = value_type *;
    using reference = value_type &;

    reference operator*() const { return *PredMBBIter; }

    pointer operator->() const { return &*PredMBBIter; }

    node_iterator operator++() {
      ++PredMBBIter;
      if (PredMBBIter == LinearMBBIter->end()) {
        assert(LinearMBBIter != PredMFIter->end() && "went over the end");
        ++LinearMBBIter;
        if (LinearMBBIter == PredMFIter->end()) {
          assert(PredMFIter != PredMFIter->getParent().end() &&
                 "went over the end");
          ++PredMFIter;
          LinearMBBIter = PredMFIter->begin();
        }
        PredMBBIter = LinearMBBIter->begin();
      }
      return *this;
    }

    node_iterator operator++(int) {
      auto Copy = *this;
      ++(*this);
      return Copy;
    }

    bool operator==(const node_iterator &Other) const {
      return PredMFIter == Other.PredMFIter &&
             PredMBBIter == Other.PredMBBIter &&
             LinearMBBIter == Other.LinearMBBIter;
    }

    bool operator!=(const node_iterator &Other) const {
      return !(*this == Other);
    }
  };

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph

  static node_iterator nodes_begin(const luthier::IPPredicatedCFG *F) {
    auto MFPredIt = F->begin();
    auto LinearMBBIt =
        MFPredIt != F->end()
            ? MFPredIt->begin()
            : luthier::PredicatedMachineFunction::const_iterator{};
    auto PredMBBIt =
        LinearMBBIt != luthier::PredicatedMachineFunction::const_iterator{}
            ? LinearMBBIt->begin()
            : luthier::LinearMachineBasicBlock::const_iterator{};
    return node_iterator{MFPredIt, LinearMBBIt, PredMBBIt};
  }

  static node_iterator nodes_end(const luthier::IPPredicatedCFG *F) {
    auto MFPredIt = F->end();
    auto LinearMBBIt =
        F->empty() ? luthier::PredicatedMachineFunction::const_iterator{}
                   : F->back().end();
    auto PredMBBIt =
        LinearMBBIt != luthier::PredicatedMachineFunction::const_iterator{}
            ? F->back().back().end()
            : luthier::LinearMachineBasicBlock::const_iterator{};
    return node_iterator{MFPredIt, LinearMBBIt, PredMBBIt};
  }

  static unsigned size(luthier::IPPredicatedCFG *F) {
    return F->getNumVecMBBs();
  }

  static unsigned getMaxNumber(luthier::IPPredicatedCFG *F) {
    return F->getNumVecMBBs();
  }
  static unsigned getNumberEpoch(luthier::IPPredicatedCFG *F) {
    return F->getNumVecMBBs();
  }
};

/// Specialization of \c DomTreeNodeTraits because the normal one works with
/// "Normal" IR/MIR
template <> struct DomTreeNodeTraits<luthier::PredicatedMachineBasicBlock> {
  using NodeType = luthier::PredicatedMachineBasicBlock;
  using NodePtr = luthier::PredicatedMachineBasicBlock *;
  using ParentPtr = luthier::IPPredicatedCFG *;
  using ParentType = std::remove_pointer_t<ParentPtr>;

  static luthier::PredicatedMachineBasicBlock *getEntryNode(ParentPtr Parent) {
    return GraphTraits<luthier::IPPredicatedCFG *>::getEntryNode(Parent);
  }

  static ParentPtr getParent(NodePtr BB) {
    return &BB->getParent().getParent().getParent();
  }
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

  static ParentPtr getParent(NodePtr BB) {
    return &BB->getParent().getParent().getParent();
  }
};

} // namespace llvm

#endif