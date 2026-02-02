//===-- LinearMachineBasicBlock.h -------------------------------*- C++ -*-===//
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
/// \file LinearMachineBasicBlock.h
/// Defines the \c LinearMachineBasicBlock class.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_LINEAR_MACHINE_BASIC_BLOCK_H
#define LUTHIER_TOOLING_LINEAR_MACHINE_BASIC_BLOCK_H
#include "luthier/Tooling/PredicatedMachineBasicBlock.h"
#include <llvm/CodeGen/MachineBasicBlock.h>

namespace luthier {

class PredicatedMachineFunction;

class LinearMBBBuilder;

class PredMBBBuilder;

/// \brief a wrapper around a group of <tt>VectorMBB</tt>s inside a single
/// \c llvm::MachineBasicBlock for easier construction of the \c VectorCFG
class LinearMachineBasicBlock {
  friend LinearMBBBuilder;

  /// The CFG this scalar MBB belongs to
  PredicatedMachineFunction &ParentCFG;
  /// The MIR MBB this scalar block wraps around; If \c nullptr then this
  /// is an entry/exit block of the \c IPVectorCFG
  llvm::MachineBasicBlock *ParentMBB{nullptr};

  llvm::SmallVector<std::unique_ptr<PredMBBBuilder>, 6> PredMBBs{};

  LinearMachineBasicBlock(llvm::MachineBasicBlock &ParentMBB,
                          PredicatedMachineFunction &ParentCFG)
      : ParentCFG(ParentCFG), ParentMBB(&ParentMBB) {};

  explicit LinearMachineBasicBlock(PredicatedMachineFunction &ParentCFG)
      : ParentCFG(ParentCFG) {};

public:
  /// Disallowed copy construction
  LinearMachineBasicBlock(const LinearMachineBasicBlock &) = delete;

  /// Disallowed assignment operation
  LinearMachineBasicBlock &operator=(const LinearMachineBasicBlock &) = delete;

  class iterator {
    decltype(PredMBBs)::iterator It{};

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = PredicatedMachineBasicBlock;
    using reference = value_type &;
    using pointer = value_type *;

    explicit iterator(const decltype(PredMBBs)::iterator It) : It(It) {}

    iterator() = default;

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

    iterator operator--() {
      --It;
      return *this;
    }

    iterator operator--(int) {
      auto Copy = *this;
      --(*this);
      return Copy;
    }

    bool operator==(const iterator &Other) const { return It == Other.It; }

    bool operator!=(const iterator &Other) const { return !(*this == Other); }
  };

  class const_iterator {
    decltype(PredMBBs)::const_iterator It{};

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::forward_iterator_tag;
    using value_type = const PredicatedMachineBasicBlock;
    using reference = value_type &;
    using pointer = value_type *;

    explicit const_iterator(const decltype(PredMBBs)::const_iterator &It)
        : It(It) {}

    const_iterator() = default;

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

    const_iterator operator--() {
      ++It;
      return *this;
    }

    const_iterator operator--(int) {
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

  [[nodiscard]] const PredicatedMachineFunction &getParent() const {
    return ParentCFG;
  }

  [[nodiscard]] PredicatedMachineFunction &getParent() { return ParentCFG; }

  [[nodiscard]] const llvm::MachineBasicBlock *getMBB() const {
    return ParentMBB;
  }

  [[nodiscard]] llvm::MachineBasicBlock *getMBB() { return ParentMBB; }

  iterator begin() { return iterator(PredMBBs.begin()); }

  [[nodiscard]] const_iterator begin() const {
    return const_iterator(PredMBBs.begin());
  }

  iterator end() { return iterator(PredMBBs.end()); }

  [[nodiscard]] const_iterator end() const {
    return const_iterator(PredMBBs.end());
  }

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  reverse_iterator rbegin() {
    return reverse_iterator(iterator(PredMBBs.end()));
  }

  [[nodiscard]] const_reverse_iterator rbegin() const {
    return const_reverse_iterator(const_iterator(PredMBBs.end()));
  }

  reverse_iterator rend() {
    return reverse_iterator(iterator(PredMBBs.begin()));
  }

  [[nodiscard]] const_reverse_iterator rend() const {
    return const_reverse_iterator(const_iterator(PredMBBs.begin()));
  }

  [[nodiscard]] unsigned size() const { return PredMBBs.size(); }

  [[nodiscard]] bool empty() const { return PredMBBs.empty(); }

  [[nodiscard]] const PredicatedMachineBasicBlock &front() const {
    return PredMBBs.front()->getPredMBB();
  }

  [[nodiscard]] const PredicatedMachineBasicBlock &back() const {
    return PredMBBs.back()->getPredMBB();
  }

  void print(llvm::raw_ostream &OS, unsigned int Indent) const;

  LLVM_DUMP_METHOD void dump() const;
};

class LinearMBBBuilder {
  LinearMachineBasicBlock Out;

  PredMBBBuilder *EntryPredOneBlock{nullptr};
  PredMBBBuilder *EntryPredZeroOrOneBlock{nullptr};
  PredMBBBuilder *ExitPredOneBlock{nullptr};
  PredMBBBuilder *ExitPredZeroOrOneBlock{nullptr};

  explicit LinearMBBBuilder(PredicatedMachineFunction &ParentCFG)
      : Out(ParentCFG) {};

  explicit LinearMBBBuilder(llvm::MachineBasicBlock &MBB,
                            PredicatedMachineFunction &ParentCFG)
      : Out(MBB, ParentCFG) {};

public:
  LinearMBBBuilder(const LinearMBBBuilder &Other) = delete;

  LinearMBBBuilder &operator=(const LinearMBBBuilder &Other) = delete;

  LinearMachineBasicBlock &getLinearMBB() { return Out; }

  [[nodiscard]] const LinearMachineBasicBlock &getLinearMBB() const {
    return Out;
  }

  static std::unique_ptr<LinearMBBBuilder>
  createLinearMBB(llvm::MachineBasicBlock &ParentMBB,
                  PredicatedMachineFunction &ParentCFG);

  static std::unique_ptr<LinearMBBBuilder>
  createEntryLinearMBB(PredicatedMachineFunction &ParentCFG);

  [[nodiscard]] bool hasFinalizedLinking() const {
    return !EntryPredOneBlock || !EntryPredZeroOrOneBlock ||
           !ExitPredOneBlock || !ExitPredZeroOrOneBlock;
  }

  using iterator = decltype(Out.PredMBBs)::iterator;
  using const_iterator = decltype(Out.PredMBBs)::const_iterator;

  iterator begin() { return Out.PredMBBs.begin(); }

  [[nodiscard]] const_iterator begin() const { return Out.PredMBBs.begin(); }

  iterator end() { return Out.PredMBBs.end(); }

  [[nodiscard]] const_iterator end() const { return Out.PredMBBs.end(); }

  using reverse_iterator = decltype(Out.PredMBBs)::reverse_iterator;
  using const_reverse_iterator = decltype(Out.PredMBBs)::const_reverse_iterator;

  reverse_iterator rbegin() { return Out.PredMBBs.rbegin(); }

  [[nodiscard]] const_reverse_iterator rbegin() const {
    return Out.PredMBBs.rbegin();
  }

  reverse_iterator rend() { return Out.PredMBBs.rend(); }

  [[nodiscard]] const_reverse_iterator rend() const {
    return Out.PredMBBs.rend();
  }

  [[nodiscard]] unsigned size() const { return Out.PredMBBs.size(); }

  [[nodiscard]] bool empty() const { return Out.PredMBBs.empty(); }

  void addSuccessor(LinearMBBBuilder &LMBB);

  void addPredecessor(LinearMBBBuilder &LMBB);

  void pruneTrivialBlocks();
};

} // namespace luthier

#endif