//===-- PredicatedMachineFunction.h ------------------------------*- C++-*-===//
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
/// \file PredicatedMachineFunction.h
/// Describes the \c PredicatedMachineFunction class and its builder.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_PREDICATED_MACHINE_FUNCTION_H
#define LUTHIER_TOOLING_PREDICATED_MACHINE_FUNCTION_H
#include "luthier/Common/DenseMapInfo.h"
#include "luthier/Tooling/LinearMachineBasicBlock.h"

namespace luthier {

class PredicatedMachineBasicBlock;

class LinearMachineBasicBlock;

class IPPredicatedCFG;

class PredMFBuilder;

class PredicatedMachineFunction {
  friend PredMFBuilder;

  IPPredicatedCFG &ParentCFG;

  llvm::MachineFunction &MF;

  llvm::SmallVector<std::unique_ptr<LinearMBBBuilder>> LinearMBBs{};

  llvm::SmallDenseMap<std::reference_wrapper<const llvm::MachineBasicBlock>,
                      std::reference_wrapper<LinearMBBBuilder>>
      MBBToLinearMBBs{};

  PredicatedMachineFunction(IPPredicatedCFG &ParentCFG,
                            llvm::MachineFunction &MF)
      : ParentCFG(ParentCFG), MF(MF) {};

public:
  /// Disallowed copy construction
  PredicatedMachineFunction(const PredicatedMachineFunction &) = delete;

  /// Disallowed assignment operation
  PredicatedMachineFunction &
  operator=(const PredicatedMachineFunction &) = delete;

  class iterator {
    decltype(LinearMBBs)::iterator It{};

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = LinearMachineBasicBlock;
    using reference = value_type &;
    using pointer = value_type *;

    explicit iterator(const decltype(LinearMBBs)::iterator &It) : It(It) {}

    iterator() = default;

    reference operator*() const { return (*It)->getLinearMBB(); }

    pointer operator->() const { return &(*It)->getLinearMBB(); }

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
    decltype(LinearMBBs)::const_iterator It{};

  public:
    using difference_type = ptrdiff_t;
    using iterator_category = std::bidirectional_iterator_tag;
    using value_type = const LinearMachineBasicBlock;
    using reference = value_type &;
    using pointer = value_type *;

    explicit const_iterator(const decltype(LinearMBBs)::const_iterator &It)
        : It(It) {}

    const_iterator() = default;

    reference operator*() const { return (*It)->getLinearMBB(); }

    pointer operator->() const { return &(*It)->getLinearMBB(); }

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
      --It;
      return *this;
    }

    const_iterator operator--(int) {
      auto Copy = *this;
      --(*this);
      return Copy;
    }

    bool operator==(const const_iterator &Other) const {
      return It == Other.It;
    }

    bool operator!=(const const_iterator &Other) const {
      return !(*this == Other);
    }
  };

  iterator begin() { return iterator(LinearMBBs.begin()); }

  [[nodiscard]] const_iterator begin() const {
    return const_iterator(LinearMBBs.begin());
  }

  iterator end() { return iterator(LinearMBBs.end()); }

  [[nodiscard]] const_iterator end() const {
    return const_iterator(LinearMBBs.end());
  }

  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  reverse_iterator rbegin() {
    return reverse_iterator(iterator(LinearMBBs.end()));
  }

  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(const_iterator(LinearMBBs.end()));
  }

  reverse_iterator rend() {
    return reverse_iterator(iterator(LinearMBBs.begin()));
  }

  const_reverse_iterator rend() const {
    return const_reverse_iterator(const_iterator(LinearMBBs.begin()));
  }

  unsigned size() const { return LinearMBBs.size(); }

  bool empty() const { return LinearMBBs.empty(); }

  [[nodiscard]] const LinearMachineBasicBlock &front() const {
    return LinearMBBs.front()->getLinearMBB();
  }

  [[nodiscard]] LinearMachineBasicBlock &front() {
    return LinearMBBs.front()->getLinearMBB();
  }

  [[nodiscard]] const LinearMachineBasicBlock &back() const {
    return LinearMBBs.back()->getLinearMBB();
  }

  [[nodiscard]] LinearMachineBasicBlock &back() {
    return LinearMBBs.back()->getLinearMBB();
  }

  [[nodiscard]] const LinearMachineBasicBlock &
  getScalarMBB(const llvm::MachineBasicBlock &MBB) const {
    return MBBToLinearMBBs.at(MBB).get().getLinearMBB();
  }

  LinearMachineBasicBlock &getScalarMBB(const llvm::MachineBasicBlock &MBB) {
    return MBBToLinearMBBs.at(MBB).get().getLinearMBB();
  }

  [[nodiscard]] const llvm::MachineFunction &getMF() const { return MF; }

  [[nodiscard]] const IPPredicatedCFG &getParent() const { return ParentCFG; }

  [[nodiscard]] IPPredicatedCFG &getParent() { return ParentCFG; }

  void print(llvm::raw_ostream &OS) const;

  LLVM_DUMP_METHOD void dump() const;
};

class PredMFBuilder {
  PredicatedMachineFunction PredMF;

  PredMFBuilder(IPPredicatedCFG &ParentCFG, llvm::MachineFunction &MF)
      : PredMF(ParentCFG, MF) {};

public:
  PredMFBuilder(const PredMFBuilder &Other) = delete;

  PredMFBuilder &operator=(const PredMFBuilder &Other) = delete;

  static std::unique_ptr<PredMFBuilder> createPredMF(IPPredicatedCFG &ParentCFG,
                                                     llvm::MachineFunction &MF);

  using iterator = decltype(PredMF.LinearMBBs)::iterator;
  using const_iterator = decltype(PredMF.LinearMBBs)::const_iterator;

  iterator begin() { return PredMF.LinearMBBs.begin(); }

  [[nodiscard]] const_iterator begin() const {
    return PredMF.LinearMBBs.begin();
  }

  iterator end() { return PredMF.LinearMBBs.end(); }

  [[nodiscard]] const_iterator end() const { return PredMF.LinearMBBs.end(); }

  using reverse_iterator = decltype(PredMF.LinearMBBs)::reverse_iterator;
  using const_reverse_iterator =
      decltype(PredMF.LinearMBBs)::const_reverse_iterator;

  reverse_iterator rbegin() { return PredMF.LinearMBBs.rbegin(); }

  [[nodiscard]] const_reverse_iterator rbegin() const {
    return PredMF.LinearMBBs.rbegin();
  }

  reverse_iterator rend() { return PredMF.LinearMBBs.rend(); }

  [[nodiscard]] const_reverse_iterator rend() const {
    return PredMF.LinearMBBs.rend();
  }

  [[nodiscard]] unsigned size() const { return PredMF.LinearMBBs.size(); }

  [[nodiscard]] bool empty() const { return PredMF.LinearMBBs.empty(); }

  [[nodiscard]] const PredicatedMachineFunction &getPredMF() const {
    return PredMF;
  }

  PredicatedMachineFunction &getPredMF() { return PredMF; }
};

} // namespace luthier

#endif