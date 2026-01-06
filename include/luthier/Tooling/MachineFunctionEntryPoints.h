//===-- MachineFunctionEntryPoints.h ------------------------------*-C++-*-===//
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
/// \file
/// Describes the \c MachineFunctionEntryPoints analysis which provides a
/// mapping between the \c llvm::MachineFunction handles in a target machine
/// module and their associated entry points.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_MACHINE_FUNCTION_ENTRY_POINTS_H
#define LUTHIER_TOOLING_MACHINE_FUNCTION_ENTRY_POINTS_H
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>

namespace luthier {

/// \brief A target \c llvm::Module analysis which provides the
/// \c llvm::MachineFunction corresponding to each entry point discovered
/// during the lifting operation
class MachineFunctionEntryPoints
    : public llvm::AnalysisInfoMixin<MachineFunctionEntryPoints> {
  friend AnalysisInfoMixin<MachineFunctionEntryPoints>;

  static llvm::AnalysisKey Key;

public:
  class Result {
    friend MachineFunctionEntryPoints;

    using FunctionToEntryPointMap =
        llvm::SmallDenseMap<const llvm::MachineFunction *, EntryPoint, 4>;

    FunctionToEntryPointMap MFToEntryPointsMap;

    Result() = default;

  public:
    bool invalidate(llvm::MachineFunction &, const llvm::PreservedAnalyses &,
                    llvm::MachineFunctionAnalysisManager::Invalidator &) {
      return false;
    }

    void insert(const llvm::MachineFunction &MF, EntryPoint EntryPoint) {
      MFToEntryPointsMap.insert({&MF, EntryPoint});
    }

    FunctionToEntryPointMap::const_iterator begin() const {
      return MFToEntryPointsMap.begin();
    }

    FunctionToEntryPointMap::iterator begin() {
      return MFToEntryPointsMap.begin();
    }

    FunctionToEntryPointMap::const_iterator end() const {
      return MFToEntryPointsMap.end();
    }

    FunctionToEntryPointMap::iterator end() { return MFToEntryPointsMap.end(); }

    unsigned size() const { return MFToEntryPointsMap.size(); }

    bool empty() const { return MFToEntryPointsMap.empty(); }

    bool contains(const llvm::MachineFunction &MF) const {
      return MFToEntryPointsMap.contains(&MF);
    }

    FunctionToEntryPointMap::const_iterator
    find(const llvm::MachineFunction &MF) const {
      return MFToEntryPointsMap.find(&MF);
    }

    FunctionToEntryPointMap::iterator find(const llvm::MachineFunction &MF) {
      return MFToEntryPointsMap.find(&MF);
    }

    EntryPoint operator[](const llvm::MachineFunction &MF) {
      return MFToEntryPointsMap[&MF];
    }
  };

  MachineFunctionEntryPoints() = default;

  Result run(llvm::MachineFunction &, llvm::MachineFunctionAnalysisManager &) {
    return Result{};
  }
};

} // namespace luthier

template <> struct llvm::DenseMapInfo<luthier::EntryPoint> {
  static luthier::EntryPoint getEmptyKey() {
    return luthier::EntryPoint(DenseMapInfo<uint64_t>::getEmptyKey());
  }

  static luthier::EntryPoint getTombstoneKey() {
    return luthier::EntryPoint(DenseMapInfo<uint64_t>::getTombstoneKey());
  }

  static unsigned getHashValue(const luthier::EntryPoint &EP) {
    if (const amdhsa::kernel_descriptor_t *KD = EP.getKernelDescriptor()) {
      return DenseMapInfo<amdhsa::kernel_descriptor_t *>::getHashValue(KD);
    } else {
      return DenseMapInfo<uint64_t>::getHashValue(EP.getEntryPointAddress());
    }
  }

  static bool isEqual(const luthier::EntryPoint &Lhs,
                      const luthier::EntryPoint &Rhs) {
    return (Lhs.getEntryPointAddress() == Rhs.getEntryPointAddress()) &&
           (Lhs.isKernel() == Rhs.isKernel());
  }
};

#endif