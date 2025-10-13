//===-- EntryPointsAnalysis.h -------------------------------------*-C++-*-===//
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
/// Describes the \c EntryPointsAnalysis class which provides a mapping
/// between entry points inside the target module and their corresponding
/// \c llvm::MachineFunction handles.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_ENTRY_POINT_ANALYSIS_H
#define LUTHIER_TOOLING_ENTRY_POINT_ANALYSIS_H
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/IR/PassManager.h>
#include <llvm/MC/MCInst.h>
#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <map>

namespace luthier {

/// \brief A \c llvm::MachineFunction analysis which provides the instruction
/// trace group discovered from the entry point corresponding to the machine
/// function in the target module
class EntryPointsAnalysis
    : public llvm::AnalysisInfoMixin<EntryPointsAnalysis> {
  friend llvm::AnalysisInfoMixin<EntryPointsAnalysis>;

  static llvm::AnalysisKey Key;

  /// Entry points in the target module are either kernels (kernel descriptor)
  /// or device function (address on the device)
  using EntryPointType =
      std::variant<const llvm::amdhsa::kernel_descriptor_t *, uint64_t>;

  using EntryPointMap =
      llvm::SmallDenseMap<const llvm::MachineFunction *, EntryPointType>;

  EntryPointMap MFToEntryPointsMap;

public:
  class Result {
    friend EntryPointsAnalysis;

    EntryPointMap &MFToEntryPointsMap;

    explicit Result(EntryPointMap &MFToEntryPointsMap)
        : MFToEntryPointsMap(MFToEntryPointsMap) {};

  public:
    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }

    void insert(const llvm::MachineFunction &MF, EntryPointType EntryPoint) {
      MFToEntryPointsMap.insert({&MF, EntryPoint});
    }

    EntryPointMap::const_iterator begin() const {
      return MFToEntryPointsMap.begin();
    }

    EntryPointMap::iterator begin() { return MFToEntryPointsMap.begin(); }

    EntryPointMap::const_iterator end() const {
      return MFToEntryPointsMap.end();
    }

    EntryPointMap::iterator end() { return MFToEntryPointsMap.end(); }

    unsigned size() const { return MFToEntryPointsMap.size(); }

    bool empty() const { return MFToEntryPointsMap.empty(); }

    bool contains(const llvm::MachineFunction &MF) const {
      return MFToEntryPointsMap.contains(&MF);
    }

    EntryPointMap::iterator find(const llvm::MachineFunction &MF) const {
      return MFToEntryPointsMap.find(&MF);
    }

    EntryPointMap::iterator find(const llvm::MachineFunction &MF) {
      return MFToEntryPointsMap.find(&MF);
    }

    EntryPointType operator[](const llvm::MachineFunction &MF) {
      return MFToEntryPointsMap[&MF];
    }
  };

  EntryPointsAnalysis() = default;

  Result run(llvm::MachineFunction &,
             llvm::MachineFunctionAnalysisManager &) {
    return Result{MFToEntryPointsMap};
  }
};

} // namespace luthier

#endif