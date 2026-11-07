//===-- PreviouslyDiscoveredEntryPointsAnalysis.h -----------------*-C++-*-===//
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
/// \file
/// Analysis that records entry points already instrumented by prior pipeline
/// runs and their load locations.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_PREVIOUSLY_DISCOVERED_ENTRY_POINTS_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_PREVIOUSLY_DISCOVERED_ENTRY_POINTS_ANALYSIS_H

#include "luthier/ToolCodeGen/PatchPCUsagesPass.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief Immutable target module analysis that holds the set of entry points
/// already instrumented by previous instrumentation pipeline runs, as well
/// as their associated load address.
class PreviouslyDiscoveredEntryPointsAnalysis
    : public llvm::AnalysisInfoMixin<PreviouslyDiscoveredEntryPointsAnalysis> {
  friend llvm::AnalysisInfoMixin<PreviouslyDiscoveredEntryPointsAnalysis>;
  static llvm::AnalysisKey Key;

public:
  using EntryType = PatchPCUsagesPass::EntryPointTraceFunctionEntry;
  using MapTy = llvm::SmallDenseMap<uint64_t, uint64_t>;

  class Result {
    const MapTy &Map;

  public:
    explicit Result(const MapTy &M) : Map(M) {}

    MapTy::const_iterator begin() const { return Map.begin(); }

    MapTy::const_iterator end() const { return Map.end(); }

    MapTy::const_iterator find(uint64_t Addr) const { return Map.find(Addr); }

    bool contains(uint64_t Addr) const { return Map.contains(Addr); }

    bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                    llvm::ModuleAnalysisManager::Invalidator &) {
      return false;
    }
  };

  explicit PreviouslyDiscoveredEntryPointsAnalysis(
      llvm::ArrayRef<EntryType> Entries) {
    for (const auto &E : Entries)
      KnownEntryPoints[E.TraceAddr] = E.FnHandleAddr;
  }

  PreviouslyDiscoveredEntryPointsAnalysis() = default;

  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) {
    return Result(KnownEntryPoints);
  }

private:
  MapTy KnownEntryPoints;
};

} // namespace luthier

#endif // LUTHIER_TOOL_CODE_GEN_PREVIOUSLY_DISCOVERED_ENTRY_POINTS_ANALYSIS_H
