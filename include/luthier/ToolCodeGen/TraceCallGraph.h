//===-- TraceCallGraph.h - Luthier IR call graph analysis -------*- C++ -*-===//
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
/// Declares the \c TraceCallGraph target module analysis that recovers the
/// call graph of a Luthier-translated target IR module.
///
/// Unlike LLVM's LazyCallGraph, this analysis resolves indirect call targets
/// by symbolically evaluating the callee operands, handling
/// \c amdgcn_s_getpc intrinsic chains (whose folded value is read from the
/// \c MD_pcsections metadata) and performing inter-procedural constant
/// propagation through function arguments. A single call site may map to
/// multiple target \c Function* values; call sites that cannot be fully
/// resolved are flagged as incomplete.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TRACE_CALL_GRAPH_H
#define LUTHIER_TOOL_CODE_GEN_TRACE_CALL_GRAPH_H
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Compiler.h>

namespace llvm {
class CallInst;
class Function;
class Module;
class raw_ostream;
} // namespace llvm

namespace luthier {

/// Result of the \c TraceCallGraphAnalysis.
class TraceCallGraph {
public:
  using CallTargetsMapT =
      llvm::DenseMap<llvm::CallInst *, llvm::SmallVector<llvm::Function *>>;
  using DiscoveredAddrsSetT = llvm::DenseSet<uint64_t>;
  using IncompleteCallSitesSetT = llvm::DenseSet<llvm::CallInst *>;

private:
  friend class TraceCallGraphAnalysis;

  CallTargetsMapT CallTargets;

  DiscoveredAddrsSetT DiscoveredCallTargetAddresses;

  IncompleteCallSitesSetT IncompleteCallSites;

  bool FullyRecovered = true;

public:
  /// == Call targets (indirect CallInst → list of resolved Function*) =========

  using call_targets_iterator = CallTargetsMapT::const_iterator;

  call_targets_iterator call_targets_begin() const {
    return CallTargets.begin();
  }
  call_targets_iterator call_targets_end() const { return CallTargets.end(); }
  llvm::iterator_range<call_targets_iterator> call_targets() const {
    return {call_targets_begin(), call_targets_end()};
  }
  size_t call_targets_size() const { return CallTargets.size(); }
  bool call_targets_empty() const { return CallTargets.empty(); }

  /// Lookup by call site; returns end() if \p CI has no resolved targets.
  call_targets_iterator findCallTargets(llvm::CallInst *CI) const {
    return CallTargets.find(CI);
  }
  /// Returns the resolved targets of \p CI. Asserts that \p CI is present.
  llvm::ArrayRef<llvm::Function *> atCallTargets(llvm::CallInst *CI) const {
    return CallTargets.at(CI);
  }

  /// == Discovered call-target addresses ======================================
  /// All binary addresses discovered as indirect call targets, regardless of
  /// whether the corresponding Function* already exists in the module. Used by
  /// CodeDiscoveryPass to enqueue new entry points before the callee stubs
  /// have been created.

  using discovered_addrs_iterator = DiscoveredAddrsSetT::const_iterator;

  discovered_addrs_iterator discovered_addrs_begin() const {
    return DiscoveredCallTargetAddresses.begin();
  }
  discovered_addrs_iterator discovered_addrs_end() const {
    return DiscoveredCallTargetAddresses.end();
  }
  llvm::iterator_range<discovered_addrs_iterator> discovered_addrs() const {
    return {discovered_addrs_begin(), discovered_addrs_end()};
  }
  size_t discovered_addrs_size() const {
    return DiscoveredCallTargetAddresses.size();
  }
  bool discovered_addrs_empty() const {
    return DiscoveredCallTargetAddresses.empty();
  }
  bool containsDiscoveredAddr(uint64_t Addr) const {
    return DiscoveredCallTargetAddresses.contains(Addr);
  }

  /// == Incomplete call sites =================================================
  /// Call sites for which the analysis could not determine ALL targets. A call
  /// site may be partially resolved (some targets in the call-targets map) yet
  /// still appear here if other targets remain unknown.

  using incomplete_call_sites_iterator =
      IncompleteCallSitesSetT::const_iterator;

  incomplete_call_sites_iterator incomplete_call_sites_begin() const {
    return IncompleteCallSites.begin();
  }
  incomplete_call_sites_iterator incomplete_call_sites_end() const {
    return IncompleteCallSites.end();
  }
  llvm::iterator_range<incomplete_call_sites_iterator>
  incomplete_call_sites() const {
    return {incomplete_call_sites_begin(), incomplete_call_sites_end()};
  }
  size_t incomplete_call_sites_size() const {
    return IncompleteCallSites.size();
  }
  bool incomplete_call_sites_empty() const {
    return IncompleteCallSites.empty();
  }
  bool containsIncompleteCallSite(llvm::CallInst *CI) const {
    return IncompleteCallSites.contains(CI);
  }

  /// \return \c True iff every indirect call site in the module has been fully
  /// resolved
  bool isFullyRecovered() const { return FullyRecovered; }

  /// Print the recovered call graph — resolved call sites, incomplete call
  /// sites, and discovered target addresses — to \p OS. Output is sorted so it
  /// is deterministic across runs.
  void print(llvm::raw_ostream &OS) const;

  /// Dump the call graph to \c luthier::dbgs()
  LLVM_DUMP_METHOD void dump() const;

  /// The analysis is invalidated whenever the module IR is modified
  /// (e.g. a new function is added or a CFG edge changes)
  bool invalidate(llvm::Module &M, const llvm::PreservedAnalyses &PA,
                  llvm::ModuleAnalysisManager::Invalidator &Inv);
};

/// Module analysis that recovers the IR-level call graph of a
/// Luthier-translated module.
class TraceCallGraphAnalysis
    : public llvm::AnalysisInfoMixin<TraceCallGraphAnalysis> {
  friend llvm::AnalysisInfoMixin<TraceCallGraphAnalysis>;
  static llvm::AnalysisKey Key;

public:
  using Result = TraceCallGraph;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
};

/// Pass that prints the \c TraceCallGraph result to an output stream.
/// Intended to be inserted after \c luthier-code-discovery in the pipeline.
class TraceCallGraphPrinter
    : public llvm::PassInfoMixin<TraceCallGraphPrinter> {
  llvm::raw_ostream &OS;

public:
  explicit TraceCallGraphPrinter(llvm::raw_ostream &OS) : OS(OS) {}

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

} // namespace luthier

#endif
