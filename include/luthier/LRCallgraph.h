//===-- LRCallGraph.h - Lifted Representation Callgraph ---------*- C++ -*-===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// \file
/// This file describes the \c LRCallgraph class, which tries to recover
/// the callgraph of each \c hsa_loaded_code_object_t in a
/// <tt>LiftedRepresentation</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_LR_CALLGRAPH_H
#define LUTHIER_LR_CALLGRAPH_H
#include <hsa/hsa.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/PassManager.h>
#include <luthier/hsa/DenseMapInfo.h>

namespace luthier {

class LiftedRepresentation;

/// \brief a struct for constructing a machine call graph
struct CallGraphNode {
  /// The function associated with the callgraph node
  const llvm::MachineFunction *Node;
  /// The functions \c Node calls as well as the instruction that performs
  /// the call
  llvm::SmallVector<
      std::pair<const llvm::MachineInstr *, const llvm::MachineFunction *>>
      CalledFunctions;
  /// The functions that call \c Node as well as the instruction that
  /// performs the call
  llvm::SmallVector<
      std::pair<const llvm::MachineInstr *, const llvm::MachineFunction *>>
      CalleeFunctions;
};

/// \brief a class that tries to analyze and recover the callgraph of all the
/// <tt>hsa_loaded_code_object</tt>s in a \c LiftedRepresentation
class LRCallGraph {
private:
  /// A map which keeps track of the \c CallGraphNode of each
  /// \c llvm::MachineFunction in the lifted representation; It is
  /// constructed by determining the target of all call instructions of the
  /// functions in the lifted representation
  llvm::DenseMap<const llvm::MachineFunction *, std::unique_ptr<CallGraphNode>>
      CallGraph{};

  /// Whether code lifter analysis was able to find the target of all
  /// call instructions in the LCO or not
  bool HasNonDeterministicCallGraph{false};

public:
  LRCallGraph() = default;

  /// Performs the callgraph analysis
  llvm::Error analyse(const llvm::Module &M,
                      const llvm::MachineModuleInfo &MMI);

  /// \return the \c CallGraphNode associated with the \p MF
  const CallGraphNode &getCallGraphNode(llvm::MachineFunction *MF) const {
    return *CallGraph.at(MF);
  }

  /// \return \c true if callgraph analysis was able to determine the target
  /// of all call instructions in the \p LCO, \c false otherwise
  [[nodiscard]] bool
  hasNonDeterministicCallGraph() const {
    return HasNonDeterministicCallGraph;
  }
};

class LRCallGraphAnalysis
    : public llvm::AnalysisInfoMixin<LRCallGraphAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<LRCallGraphAnalysis>;

  static llvm::AnalysisKey Key;

  LRCallGraph CG{};

public:
  using Result = LRCallGraph &;

  LRCallGraphAnalysis() = default;

  /// Run the analysis pass that would
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  /// Never invalidate the results
  bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                  llvm::ModuleAnalysisManager::Invalidator &) {
    return false;
  }
};

} // namespace luthier

#endif