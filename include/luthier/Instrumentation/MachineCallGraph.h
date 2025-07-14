//===-- MachineCallGraph.h --------------------------------------*- C++ -*-===//
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
///
/// \file
/// Describes the \c MachineCallGraph class and its associated analysis pass,
/// as well as the \c MachineCallGraphAnalysis pass which can be extended to
/// calculate the machine callgraph.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_INSTRUMENTATION_MACHINE_CALLGRAPH_H
#define LUTHIER_INSTRUMENTATION_MACHINE_CALLGRAPH_H
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/PassManager.h>

namespace luthier {

/// \brief Represents an edge inside the \c MachineCallGraph
struct MachineCallGraphEdge {
private:
  /// The \c MachineInstr that performs the call
  const llvm::MachineInstr &CallMI;

  /// List of all possible functions that can be called from this call site
  const llvm::SmallDenseSet<const llvm::MachineFunction &, 1>
      PossibleTargetFunctions;

public:
  using called_function_iterator =
      decltype(PossibleTargetFunctions)::const_iterator;

  MachineCallGraphEdge(
      const llvm::MachineInstr &CallMI,
      std::initializer_list<const llvm::MachineFunction &> PossibleTargets)
      : CallMI(CallMI), PossibleTargetFunctions(std::move(PossibleTargets)) {};

  /// \return the call instruction used to perform the call
  const llvm::MachineInstr &getCallMI() const { return CallMI; }

  /// \return iterator spanning all called functions
  llvm::iterator_range<called_function_iterator>
  possibleTargetFunctions() const {
    return PossibleTargetFunctions;
  }
};

class MachineCallGraph;

/// \brief Represents a node inside the \c MachineCallGraph
struct MachineCallGraphNode {
private:
  friend MachineCallGraph;
  /// The function associated with the callgraph node
  const llvm::MachineFunction &Node;

  using EdgeMap =
      llvm::SmallDenseMap<const llvm::MachineInstr &, MachineCallGraphEdge, 1>;

  /// The functions the current node calls as well as the instruction
  /// inside the current node's function that performs the call
  /// A nullptr machine function indicates that the call target
  EdgeMap Children;
  /// The functions that call \c Node as well as the instruction that
  /// performs the call
  EdgeMap Parents;

public:
  explicit MachineCallGraphNode(const llvm::MachineFunction &MF) : Node(MF) {};

  using child_iterator = decltype(Children)::const_iterator;

  using parent_iterator = decltype(Parents)::const_iterator;

  const llvm::MachineFunction &getFunction() const { return Node; }

  llvm::iterator_range<child_iterator> children() const { return Children; }

  llvm::iterator_range<child_iterator> parents() const { return Parents; }
};

/// \brief a class containing the recovered callgraph of a
/// lifted \c llvm::MachineModuleInfo
/// \details LLVM MIR does not have a callgraph, which is why \c
/// MachineCallGraph is implemented here in the first place.
/// \c MachineCallGraph is meant to be used for a lifted representation, and
/// is meant to be populated by a
class MachineCallGraph {
private:
  /// A map which keeps track of the \c CallGraphNode of each
  /// \c llvm::MachineFunction
  llvm::DenseMap<const llvm::MachineFunction &,
                 std::unique_ptr<MachineCallGraphNode>>
      Nodes;

public:
  MachineCallGraph(const llvm::Module &M, const llvm::MachineModuleInfo &MMI);

  /// \return the \c CallGraphNode associated with the \p MF
  const MachineCallGraphNode &getNode(const llvm::MachineFunction &MF) const {
    return *Nodes.at(MF);
  }

  llvm::Error addEdge(const llvm::MachineInstr &MI,
                      const llvm::MachineFunction &TargetMF);

  /// Never invalidate the results
  __attribute__((used)) bool
  invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
             llvm::ModuleAnalysisManager::Invalidator &) {
    return false;
  }
};

class MachineCallGraphAnalysis
    : public llvm::AnalysisInfoMixin<MachineCallGraphAnalysis> {
private:
  friend llvm::AnalysisInfoMixin<MachineCallGraphAnalysis>;

  static llvm::AnalysisKey Key;

public:
  using Result = MachineCallGraph;

  MachineCallGraphAnalysis() = default;

  /// Run the analysis pass that would
  Result run(llvm::Module &, llvm::ModuleAnalysisManager &) { return Result{}; }
};

} // namespace luthier

#endif