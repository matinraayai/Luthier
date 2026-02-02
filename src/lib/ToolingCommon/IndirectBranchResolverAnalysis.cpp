//===-- InstructionTracesAnalysis.cpp -------------------------------------===//
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
/// Implements the \c InstructionTracesAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/IndirectBranchResolverAnalysis.h"
#include "luthier/Tooling/MachineFunctionEntryPoint.h"
#include <llvm/Support/GenericDomTree.h>
// #include <luthier/Tooling/IPReachingDefAnalysis.h>
// #include <luthier/Tooling/IPVectorCFG.h>
// #include <luthier/Tooling/IPVectorRegLiveness.h>

namespace luthier {

llvm::AnalysisKey IndirectBranchResolverAnalysis::Key;

bool IndirectBranchResolverAnalysis::Result::invalidate(
    llvm::Module &, const llvm::PreservedAnalyses &PA,
    llvm::ModuleAnalysisManager::Invalidator &) {
  // Unless it is invalidated explicitly, it should remain preserved.
  auto PAC = PA.getChecker<IndirectBranchResolverAnalysis>();
  return !PAC.preservedWhenStateless();
}

IndirectBranchResolverAnalysis::Result
IndirectBranchResolverAnalysis::run(llvm::Module &TargetModule,
                                    llvm::ModuleAnalysisManager &TargetMAM) {
  // /// Get the IP Vector CFG; calculate the liveness and reaching definitions
  // auto &IPVecCFG =
  //     TargetMAM.getResult<IPVectorCFGAnalysis>(TargetModule).getVecCFG();
  // auto &IPReachingDefs =
  // TargetMAM.getResult<ReachingDefAnalysis>(TargetModule);
  // IPReachingDefs.print(llvm::outs());
  // /// Iterate over the IPVectorCFG in the post dom order and locate the
  // /// indirect branch and all call instructions
  // llvm::DominatorTreeBase<VectorMBB, false> DOM;
  //
  // DOM.recalculate(IPVecCFG);

  // for (const VectorMBB *VecMBB : DOM.roots()) {
  //   for (const llvm::MachineInstr &MI : *VecMBB) {
  //     if (MI.isCall() || MI.isIndirectBranch()) {
  //     }
  //   }
  // }
}
} // namespace luthier
