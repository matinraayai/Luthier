//===-- IndirectBranchResolverAnalysis.cpp --------------------------------===//
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
/// \file IndirectBranchResolverAnalysis.cpp
/// Implements the \c IndirectBranchResolverAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/IndirectBranchResolverAnalysis.h"
#include "luthier/Tooling/IPPredicatedCFG.h"
#include "luthier/Tooling/IPReachingDefAnalysis.h"
#include "luthier/Tooling/IPVectorRegLiveness.h"
#include "luthier/Tooling/MachineFunctionEntryPoint.h"
#include <llvm/Support/GenericDomTree.h>

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
  auto &IPVecCFG =
      TargetMAM.getResult<IPPredCFGAnalysis>(TargetModule).getVecCFG();
  auto &IPReachingDefs =
      TargetMAM.getResult<IPReachingDefAnalysis>(TargetModule);
  /// Iterate over the IPVectorCFG in the dom order and locate the
  /// indirect branch and all call instructions
  llvm::DominatorTreeBase<PredicatedMachineBasicBlock, false> DOM;

  DOM.recalculate(IPVecCFG);
  for (const PredicatedMachineBasicBlock *PredMBB : DOM.roots()) {
    for (const llvm::MachineInstr &MI : *PredMBB) {
      if (MI.isCall() || MI.isIndirectBranch()) {
      }
    }
  }
}
} // namespace luthier
