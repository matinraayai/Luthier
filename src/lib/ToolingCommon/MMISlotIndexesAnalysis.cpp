//===-- MMISlotIndexesAnalysis.cpp ----------------------------------------===//
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
/// This file implements the <tt>MMISlotIndexesAnalysis</tt> pass.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/MMISlotIndexesAnalysis.h"
#include "luthier/Tooling/IPPredicatedCFG.h"
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/Module.h>

namespace luthier {

llvm::AnalysisKey MMISlotIndexesAnalysis::Key;

MMISlotIndexesAnalysis::Result
MMISlotIndexesAnalysis::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  MMISlotIndexesAnalysis::Result Out;
  auto &MMI = MAM.getCachedResult<llvm::MachineModuleAnalysis>(M)->getMMI();
  auto &IPVecCFG = MAM.getResult<IPPredCFGAnalysis>(M).getVecCFG();

  for (const auto &F : M) {
    auto *MF = MMI.getMachineFunction(F);
    auto& PMF = IPVecCFG[*MF];
    if (!MF)
      continue;
    Out.Res.insert({&PMF, SlotIndexes(PMF)});
  }
  return Out;
}

llvm::PreservedAnalyses
MMISlotIndexesPrinterPass::run(llvm::Module &M,
                            llvm::ModuleAnalysisManager &MAM) {
  auto &SIA = MAM.getResult<MMISlotIndexesAnalysis>(M);
  for(auto &Entry : SIA){
    auto* MF = Entry.getFirst();
    auto& SI = Entry.getSecond();
    OS << "Slot indexes in machine function: " << MF->getMF().getName() << '\n';
    SI.print(OS);
  }
  
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
