//===-- MIToIRMappingAnalysis.cpp -----------------------------------------===//
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
/// Implements the \c MIToIRMappingAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/MIToIRMappingAnalysis.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Debug.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-mi-to-ir-mapping"

namespace luthier {

bool MIToIRMapping::invalidate(llvm::Function &,
                               const llvm::PreservedAnalyses &PA,
                               llvm::FunctionAnalysisManager::Invalidator &) {
  // The mapping holds raw MachineInstr/Instruction pointers, so any pass that
  // adds or removes instructions in either representation invalidates it.
  // Unless it is preserved explicitly, recompute it.
  auto PAC = PA.getChecker<MIToIRMappingAnalysis>();
  return !PAC.preservedWhenStateless();
}

llvm::AnalysisKey MIToIRMappingAnalysis::Key;

MIToIRMappingAnalysis::Result
MIToIRMappingAnalysis::run(llvm::Function &F,
                           llvm::FunctionAnalysisManager &FAM) {
  MIToIRMapping Result;

  llvm::MachineFunction &MF =
      FAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();

  LLVM_DEBUG(luthier::dbgs() << "[MIToIRMapping] Running analysis for "
                             << MF.getName() << "\n";);

  // Pass 1: index this function's trace MIs by their PC sections MDNode. Each
  // trace MI gets a freshly-created, unique TargetMachineInstrMDNode, so the
  // node pointer is a bijective key on the MI side. Non-trace pcsections (and
  // MIs without any) are skipped.
  llvm::DenseMap<const llvm::MDNode *, llvm::MachineInstr *> PCSToMI;
  for (llvm::MachineBasicBlock &MBB : MF) {
    for (llvm::MachineInstr &MI : MBB) {
      TargetMachineInstrMDNode *MD =
          TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI);
      if (!MD)
        continue;
      // try_emplace: first MI wins. A duplicate key would mean two MIs share a
      // node pointer (only possible if an MI was cloned with its metadata after
      // discovery); warn rather than silently mis-map.
      if (!PCSToMI.try_emplace(MD, &MI).second) {
        LLVM_DEBUG(luthier::dbgs()
                       << "[MIToIRMapping] Duplicate trace MDNode for MI " << MI
                       << "; keeping the first MI mapped to it.\n";);
      }
    }
  }

  // Pass 2: walk the lifted IR body (the same Function that backs this machine
  // function) in program order and join each IR instruction to its source MI
  // through the shared PC sections MDNode pointer.
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      auto *MD = llvm::dyn_cast_or_null<TargetMachineInstrMDNode>(
          I.getMetadata(llvm::LLVMContext::MD_pcsections));
      if (!MD || !MD->isTraceInstr())
        continue;
      if (auto It = PCSToMI.find(MD); It != PCSToMI.end())
        Result.addEntry(*It->second, I);
    }
  }

  LLVM_DEBUG(luthier::dbgs()
                 << "[MIToIRMapping] Mapped " << Result.size()
                 << " IR instructions in " << MF.getName() << "\n";);

  return Result;
}

} // namespace luthier
