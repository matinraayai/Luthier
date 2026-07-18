//===-- IPPredicatedCFG.cpp -----------------------------------------------===//
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
///
/// \file IPPredicatedCFG.cpp
/// Implements the \c IPPredicatedCFG class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/EntryPoint.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/InitialEntryPointAnalysis.h"
#include "luthier/ToolCodeGen/MIRConvenience.h"
#include "luthier/ToolCodeGen/PredicatedMachineBasicBlock.h"
#include "luthier/ToolCodeGen/TraceCallGraph.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier {

void IPPredicatedCFG::print(llvm::raw_ostream &OS) const {
  for (const auto &PredMBB : *this)
    PredMBB.print(OS, 0);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void IPPredicatedCFG::dump() const { print(luthier::dbgs()); }
#endif

PredicatedMachineBasicBlock &
IPPredicatedCFG::getPredMBB(const llvm::MachineInstr &MI) {
  const llvm::MachineBasicBlock *MBB = MI.getParent();
  assert(MBB && "MI doesn't have a parent MBB");
  auto It = MBBToPredMBB.find(*MBB);
  assert(It != MBBToPredMBB.end() &&
         "MBB not found in IPPredicatedCFG; was it built from this module?");
  return It->second->getPredMBB();
}

llvm::Expected<std::unique_ptr<IPPredicatedCFG>>
IPPredicatedCFG::getIPPredCFG(Prototype &IP,
                              PrototypeAnalysisManager &IPAM) {
  llvm::Module &TargetModule = IP.getTargetModule();

  // The IP-side proxy hands out the outer ModuleAnalysisManager; the same
  // MAM caches results for both modules, keyed by the module reference.
  llvm::ModuleAnalysisManager &MAM =
      IPAM.getResult<ModuleAnalysisManagerPrototypeProxy>(IP)
          .getManager();
  llvm::FunctionAnalysisManager &FAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule)
          .getManager();

  auto Out = std::unique_ptr<IPPredicatedCFG>(new IPPredicatedCFG());

  // Reverse map from IR BasicBlock → MachineBasicBlock, used to look up
  // which MBB owns a CallInst that appears in the TraceCallGraph.
  llvm::DenseMap<const llvm::BasicBlock *, llvm::MachineBasicBlock *> IRBBToMBB;

  llvm::Function *EntryFunc = nullptr;

  // ── Phase 1: create one PredMBBBuilder per MBB ──────────────────────────
  for (llvm::Function &F : TargetModule) {
    llvm::MachineFunction &MF =
        FAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();

    if (F.hasFnAttribute(InitialEntryPointAttr)) {
      if (EntryFunc)
        return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
            "Functions {0} and {1} are both designated as initial entry points",
            F.getName(), EntryFunc->getName()));
      EntryFunc = &F;
    }

    PredMBBBuilder *FirstBuilder = nullptr;
    for (llvm::MachineBasicBlock &MBB : MF) {
      auto &Builder = *Out->AllPredMBBs.emplace_back(
          std::make_unique<PredMBBBuilder>(MBB, *Out, 0));
      Out->MBBToPredMBB[MBB] = &Builder;
      if (!FirstBuilder)
        FirstBuilder = &Builder;
      if (const llvm::BasicBlock *BB = MBB.getBasicBlock())
        IRBBToMBB[BB] = &MBB;
    }
    if (FirstBuilder)
      Out->MFToEntryPredMBB[&MF] = FirstBuilder;
  }

  if (!EntryFunc)
    return LUTHIER_MAKE_GENERIC_ERROR("Failed to find an entry function.");

  {
    llvm::MachineFunction &EntryMF =
        FAM.getResult<llvm::MachineFunctionAnalysis>(*EntryFunc).getMF();
    auto It = Out->MFToEntryPredMBB.find(&EntryMF);
    assert(It != Out->MFToEntryPredMBB.end() && "Entry MF has no MBBs");
    Out->EntryPredMBB = It->second;
  }

  // ── Phase 2: intra-procedural edges from MBB successor links ────────────
  for (auto &Builder : Out->AllPredMBBs) {
    llvm::MachineBasicBlock &MBB = Builder->getPredMBB().getMBB();
    for (llvm::MachineBasicBlock *Succ : MBB.successors()) {
      if (auto *SuccBuilder = Out->MBBToPredMBB.lookup(*Succ))
        Builder->addSuccessorBlock(*SuccBuilder);
    }
  }

  // ── Phase 3: inter-procedural edges from TraceCallGraph ─────────────────
  auto &CG = IPAM.getResult<TraceCallGraphAnalysis>(IP);

  for (auto &[CI, Targets] : CG.call_targets()) {
    auto *SrcMBB = IRBBToMBB.lookup(CI->getParent());
    if (!SrcMBB)
      continue;
    auto *SrcBuilder = Out->MBBToPredMBB.lookup(*SrcMBB);
    if (!SrcBuilder)
      continue;
    for (llvm::Function *Callee : Targets) {
      llvm::MachineFunction &CalleeMF =
          FAM.getResult<llvm::MachineFunctionAnalysis>(*Callee).getMF();
      if (auto *EntryBuilder = Out->MFToEntryPredMBB.lookup(&CalleeMF))
        SrcBuilder->addSuccessorBlock(*EntryBuilder);
    }
  }

  for (llvm::CallInst *CI : CG.incomplete_call_sites()) {
    auto *SrcMBB = IRBBToMBB.lookup(CI->getParent());
    if (!SrcMBB)
      continue;
    if (auto *SrcBuilder = Out->MBBToPredMBB.lookup(*SrcMBB))
      SrcBuilder->setHasUnresolvedEdges(true);
  }

  // ── Phase 4: assign global indices ──────────────────────────────────────
  unsigned Idx = 0;
  for (auto &Builder : Out->AllPredMBBs)
    Builder->setGlobalIndex(Idx++);
  Out->NumPredMBBs = Idx;

  return Out;
}

llvm::AnalysisKey IPPredCFGAnalysis::Key;

bool IPPredCFGAnalysis::Result::invalidate(
    Prototype &, const llvm::PreservedAnalyses &PA,
    PrototypeAnalysisManager::Invalidator &) {
  auto PAC = PA.getChecker<IPPredCFGAnalysis>();
  return !PAC.preserved() &&
         !PAC.preservedSet<llvm::AllAnalysesOn<Prototype>>();
}

IPPredCFGAnalysis::Result
IPPredCFGAnalysis::run(Prototype &IP,
                       PrototypeAnalysisManager &IPAM) {
  llvm::LLVMContext &Ctx = IP.getTargetModule().getContext();
  llvm::Expected<std::unique_ptr<IPPredicatedCFG>> ResOrErr =
      IPPredicatedCFG::getIPPredCFG(IP, IPAM);
  if (auto Err = ResOrErr.takeError()) {
    Ctx.emitError(llvm::toString(std::move(Err)));
    return Result{nullptr};
  }
  return Result{std::move(*ResOrErr)};
}

llvm::PreservedAnalyses
IPPredCFGPrinter::run(Prototype &IP,
                      PrototypeAnalysisManager &IPAM) {
  auto &IPVecCFG = IPAM.getResult<IPPredCFGAnalysis>(IP).getVecCFG();
  IPVecCFG.print(OS);
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
