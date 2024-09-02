//===-- LRCallGraph.cpp - Lifted Representation Callgraph -----------------===//
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
/// This file implements the \c LRCallgraph class.
//===----------------------------------------------------------------------===//
#include "hsa/LoadedCodeObject.hpp"
#include <luthier/LRCallgraph.h>
#include <luthier/LiftedRepresentation.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-callgraph-analysis"

namespace luthier {

/// Given a \c llvm::MachineInstr of call type, finds
/// all the <tt>llvm::MachineFunction</tt>s that it calls.
/// \param CallMI the call machine instruction to be analyzed
/// \return the machine function called by the \p CallMI or \c nullptr
/// if the callee is not deterministic or \c CallMI is not a call instruction
/// TODO: Make this analysis more thorough instead of just checking the
/// values inside its own basic block
static llvm::MachineFunction *
findCalleeMFOfCallInst(const llvm::MachineInstr &CallMI,
                       llvm::MachineModuleInfo &MMI) {
  if (!CallMI.isCall())
    return nullptr;
  auto Callee = CallMI.getOperand(0);

  // Since we require relocation info, callee should not have an immediate
  // type
  if (Callee.isImm())
    llvm_unreachable("Callee of type immediate should not show up here");
  // If the callee is a global value, then we have found the machine function
  // this instruction calls
  if (Callee.isGlobal()) {
    auto *CalleeFunc = llvm::dyn_cast<llvm::Function>(Callee.getGlobal());
    if (CalleeFunc == nullptr)
      llvm_unreachable("Callee is a global value of not type function");
    return MMI.getMachineFunction(*CalleeFunc);
  }
  if (Callee.isReg()) {
    auto CalleeSGPRPair = Callee.getReg();
    // Called target is an SGPR pair; We first try to find the instruction
    // that defines it inside the current basic block
    auto &MRI = CallMI.getParent()->getParent()->getRegInfo();
    llvm::SmallDenseSet<llvm::MachineInstr *, 1> UsesInCurrentMBB;
    const llvm::MachineInstr *SelectedUseInCurrentMBB{nullptr};
    for (auto &Def : MRI.def_instructions(CalleeSGPRPair)) {
      if (Def.getParent() == CallMI.getParent())
        UsesInCurrentMBB.insert(&Def);
    }
    // if the def list is zero, return for now
    // TODO: make this more thorough
    if (UsesInCurrentMBB.empty()) {
      return nullptr;
    } else if (UsesInCurrentMBB.size() > 1) {
      // Limit our analysis to the instruction closest to the CallInst
      for (auto It = CallMI.getReverseIterator();
           It != CallMI.getParent()->rend(); It++) {
        if (UsesInCurrentMBB.contains(&(*It))) {
          SelectedUseInCurrentMBB = &(*It);
          break;
        }
      }
      if (SelectedUseInCurrentMBB == nullptr)
        llvm_unreachable("should not reach here");
    } else {
      SelectedUseInCurrentMBB = *UsesInCurrentMBB.begin();
    }

    llvm::SmallDenseSet<llvm::MachineFunction *, 1> DiscoveredFunctions;
    // Now that we found the defining instruction, we can find the
    // global value operands of type function involved in construction of
    // the value until now
    auto *TRI =
        CallMI.getParent()->getParent()->getSubtarget().getRegisterInfo();
    bool DeterministicCall{true};
    for (auto It = SelectedUseInCurrentMBB->getIterator();
         It != CallMI.getIterator(); It++) {
      auto NumDefs = It->getNumExplicitDefs();
      bool ModifiesCallReg{false};
      for (int I = 0; I < NumDefs; I++) {
        if (It->getOperand(I).isReg() &&
            TRI->regsOverlap(CalleeSGPRPair, It->getOperand(I).getReg())) {
          ModifiesCallReg = true;
        }
      }
      if (ModifiesCallReg) {
        for (int I = NumDefs; I < It->getNumExplicitOperands(); I++) {
          auto Operand = It->getOperand(I);
          if (Operand.isGlobal()) {
            auto *GV = Operand.getGlobal();
            if (auto *F = llvm::dyn_cast<llvm::Function>(GV)) {
              DiscoveredFunctions.insert(MMI.getMachineFunction(*F));
            } else {
              // If a GV is not a function, then probably it has a list of
              // function pointers defined somewhere and using them; Hence
              // we can't be certain about what is being called.
              DeterministicCall = false;
            }
          }
        }
      }
    }
    if (!DeterministicCall)
      return nullptr;
    else if (DiscoveredFunctions.size() == 1)
      return *DiscoveredFunctions.begin();
    else
      return nullptr;
  } else
    return nullptr;
}

void constructCallGraph(
    const llvm::DenseMap<const llvm::MachineInstr *, const llvm::MachineFunction *>
        &CalledInstrToCalleeFuncMap,
    llvm::DenseMap<const llvm::MachineFunction *, std::unique_ptr<CallGraphNode>>
        &CallGraph) {
  for (const auto &[MI, CalleeMF] : CalledInstrToCalleeFuncMap) {
    const auto CallerMF = MI->getParent()->getParent();
    CallGraphNode *CallerNode;
    CallGraphNode *CalleeNode;

    if (!CallGraph.contains(CallerMF)) {
      CallGraph.insert(
          {CallerMF, std::move(std::make_unique<CallGraphNode>())});
      CallGraph.at(CallerMF)->Node = CallerMF;
    }
    if (!CallGraph.contains(CalleeMF)) {
      CallGraph.insert(
          {CalleeMF, std::move(std::make_unique<CallGraphNode>())});
      CallGraph.at(CalleeMF)->Node = CalleeMF;
    }

    CallerNode = CallGraph.at(CallerMF).get();
    CallerNode->CalledFunctions.push_back({MI, CalleeMF});
    CalleeNode = CallGraph.at(CalleeMF).get();
    CalleeNode->CalleeFunctions.push_back({MI, CallerMF});
  }
}

llvm::Error LRCallGraph::analyse() {
  // Populate the non-deterministic callgraph analysis field
  for (const auto &[LCO, ModuleAndMMI] : LR.loaded_code_objects()) {
    HasNonDeterministicCallGraph.insert({LCO, false});
  }

  // Iterate over the functions of LR, and recover the target of all call
  // instruction
  llvm::DenseMap<const llvm::MachineInstr *, const llvm::MachineFunction *>
      CallInstrToCalleeMap;
  for (const auto &[FuncSymbol, MF] : LR.functions()) {
    auto LCO = FuncSymbol->getLoadedCodeObject();
    auto [M, MMI] = LR.getModuleAndMMI(LCO);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(MMI != nullptr));

    for (auto &MBB : *MF) {
      for (auto &MI : MBB) {
        if (MI.isCall()) {
          llvm::MachineFunction *CalleeMF = findCalleeMFOfCallInst(MI, *MMI);
          CallInstrToCalleeMap.insert({&MI, CalleeMF});
          LLVM_DEBUG(if (CalleeMF != nullptr) llvm::dbgs()
                         << "Found MI at: " << &MI
                         << "to be a call to function: " << CalleeMF->getName()
                         << "\n";
                     else llvm::dbgs()
                     << "MI at : " << &MI << "has an unknown call target.\n";);
          if (CalleeMF == nullptr)
            HasNonDeterministicCallGraph[LCO] = true;
        }
      }
    }
  }
  // Populate the callgraph
  constructCallGraph(CallInstrToCalleeMap, this->CallGraph);
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<LRCallGraph>>
LRCallGraph::analyse(const LiftedRepresentation &LR) {
  std::unique_ptr<LRCallGraph> CG(new LRCallGraph(LR));
  LUTHIER_RETURN_ON_ERROR(CG->analyse());
  return std::move(CG);
}

} // namespace luthier