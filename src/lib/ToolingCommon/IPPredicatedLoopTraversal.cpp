//===-- IPPredicatedLoopTraversal.cpp -------------------------------------===//
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
/// \file IPPredicatedLoopTraversal.cpp
/// Implements the \c IPPredicatedLoopTraversal class.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/IPPredicatedLoopTraversal.h"
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/CodeGen/MachineFunction.h>

namespace luthier {

bool IPPredicatedLoopTraversal::isBlockDone(
    const PredicatedMachineBasicBlock *MBB) {
  unsigned MBBNumber = MBB->getGlobalNumber();
  assert(MBBNumber < MBBInfos.size() && "Unexpected basic block number.");
  return MBBInfos[MBBNumber].PrimaryCompleted &&
         MBBInfos[MBBNumber].IncomingCompleted ==
             MBBInfos[MBBNumber].PrimaryIncoming &&
         MBBInfos[MBBNumber].IncomingProcessed == MBB->preds_size();
}

IPPredicatedLoopTraversal::IPTraversalOrder
IPPredicatedLoopTraversal::traverse(const IPPredicatedCFG &IPPredCFG) {
  // Initialize the MMBInfos
  MBBInfos.assign(IPPredCFG.getNumVecMBBs(), PredicatedMBBInfo());

  const PredicatedMachineFunction &EntryPredMF = IPPredCFG.getEntry();

  const PredicatedMachineBasicBlock *Entry =
      EntryPredMF.empty() ? nullptr : &*EntryPredMF.begin()->begin();
  assert(Entry && "Entry block is nullptr");
  llvm::ReversePostOrderTraversal<
      const luthier::PredicatedMachineBasicBlock *,
      llvm::GraphTraits<const PredicatedMachineBasicBlock *>>
      RPOT(Entry);
  llvm::SmallVector<const PredicatedMachineBasicBlock *, 4> Workqueue;
  llvm::SmallVector<TraversedPredMBBInfo, 4> MBBTraversalOrder;
  for (const luthier::PredicatedMachineBasicBlock *MBB : RPOT) {
    // N.B: IncomingProcessed and IncomingCompleted were already updated while
    // processing this block's predecessors.
    unsigned MBBNumber = MBB->getGlobalNumber();
    assert(MBBNumber < MBBInfos.size() && "Unexpected basic block number.");
    MBBInfos[MBBNumber].PrimaryCompleted = true;
    MBBInfos[MBBNumber].PrimaryIncoming = MBBInfos[MBBNumber].IncomingProcessed;
    bool Primary = true;
    Workqueue.push_back(MBB);
    while (!Workqueue.empty()) {
      const PredicatedMachineBasicBlock *ActiveMBB = Workqueue.pop_back_val();
      bool Done = isBlockDone(ActiveMBB);
      MBBTraversalOrder.push_back(
          TraversedPredMBBInfo(ActiveMBB, Primary, Done));
      for (const PredicatedMachineBasicBlock &Succ : ActiveMBB->successors()) {
        unsigned SuccNumber = Succ.getGlobalNumber();
        assert(SuccNumber < MBBInfos.size() &&
               "Unexpected basic block number.");
        if (!isBlockDone(&Succ)) {
          if (Primary)
            MBBInfos[SuccNumber].IncomingProcessed++;
          if (Done)
            MBBInfos[SuccNumber].IncomingCompleted++;
          if (isBlockDone(&Succ))
            Workqueue.push_back(&Succ);
        }
      }
      Primary = false;
    }
  }

  // We need to go through again and finalize any blocks that are not done yet.
  // This is possible if blocks have dead predecessors, so we didn't visit them
  // above.
  for (const PredicatedMachineBasicBlock *MBB : RPOT) {
    if (!isBlockDone(MBB))
      MBBTraversalOrder.push_back(TraversedPredMBBInfo(MBB, false, true));
    // Don't update successors here. We'll get to them anyway through this
    // loop.
  }

  MBBInfos.clear();

  return MBBTraversalOrder;
}

} // namespace luthier
