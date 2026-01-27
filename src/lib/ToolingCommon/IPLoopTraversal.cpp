//===- LoopTraversal.cpp - Optimal basic block traversal order --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "luthier/Tooling/IPLoopTraversal.h"
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/CodeGen/MachineFunction.h>

namespace luthier {
bool LoopTraversal::isBlockDone(const VectorMBB *MBB) {
  unsigned MBBNumber = MBB->getIndex();
  assert(MBBNumber < MBBInfos.size() && "Unexpected basic block number.");
  return MBBInfos[MBBNumber].PrimaryCompleted &&
         MBBInfos[MBBNumber].IncomingCompleted ==
             MBBInfos[MBBNumber].PrimaryIncoming &&
         MBBInfos[MBBNumber].IncomingProcessed == MBB->preds_size();
}

LoopTraversal::TraversalOrder LoopTraversal::traverse(const IPVectorCFG &MF) {
  // Initialize the MMBInfos
  MBBInfos.assign(MF.getNumVecMBBs(), MBBInfo());

  const VectorMBB *Entry = &*MF.begin()->begin()->begin();
  llvm::ReversePostOrderTraversal<const luthier::VectorMBB *,
                                  llvm::GraphTraits<const VectorMBB *>>
      RPOT(Entry);
  llvm::SmallVector<const VectorMBB *, 4> Workqueue;
  llvm::SmallVector<TraversedMBBInfo, 4> MBBTraversalOrder;
  for (const luthier::VectorMBB *MBB : RPOT) {
    // N.B: IncomingProcessed and IncomingCompleted were already updated while
    // processing this block's predecessors.
    unsigned MBBNumber = MBB->getIndex();
    assert(MBBNumber < MBBInfos.size() && "Unexpected basic block number.");
    MBBInfos[MBBNumber].PrimaryCompleted = true;
    MBBInfos[MBBNumber].PrimaryIncoming = MBBInfos[MBBNumber].IncomingProcessed;
    bool Primary = true;
    Workqueue.push_back(MBB);
    while (!Workqueue.empty()) {
      const luthier::VectorMBB *ActiveMBB = Workqueue.pop_back_val();
      bool Done = isBlockDone(ActiveMBB);
      MBBTraversalOrder.push_back(TraversedMBBInfo(ActiveMBB, Primary, Done));
      for (const luthier::VectorMBB *Succ : ActiveMBB->successors()) {
        unsigned SuccNumber = Succ->getIndex();
        assert(SuccNumber < MBBInfos.size() &&
               "Unexpected basic block number.");
        if (!isBlockDone(Succ)) {
          if (Primary)
            MBBInfos[SuccNumber].IncomingProcessed++;
          if (Done)
            MBBInfos[SuccNumber].IncomingCompleted++;
          if (isBlockDone(Succ))
            Workqueue.push_back(Succ);
        }
      }
      Primary = false;
    }
  }

  // We need to go through again and finalize any blocks that are not done yet.
  // This is possible if blocks have dead predecessors, so we didn't visit them
  // above.
  for (const VectorMBB *MBB : RPOT) {
    if (!isBlockDone(MBB))
      MBBTraversalOrder.push_back(TraversedMBBInfo(MBB, false, true));
    // Don't update successors here. We'll get to them anyway through this
    // loop.
  }

  MBBInfos.clear();

  return MBBTraversalOrder;
}

} // namespace luthier
