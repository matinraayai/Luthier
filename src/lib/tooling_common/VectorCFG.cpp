//===-- VectorCFG.cpp -----------------------------------------------------===//
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
/// \file This file implements the \c VectorMBB and \c VectorCFG classes.
//===----------------------------------------------------------------------===//
#include "luthier/VectorCFG.h"
#include "luthier/CodeGenHelpers.h"
#include <SIInstrInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/MachineFunction.h>

namespace luthier {

void VectorMBB::setInstRange(llvm::MachineBasicBlock::const_iterator Begin,
                             llvm::MachineBasicBlock::const_iterator End) {
  assert(Begin->getParent() == End->getParent());
  Instructions = {Begin, End};
}

VectorMBB &VectorCFG::createVectorMBB() {
  return *MBBs.emplace_back(std::make_unique<VectorMBB>(*this));
}

std::unique_ptr<VectorCFG>
VectorCFG::getVectorCFG(const llvm::MachineFunction &MF) {
  auto Out = std::unique_ptr<VectorCFG>(new VectorCFG());
  // Create an empty VectorMBB as the entry point for the Vector CFG
  auto &EntryVectorMBB = Out->createVectorMBB();

  typedef struct {
    VectorMBB &NotTakenBlock;
    VectorMBB &TakenBlock;
  } ConnectorVectorMBBs;

  // A mapping between the llvm::MachineBasicBlock in the MF and the
  // Entry taken/not-taken VectorMBBs
  llvm::DenseMap<const llvm::MachineBasicBlock *, ConnectorVectorMBBs>
      EntryVectorMBBsPerLLVMMBB;
  // A mapping between the llvm::MachineBasicBlock in the MF and the
  // Exit taken/not-taken VectorMBBs
  llvm::DenseMap<const llvm::MachineBasicBlock *, ConnectorVectorMBBs>
      ExitVectorMBBsPerLLVMMBB;

  // Extract the vector MBBs inside each basic block
  for (const auto &MBB : MF) {
    // Create entry and exit blocks
    auto &EntryNotTakenBlock = Out->createVectorMBB();
    auto &EntryTakenBlock = Out->createVectorMBB();

    // If the MBB is the entry block, then add them to the list of successors
    // of the VectorCFG entry VectorMBB
    if (MBB.isEntryBlock()) {
      EntryVectorMBB.addSuccessorBlock(EntryTakenBlock);
      EntryVectorMBB.addSuccessorBlock(EntryNotTakenBlock);
    }

    EntryVectorMBBsPerLLVMMBB.insert(
        {&MBB, {EntryNotTakenBlock, EntryTakenBlock}});

    auto &ExitNotTakenBlock = Out->createVectorMBB();
    auto &ExitTakenBlock = Out->createVectorMBB();

    ExitVectorMBBsPerLLVMMBB.insert(
        {&MBB, {ExitNotTakenBlock, ExitTakenBlock}});

    // The currently taken block in the MBB
    auto *CurrentTakenBlock = &EntryTakenBlock;
    // Set of VectorMBBs that are waiting to be connected to the next non-taken
    // block or joined into the next block that starts with a scalar instruction
    llvm::SmallDenseSet<VectorMBB *> NotTakenBlocksWithHangingEdges{
        &EntryNotTakenBlock};

    for (const auto &MI : MBB) {
      // Check if the MI is vector (i.e. not scalar nor lane access),
      // whether it writes to the exec mask, and whether the last MI
      // (if exists) was a scalar instruction
      bool IsVector = isVector(MI);
      bool WritesExecMask = MI.modifiesRegister(
          llvm::AMDGPU::EXEC, MF.getSubtarget().getRegisterInfo());
      bool IsFormerMIScalar =
          MI.getPrevNode() &&
          (isScalar(*MI.getPrevNode()) || isLaneAccess(*MI.getPrevNode()));

      if (IsVector && (WritesExecMask || IsFormerMIScalar)) {
        // If the current instruction is a vector inst, and if it writes
        // to the exec mask or if the last instruction was a scalar inst,
        // then we need to do a "split": Create a new VectorMBB to
        // replace the current taken block and make the new block the successor
        // of the current block, and then put the just-replaced taken block
        // into the list of not-taken blocks that are yet to be connected
        auto &NewCurrentTakenBlock = Out->createVectorMBB();
        NewCurrentTakenBlock.addPredecessorBlock(*CurrentTakenBlock);
        NotTakenBlocksWithHangingEdges.insert(CurrentTakenBlock);
        CurrentTakenBlock = &NewCurrentTakenBlock;

        if (CurrentTakenBlock->empty())
          CurrentTakenBlock->setInstRange(MI.getIterator(),
                                          MI.getNextNode()->getIterator());
        else
          CurrentTakenBlock->setInstRange(CurrentTakenBlock->begin(),
                                          MI.getIterator());
      } else if (!IsVector) {
        // Otherwise, if we observe a scalar instruction, we have to do a "join"
        // operation: Create a new VectorMBB to replace the current taken block,
        // and make it the successor of the current taken block. Also make
        // all not taken blocks with hanging edges the predecessor of the new
        // taken block, and then clear the not-taken set
        auto &NewCurrentTakenBlock = Out->createVectorMBB();
        NewCurrentTakenBlock.addPredecessorBlock(*CurrentTakenBlock);
        for (auto &NonTakenBlock : NotTakenBlocksWithHangingEdges) {
          NonTakenBlock->addSuccessorBlock(NewCurrentTakenBlock);
        }
        NotTakenBlocksWithHangingEdges.clear();
        CurrentTakenBlock = &NewCurrentTakenBlock;
      }
      // Add the current instruction to the current taken block
      if (CurrentTakenBlock->empty())
        CurrentTakenBlock->setInstRange(MI.getIterator(),
                                        MI.getNextNode()->getIterator());
      else
        CurrentTakenBlock->setInstRange(CurrentTakenBlock->begin(),
                                        MI.getNextNode()->getIterator());
    }
    // Connect the current taken block to the exit taken block of the current
    // MBB
    CurrentTakenBlock->addSuccessorBlock(ExitTakenBlock);
    // Connect all the non-taken blocks to the exit non-taken block of the
    // current MBB
    for (auto &NonTakenBlock : NotTakenBlocksWithHangingEdges) {
      NonTakenBlock->addSuccessorBlock(ExitNotTakenBlock);
    }
  }
  // Now that we've figured out the relations between VectorMBBs inside
  // each MBB, it's time to connect the entry/exit taken/non-taken blocks
  // of each MBB together based on the edges of the original CFG
  for (const auto &MBB : MF) {
    auto &MBBTakenEntryBlock = EntryVectorMBBsPerLLVMMBB.at(&MBB).TakenBlock;
    auto &MBBNotTakenEntryBlock =
        EntryVectorMBBsPerLLVMMBB.at(&MBB).NotTakenBlock;
    for (const auto &Pred : MBB.predecessors()) {
      MBBTakenEntryBlock.addPredecessorBlock(
          EntryVectorMBBsPerLLVMMBB.at(Pred).TakenBlock);
      MBBNotTakenEntryBlock.addPredecessorBlock(
          EntryVectorMBBsPerLLVMMBB.at(Pred).NotTakenBlock);
    }
    for (const auto &Succ : MBB.successors()) {
      MBBTakenEntryBlock.addSuccessorBlock(
          EntryVectorMBBsPerLLVMMBB.at(Succ).TakenBlock);
      MBBNotTakenEntryBlock.addSuccessorBlock(
          EntryVectorMBBsPerLLVMMBB.at(Succ).NotTakenBlock);
    }
  }

  return std::move(Out);
}

} // namespace luthier