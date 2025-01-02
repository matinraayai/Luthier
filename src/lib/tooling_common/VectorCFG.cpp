//===-- VectorCFG.cpp -----------------------------------------------------===//
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
/// \file This file implements the \c VectorMBB and \c VectorCFG classes.
//===----------------------------------------------------------------------===//
#include "luthier/VectorCFG.h"
#include "luthier/CodeGenHelpers.h"
#include <SIInstrInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier {

void VectorMBB::setInstRange(llvm::MachineBasicBlock::const_iterator Begin,
                             llvm::MachineBasicBlock::const_iterator End) {
  Instructions = {Begin, End};
}

void VectorMBB::addLiveIn(llvm::MCRegister PhysReg,
                          llvm::LaneBitmask LaneMask) {
  LiveIns.emplace_back(PhysReg, LaneMask);
}

void VectorMBB::clearLiveIns(
    std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &OldLiveIns) {
  std::swap(LiveIns, OldLiveIns);
}

void VectorMBB::sortUniqueLiveIns() {
  llvm::sort(LiveIns, [](const llvm::MachineBasicBlock::RegisterMaskPair &LI0,
                         const llvm::MachineBasicBlock::RegisterMaskPair &LI1) {
    return LI0.PhysReg < LI1.PhysReg;
  });
  // Liveins are sorted by physreg now we can merge their lanemasks.
  LiveInVector::const_iterator I = LiveIns.begin();
  LiveInVector::const_iterator J;
  auto Out = LiveIns.begin();
  for (; I != LiveIns.end(); ++Out, I = J) {
    llvm::MCRegister PhysReg = I->PhysReg;
    llvm::LaneBitmask LaneMask = I->LaneMask;
    for (J = std::next(I); J != LiveIns.end() && J->PhysReg == PhysReg; ++J)
      LaneMask |= J->LaneMask;
    Out->PhysReg = PhysReg;
    Out->LaneMask = LaneMask;
  }
  LiveIns.erase(Out, LiveIns.end());
}

void VectorMBB::print(llvm::raw_ostream &OS) const {
  const auto &ST = this->getParent().getMF().getSubtarget();
  const auto TII = ST.getInstrInfo();
  const auto TRI = ST.getRegisterInfo();
  OS << "Vector MBB " << Name << "\n";
  OS << "Live-ins:[";
  llvm::interleave(
      LiveIns.begin(), LiveIns.end(),
      [&](const llvm::MachineBasicBlock::RegisterMaskPair &Vec) {
        OS << printReg(llvm::Register(Vec.PhysReg), TRI);
      },
      [&]() { OS << ", "; });
  OS << "]\n";
  OS << "Contents:\n";
  for (const auto &MI : *this) {
    MI.print(OS, true, false, false, true, TII);
  }
  OS << "Successors: [";
  llvm::interleave(
      Successors.begin(), Successors.end(),
      [&](const VectorMBB *MBB) { OS << "MBB " << MBB->getNum(); },
      [&]() { OS << ", "; });
  OS << "]\n";
}

VectorMBB &VectorCFG::createVectorMBB() {
  return *MBBs.emplace_back(std::make_unique<VectorMBB>(
      *this, llvm::formatv("{0}", MBBs.size()).str()));
}

std::unique_ptr<VectorCFG>
VectorCFG::getVectorCFG(const llvm::MachineFunction &MF) {
  auto Out = std::unique_ptr<VectorCFG>(new VectorCFG(MF));
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
          MI.getPrevNode() != nullptr &&
          (isScalar(*MI.getPrevNode()) || isLaneAccess(*MI.getPrevNode()));
      llvm::outs() << "=================\n";
      llvm::outs() << MI << "\n";
      if (IsVector && (WritesExecMask || IsFormerMIScalar)) {
        llvm::outs() << "Split point detected\n";
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
      } else if (!IsVector && !IsFormerMIScalar) {
        llvm::outs() << "join point detected\n";
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
      auto NextIterator = MI.getNextNode() != nullptr
                              ? MI.getNextNode()->getIterator()
                              : MI.getParent()->end();
      if (CurrentTakenBlock->empty())
        CurrentTakenBlock->setInstRange(MI.getIterator(), NextIterator);
      else
        CurrentTakenBlock->setInstRange(CurrentTakenBlock->begin(),
                                        NextIterator);
      Out->print(llvm::outs());
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
void VectorCFG::print(llvm::raw_ostream &OS) const {
  OS << "Vector CFG for Machine Function " << MF.getName() << ":\n";
  for (const auto &MBB : *this) {
    MBB->print(OS);
  }
}

void addLiveOutsNoPristines(llvm::LivePhysRegs &LPR, const VectorMBB &MBB) {
  // To get the live-outs we simply merge the live-ins of all successors.
  for (const VectorMBB *Succ : MBB.successors())
    addBlockLiveIns(LPR, *Succ);
}

void addBlockLiveIns(llvm::LivePhysRegs &LPR, const VectorMBB &VecMBB) {
  for (const auto &LI : VecMBB.liveins()) {
    llvm::MCPhysReg Reg = LI.PhysReg;
    llvm::LaneBitmask Mask = LI.LaneMask;
    auto *TRI = VecMBB.getParent().getMF().getSubtarget().getRegisterInfo();
    llvm::MCSubRegIndexIterator S(Reg, TRI);
    assert(Mask.any() && "Invalid livein mask");
    if (Mask.all() || !S.isValid()) {
      LPR.addReg(Reg);
      continue;
    }
    for (; S.isValid(); ++S) {
      unsigned SI = S.getSubRegIndex();
      if ((Mask & TRI->getSubRegIndexLaneMask(SI)).any())
        LPR.addReg(S.getSubReg());
    }
  }
}

void addLiveIns(VectorMBB &MBB, const llvm::LivePhysRegs &LiveRegs) {
  const auto &MF = MBB.getParent().getMF();
  const auto &MRI = MF.getRegInfo();
  const auto &TRI = *MRI.getTargetRegisterInfo();
  for (llvm::MCPhysReg Reg : LiveRegs) {
    if (MRI.isReserved(Reg))
      continue;
    // Skip the register if we are about to add one of its super registers.
    if (any_of(TRI.superregs(Reg), [&](llvm::MCPhysReg SReg) {
          return LiveRegs.contains(SReg) && !MRI.isReserved(SReg);
        }))
      continue;
    MBB.addLiveIn(Reg);
  }
}

// ScalarMBB::ScalarMBB(const llvm::MachineBasicBlock &ParentMBB,
//                      VectorCFG &ParentCFG, VectorMBB &EntryTakenMBB,
//                      VectorMBB &EntryNotTakenMBB, VectorMBB &ExitTakenMBB,
//                      VectorMBB &ExitNotTakenMBB)
//     : ParentCFG(ParentCFG), ParentMBB(ParentMBB) {
//
// }
} // namespace luthier