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
#include <SIInstrInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Support/FormatVariadic.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/common/LuthierError.h>
#include <luthier/llvm/CodeGenHelpers.h>
#include <luthier/llvm/streams.h>
#include <luthier/tooling/VectorCFG.h>

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

void VectorMBB::print(llvm::raw_ostream &OS, unsigned int Indent) const {
  const auto &ST = this->getParent().getMF().getSubtarget();
  const auto TII = ST.getInstrInfo();
  const auto TRI = ST.getRegisterInfo();
  OS.indent(Indent) << "Vector MBB " << Name << "\n";
  OS.indent(Indent) << "Live-ins:[";
  llvm::interleave(
      LiveIns.begin(), LiveIns.end(),
      [&](const llvm::MachineBasicBlock::RegisterMaskPair &Vec) {
        OS << printReg(llvm::Register(Vec.PhysReg), TRI);
      },
      [&]() { OS << ", "; });
  OS << "]\n";

  if (!this->empty()) {
    OS.indent(Indent) << "Contents:\n";
    for (const auto &MI : *this) {
      MI.print(OS.indent(Indent + 2), true, false, false, true, TII);
    }
  }

  OS.indent(Indent) << "Successors: [";
  llvm::interleave(
      Successors.begin(), Successors.end(),
      [&](const VectorMBB *MBB) { OS << "MBB " << MBB->getName(); },
      [&]() { OS << ", "; });
  OS << "]\n";
}

llvm::Expected<std::unique_ptr<VectorCFG>>
VectorCFG::getVectorCFG(const llvm::MachineFunction &MF) {
  auto Out = std::unique_ptr<VectorCFG>(new VectorCFG(MF));
  // Create scalar MBBs
  for (const auto &MBB : MF) {
    auto SMBB = ScalarMBB::create(MBB, *Out);
    LUTHIER_RETURN_ON_ERROR(SMBB.takeError());
    Out->MBBs.insert({&MBB, std::move(*SMBB)});
  }
  // Link scalar MBBs and the entry/exit vector blocks
  for (auto &[MBB, SMBB] : Out->MBBs) {
    if (MBB->isEntryBlock()) {
      Out->EntryBlock->addSuccessorBlock(SMBB->Entry.TakenBlock);
      Out->EntryBlock->addSuccessorBlock(SMBB->Entry.NotTakenBlock);
    }
    if (MBB->isReturnBlock()) {
      Out->ExitBlock->addPredecessorBlock(SMBB->Exit.TakenBlock);
      Out->ExitBlock->addPredecessorBlock(SMBB->Exit.NotTakenBlock);
    }
    for (const auto &MBBSucc : MBB->successors()) {
      SMBB->Exit.TakenBlock.addSuccessorBlock(
          Out->MBBs.at(MBBSucc)->Entry.TakenBlock);
      SMBB->Exit.NotTakenBlock.addSuccessorBlock(
          Out->MBBs.at(MBBSucc)->Entry.NotTakenBlock);
    }
    for (const auto &MBBPred : MBB->predecessors()) {
      SMBB->Entry.TakenBlock.addPredecessorBlock(
          Out->MBBs.at(MBBPred)->Exit.TakenBlock);
      SMBB->Entry.NotTakenBlock.addSuccessorBlock(
          Out->MBBs.at(MBBPred)->Exit.NotTakenBlock);
    }
  }
  return Out;
}
void VectorCFG::print(llvm::raw_ostream &OS) const {
  OS << "# Vector CFG for Machine Function " << MF.getName() << ":\n";
  OS << "\n";
  EntryBlock->print(OS, 2);
  OS << "\n";
  for (const auto &[MBB, SMBB] : *this) {
    SMBB->print(OS, 2);
  }
  OS << "\n";
  ExitBlock->print(OS, 2);
  OS << "\n# End Vector CFG for Machine Function " << MF.getName() << ".\n\n";
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

ScalarMBB::ScalarMBB(const llvm::MachineBasicBlock &ParentMBB,
                     VectorCFG &ParentCFG)
    : ParentCFG(ParentCFG), ParentMBB(ParentMBB), Entry([&]() {
        MBBs.emplace_back(std::make_unique<VectorMBB>(
            ParentCFG, (ParentMBB.getFullName() + llvm::Twine(".") +
                        llvm::Twine("entry-taken"))
                           .str()));
        VectorMBB &TakenBlock = *MBBs.back();

        MBBs.emplace_back(std::make_unique<VectorMBB>(
            ParentCFG, (ParentMBB.getFullName() + llvm::Twine(".") +
                        llvm::Twine("entry-not-taken"))
                           .str()));
        VectorMBB &NotTakenBlock = *MBBs.back();

        return ScalarEntryOrExitBlocks{TakenBlock, NotTakenBlock};
      }()),
      Exit([&]() {
        MBBs.emplace_back(std::make_unique<VectorMBB>(
            ParentCFG, (ParentMBB.getFullName() + llvm::Twine(".") +
                        llvm::Twine("exit-taken"))
                           .str()));
        VectorMBB &TakenBlock = *MBBs.back();

        MBBs.emplace_back(std::make_unique<VectorMBB>(
            ParentCFG, (ParentMBB.getFullName() + llvm::Twine(".") +
                        llvm::Twine("exit-not-taken"))
                           .str()));
        VectorMBB &NotTakenBlock = *MBBs.back();

        return ScalarEntryOrExitBlocks{TakenBlock, NotTakenBlock};
      }()) {}

llvm::Expected<std::unique_ptr<ScalarMBB>>
ScalarMBB::create(const llvm::MachineBasicBlock &ParentMBB,
                  VectorCFG &ParentCFG) {
  std::unique_ptr<ScalarMBB> Out{new ScalarMBB(ParentMBB, ParentCFG)};

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      ParentMBB.getParent() != nullptr, "MBB {0} does not have a MF parent.",
      ParentMBB.getFullName()));
  llvm::outs() << "Taken blocks: " << "\n";
  Out->Entry.TakenBlock.print(llvm::outs(), 0);
  Out->Entry.NotTakenBlock.print(llvm::outs(), 0);
  llvm::outs() << "Non-Taken blocks: " << "\n";
  Out->Exit.TakenBlock.print(llvm::outs(), 0);
  Out->Exit.NotTakenBlock.print(llvm::outs(), 0);

  auto *TRI = ParentMBB.getParent()->getSubtarget().getRegisterInfo();
  // The currently taken block in the MBB
  auto *CurrentTakenBlock = &Out->Entry.TakenBlock;
  // Set of VectorMBBs that are waiting to be connected to the next non-taken
  // block or joined into the next block that starts with a scalar instruction
  llvm::SmallDenseSet<VectorMBB *> NotTakenBlocksWithHangingEdges{
      &Out->Entry.NotTakenBlock};

  for (const auto &MI : ParentMBB) {
    // Check if the MI is vector (i.e. not scalar nor lane access),
    // whether it writes to the exec mask, and whether the last MI
    // (if exists) was a scalar instruction
    bool IsVector = isVector(MI);
    bool WritesExecMask = MI.modifiesRegister(llvm::AMDGPU::EXEC, TRI);
    bool IsFormerMIScalar =
        MI.getPrevNode() != nullptr &&
        (isScalar(*MI.getPrevNode()) || isLaneAccess(*MI.getPrevNode()));
    luthier::outs() << "=================\n";
    luthier::outs() << MI << "\n";
    if (IsVector && (WritesExecMask || IsFormerMIScalar)) {
      luthier::outs() << "Split point detected\n";
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
      luthier::outs() << "join point detected\n";
      // Otherwise, if we observe a scalar instruction, we have to do a "join"
      // operation: Create a new VectorMBB to replace the current taken block,
      // and make it the successor of the current taken block. Also make
      // all not taken blocks with hanging edges the predecessor of the new
      // taken block, and then clear the not-taken set
      auto &NewCurrentTakenBlock = Out->createVectorMBB();
      NewCurrentTakenBlock.addPredecessorBlock(*CurrentTakenBlock);
      llvm::outs() << "Resolving hanging edges:\n";
      for (auto &NonTakenBlock : NotTakenBlocksWithHangingEdges) {
        llvm::outs() << NonTakenBlock->getName() << "\n";
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
      CurrentTakenBlock->setInstRange(CurrentTakenBlock->begin(), NextIterator);
    Out->print(luthier::outs(), 0);
  }
  // Connect the current taken block to the exit taken block of the current
  // MBB
  CurrentTakenBlock->addSuccessorBlock(Out->Exit.TakenBlock);
  // Connect all the non-taken blocks to the exit non-taken block of the
  // current MBB
  for (auto &NonTakenBlock : NotTakenBlocksWithHangingEdges) {
    NonTakenBlock->addSuccessorBlock(Out->Exit.NotTakenBlock);
  }
  Out->print(luthier::outs(), 0);
  return Out;
}

void ScalarMBB::print(llvm::raw_ostream &OS, unsigned int Indent) const {
  OS.indent(Indent) << "Scalar MBB " << ParentMBB.getFullName() << ":\n";
  OS << "\n";
  Entry.TakenBlock.print(OS, Indent + 2);
  OS << "\n";
  Entry.NotTakenBlock.print(OS, Indent + 2);
  OS << "\n";
  for (const auto &MBB : MBBs) {
    if (&*MBB == &Entry.TakenBlock || &*MBB == &Entry.NotTakenBlock ||
        &*MBB == &Exit.TakenBlock || &*MBB == &Exit.NotTakenBlock) {
      continue;
    }
    MBB->print(OS, Indent + 2);
    OS << "\n";
  }

  Exit.TakenBlock.print(OS, Indent + 2);
  Exit.NotTakenBlock.print(OS, Indent + 2);
}

} // namespace luthier
