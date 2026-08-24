//===-- TargetModuleScavenger.cpp ---------------------------------*- C++-*-===//
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
/// \file TargetModuleScavenger.cpp
/// Sibling-class fork of \c llvm::RegScavenger. See the header for
/// rationale. Code transposed from
/// \c llvm/lib/CodeGen/RegisterScavenging.cpp with these specific changes:
///   * \c findSurvivorBackwards consults the extra \c ReservedRegs in
///     addition to \c MRI.isReserved at every candidate check.
///   * \c isReserved returns true for both \c MRI.isReserved and
///     \c ReservedRegs members.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TargetModuleScavenger.h"

#include "luthier/LLVM/streams.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/LiveRegUnits.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachineOperand.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-reg-scavenging"

namespace luthier {

void TargetModuleScavenger::assignRegToScavengingIndex(
    int FI, llvm::Register Reg, llvm::MachineInstr *Restore) {
  for (ScavengedInfo &Slot : Scavenged) {
    if (Slot.FrameIndex == FI) {
      assert(!Slot.Reg || Slot.Reg == Reg);
      Slot.Reg = Reg;
      Slot.Restore = Restore;
      return;
    }
  }
  llvm_unreachable("did not find scavenging index");
}

void TargetModuleScavenger::setRegUsed(llvm::Register Reg,
                                     llvm::LaneBitmask LaneMask) {
  LiveUnits.addRegMasked(Reg, LaneMask);
}

void TargetModuleScavenger::init(llvm::MachineBasicBlock &MBBIn) {
  llvm::MachineFunction &MF = *MBBIn.getParent();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  LiveUnits.init(*TRI);
  this->MBB = &MBBIn;
  for (ScavengedInfo &SI : Scavenged) {
    SI.Reg = 0;
    SI.Restore = nullptr;
  }
}

void TargetModuleScavenger::enterBasicBlock(llvm::MachineBasicBlock &MBBIn) {
  init(MBBIn);
  LiveUnits.addLiveIns(MBBIn);
  MBBI = MBBIn.begin();
}

void TargetModuleScavenger::enterBasicBlockEnd(llvm::MachineBasicBlock &MBBIn) {
  init(MBBIn);
  LiveUnits.addLiveOuts(MBBIn);
  MBBI = MBBIn.end();
}

void TargetModuleScavenger::backward() {
  const llvm::MachineInstr &MI = *--MBBI;
  LiveUnits.stepBackward(MI);
  for (ScavengedInfo &I : Scavenged) {
    if (I.Restore == &MI) {
      I.Reg = 0;
      I.Restore = nullptr;
    }
  }
}

bool TargetModuleScavenger::isReserved(llvm::Register Reg) const {
  for (llvm::MCPhysReg R : ReservedRegs)
    if (TRI->regsOverlap(Reg, R))
      return true;
  return MRI->isReserved(Reg);
}

bool TargetModuleScavenger::isRegUsed(llvm::Register Reg,
                                    bool IncludeReserved) const {
  if (isReserved(Reg))
    return IncludeReserved;
  return !LiveUnits.available(Reg);
}

llvm::Register
TargetModuleScavenger::FindUnusedReg(const llvm::TargetRegisterClass *RC) const {
  for (llvm::Register Reg : *RC) {
    if (!isRegUsed(Reg)) {
      LLVM_DEBUG(luthier::dbgs() << "LuthierScavenger found unused reg: "
                                 << llvm::printReg(Reg, TRI) << "\n");
      return Reg;
    }
  }
  return 0;
}

llvm::BitVector
TargetModuleScavenger::getRegsAvailable(const llvm::TargetRegisterClass *RC) {
  llvm::BitVector Mask(TRI->getNumRegs());
  for (llvm::Register Reg : *RC)
    if (!isRegUsed(Reg))
      Mask.set(Reg.id());
  return Mask;
}

/// See the stock \c findSurvivorBackwards. Sole modification: the
/// \c MRI.isReserved guard is replaced by a predicate that also rejects
/// any candidate whose regunits overlap a reg in \p ReservedRegs (via
/// \c TRI.regsOverlap), so a paired/aliased candidate that overlaps an
/// individually-reserved reg is caught.
static std::pair<llvm::MCPhysReg, llvm::MachineBasicBlock::iterator>
findSurvivorBackwards(
    const llvm::MachineRegisterInfo &MRI,
    const llvm::DenseSet<llvm::MCPhysReg> &ReservedRegs,
    llvm::MachineBasicBlock::iterator From,
    llvm::MachineBasicBlock::iterator To, const llvm::LiveRegUnits &LiveOut,
    llvm::ArrayRef<llvm::MCPhysReg> AllocationOrder, bool RestoreAfter) {
  bool FoundTo = false;
  llvm::MCPhysReg Survivor = 0;
  llvm::MachineBasicBlock::iterator Pos;
  llvm::MachineBasicBlock &MBB = *From->getParent();
  unsigned InstrLimit = 25;
  unsigned InstrCountDown = InstrLimit;
  const llvm::TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();
  llvm::LiveRegUnits Used(TRI);

  auto Forbidden = [&](llvm::MCPhysReg Reg) {
    if (MRI.isReserved(Reg))
      return true;
    for (llvm::MCPhysReg R : ReservedRegs)
      if (TRI.regsOverlap(Reg, R))
        return true;
    return false;
  };

  assert(From->getParent() == To->getParent() &&
         "Target instruction is in other than current basic block, use "
         "enterBasicBlockEnd first");

  for (llvm::MachineBasicBlock::iterator I = From;; --I) {
    const llvm::MachineInstr &MI = *I;
    Used.accumulate(MI);
    if (I == To) {
      for (llvm::MCPhysReg Reg : AllocationOrder) {
        if (!Forbidden(Reg) && Used.available(Reg) && LiveOut.available(Reg))
          return std::make_pair(Reg, MBB.end());
      }
      FoundTo = true;
      Pos = To;
      if (RestoreAfter)
        Used.accumulate(*std::next(From));
    }
    if (FoundTo) {
      if (!From->getFlag(llvm::MachineInstr::FrameSetup) &&
          MI.getFlag(llvm::MachineInstr::FrameSetup))
        break;

      if (Survivor == 0 || !Used.available(Survivor)) {
        llvm::MCPhysReg AvilableReg = 0;
        for (llvm::MCPhysReg Reg : AllocationOrder) {
          if (!Forbidden(Reg) && Used.available(Reg)) {
            AvilableReg = Reg;
            break;
          }
        }
        if (AvilableReg == 0)
          break;
        Survivor = AvilableReg;
      }
      if (--InstrCountDown == 0)
        break;

      bool FoundVReg = false;
      for (const llvm::MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.getReg().isVirtual()) {
          FoundVReg = true;
          break;
        }
      }
      if (FoundVReg) {
        InstrCountDown = InstrLimit;
        Pos = I;
      }
      if (I == MBB.begin())
        break;
    }
    assert(I != MBB.begin() && "Did not find target instruction while "
                               "iterating backwards");
  }

  return std::make_pair(Survivor, Pos);
}

static unsigned getFrameIndexOperandNum(llvm::MachineInstr &MI) {
  unsigned i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }
  return i;
}

TargetModuleScavenger::ScavengedInfo &
TargetModuleScavenger::spill(llvm::Register Reg,
                           const llvm::TargetRegisterClass &RC, int SPAdj,
                           llvm::MachineBasicBlock::iterator Before,
                           llvm::MachineBasicBlock::iterator &UseMI) {
  // Stock FrameIndex path — verbatim from llvm::RegScavenger::spill.
  // TODO: Add a spill to SVA path
  const llvm::MachineFunction &MF = *Before->getMF();
  const llvm::MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned NeedSize = TRI->getSpillSize(RC);
  llvm::Align NeedAlign = TRI->getSpillAlign(RC);

  unsigned SI = Scavenged.size(), Diff = std::numeric_limits<unsigned>::max();
  int FIB = MFI.getObjectIndexBegin(), FIE = MFI.getObjectIndexEnd();
  for (unsigned I = 0; I < Scavenged.size(); ++I) {
    if (Scavenged[I].Reg != 0)
      continue;
    int FI = Scavenged[I].FrameIndex;
    if (FI < FIB || FI >= FIE)
      continue;
    unsigned S = MFI.getObjectSize(FI);
    llvm::Align A = MFI.getObjectAlign(FI);
    if (NeedSize > S || NeedAlign > A)
      continue;
    unsigned D = (S - NeedSize) + (A.value() - NeedAlign.value());
    if (D < Diff) {
      SI = I;
      Diff = D;
    }
  }

  if (SI == Scavenged.size()) {
    Scavenged.push_back(ScavengedInfo(FIE));
  }

  Scavenged[SI].Reg = Reg;
  if (!TRI->saveScavengerRegister(*MBB, Before, UseMI, &RC, Reg)) {
    int FI = Scavenged[SI].FrameIndex;
    if (FI < FIB || FI >= FIE) {
      llvm::report_fatal_error(llvm::Twine("Error while trying to spill ") +
                               TRI->getName(Reg) + " from class " +
                               TRI->getRegClassName(&RC) +
                               ": Cannot scavenge register without an "
                               "emergency spill slot or SVA-lane sink!");
    }
    TII->storeRegToStackSlot(*MBB, Before, Reg, true, FI, &RC,
                             llvm::Register());
    llvm::MachineBasicBlock::iterator II = std::prev(Before);
    unsigned FIOperandNum = getFrameIndexOperandNum(*II);
    // Stock scavenger passes `this` (RegScavenger*) to eliminateFrameIndex.
    // Our sibling fork can't satisfy that type, so we pass nullptr — which
    // AMDGPU's SIRegisterInfo::eliminateFrameIndex accepts (the scavenger
    // arg is only used to track post-elim live state, which we manage
    // separately). If a target rejects nullptr, the SVA-lane sink path
    // above is the supported escape hatch.
    TRI->eliminateFrameIndex(II, SPAdj, FIOperandNum, /*RS=*/nullptr);

    TII->loadRegFromStackSlot(*MBB, UseMI, Reg, FI, &RC, llvm::Register());
    II = std::prev(UseMI);
    FIOperandNum = getFrameIndexOperandNum(*II);
    TRI->eliminateFrameIndex(II, SPAdj, FIOperandNum, /*RS=*/nullptr);
  }
  return Scavenged[SI];
}

llvm::Register TargetModuleScavenger::scavengeRegisterBackwards(
    const llvm::TargetRegisterClass &RC, llvm::MachineBasicBlock::iterator To,
    bool RestoreAfter, int SPAdj, bool AllowSpill) {
  const llvm::MachineBasicBlock &MBBR = *To->getParent();
  const llvm::MachineFunction &MF = *MBBR.getParent();
  llvm::ArrayRef<llvm::MCPhysReg> AllocationOrder =
      RC.getRawAllocationOrder(MF);
  std::pair<llvm::MCPhysReg, llvm::MachineBasicBlock::iterator> P =
      findSurvivorBackwards(*MRI, ReservedRegs, std::prev(MBBI), To,
                            LiveUnits, AllocationOrder, RestoreAfter);
  llvm::MCPhysReg Reg = P.first;
  llvm::MachineBasicBlock::iterator SpillBefore = P.second;
  if (Reg != 0 && SpillBefore == MBBR.end()) {
    LLVM_DEBUG(luthier::dbgs() << "LuthierScavenged free register: "
                               << llvm::printReg(Reg, TRI) << '\n');
    return Reg;
  }
  if (!AllowSpill)
    return 0;
  assert(Reg != 0 && "No register left to scavenge!");
  llvm::MachineBasicBlock::iterator ReloadBefore =
      RestoreAfter ? std::next(MBBI) : MBBI;
  ScavengedInfo &Scav = spill(Reg, RC, SPAdj, SpillBefore, ReloadBefore);
  Scav.Restore = &*std::prev(SpillBefore);
  LiveUnits.removeReg(Reg);
  LLVM_DEBUG(luthier::dbgs()
             << "LuthierScavenged register with spill: "
             << llvm::printReg(Reg, TRI) << " until " << *SpillBefore);
  return Reg;
}

} // namespace luthier
