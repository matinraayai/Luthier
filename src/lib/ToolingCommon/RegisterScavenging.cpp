//===- RegisterScavenging.cpp - Machine register scavenging ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements the machine register scavenger. It can provide
/// information, such as unused registers, at any point in a machine basic
/// block. It also provides a mechanism to make registers available by evicting
/// them to spill slots.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iterator>
#include <limits>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "reg-scavenging"

STATISTIC(NumScavengedRegs, "Number of frame index regs scavenged");

void RegScavenger::init(MachineBasicBlock &MBB) {
  MachineFunction &MF = *MBB.getParent();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  LiveUnits.init(*TRI);

  this->MBB = &MBB;

  for (ScavengedInfo &SI : Scavenged) {
    SI.Reg = 0;
    SI.Restore = nullptr;
  }
}

void RegScavenger::enterBasicBlock(VectorMBB &MBB) {
  init(MBB);
  for (const auto &LI : MBB.liveins())
    LiveUnits.addRegMasked(LI.PhysReg, LI.LaneMask);
  MBBI = MBB.begin();
}

void RegScavenger::enterBasicBlockEnd(MachineBasicBlock &MBB) {
  init(MBB);
  for (const auto &LO : MBB.liveouts())
    LiveUnits.addRegMasked(LO.PhysReg, LO.LaneMask);
  MBBI = MBB.end();
}



bool RegScavenger::isRegUsed(Register Reg, bool includeReserved) const {
  if (isReserved(Reg))
    return includeReserved;
  return !LiveUnits.available(Reg);
}

Register RegScavenger::FindUnusedReg(const TargetRegisterClass *RC) const {
  for (Register Reg : *RC) {
    if (!isRegUsed(Reg)) {
      LLVM_DEBUG(dbgs() << "Scavenger found unused reg: " << printReg(Reg, TRI)
                        << "\n");
      return Reg;
    }
  }
  return 0;
}

BitVector RegScavenger::getRegsAvailable(const TargetRegisterClass *RC) {
  BitVector Mask(TRI->getNumRegs());
  for (Register Reg : *RC)
    if (!isRegUsed(Reg))
      Mask.set(Reg.id());
  return Mask;
}

/// Given the bitvector \p Available of free register units at position
/// \p From. Search backwards to find a register that is part of \p
/// Candidates and not used/clobbered until the point \p To. If there is
/// multiple candidates continue searching and pick the one that is not used/
/// clobbered for the longest time.
/// Returns the register and the earliest position we know it to be free or
/// the position MBB.end() if no register is available.
static std::pair<MCPhysReg, MachineBasicBlock::iterator>
findSurvivorBackwards(const MachineRegisterInfo &MRI,
    MachineBasicBlock::iterator From, MachineBasicBlock::iterator To,
    const LiveRegUnits &LiveOut, ArrayRef<MCPhysReg> AllocationOrder,
    bool RestoreAfter) {
  bool FoundTo = false;
  MCPhysReg Survivor = 0;
  MachineBasicBlock::iterator Pos;
  MachineBasicBlock &MBB = *From->getParent();
  unsigned InstrLimit = 25;
  unsigned InstrCountDown = InstrLimit;
  const TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();
  LiveRegUnits Used(TRI);

  assert(From->getParent() == To->getParent() &&
         "Target instruction is in other than current basic block, use "
         "enterBasicBlockEnd first");

  for (MachineBasicBlock::iterator I = From;; --I) {
    const MachineInstr &MI = *I;

    Used.accumulate(MI);

    if (I == To) {
      // See if one of the registers in RC wasn't used so far.
      for (MCPhysReg Reg : AllocationOrder) {
        if (!MRI.isReserved(Reg) && Used.available(Reg) &&
            LiveOut.available(Reg))
          return std::make_pair(Reg, MBB.end());
      }
      // Otherwise we will continue up to InstrLimit instructions to find
      // the register which is not defined/used for the longest time.
      FoundTo = true;
      Pos = To;
      // Note: It was fine so far to start our search at From, however now that
      // we have to spill, and can only place the restore after From then
      // add the regs used/defed by std::next(From) to the set.
      if (RestoreAfter)
        Used.accumulate(*std::next(From));
    }
    if (FoundTo) {
      // Don't search to FrameSetup instructions if we were searching from
      // Non-FrameSetup instructions. Otherwise, the spill position may point
      // before FrameSetup instructions.
      if (!From->getFlag(MachineInstr::FrameSetup) &&
          MI.getFlag(MachineInstr::FrameSetup))
        break;

      if (Survivor == 0 || !Used.available(Survivor)) {
        MCPhysReg AvilableReg = 0;
        for (MCPhysReg Reg : AllocationOrder) {
          if (!MRI.isReserved(Reg) && Used.available(Reg)) {
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

      // Keep searching when we find a vreg since the spilled register will
      // be usefull for this other vreg as well later.
      bool FoundVReg = false;
      for (const MachineOperand &MO : MI.operands()) {
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

static unsigned getFrameIndexOperandNum(MachineInstr &MI) {
  unsigned i = 0;
  while (!MI.getOperand(i).isFI()) {
    ++i;
    assert(i < MI.getNumOperands() && "Instr doesn't have FrameIndex operand!");
  }
  return i;
}

RegScavenger::ScavengedInfo &
RegScavenger::spill(Register Reg, const TargetRegisterClass &RC, int SPAdj,
                    MachineBasicBlock::iterator Before,
                    MachineBasicBlock::iterator &UseMI) {
  // Find an available scavenging slot with size and alignment matching
  // the requirements of the class RC.
  const MachineFunction &MF = *Before->getMF();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  unsigned NeedSize = TRI->getSpillSize(RC);
  Align NeedAlign = TRI->getSpillAlign(RC);

  unsigned SI = Scavenged.size(), Diff = std::numeric_limits<unsigned>::max();
  int FIB = MFI.getObjectIndexBegin(), FIE = MFI.getObjectIndexEnd();
  for (unsigned I = 0; I < Scavenged.size(); ++I) {
    if (Scavenged[I].Reg != 0)
      continue;
    // Verify that this slot is valid for this register.
    int FI = Scavenged[I].FrameIndex;
    if (FI < FIB || FI >= FIE)
      continue;
    unsigned S = MFI.getObjectSize(FI);
    Align A = MFI.getObjectAlign(FI);
    if (NeedSize > S || NeedAlign > A)
      continue;
    // Avoid wasting slots with large size and/or large alignment. Pick one
    // that is the best fit for this register class (in street metric).
    // Picking a larger slot than necessary could happen if a slot for a
    // larger register is reserved before a slot for a smaller one. When
    // trying to spill a smaller register, the large slot would be found
    // first, thus making it impossible to spill the larger register later.
    unsigned D = (S - NeedSize) + (A.value() - NeedAlign.value());
    if (D < Diff) {
      SI = I;
      Diff = D;
    }
  }

  if (SI == Scavenged.size()) {
    // We need to scavenge a register but have no spill slot, the target
    // must know how to do it (if not, we'll assert below).
    Scavenged.push_back(ScavengedInfo(FIE));
  }

  // Avoid infinite regress
  Scavenged[SI].Reg = Reg;

  // If the target knows how to save/restore the register, let it do so;
  // otherwise, use the emergency stack spill slot.
  if (!TRI->saveScavengerRegister(*MBB, Before, UseMI, &RC, Reg)) {
    // Spill the scavenged register before \p Before.
    int FI = Scavenged[SI].FrameIndex;
    if (FI < FIB || FI >= FIE) {
      report_fatal_error(Twine("Error while trying to spill ") +
                         TRI->getName(Reg) + " from class " +
                         TRI->getRegClassName(&RC) +
                         ": Cannot scavenge register without an emergency "
                         "spill slot!");
    }
    TII->storeRegToStackSlot(*MBB, Before, Reg, true, FI, &RC, Register());
    MachineBasicBlock::iterator II = std::prev(Before);

    unsigned FIOperandNum = getFrameIndexOperandNum(*II);
    TRI->eliminateFrameIndex(II, SPAdj, FIOperandNum, this);

    // Restore the scavenged register before its use (or first terminator).
    TII->loadRegFromStackSlot(*MBB, UseMI, Reg, FI, &RC, Register());
    II = std::prev(UseMI);

    FIOperandNum = getFrameIndexOperandNum(*II);
    TRI->eliminateFrameIndex(II, SPAdj, FIOperandNum, this);
  }
  return Scavenged[SI];
}

Register RegScavenger::scavengeRegisterBackwards(const TargetRegisterClass &RC,
                                                 MachineBasicBlock::iterator To,
                                                 bool RestoreAfter, int SPAdj,
                                                 bool AllowSpill) {
  const MachineBasicBlock &MBB = *To->getParent();
  const MachineFunction &MF = *MBB.getParent();

  // Find the register whose use is furthest away.
  ArrayRef<MCPhysReg> AllocationOrder = RC.getRawAllocationOrder(MF);
  std::pair<MCPhysReg, MachineBasicBlock::iterator> P = findSurvivorBackwards(
      *MRI, std::prev(MBBI), To, LiveUnits, AllocationOrder, RestoreAfter);
  MCPhysReg Reg = P.first;
  MachineBasicBlock::iterator SpillBefore = P.second;
  // Found an available register?
  if (Reg != 0 && SpillBefore == MBB.end()) {
    LLVM_DEBUG(dbgs() << "Scavenged free register: " << printReg(Reg, TRI)
               << '\n');
    return Reg;
  }

  if (!AllowSpill)
    return 0;

  assert(Reg != 0 && "No register left to scavenge!");

  MachineBasicBlock::iterator ReloadBefore =
      RestoreAfter ? std::next(MBBI) : MBBI;
  if (ReloadBefore != MBB.end())
    LLVM_DEBUG(dbgs() << "Reload before: " << *ReloadBefore << '\n');
  ScavengedInfo &Scavenged = spill(Reg, RC, SPAdj, SpillBefore, ReloadBefore);
  Scavenged.Restore = &*std::prev(SpillBefore);
  LiveUnits.removeReg(Reg);
  LLVM_DEBUG(dbgs() << "Scavenged register with spill: " << printReg(Reg, TRI)
             << " until " << *SpillBefore);
  return Reg;
}




