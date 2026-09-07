//===-- MIRConvenience.cpp ------------------------------------------------===//
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
/// \file
/// This file implements a set of high-level convenience functions used to write
/// MIR instructions.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/MIRConvenience.h"
#include <GCNSubtarget.h>
#include <SIDefines.h>
#include <SIInstrInfo.h>
#include <SIRegisterInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/Support/MathExtras.h>
#include <algorithm>
#include <cassert>

namespace luthier {

bool shouldImplicitReadExec(const llvm::MachineInstr &MI) {
  if (llvm::SIInstrInfo::isVALU(MI)) {
    switch (MI.getOpcode()) {
    case llvm::AMDGPU::V_READLANE_B32:
    case llvm::AMDGPU::SI_RESTORE_S32_FROM_VGPR:
    case llvm::AMDGPU::V_WRITELANE_B32:
    case llvm::AMDGPU::SI_SPILL_S32_TO_VGPR:
      return false;
    default:
      return true;
    }
  }

  if (llvm::SIInstrInfo::isSALU(MI) || llvm::SIInstrInfo::isSMRD(MI))
    return false;

  return true;
}

bool isVectorMBB(const llvm::MachineBasicBlock &MBB) {
  auto It = MBB.getFirstNonDebugInstr();
  if (It == MBB.end())
    return false;
  return shouldImplicitReadExec(*It);
}

static void emitSGPRSwapImpl(llvm::MachineBasicBlock &MBB,
                             llvm::MachineBasicBlock::iterator InsertionPoint,
                             llvm::MCRegister SrcSGPR,
                             llvm::MCRegister DestSGPR) {
  // Swap-with-self is a no-op
  if (SrcSGPR == DestSGPR)
    return;
  const auto *TII = MBB.getParent()->getSubtarget().getInstrInfo();
  (void)llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32), SrcSGPR)
      .addReg(SrcSGPR)
      .addReg(DestSGPR);
  (void)llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32), DestSGPR)
      .addReg(SrcSGPR)
      .addReg(DestSGPR, llvm::RegState::Kill);
  (void)llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32), SrcSGPR)
      .addReg(SrcSGPR)
      .addReg(DestSGPR);
}

void emitSGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
                  llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR) {
  emitSGPRSwapImpl(*InsertionPoint->getParent(), InsertionPoint, SrcSGPR,
                   DestSGPR);
}

void emitSGPRSwap(llvm::MachineBasicBlock &MBB, llvm::MCRegister SrcSGPR,
                  llvm::MCRegister DestSGPR) {
  emitSGPRSwapImpl(MBB, MBB.end(), SrcSGPR, DestSGPR);
}

/// \return the \c HwregEncoding immediate naming one half of \c FLAT_SCR .
static int16_t flatScratchHwregEncoding(bool Hi) {
  using namespace llvm::AMDGPU::Hwreg;
  return static_cast<int16_t>(
      HwregEncoding::encode(Hi ? ID_FLAT_SCR_HI : ID_FLAT_SCR_LO, 0, 32));
}

void emitFlatScratchSwap(llvm::MachineBasicBlock &MBB,
                         llvm::MCRegister InstFSLo, llvm::MCRegister InstFSHi,
                         llvm::MCRegister TempSGPR) {
  const auto &ST = MBB.getParent()->getSubtarget<llvm::GCNSubtarget>();
  const auto &TII = *ST.getInstrInfo();
  if (!llvm::AMDGPU::isGFX10Plus(ST)) {
    // FLAT_SCR_LO / FLAT_SCR_HI are addressable SGPRs here, so the exchange is
    // two ordinary register swaps and no temporary is needed.
    emitSGPRSwap(MBB, llvm::AMDGPU::FLAT_SCR_HI, InstFSHi);
    emitSGPRSwap(MBB, llvm::AMDGPU::FLAT_SCR_LO, InstFSLo);
    return;
  }
  // GFX10+: FLAT_SCR is a hardware register reachable only through
  // S_GETREG_B32 / S_SETREG_B32, neither of which can name a second register,
  // so the XOR trick is unavailable. Rotate each half through TempSGPR:
  //   TempSGPR <- FLAT_SCR_x ; FLAT_SCR_x <- InstFSx ; InstFSx <- TempSGPR
  // which leaves the application's half in InstFSx, exactly as the GFX9 swap
  // does, so a second call restores it.
  auto SwapHalf = [&](bool Hi, llvm::MCRegister InstFS) {
    const int16_t Encoded = flatScratchHwregEncoding(Hi);
    (void)llvm::BuildMI(MBB, MBB.end(), llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_GETREG_B32), TempSGPR)
        .addImm(Encoded);
    (void)llvm::BuildMI(MBB, MBB.end(), llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_SETREG_B32))
        .addReg(InstFS)
        .addImm(Encoded);
    (void)llvm::BuildMI(MBB, MBB.end(), llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_MOV_B32), InstFS)
        .addReg(TempSGPR, llvm::RegState::Kill);
  };
  SwapHalf(/*Hi=*/true, InstFSHi);
  SwapHalf(/*Hi=*/false, InstFSLo);
}

void emitFlatScratchSaveToSVALanes(llvm::MachineBasicBlock &MBB,
                                   llvm::MCRegister SVAVGPR, unsigned Lane,
                                   llvm::MCRegister TempSGPR) {
  const auto &ST = MBB.getParent()->getSubtarget<llvm::GCNSubtarget>();
  if (!llvm::AMDGPU::isGFX10Plus(ST)) {
    emitMoveFromSGPRToVGPRLane(MBB, llvm::AMDGPU::FLAT_SCR_LO, SVAVGPR, Lane,
                               false);
    emitMoveFromSGPRToVGPRLane(MBB, llvm::AMDGPU::FLAT_SCR_HI, SVAVGPR,
                               Lane + 1, false);
    return;
  }
  assert(TempSGPR && "GFX10+ needs a temporary to read FLAT_SCR out of hwreg");
  const auto &TII = *ST.getInstrInfo();
  for (unsigned I = 0; I != 2; ++I) {
    (void)llvm::BuildMI(MBB, MBB.end(), llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_GETREG_B32), TempSGPR)
        .addImm(flatScratchHwregEncoding(/*Hi=*/I == 1));
    emitMoveFromSGPRToVGPRLane(MBB, TempSGPR, SVAVGPR, Lane + I, true);
  }
}

void emitFlatScratchLoadFromSVALanes(llvm::MachineBasicBlock &MBB,
                                     llvm::MCRegister SVAVGPR, unsigned Lane,
                                     llvm::MCRegister TempSGPR) {
  const auto &ST = MBB.getParent()->getSubtarget<llvm::GCNSubtarget>();
  if (!llvm::AMDGPU::isGFX10Plus(ST)) {
    emitMoveFromVGPRLaneToSGPR(MBB, SVAVGPR, llvm::AMDGPU::FLAT_SCR_LO, Lane,
                               false);
    emitMoveFromVGPRLaneToSGPR(MBB, SVAVGPR, llvm::AMDGPU::FLAT_SCR_HI,
                               Lane + 1, false);
    return;
  }
  assert(TempSGPR && "GFX10+ needs a temporary to write FLAT_SCR via hwreg");
  const auto &TII = *ST.getInstrInfo();
  for (unsigned I = 0; I != 2; ++I) {
    emitMoveFromVGPRLaneToSGPR(MBB, SVAVGPR, TempSGPR, Lane + I, false);
    (void)llvm::BuildMI(MBB, MBB.end(), llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_SETREG_B32))
        .addReg(TempSGPR, llvm::RegState::Kill)
        .addImm(flatScratchHwregEncoding(/*Hi=*/I == 1));
  }
}

void emitVGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
                  llvm::MCRegister SrcVGPR, llvm::MCRegister DestVGPR) {
  // Self-swap is a no-op
  if (SrcVGPR == DestVGPR)
    return;
  auto &MBB = *InsertionPoint->getParent();
  const auto *TII = MBB.getParent()->getSubtarget().getInstrInfo();
  (void)llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::V_XOR_B32_e32), SrcVGPR)
      .addReg(SrcVGPR)
      .addReg(DestVGPR);
  (void)llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::V_XOR_B32_e32), DestVGPR)
      .addReg(SrcVGPR)
      .addReg(DestVGPR, llvm::RegState::Kill);
  (void)llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::V_XOR_B32_e32), SrcVGPR)
      .addReg(SrcVGPR)
      .addReg(DestVGPR);
}

void emitExecMaskFlip(llvm::MachineBasicBlock::iterator MI) {
  const auto &ST = MI->getMF()->getSubtarget<llvm::GCNSubtarget>();
  const auto &TII = *ST.getInstrInfo();
  const llvm::MCRegister Exec = ST.getRegisterInfo()->getExec();
  const unsigned Opc =
      ST.isWave32() ? llvm::AMDGPU::S_NOT_B32 : llvm::AMDGPU::S_NOT_B64;
  (void)llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(), TII.get(Opc),
                      Exec)
      .addReg(Exec, llvm::RegState::Kill);
}

void emitExecMaskFlip(llvm::MachineBasicBlock &MBB) {
  emitExecMaskFlip(MBB, MBB.end());
}

void emitExecMaskFlip(llvm::MachineBasicBlock &MBB,
                      llvm::MachineBasicBlock::iterator InsertionPoint) {
  const auto &ST = MBB.getParent()->getSubtarget<llvm::GCNSubtarget>();
  const auto &TII = *ST.getInstrInfo();
  const llvm::MCRegister Exec = ST.getRegisterInfo()->getExec();
  const unsigned Opc =
      ST.isWave32() ? llvm::AMDGPU::S_NOT_B32 : llvm::AMDGPU::S_NOT_B64;
  (void)llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(), TII.get(Opc), Exec)
      .addReg(Exec, llvm::RegState::Kill);
}

void emitMoveFromVGPRToVGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcVGPR, llvm::MCRegister DestVGPR,
                            bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_MOV_B32_e32), DestVGPR)
      .addReg(SrcVGPR, llvm::getKillRegState(KillSource));
}

void emitMoveFromVGPRToVGPR(llvm::MachineBasicBlock &MBB,
                            llvm::MCRegister SrcVGPR, llvm::MCRegister DestVGPR,
                            bool KillSource) {
  const auto &TII = *MBB.getParent()->getSubtarget().getInstrInfo();
  llvm::BuildMI(MBB, MBB.end(), llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_MOV_B32_e32), DestVGPR)
      .addReg(SrcVGPR, llvm::getKillRegState(KillSource));
}

void emitMoveFromSGPRToSGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR,
                            bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32), DestSGPR)
      .addReg(SrcSGPR, llvm::getKillRegState(KillSource));
}

void emitMoveFromAGPRToVGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcAGPR, llvm::MCRegister DestVGPR,
                            bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), DestVGPR)
      .addReg(SrcAGPR, llvm::getKillRegState(KillSource));
}

void emitMoveFromVGPRToAGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcVGPR, llvm::MCRegister DestAGPR,
                            bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64), DestAGPR)
      .addReg(SrcVGPR, llvm::getKillRegState(KillSource));
}

void emitMoveFromSGPRToVGPRLane(llvm::MachineBasicBlock::iterator MI,
                                llvm::MCRegister SrcSGPR,
                                llvm::MCRegister DestVGPR, unsigned int Lane,
                                bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  (void)llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::V_WRITELANE_B32), DestVGPR)
      .addReg(SrcSGPR, llvm::getKillRegState(KillSource))
      .addImm(Lane)
      .addReg(DestVGPR);
}

void emitMoveFromSGPRToVGPRLane(llvm::MachineBasicBlock &MBB,
                                llvm::MCRegister SrcSGPR,
                                llvm::MCRegister DestVGPR, unsigned int Lane,
                                bool KillSource) {
  const auto &TII = *MBB.getParent()->getSubtarget().getInstrInfo();
  (void)llvm::BuildMI(MBB, MBB.end(), llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::V_WRITELANE_B32), DestVGPR)
      .addReg(SrcSGPR, llvm::getKillRegState(KillSource))
      .addImm(Lane)
      .addReg(DestVGPR);
}

void emitMoveFromVGPRLaneToSGPR(llvm::MachineBasicBlock::iterator MI,
                                llvm::MCRegister SrcVGPR,
                                llvm::MCRegister DestSGPR, unsigned int Lane,
                                bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  (void)llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::V_READLANE_B32), DestSGPR)
      .addReg(SrcVGPR, llvm::getKillRegState(KillSource))
      .addImm(Lane);
}

void emitMoveFromVGPRLaneToSGPR(llvm::MachineBasicBlock &MBB,
                                llvm::MCRegister SrcVGPR,
                                llvm::MCRegister DestSGPR, unsigned int Lane,
                                bool KillSource) {
  const auto &TII = *MBB.getParent()->getSubtarget().getInstrInfo();
  (void)llvm::BuildMI(MBB, MBB.end(), llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::V_READLANE_B32), DestSGPR)
      .addReg(SrcVGPR, llvm::getKillRegState(KillSource))
      .addImm(Lane);
}

llvm::MachineBasicBlock::iterator createSCCSafeSequenceOfMIs(
    llvm::MachineBasicBlock::iterator MI,
    const std::function<void(llvm::MachineBasicBlock &,
                             const llvm::TargetInstrInfo &)> &MIBuilder) {
  auto &MBB = *MI->getParent();
  auto &MF = *MBB.getParent();
  const auto &TII = *MF.getSubtarget().getInstrInfo();
  // First we add an SCC1 branch before the MI
  auto Builder = llvm::BuildMI(MBB, MI, llvm::DebugLoc(),
                               TII.get(llvm::AMDGPU::S_CBRANCH_SCC1));
  // We then split MBB into two at the newly inserted branch instruction;
  // The MBB is the entry block while the newly created block is the exit
  // block of this code snippet.
  llvm::MachineBasicBlock *ExitBlockPtr = MBB.splitAt(*Builder);
  if (ExitBlockPtr == &MBB) {
    ExitBlockPtr = MF.CreateMachineBasicBlock(MBB.getBasicBlock());
    MF.insert(std::next(MBB.getIterator()), ExitBlockPtr);
    ExitBlockPtr->transferSuccessorsAndUpdatePHIs(&MBB);
  }
  llvm::MachineBasicBlock &ExitBlock = *ExitBlockPtr;
  MBB.removeSuccessor(&ExitBlock);
  // Create the SCC0MBB, which will house the code for when SCC=0
  // It comes right after the entry block because the CBRANCH is not taken
  auto *SCC0MBB = MF.CreateMachineBasicBlock();
  MF.insert(ExitBlock.getIterator(), SCC0MBB);
  MBB.addSuccessor(SCC0MBB);
  SCC0MBB->addSuccessor(&ExitBlock);
  // Create the SCC1MBB, which will house the code for when SCC=1
  // It comes after SCC0MBB, and falls right through to the exit block
  auto *SCC1MBB = MF.CreateMachineBasicBlock();
  MF.insert(ExitBlock.getIterator(), SCC1MBB);
  MBB.addSuccessor(SCC1MBB);
  SCC1MBB->addSuccessor(&ExitBlock);
  // Make the S_CBRANCH_SCC1 instruction jump to the SCC1MBB branch
  Builder.addMBB(SCC1MBB);
  // Now that we've created the basic blocks, and we've implicitly saved the
  // SCC value by branching, we can now safely carry out operations that
  // clobber the SCC bit
  for (auto *SCMBB : {SCC0MBB, SCC1MBB}) {
    // Insert the user-defined instructions
    MIBuilder(*SCMBB, TII);
    // If this is the SCC0 block, we need to set SCC to zero.
    // We also need to do an unconditional branch to the exit block
    if (SCMBB == SCC0MBB) {
      (void)llvm::BuildMI(*SCMBB, SCMBB->end(), llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::S_CMP_EQ_I32))
          .addImm(0)
          .addImm(1);
      (void)llvm::BuildMI(*SCMBB, SCMBB->end(), llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::S_BRANCH))
          .addMBB(&ExitBlock);
    } else {
      // If this is the SCC1 block, we need to set SCC to one.
      (void)llvm::BuildMI(*SCMBB, SCMBB->end(), llvm::DebugLoc(),
                          TII.get(llvm::AMDGPU::S_CMP_EQ_I32))
          .addImm(0)
          .addImm(0);
    }
  }
  return ExitBlock.begin();
}

/// Wait states GFX9+ requires between a write to an SGPR and a VMEM
/// instruction that reads that SGPR as an address operand
/// (\c GCNSubtarget::hasVMEMReadSGPRVALUDefHazard, true for every subtarget
/// from Volcanic Islands on).
///
/// Only the GFX9 absolute-flat-scratch emergency-slot path still has an SGPR
/// address operand to hazard on — every other target reaches the slots with a
/// SADDR-less \c SCRATCH_* . This code is injected by the target module
/// patcher *after* codegen has finished, so neither
/// \c PostRAHazardRecognizerPass nor \c SIInsertWaitcnts ever runs over it and
/// the required waits have to be emitted here.
static constexpr unsigned VMEMReadSGPRVALUDefWaitStates = 5;

/// Emits \p NumWaitStates worth of \c S_NOP before \p Where. An \c S_NOP with
/// immediate \c N is worth \c N+1 wait states and the immediate is 4 bits, so
/// long requests are split across several.
static void emitWaitStatesImpl(llvm::MachineBasicBlock &MBB,
                               llvm::MachineBasicBlock::iterator Where,
                               unsigned NumWaitStates) {
  const auto &TII = *MBB.getParent()->getSubtarget().getInstrInfo();
  while (NumWaitStates > 0) {
    const unsigned Chunk = std::min(NumWaitStates, 16u);
    (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_NOP))
        .addImm(Chunk - 1);
    NumWaitStates -= Chunk;
  }
}

/// Emits one emergency-slot scratch access at the absolute private-segment
/// offset \p Offset, bracketed by the waits the hardware requires and that no
/// later pass will supply.
///
/// The instrumentation stack starts at offset zero of the wavefront's private
/// segment, so the two slots are simply \c offset:0 and \c offset:4 with no
/// address register to add. \c inst_offset stays non-negative: on gfx942 a
/// \c SCRATCH_{LOAD,STORE}_DWORD_SADDR carrying a negative immediate takes a
/// memory violation no matter what the SADDR holds (verified by probing the
/// same instruction with SADDR from 0 to 9192 -- every value faults with
/// \c offset:-8 and every value succeeds with \c offset:0 or \c offset:8;
/// there is no bounds check on the SADDR itself).
///
/// Which SADDR-less encoding is available depends on the target:
///   * \c hasFlatScratchSTMode (gfx10.3+, gfx940+) — the ST form takes
///     neither a VGPR nor an SGPR address operand.
///   * GFX10.1 — no ST form, but \c SGPR_NULL reads as zero and is a legal
///     SADDR.
///   * GFX9 absolute flat scratch — neither, so \p SADDRScratchSGPR is zeroed
///     and used. This is the whole reason the GFX9 absolute-FS storage schemes
///     still reserve a third SGPR.
///
/// A trailing \c S_WAITCNT is always emitted. A scratch store reads its data
/// VGPR asynchronously, and every caller of the store form immediately
/// overwrites \c VGPR0 with the state value array — a write-after-read on the
/// in-flight store's data operand, which would spill whatever the overwrite
/// left behind instead of the app's \c V0. The load form has the mirror-image
/// read-after-write problem: callers consume the loaded VGPR right away.
static void emitEmergencySlotAccess(llvm::MachineBasicBlock &MBB,
                                    llvm::MachineBasicBlock::iterator Where,
                                    llvm::MCRegister SADDRScratchSGPR,
                                    llvm::MCRegister DataVGPR, int Offset,
                                    bool IsStore, bool KillSource) {
  assert(Offset >= 0 && "emergency-slot offsets must be non-negative");
  const auto &ST = MBB.getParent()->getSubtarget<llvm::GCNSubtarget>();
  const auto &TII = *ST.getInstrInfo();

  if (ST.hasFlatScratchSTMode()) {
    const unsigned Opc = IsStore ? llvm::AMDGPU::SCRATCH_STORE_DWORD_ST
                                 : llvm::AMDGPU::SCRATCH_LOAD_DWORD_ST;
    if (IsStore)
      (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(), TII.get(Opc))
          .addReg(DataVGPR, llvm::getKillRegState(KillSource))
          .addImm(Offset)
          .addImm(0);
    else
      (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(), TII.get(Opc), DataVGPR)
          .addImm(Offset)
          .addImm(0);
    (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
    return;
  }

  llvm::MCRegister SADDR;
  if (llvm::AMDGPU::isGFX10Plus(ST)) {
    SADDR = llvm::AMDGPU::SGPR_NULL;
  } else {
    assert(SADDRScratchSGPR &&
           "GFX9 absolute-flat-scratch emergency-slot access needs a scratch "
           "SGPR to hold the zero SADDR");
    SADDR = SADDRScratchSGPR;
    (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_MOV_B32), SADDR)
        .addImm(0);
    // GFX9+ requires wait states between an SALU write to an SGPR and a VMEM
    // instruction that reads it as an address operand
    // (\c GCNSubtarget::hasVMEMReadSGPRVALUDefHazard). This code is injected
    // after codegen has finished, so neither \c PostRAHazardRecognizerPass nor
    // \c SIInsertWaitcnts ever runs over it and the waits have to be emitted
    // here.
    emitWaitStatesImpl(MBB, Where, VMEMReadSGPRVALUDefWaitStates);
  }

  const unsigned Opc = IsStore ? llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR
                               : llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR;
  if (IsStore)
    (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(), TII.get(Opc))
        .addReg(DataVGPR, llvm::getKillRegState(KillSource))
        .addReg(SADDR)
        .addImm(Offset)
        .addImm(0);
  else
    (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(), TII.get(Opc), DataVGPR)
        .addReg(SADDR)
        .addImm(Offset)
        .addImm(0);
  (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::S_WAITCNT))
      .addImm(0);
}

void emitLoadFromEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister DestVGPR) {
  emitEmergencySlotAccess(*MI->getParent(), MI, SADDRScratchSGPR, DestVGPR, 0,
                          /*IsStore=*/false, /*KillSource=*/false);
}

void emitLoadFromEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister DestVGPR) {
  emitEmergencySlotAccess(MBB, MBB.end(), SADDRScratchSGPR, DestVGPR, 0,
                          /*IsStore=*/false, /*KillSource=*/false);
}

void emitStoreToEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister SrcVGPR, bool KillSource) {
  emitEmergencySlotAccess(*MI->getParent(), MI, SADDRScratchSGPR, SrcVGPR, 0,
                          /*IsStore=*/true, KillSource);
}

void emitStoreToEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister SrcVGPR, bool KillSource) {
  emitEmergencySlotAccess(MBB, MBB.end(), SADDRScratchSGPR, SrcVGPR, 0,
                          /*IsStore=*/true, KillSource);
}

void emitLoadFromEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister DestVGPR) {
  emitEmergencySlotAccess(*MI->getParent(), MI, SADDRScratchSGPR, DestVGPR, 4,
                          /*IsStore=*/false, /*KillSource=*/false);
}

void emitStoreToEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister SADDRScratchSGPR,
    llvm::MCRegister SrcVGPR, bool KillSource) {
  emitEmergencySlotAccess(*MI->getParent(), MI, SADDRScratchSGPR, SrcVGPR, 4,
                          /*IsStore=*/true, KillSource);
}

unsigned getScratchScaleFactor(const llvm::GCNSubtarget &ST) {
  return ST.hasFlatScratchEnabled() ? 1 : ST.getWavefrontSize();
}

void emitWaitCnt(llvm::MachineBasicBlock::iterator MI, unsigned Encoding) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  (void)llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::S_WAITCNT))
      .addImm(Encoding);
}

} // namespace luthier