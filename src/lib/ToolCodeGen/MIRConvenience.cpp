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

void emitSGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
                  llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR) {
  // Swap-with-self is a no-op
  if (SrcSGPR == DestSGPR)
    return;
  auto &MBB = *InsertionPoint->getParent();
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

/// Wait states GFX9+ requires between a VALU write to an SGPR and a VMEM
/// instruction that reads that SGPR as an address operand
/// (\c GCNSubtarget::hasVMEMReadSGPRVALUDefHazard, true for every subtarget
/// from Volcanic Islands on).
///
/// Every emergency-slot access below takes its SADDR from \c StackPtr, which
/// the SVA hand-off protocol materializes with a \c V_READLANE_B32 out of the
/// state value array a couple of instructions earlier. This code is injected
/// by the target module patcher *after* codegen has finished, so neither
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

/// Emits one emergency-slot scratch access at \c [StackPtr + Offset], bracketed
/// by the waits the hardware requires and that no later pass will supply:
///
///   * Before: \c VMEMReadSGPRVALUDefWaitStates wait states, because \p
///     StackPtr is VALU-defined (see above).
///   * After: an \c S_WAITCNT. A scratch store reads its data VGPR
///     asynchronously, and every caller of the store form immediately
///     overwrites \c VGPR0 with the state value array — a write-after-read on
///     the in-flight store's data operand, which would spill whatever the
///     overwrite left behind instead of the app's \c V0. The load form has the
///     mirror-image read-after-write problem: callers consume the loaded VGPR
///     right away.
///
/// \p Offset must be non-negative, and \c StackPtr is the *bottom* of the
/// instrumentation stack, so the two slots are reached with \c offset:0 and
/// \c offset:4. A negative \c inst_offset must never be used here: on gfx942 a
/// \c SCRATCH_{LOAD,STORE}_DWORD_SADDR carrying a negative immediate takes a
/// memory violation no matter what the SADDR holds (verified by probing the
/// same instruction with SADDR from 0 to 9192 -- every value faults with
/// \c offset:-8 and every value succeeds with \c offset:0 or \c offset:8;
/// there is no bounds check on the SADDR itself).
///
/// \p StackPtr arrives in the units the *application's* frame uses, because
/// that is the one value everyone shares: the kernel prolog parks it in the
/// SVA's stack-pointer lane, and \c InjectedPayloadPEIPass reads it straight
/// into the payload's \c SGPR32 where the payload's compiler-generated frame
/// code consumes it. On a subtarget that reaches scratch through the private
/// segment buffer that is a wave-swizzled offset -- bytes times the wavefront
/// size, which is exactly what \c getScratchScaleFactor reports and what
/// \c SIFrameLowering scales an entry function's stack size by.
/// \c SCRATCH_{LOAD,STORE}_DWORD_SADDR is a *flat* scratch instruction and
/// takes a plain per-lane byte offset instead, so the two disagree by the
/// wavefront size and the access lands a whole wave's worth of scratch past
/// where it belongs. Unswizzle into \p StackPtr for the duration of the
/// access and put it back afterwards, so the register still holds what every
/// other consumer expects.
///
/// The shift pair clobbers \c SCC. Every caller is either already inside a
/// \c createSCCSafeSequenceOfMIs region (the spilled and AGPR storage schemes
/// bracket their load/store bodies with one) or sits at a point where the
/// AMDGPU ABI leaves \c SCC dead -- immediately before a call, or at a device
/// function's entry, which is where the V0-courier hand-off protocol runs.
static void emitEmergencySlotAccess(llvm::MachineBasicBlock &MBB,
                                    llvm::MachineBasicBlock::iterator Where,
                                    llvm::MCRegister StackPtr,
                                    llvm::MCRegister DataVGPR, int Offset,
                                    bool IsStore, bool KillSource) {
  assert(Offset >= 0 && "emergency-slot offsets must be non-negative");
  const auto &ST = MBB.getParent()->getSubtarget<llvm::GCNSubtarget>();
  const auto &TII = *ST.getInstrInfo();
  // Emitted before the shift so the wait states cover the first SALU read of
  // the VALU-defined StackPtr, which is now the S_LSHR_B32 rather than the
  // scratch op itself.
  emitWaitStatesImpl(MBB, Where, VMEMReadSGPRVALUDefWaitStates);
  const unsigned Log2WaveSize = llvm::Log2_32(ST.getWavefrontSize());
  const bool NeedsUnswizzle = getScratchScaleFactor(ST) != 1;
  assert((!NeedsUnswizzle ||
          getScratchScaleFactor(ST) == ST.getWavefrontSize()) &&
         "scratch scale is either 1 or the wavefront size");
  if (NeedsUnswizzle)
    (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_LSHR_B32), StackPtr)
        .addReg(StackPtr)
        .addImm(Log2WaveSize);
  if (IsStore)
    (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DataVGPR, llvm::getKillRegState(KillSource))
        .addReg(StackPtr)
        .addImm(Offset)
        .addImm(0);
  else
    (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR),
                        DataVGPR)
        .addReg(StackPtr)
        .addImm(Offset)
        .addImm(0);
  (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::S_WAITCNT))
      .addImm(0);
  // Re-swizzle: the caller's register is the shared instrumentation SP and
  // every other reader of it wants the application's units back.
  if (NeedsUnswizzle)
    (void)llvm::BuildMI(MBB, Where, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_LSHL_B32), StackPtr)
        .addReg(StackPtr)
        .addImm(Log2WaveSize);
}

void emitLoadFromEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister DestVGPR) {
  emitEmergencySlotAccess(*MI->getParent(), MI, StackPtr, DestVGPR, 0,
                          /*IsStore=*/false, /*KillSource=*/false);
}

void emitLoadFromEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister StackPtr,
    llvm::MCRegister DestVGPR) {
  emitEmergencySlotAccess(MBB, MBB.end(), StackPtr, DestVGPR, 0,
                          /*IsStore=*/false, /*KillSource=*/false);
}

void emitStoreToEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource) {
  emitEmergencySlotAccess(*MI->getParent(), MI, StackPtr, SrcVGPR, 0,
                          /*IsStore=*/true, KillSource);
}

void emitStoreToEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock &MBB, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource) {
  emitEmergencySlotAccess(MBB, MBB.end(), StackPtr, SrcVGPR, 0,
                          /*IsStore=*/true, KillSource);
}

void emitLoadFromEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister DestVGPR) {
  emitEmergencySlotAccess(*MI->getParent(), MI, StackPtr, DestVGPR, 4,
                          /*IsStore=*/false, /*KillSource=*/false);
}

void emitStoreToEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource) {
  emitEmergencySlotAccess(*MI->getParent(), MI, StackPtr, SrcVGPR, 4,
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