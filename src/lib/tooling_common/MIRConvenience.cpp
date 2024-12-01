//===-- MIRConvenience.cpp ------------------------------------------------===//
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
/// This file implements a set of high-level convenience functions used to write
/// MIR instructions.
//===----------------------------------------------------------------------===//
#include "tooling_common/MIRConvenience.hpp"
#include <SIInstrInfo.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>

namespace luthier {

void emitSGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
                  llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR) {
  auto &MBB = *InsertionPoint->getParent();
  const auto *TII = MBB.getParent()->getSubtarget().getInstrInfo();
  llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::S_XOR_B32), SrcSGPR)
      .addReg(SrcSGPR)
      .addReg(DestSGPR);
  llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::S_XOR_B32), DestSGPR)
      .addReg(SrcSGPR)
      .addReg(DestSGPR, llvm::RegState::Kill);
  llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::S_XOR_B32), SrcSGPR)
      .addReg(SrcSGPR)
      .addReg(DestSGPR);
}

void emitVGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
                  llvm::MCRegister SrcVGPR, llvm::MCRegister DestVGPR) {
  auto &MBB = *InsertionPoint->getParent();
  const auto *TII = MBB.getParent()->getSubtarget().getInstrInfo();
  llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::V_XOR_B32_e32), SrcVGPR)
      .addReg(SrcVGPR)
      .addReg(DestVGPR);
  llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::V_XOR_B32_e32), DestVGPR)
      .addReg(SrcVGPR)
      .addReg(DestVGPR, llvm::RegState::Kill);
  llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::V_XOR_B32_e32), SrcVGPR)
      .addReg(SrcVGPR)
      .addReg(DestVGPR);
}

void emitExecMaskFlip(llvm::MachineBasicBlock::iterator MI) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();

  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
      .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
}

void emitMoveFromVGPRToVGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcVGPR, llvm::MCRegister DestVGPR,
                            bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_MOV_B32_e32), DestVGPR)
      .addReg(SrcVGPR, KillSource ? llvm::RegState::Kill : 0);
}

void emitMoveFromSGPRToSGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR,
                            bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32), DestSGPR)
      .addReg(SrcSGPR, KillSource ? llvm::RegState::Kill : 0);
}

void emitMoveFromAGPRToVGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcAGPR, llvm::MCRegister DestVGPR,
                            bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), DestVGPR)
      .addReg(SrcAGPR, KillSource ? llvm::RegState::Kill : 0);
}

void emitMoveFromVGPRToAGPR(llvm::MachineBasicBlock::iterator MI,
                            llvm::MCRegister SrcVGPR, llvm::MCRegister DestAGPR,
                            bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64), DestAGPR)
      .addReg(SrcVGPR, KillSource ? llvm::RegState::Kill : 0);
}

void emitMoveFromSGPRToVGPRLane(llvm::MachineBasicBlock::iterator MI,
                                llvm::MCRegister SrcSGPR,
                                llvm::MCRegister DestVGPR, unsigned int Lane,
                                bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_WRITELANE_B32), DestVGPR)
      .addReg(SrcSGPR, KillSource ? llvm::RegState::Kill : 0)
      .addImm(Lane)
      .addReg(DestVGPR);
}

void emitMoveFromVGPRLaneToSGPR(llvm::MachineBasicBlock::iterator MI,
                                llvm::MCRegister SrcVGPR,
                                llvm::MCRegister DestSGPR, unsigned int Lane,
                                bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_READLANE_B32), DestSGPR)
      .addReg(SrcVGPR, KillSource ? llvm::RegState::Kill : 0)
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
  // block of this code snippet
  llvm::MachineBasicBlock &ExitBlock = *MBB.splitAt(*Builder);
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
      llvm::BuildMI(*SCMBB, SCMBB->end(), llvm::DebugLoc(),
                    TII.get(llvm::AMDGPU::S_CMP_EQ_I32))
          .addImm(0)
          .addImm(1);
      llvm::BuildMI(*SCMBB, SCMBB->end(), llvm::DebugLoc(),
                    TII.get(llvm::AMDGPU::S_BRANCH))
          .addMBB(&ExitBlock);
    } else {
      // If this is the SCC1 block, we need to set SCC to one.
      llvm::BuildMI(*SCMBB, SCMBB->end(), llvm::DebugLoc(),
                    TII.get(llvm::AMDGPU::S_CMP_EQ_I32))
          .addImm(0)
          .addImm(0);
    }
  }
  return ExitBlock.begin();
}

void emitLoadFromEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister DestVGPR) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), DestVGPR)
      .addReg(StackPtr)
      .addImm(-8)
      .addImm(0);
}

void emitStoreToEmergencyVGPRScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
      .addReg(SrcVGPR, KillSource ? llvm::RegState::Kill : 0)
      .addReg(StackPtr)
      .addImm(-8)
      .addImm(0);
}

void emitLoadFromEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister DestVGPR) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), DestVGPR)
      .addReg(StackPtr)
      .addImm(-4)
      .addImm(0);
}

void emitStoreToEmergencySVSScratchSpillLocation(
    llvm::MachineBasicBlock::iterator MI, llvm::MCRegister StackPtr,
    llvm::MCRegister SrcVGPR, bool KillSource) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
      .addReg(SrcVGPR, KillSource ? llvm::RegState::Kill : 0)
      .addReg(StackPtr)
      .addImm(-4)
      .addImm(0);
}

void emitWaitCnt(llvm::MachineBasicBlock::iterator MI) {
  const auto &TII = *MI->getMF()->getSubtarget().getInstrInfo();
  llvm::BuildMI(*MI->getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_WAITCNT))
      .addImm(0);
}

} // namespace luthier