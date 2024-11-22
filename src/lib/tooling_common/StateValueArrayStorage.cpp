//===-- StateValueArrayStorage.cpp ----------------------------------------===//
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
/// This file implement different storage mechanisms for the state value array.
//===----------------------------------------------------------------------===//

#include "tooling_common/StateValueArrayStorage.hpp"
#include <llvm/CodeGen/MachineInstrBuilder.h>

namespace luthier {

/// Swaps the value between \p ScrSGPR and \p DestSGPR by inserting 3
/// <tt>S_XOR_B32</tt>s before \p InsertionPoint
static void buildSGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
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

/// Generates a set of MBBs that ensures the \c SCC bit does not get clobbered
/// due to the sequence of instructions built by \p MIBuilder before the
/// insertion point \p MI
/// This is a common pattern used when loading and storing the state value
/// array that allows flipping the exec mask without clobbering the
/// \c SCC bit and not requiring temporary registers
static void createSCCSafeSequenceOfMIs(
    llvm::MachineInstr &MI,
    const std::function<void(llvm::MachineBasicBlock &,
                             const llvm::TargetInstrInfo &)> &MIBuilder) {
  auto &MBB = *MI.getParent();
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
}

void TwoAGPRValueStorage::emitCodeToLoadSVA(llvm::MachineInstr &MI,
                                            llvm::MCRegister DestVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill the Dest VGPR in the active lanes to the temp AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64), TempAGPR)
        .addReg(DestVGPR);
    // Copy the state value from AGPR to the dest VGPR in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), DestVGPR)
        .addReg(StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill the Dest VGPR in the remaining non-active lanes to the temp
    // AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64), TempAGPR)
        .addReg(DestVGPR);
    // Copy the state value from AGPR to the dest VGPR in the non-active
    // lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), DestVGPR)
        .addReg(StorageAGPR, llvm::RegState::Kill);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

void TwoAGPRValueStorage::emitCodeToStoreSVA(llvm::MachineInstr &MI,
                                             llvm::MCRegister SrcVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill the Src VGPR in the active lanes to the storage AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64), StorageAGPR)
        .addReg(SrcVGPR);
    // Restore the temp AGPR contents into the src VGPR in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), SrcVGPR)
        .addReg(TempAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill the Src VGPR in the inactive lanes to the storage AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64), StorageAGPR)
        .addReg(SrcVGPR);
    // Restore the temp AGPR contents into the src VGPR in inactive laness
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), SrcVGPR)
        .addReg(TempAGPR, llvm::RegState::Kill);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

void AGPRWithThreeSGPRSValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// FS swap
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                  FlatScratchSGPRLow);
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                  FlatScratchSGPRHigh);
    /// Spill the DestVGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(0)
        .addImm(0);
    /// Restore the state value array from the storage AGPR to the dest VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), DestVGPR)
        .addReg(StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    /// Spill the DestVGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DestVGPR, llvm::RegState::Kill)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(0)
        .addImm(0);
    /// Restore the state value array from the storage AGPR to the dest VPGR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), DestVGPR)
        .addReg(StorageAGPR);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

void AGPRWithThreeSGPRSValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// FS swap
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                  FlatScratchSGPRLow);
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                  FlatScratchSGPRHigh);
    /// Spill the Src VGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(4)
        .addImm(0);
    /// Restore the app VGPR from the storage AGPR in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), SrcVGPR)
        .addReg(StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);

    /// Spill the Src to the emergency spill slot in the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(4)
        .addImm(0);
    /// Restore the app VGPR from the storage AGPR in the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), SrcVGPR)
        .addReg(StorageAGPR);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

void SpilledWithThreeSGPRsValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// FS swap
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                  FlatScratchSGPRLow);
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                  FlatScratchSGPRHigh);
    /// Spill the DestVGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(0)
        .addImm(0);
    /// Restore the state value array from its fixed storage to the dest VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(4)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    /// Spill the DestVGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DestVGPR, llvm::RegState::Kill)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(0)
        .addImm(0);
    /// Restore the state value array from its fixed storage to the dest VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(4)
        .addImm(0);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

void SpilledWithThreeSGPRsValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// FS swap
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                  FlatScratchSGPRLow);
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                  FlatScratchSGPRHigh);
    /// Spill the Src to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(4)
        .addImm(0);
    /// Restore the app VGPR from its fixed storage to the src VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(0)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);

    /// Spill the Src to the emergency spill slot in the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(4)
        .addImm(0);
    /// Restore the app VGPR from its fixed storage to the src VGPR
    /// in the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(0)
        .addImm(0);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

} // namespace luthier