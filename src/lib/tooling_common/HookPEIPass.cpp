//===-- HookPEIPass.cpp ---------------------------------------------------===//
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
/// This file implements Luthier's Hook Prologue and Epilogue insertion pass.
//===----------------------------------------------------------------------===//

#include "tooling_common/HookPEIPass.hpp"
#include "tooling_common/PhysicalRegAccessVirtualizationPass.hpp"
#include <GCNSubtarget.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/Passes.h>

namespace luthier {

const static llvm::SmallDenseMap<llvm::MCRegister, unsigned int>
    ValueRegisterSpillSlots{
        {llvm::AMDGPU::SGPR0, 0},      {llvm::AMDGPU::SGPR1, 1},
        {llvm::AMDGPU::SGPR2, 2},      {llvm::AMDGPU::SGPR3, 3},
        {llvm::AMDGPU::SGPR32, 4},     {llvm::AMDGPU::FLAT_SCR_LO, 5},
        {llvm::AMDGPU::FLAT_SCR_HI, 6}};

char HookPEIPass::ID = 0;

HookPEIPass::HookPEIPass(
    const luthier::LiftedRepresentation &LR,
    const luthier::LRStateValueLocations &StateValueLocations,
    const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
        &HookFuncToInstPointMI,
    const LRRegisterLiveness &RegLiveness,
    const llvm::LivePhysRegs &PhysicalRegsNotTobeClobbered)
    : LR(LR), StateValueLocations(StateValueLocations),
      HookFuncToInstPointMI(HookFuncToInstPointMI), RegLiveness(RegLiveness),
      PhysicalRegsNotTobeClobbered(PhysicalRegsNotTobeClobbered),
      llvm::MachineFunctionPass(ID) {}

bool HookPEIPass::runOnMachineFunction(llvm::MachineFunction &MF) {
  bool Changed{false};
  // Get the state value location for this hook
  auto &HookStateValueLocation =
      StateValueLocations.getStateValueDescriptorOfHookInsertionPoint(
          *HookFuncToInstPointMI.at(&MF.getFunction()));
  // Get the liveness information for the hook
  auto &InstPointLiveRegs =
      getAnalysis<PhysicalRegAccessVirtualizationPass>().get32BitLiveInRegs(MF);
  auto *TII = MF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
  // Emit Value State Register Load prologue: ==================================
  // Keep track of the first instruction of the hook
  auto EntryInstruction = MF.front().begin();
  auto &StateValueLocation = HookStateValueLocation.StateValueLocation;
  auto &StateValueStorage = StateValueLocation.getSVS();
  // If the state value is located in a VGPR, then there's no need to
  // emit anything
  //
  // If the state value is located in an AGPR with a temp AGPR to spare,
  // we only need to do a couple of moves in the beginning and at the end
  // between where the state value will be located and the AGPRs
  if (auto *TwoAGPRStorage =
          llvm::dyn_cast<TwoAGPRValueStorage>(&StateValueStorage)) {
    auto &EntryMBB = MF.front();
    // Spill app's VGPR into the temp AGPR
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TwoAGPRStorage->TempAGPR)
        .addReg(HookStateValueLocation.StateValueVGPR, llvm::RegState::Kill);
    // Read the state value into the VGPR
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  HookStateValueLocation.StateValueVGPR)
        .addReg(TwoAGPRStorage->StorageAGPR);
    Changed |= true;
  } else if (auto *OneAGPRThreeSGPRStorage =
                 llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(
                     &StateValueStorage)) {
    RequiresStackInPreKernel = true;
    auto &EntryMBB = MF.front();
    // Swap the values between the Two flat scratch register pair and the
    // flat scratch register pair

    /// FS_LO Swap
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32),
                  OneAGPRThreeSGPRStorage->FlatScratchSGPRLow)
        .addReg(llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow,
                llvm::RegState::Kill);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow);

    /// FS_HI Swap
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32),
                  OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh)
        .addReg(llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh,
                llvm::RegState::Kill);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh);
    // Spill the VGPR that will hold the state value into the beginning of
    // the scratch
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(HookStateValueLocation.StateValueVGPR, llvm::RegState::Kill)
        .addReg(OneAGPRThreeSGPRStorage->InstrumentationStackPointer)
        .addImm(0)
        .addImm(0);

    // Move the value state from the storage AGPR to the destination VGPR
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  HookStateValueLocation.StateValueVGPR)
        .addReg(TwoAGPRStorage->StorageAGPR);
    Changed |= true;
  } else if (auto *ThreeSGPRStorage =
                 llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(
                     &StateValueStorage)) {
    RequiresStackInPreKernel = true;
    auto &EntryMBB = MF.front();
    // Swap the values between the Two flat scratch register pair and the
    // flat scratch register pair

    /// FS_LO Swap
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32),
                  OneAGPRThreeSGPRStorage->FlatScratchSGPRLow)
        .addReg(llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow,
                llvm::RegState::Kill);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow);

    /// FS_HI Swap
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32),
                  OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh)
        .addReg(llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh,
                llvm::RegState::Kill);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh);
    // Spill the VGPR that will hold the state value into the beginning of
    // the scratch
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(HookStateValueLocation.StateValueVGPR, llvm::RegState::Kill)
        .addReg(OneAGPRThreeSGPRStorage->InstrumentationStackPointer)
        .addImm(0)
        .addImm(0);

    // Move the value state from the storage AGPR to the destination VGPR
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR),
                  HookStateValueLocation.StateValueVGPR)
        .addReg(OneAGPRThreeSGPRStorage->InstrumentationStackPointer)
        .addImm(4)
        .addImm(0);
    Changed |= true;
  }
  // If the app has either s0, s1, s2, s3, s32, and FLAT_SCRATCH_LO/HI
  // live/we shouldn't clobber them,
  // then we need to spill it to the value register before the hook runs
  for (const auto &[PhysReg, SpillLane] : ValueRegisterSpillSlots) {
    if (InstPointLiveRegs.contains(PhysReg) ||
        PhysicalRegsNotTobeClobbered.contains(PhysReg)) {
      llvm::BuildMI(MF.front(), EntryInstruction, llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                    HookStateValueLocation.StateValueVGPR)
          .addReg(PhysReg, llvm::RegState::Kill)
          .addImm(SpillLane)
          .addReg(HookStateValueLocation.StateValueVGPR);
    }
  }
  // If the hook uses s0, s1, s2, and s3 individually, then
  // If the hook uses s[0:3], s32, and FLAT_SCRATCH_LO/HI,
  // then we need to signal the LR pre-kernel inserter that we need them +
  // Generate code to load it
  auto &MRI = MF.getRegInfo();

  for (const auto &[PhysReg, SpillLane] : ValueRegisterSpillSlots) {
    if (MRI.isPhysRegUsed(PhysReg)) {
      RequiresStackInPreKernel = true;
    }
  }
  if (RequiresStackInPreKernel) {
    for (const auto &[PhysReg, SpillLane] : ValueRegisterSpillSlots) {
      llvm::BuildMI(MF.front(), EntryInstruction, llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::V_READLANE_B32), PhysReg)
          .addReg(HookStateValueLocation.StateValueVGPR)
          .addImm(SpillLane);
    }
  }

  // Emit epilogue (do everything we just did now in reverse for all return
  // blocks)
  for (auto &MBB : MF) {
    if (MBB.isReturnBlock()) {
      auto FirstTermInst = MBB.getFirstTerminator();
      // There's no need to save s[0:3]/s32/FS of instrumentation
      // Restore s[0:3]/s32/FS of the app if saved in the prologue
      for (const auto &[PhysReg, SpillLane] : ValueRegisterSpillSlots) {
        if (InstPointLiveRegs.contains(PhysReg) ||
            PhysicalRegsNotTobeClobbered.contains(PhysReg)) {
          llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_READLANE_B32), PhysReg)
              .addReg(HookStateValueLocation.StateValueVGPR)
              .addImm(SpillLane);
        }
      }
      // Restore the app's state
      if (auto *TwoAGPRStorage =
              llvm::dyn_cast<TwoAGPRValueStorage>(&StateValueStorage)) {
        // Read the state value into the Storage AGPR
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                      TwoAGPRStorage->StorageAGPR)
            .addReg(HookStateValueLocation.StateValueVGPR);
        // Restore the app's VGPR from the temp AGPR
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                      HookStateValueLocation.StateValueVGPR)
            .addReg(TwoAGPRStorage->TempAGPR, llvm::RegState::Kill);

      } else if (auto *OneAGPRThreeSGPRStorage =
                     llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(
                         &StateValueStorage)) {
        // Swap the values between the Two flat scratch register pair and the
        // flat scratch register pair

        /// FS_LO Swap
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      OneAGPRThreeSGPRStorage->FlatScratchSGPRLow)
            .addReg(llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow,
                    llvm::RegState::Kill);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow);

        /// FS_HI Swap
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh)
            .addReg(llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh,
                    llvm::RegState::Kill);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh);
        // Load the VGPR that was spilled to hold the state value VGPR
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR),
                      HookStateValueLocation.StateValueVGPR)
            .addReg(OneAGPRThreeSGPRStorage->InstrumentationStackPointer)
            .addImm(0)
            .addImm(0);
      } else if (auto *ThreeSGPRStorage =
                     llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(
                         &StateValueStorage)) {

        /// FS_LO Swap
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      OneAGPRThreeSGPRStorage->FlatScratchSGPRLow)
            .addReg(llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow,
                    llvm::RegState::Kill);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRLow);

        /// FS_HI Swap
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh)
            .addReg(llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh,
                    llvm::RegState::Kill);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(OneAGPRThreeSGPRStorage->FlatScratchSGPRHigh);
        // Spill the VGPR that will hold the state value into the beginning of
        // the scratch
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR),
                      HookStateValueLocation.StateValueVGPR)
            .addReg(OneAGPRThreeSGPRStorage->InstrumentationStackPointer)
            .addImm(0)
            .addImm(0);
      }
    }
  }

  return Changed;
}


void HookPEIPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreservedID(llvm::MachineLoopInfoID);
  AU.addPreserved<llvm::SlotIndexesWrapperPass>();
  AU.addRequired<PhysicalRegAccessVirtualizationPass>();
  llvm::MachineFunctionPass::getAnalysisUsage(AU);
}
} // namespace luthier