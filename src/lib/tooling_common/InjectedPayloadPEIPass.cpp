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

#include "tooling_common/InjectedPayloadPEIPass.hpp"
#include "tooling_common/IntrinsicMIRLoweringPass.hpp"
#include "tooling_common/LRStateValueStorageAndLoadLocations.hpp"
#include "tooling_common/PhysicalRegAccessVirtualizationPass.hpp"
#include <GCNSubtarget.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/Passes.h>
#include <luthier/LiftedRepresentation.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-injected-payload-pei-pass"

namespace luthier {

const static llvm::SmallDenseMap<llvm::MCRegister, unsigned int>
    ValueRegisterSpillSlots{
        {llvm::AMDGPU::SGPR0, 0},      {llvm::AMDGPU::SGPR1, 1},
        {llvm::AMDGPU::SGPR2, 2},      {llvm::AMDGPU::SGPR3, 3},
        {llvm::AMDGPU::SGPR32, 4},     {llvm::AMDGPU::FLAT_SCR_LO, 5},
        {llvm::AMDGPU::FLAT_SCR_HI, 6}};

const static llvm::SmallDenseMap<llvm::MCRegister, unsigned int>
    ValueRegisterInstrumentationSlots{
        {llvm::AMDGPU::SGPR0, 7},       {llvm::AMDGPU::SGPR1, 8},
        {llvm::AMDGPU::SGPR2, 9},       {llvm::AMDGPU::SGPR3, 10},
        {llvm::AMDGPU::SGPR32, 11},     {llvm::AMDGPU::FLAT_SCR_LO, 12},
        {llvm::AMDGPU::FLAT_SCR_HI, 13}};

char InjectedPayloadPEIPass::ID = 0;

InjectedPayloadPEIPass::InjectedPayloadPEIPass(
    const LiftedRepresentation &LR,
    const LRStateValueStorageAndLoadLocations &StateValueLocations,
    PhysicalRegAccessVirtualizationPass &PhysRegVirtAccessPass,
    const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
        &HookFuncToInstPointMI,
    const LRRegisterLiveness &RegLiveness,
    const llvm::LivePhysRegs &PhysicalRegsNotTobeClobbered,
    PreKernelEmissionDescriptor &PKInfo)
    : LR(LR), StateValueLocations(StateValueLocations),
      HookFuncToInstPointMI(HookFuncToInstPointMI), RegLiveness(RegLiveness),
      PhysicalRegsNotTobeClobbered(PhysicalRegsNotTobeClobbered),
      PKInfo(PKInfo), PhysRegVirtAccessPass(PhysRegVirtAccessPass),
      llvm::MachineFunctionPass(ID) {}

bool InjectedPayloadPEIPass::runOnMachineFunction(llvm::MachineFunction &MF) {

  LLVM_DEBUG(llvm::dbgs() << "Running the injected payload prologue/epilogue "
                             "insertion pass.\n";
             llvm::dbgs()
             << "Contents of the function before adding prologue/epilogue:\n";
             MF.print(llvm::dbgs()););

  // If the function being processed is not an injected payload (i.e a device
  // function getting called inside a hook) it does not require the custom
  // prologue/epilogue insertion pass so skip it.
  if (!MF.getFunction().hasFnAttribute(LUTHIER_HOOK_ATTRIBUTE) &&
      !MF.getFunction().hasFnAttribute(LUTHIER_INJECTED_PAYLOAD_ATTRIBUTE)) {

    LLVM_DEBUG(
        llvm::dbgs()
            << "Function " << MF.getName()
            << " is not a hook or an injected payload. skipping it....";);

    return false;
  }

  bool Changed{false};
  // Get the state value location for this hook
  auto &StateValueLoadPlan =
      *StateValueLocations.getStateValueArrayLoadPlanForInstPoint(
          *HookFuncToInstPointMI.at(&MF.getFunction()));
  // Get the liveness information for the hook
  auto &InstPointLiveRegs = PhysRegVirtAccessPass.get32BitLiveInRegs(MF);
  auto *TII = MF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
  auto &StateValueStorage = StateValueLoadPlan.StateValueStorageLocation;
  // We need to first determine if we need to even emit a prologue/epilogue for
  // this hook; If the hooks makes use of the state value VGPR
  // (reads from it/writes to it), or is using s[0:3], s32, and FS, then it
  // requires a prologue/epilogue
  // If any of the hooks require the presence of the state value register,
  // a pre-kernel must be emitted in the LR
  auto &MRI = MF.getRegInfo();
  bool HookMakesUseOfStateValueArray{false};
  // Loop over the uses of the state value array load VGPR, and find one that's
  // not the implicit use in the last return instruction
  for (const llvm::MachineInstr &StateValueUse :
       MRI.reg_instructions(StateValueLoadPlan.StateValueArrayLoadVGPR)) {
    if (!StateValueUse.isReturn() &&
        !StateValueUse.hasRegisterImplicitUseOperand(
            StateValueLoadPlan.StateValueArrayLoadVGPR)) {
      LLVM_DEBUG(
          llvm::dbgs()
              << "Found an explicit use of the state value array VGPR.\n";);
      HookMakesUseOfStateValueArray = true;
      break;
    }
  }

  for (const auto &[PhysReg, SpillLaneID] : ValueRegisterSpillSlots) {
    if (MRI.isPhysRegUsed(PhysReg)) {
      HookMakesUseOfStateValueArray = true;
      LLVM_DEBUG(auto *TRI = MF.getSubtarget().getRegisterInfo();
                 llvm::dbgs()
                 << "Found a use of a state value array spill slot. Register "
                 << llvm::printReg(PhysReg, TRI) << " in spill slot "
                 << SpillLaneID << ".\n";);
      break;
    }
  }

  if (!HookMakesUseOfStateValueArray) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Hook doesn't make use of the state value array load "
                      "VGPR. Skipping "
                      "emission of prologue and epiloge for this function.\n";);
    return false;
  }

  // Emit State Value Load prologue: ===========================================

  // Keep track of the first instruction of the injected payload
  auto EntryInstruction = MF.front().begin();

  // If the state value is located in a VGPR, then there's no need to
  // emit anything

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
        .addReg(StateValueLoadPlan.StateValueArrayLoadVGPR,
                llvm::RegState::Kill);
    // Read the state value into the VGPR
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  StateValueLoadPlan.StateValueArrayLoadVGPR)
        .addReg(TwoAGPRStorage->StorageAGPR);
    Changed |= true;
  } else if (auto *OneAGPRThreeSGPRStorage =
                 llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(
                     &StateValueStorage)) {
    PKInfo.EnableScratchAndStoreStackInfo = true;
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
        .addReg(StateValueLoadPlan.StateValueArrayLoadVGPR,
                llvm::RegState::Kill)
        .addReg(OneAGPRThreeSGPRStorage->EmergencyVGPRSpillSlotOffset)
        .addImm(0)
        .addImm(0);

    // Move the value state from the storage AGPR to the destination VGPR
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  StateValueLoadPlan.StateValueArrayLoadVGPR)
        .addReg(OneAGPRThreeSGPRStorage->StorageAGPR);
    Changed |= true;
  } else if (auto *ThreeSGPRStorage =
                 llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(
                     &StateValueStorage)) {
    PKInfo.EnableScratchAndStoreStackInfo = true;
    auto &EntryMBB = MF.front();
    // Swap the values between the Two flat scratch register pair and the
    // flat scratch register pair

    /// FS_LO Swap
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(ThreeSGPRStorage->FlatScratchSGPRLow);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32),
                  ThreeSGPRStorage->FlatScratchSGPRLow)
        .addReg(llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(ThreeSGPRStorage->FlatScratchSGPRLow, llvm::RegState::Kill);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(llvm::AMDGPU::FLAT_SCR_LO)
        .addReg(ThreeSGPRStorage->FlatScratchSGPRLow);

    /// FS_HI Swap
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(ThreeSGPRStorage->FlatScratchSGPRHigh);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32),
                  ThreeSGPRStorage->FlatScratchSGPRHigh)
        .addReg(llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(ThreeSGPRStorage->FlatScratchSGPRHigh, llvm::RegState::Kill);
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::S_XOR_B32), llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(llvm::AMDGPU::FLAT_SCR_HI)
        .addReg(ThreeSGPRStorage->FlatScratchSGPRHigh);
    // Spill the VGPR that will hold the state value into the beginning of
    // the scratch
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(StateValueLoadPlan.StateValueArrayLoadVGPR,
                llvm::RegState::Kill)
        .addReg(ThreeSGPRStorage->EmergencyVGPRSpillSlotOffset)
        .addImm(0)
        .addImm(0);

    // Move the value state from the storage AGPR to the destination VGPR
    llvm::BuildMI(EntryMBB, EntryInstruction, llvm::DebugLoc(),
                  TII->get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR),
                  StateValueLoadPlan.StateValueArrayLoadVGPR)
        .addReg(ThreeSGPRStorage->EmergencyVGPRSpillSlotOffset)
        .addImm(4)
        .addImm(0);
    Changed |= true;
  }

  // If the hook has stack usage
  // then we need to signal the LR pre-kernel inserter that we need them +
  // Generate code to load it
  auto &FrameInfo = MF.getFrameInfo();
  // TODO: Make sure this is correct
  if (FrameInfo.hasStackObjects() && FrameInfo.getStackSize() != 0) {
    PKInfo.EnableScratchAndStoreStackInfo = true;
  }

  // If the app has either s0, s1, s2, s3, s32, and FLAT_SCRATCH_LO/HI
  // live/we shouldn't clobber them,
  // then we need to spill it to the value register before the hook runs
  for (const auto &[PhysReg, SpillLane] : ValueRegisterSpillSlots) {
    if (InstPointLiveRegs.contains(PhysReg) ||
        (!PhysicalRegsNotTobeClobbered.empty() &&
         PhysicalRegsNotTobeClobbered.contains(PhysReg))) {
      llvm::BuildMI(MF.front(), EntryInstruction, llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                    StateValueLoadPlan.StateValueArrayLoadVGPR)
          .addReg(PhysReg, llvm::RegState::Kill)
          .addImm(SpillLane)
          .addReg(StateValueLoadPlan.StateValueArrayLoadVGPR);
    }
  }

  if (PKInfo.EnableScratchAndStoreStackInfo) {
    for (const auto &[PhysReg, SpillLane] : ValueRegisterInstrumentationSlots) {
      llvm::BuildMI(MF.front(), EntryInstruction, llvm::DebugLoc(),
                    TII->get(llvm::AMDGPU::V_READLANE_B32), PhysReg)
          .addReg(StateValueLoadPlan.StateValueArrayLoadVGPR)
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
            !PhysicalRegsNotTobeClobbered.empty() &&
                PhysicalRegsNotTobeClobbered.contains(PhysReg)) {
          llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_READLANE_B32), PhysReg)
              .addReg(StateValueLoadPlan.StateValueArrayLoadVGPR)
              .addImm(SpillLane);
          FirstTermInst->addOperand(
              llvm::MachineOperand::CreateReg(PhysReg, false, true));
        }
      }
      // Restore the app's state
      if (auto *TwoAGPRStorage =
              llvm::dyn_cast<TwoAGPRValueStorage>(&StateValueStorage)) {
        // Read the state value into the Storage AGPR
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                      TwoAGPRStorage->StorageAGPR)
            .addReg(StateValueLoadPlan.StateValueArrayLoadVGPR);
        // Restore the app's VGPR from the temp AGPR
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                      StateValueLoadPlan.StateValueArrayLoadVGPR)
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
                      StateValueLoadPlan.StateValueArrayLoadVGPR)
            .addReg(OneAGPRThreeSGPRStorage->EmergencyVGPRSpillSlotOffset)
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
            .addReg(ThreeSGPRStorage->FlatScratchSGPRLow);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      ThreeSGPRStorage->FlatScratchSGPRLow)
            .addReg(llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(ThreeSGPRStorage->FlatScratchSGPRLow, llvm::RegState::Kill);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(llvm::AMDGPU::FLAT_SCR_LO)
            .addReg(ThreeSGPRStorage->FlatScratchSGPRLow);

        /// FS_HI Swap
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(ThreeSGPRStorage->FlatScratchSGPRHigh);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      ThreeSGPRStorage->FlatScratchSGPRHigh)
            .addReg(llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(ThreeSGPRStorage->FlatScratchSGPRHigh,
                    llvm::RegState::Kill);
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::S_XOR_B32),
                      llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(llvm::AMDGPU::FLAT_SCR_HI)
            .addReg(ThreeSGPRStorage->FlatScratchSGPRHigh);
        // Spill the VGPR that will hold the state value into the beginning of
        // the scratch
        llvm::BuildMI(MBB, FirstTermInst, llvm::DebugLoc(),
                      TII->get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR),
                      StateValueLoadPlan.StateValueArrayLoadVGPR)
            .addReg(ThreeSGPRStorage->EmergencyVGPRSpillSlotOffset)
            .addImm(0)
            .addImm(0);
      }
    }
  }

  LLVM_DEBUG(
      llvm::dbgs()
          << "Machine function contents after inserting prologue/epilogue:\n";
      MF.print(llvm::dbgs()););

  return Changed;
}

void InjectedPayloadPEIPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addPreservedID(llvm::MachineLoopInfoID);
  AU.addPreserved<llvm::SlotIndexesWrapperPass>();
  llvm::MachineFunctionPass::getAnalysisUsage(AU);
}
} // namespace luthier