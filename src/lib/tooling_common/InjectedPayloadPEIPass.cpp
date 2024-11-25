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
#include "tooling_common/StateValueArraySpecs.hpp"
#include <GCNSubtarget.h>
#include <llvm/CodeGen/MachineFrameInfo.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/Passes.h>
#include <luthier/LiftedRepresentation.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-injected-payload-pei-pass"

namespace luthier {

char InjectedPayloadPEIPass::ID = 0;

InjectedPayloadPEIPass::InjectedPayloadPEIPass(
    const LiftedRepresentation &LR,
    const LRStateValueStorageAndLoadLocations &StateValueLocations,
    PhysicalRegAccessVirtualizationPass &PhysRegVirtAccessPass,
    const llvm::DenseMap<llvm::Function *, llvm::MachineInstr *>
        &HookFuncToInstPointMI,
    const llvm::LivePhysRegs &PhysicalRegsNotTobeClobbered,
    FunctionPreambleDescriptor &PKInfo)
    : LR(LR), StateValueLocations(StateValueLocations),
      HookFuncToInstPointMI(HookFuncToInstPointMI),
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

  for (const auto &[PhysReg, SpillLaneID] :
       stateValueArray::getFrameSpillSlots()) {
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

  // Keep track of the first instruction of the injected payload
  auto &EntryInstruction = *MF.front().begin();
  // emit the prologue
  if (StateValueStorage.requiresLoadAndStoreBeforeUse()) {
    StateValueStorage.emitCodeToLoadSVA(
        EntryInstruction, StateValueLoadPlan.StateValueArrayLoadVGPR);
    Changed |= true;
  }

  bool RequiresAccessToStack{false};
  // If the SVA is stored on the stack, then this function requires access
  // to FS
  if (StateValueStorage.getStateValueStorageReg() == 0) {
    RequiresAccessToStack = true;
    if (MF.getFunction().getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL)
      PKInfo.Kernels[&MF].RequiresScratchAndStackSetup = true;
    else
      PKInfo.DeviceFunctions[&MF].RequiresScratchAndStackSetup = true;
  }

  // If the hook has stack usage
  // then we need to signal the LR pre-kernel inserter that we need them +
  // Generate code to load it
  auto &FrameInfo = MF.getFrameInfo();
  // TODO: Make sure this is correct
  if (FrameInfo.hasStackObjects() && FrameInfo.getStackSize() != 0) {
    RequiresAccessToStack = true;
    if (MF.getFunction().getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL)
      PKInfo.Kernels[&MF].RequiresScratchAndStackSetup = true;
    else
      PKInfo.DeviceFunctions[&MF].RequiresScratchAndStackSetup = true;
  }

  // If the app has either s0, s1, s2, s3, s32, and FLAT_SCRATCH_LO/HI
  // live/we shouldn't clobber them,
  // then we need to spill it to the value register before the hook runs
  for (const auto &[PhysReg, SpillLane] :
       stateValueArray::getFrameSpillSlots()) {
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
  // If the injected payload requires access to stack, then read the
  // frame registers from the SVA lanes
  if (RequiresAccessToStack) {
    for (const auto &[PhysReg, SpillLane] :
         stateValueArray::getFrameStoreSlots()) {
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
      for (const auto &[PhysReg, SpillLane] :
           stateValueArray::getFrameSpillSlots()) {
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
      if (StateValueStorage.requiresLoadAndStoreBeforeUse()) {
        StateValueStorage.emitCodeToStoreSVA(
            EntryInstruction, StateValueLoadPlan.StateValueArrayLoadVGPR);
        Changed |= true;
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