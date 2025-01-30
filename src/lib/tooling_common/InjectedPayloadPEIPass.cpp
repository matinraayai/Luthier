//===-- InjectedPayloadPEIPass.cpp ----------------------------------------===//
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
/// \file
/// This file implements Luthier's Injected Payload Prologue and Epilogue
/// insertion pass.
//===----------------------------------------------------------------------===//

#include "tooling_common/InjectedPayloadPEIPass.hpp"
#include "luthier/consts.h"
#include "luthier/llvm/streams.h"
#include "luthier/tooling/LiftedRepresentation.h"
#include "tooling_common/IntrinsicMIRLoweringPass.hpp"
#include "tooling_common/PhysRegsNotInLiveInsAnalysis.hpp"
#include "tooling_common/SVStorageAndLoadLocations.hpp"
#include "tooling_common/StateValueArraySpecs.hpp"
#include "tooling_common/WrapperAnalysisPasses.hpp"
#include <GCNSubtarget.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/CodeGen/MachineInstrBuilder.h>
#include <llvm/CodeGen/Passes.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-injected-payload-pei-pass"

namespace luthier {

char InjectedPayloadPEIPass::ID = 0;

static llvm::RegisterPass<InjectedPayloadPEIPass>
    X("injected-payload-pei", "Injected Payload PEI Pass",
      true /* Only looks at CFG */, false /* Analysis Pass */);

bool InjectedPayloadPEIPass::runOnMachineFunction(llvm::MachineFunction &MF) {

  LLVM_DEBUG(llvm::dbgs() << "Running the injected payload prologue/epilogue "
                             "insertion pass.\n";
             llvm::dbgs()
             << "Contents of the function before adding prologue/epilogue:\n";
             MF.print(llvm::dbgs()););

  // If the function being processed is not an injected payload (i.e a device
  // function getting called inside a hook) it does not require the custom
  // prologue/epilogue insertion pass so skip it.
  if (!MF.getFunction().hasFnAttribute(HookAttribute) &&
      !MF.getFunction().hasFnAttribute(InjectedPayloadAttribute)) {

    LLVM_DEBUG(
        llvm::dbgs()
            << "Function " << MF.getName()
            << " is not a hook or an injected payload. skipping it....";);

    return false;
  }

  auto &IModule = const_cast<llvm::Module &>(
      *getAnalysis<llvm::MachineModuleInfoWrapperPass>().getMMI().getModule());

  auto &IMAM = getAnalysis<IModuleMAMWrapperPass>().getMAM();

  const auto &IPIP =
      *IMAM.getCachedResult<InjectedPayloadAndInstPointAnalysis>(IModule);

  auto &TargetModule =
      IMAM.getCachedResult<TargetAppModuleAndMAMAnalysis>(IModule)
          ->getTargetAppModule();
  auto &TargetMAM = IMAM.getCachedResult<TargetAppModuleAndMAMAnalysis>(IModule)
                        ->getTargetAppMAM();

  const auto &StateValueLocations =
      *TargetMAM.getCachedResult<LRStateValueStorageAndLoadLocationsAnalysis>(
          TargetModule);

  auto &PKInfo =
      TargetMAM.getResult<FunctionPreambleDescriptorAnalysis>(TargetModule);

  const auto &PhysicalRegsNotTobeClobbered =
      IMAM.getCachedResult<PhysRegsNotInLiveInsAnalysis>(IModule)
          ->getPhysRegsNotInLiveIns();

  bool Changed{false};
  // Get the state value location for this hook
  auto &StateValueLoadPlan =
      *StateValueLocations.getStateValueArrayLoadPlanForInstPoint(
          *IPIP.at(MF.getFunction()));
  // Get the liveness information for the hook
  auto &InstPointLiveRegs = PhysRegVirtAccessPass.get32BitLiveInRegs();
  auto *TII = MF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
  auto &StateValueStorage = StateValueLoadPlan.StateValueStorageLocation;

  // Target Machine function which this injected payload will be patched into
  auto TargetMF = IPIP.at(MF.getFunction())->getMF();

  // We need to first determine if we need to even emit a prologue/epilogue for
  // this hook; If the hooks makes use of the state value VGPR
  // (reads from it/writes to it), or is using s[0:3], s32, and FS, then it
  // requires a prologue/epilogue
  // If any of the hooks require the presence of the state value register,
  // a pre-kernel must be emitted in the LR
  const auto &MRI = MF.getRegInfo();
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
    if (TargetMF->getFunction().getCallingConv() ==
        llvm::CallingConv::AMDGPU_KERNEL) {
      PKInfo.Kernels[TargetMF].RequiresScratchAndStackSetup = true;
      luthier::outs() << "Pre-kernel is set to: "
                      << PKInfo.Kernels[TargetMF].RequiresScratchAndStackSetup
                      << "\n";
    } else
      PKInfo.DeviceFunctions[TargetMF].RequiresScratchAndStackSetup = true;
  }

  // If the hook has stack usage
  // then we need to signal the LR pre-kernel inserter that we need them +
  // Generate code to load it
  auto &FrameInfo = MF.getFrameInfo();
  // TODO: Make sure this is correct
  if (FrameInfo.hasStackObjects() && FrameInfo.getStackSize() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "Found a use of stack.\n";);
    RequiresAccessToStack = true;
    if (TargetMF->getFunction().getCallingConv() ==
        llvm::CallingConv::AMDGPU_KERNEL) {
      PKInfo.Kernels[TargetMF].RequiresScratchAndStackSetup = true;
      luthier::outs() << "Pre-kernel is set to: "
                      << PKInfo.Kernels[TargetMF].RequiresScratchAndStackSetup
                      << "\n";
    } else
      PKInfo.DeviceFunctions[TargetMF].RequiresScratchAndStackSetup = true;
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
  AU.setPreservesAll();
  AU.addRequiredID(IModuleMAMWrapperPass::ID);
  llvm::MachineFunctionPass::getAnalysisUsage(AU);
}
} // namespace luthier