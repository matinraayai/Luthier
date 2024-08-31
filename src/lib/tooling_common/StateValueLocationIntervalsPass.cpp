//===-- StateValueLocationIntervalsPass.cpp -------------------------------===//
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
/// This file implements the State Value Location Intervals Pass.
//===----------------------------------------------------------------------===//
#include "tooling_common/StateValueLocationIntervalsPass.hpp"

#include <GCNSubtarget.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>

namespace luthier {

char StateValueLocationIntervalsPass::ID = 0;

StateValueLocationIntervalsPass::StateValueLocationIntervalsPass(
    const luthier::LiftedRepresentation &LR,
    llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &MIToHookMap)
    : llvm::MachineFunctionPass(ID), LR(LR), MIToHookMap(MIToHookMap) {

  for (const auto &[FuncSymbol, MF] : LR.functions()) {
    if (auto *KernelSymbol =
            llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(FuncSymbol)) {
      Kernel = std::make_pair(KernelSymbol, MF);
    }
    // Populate the slot index tracking for each instruction of each MF
    auto &SlotIndexes =
        FunctionsSlotIndexes.insert({MF, std::unique_ptr<llvm::SlotIndexes>()})
            .first->getSecond();
    SlotIndexes = std::make_unique<llvm::SlotIndexes>(*MF);
  }

  // Value state and flat scratch register selection ===========================
  // Try to find a fixed location to store the value state register and
  // the instrumentation stack flat scratch
  auto [FixedValueStateRegLocation,
        FixedInstrumentationStackFlatScratchLocation] =
      findFixedRegisterLocationsToStoreInstrValueVGPRAndInstrFlatScratchReg();

  auto &TRI =
      *Kernel.second->getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();

  /// If true, then it means we don't need to emit instructions in the
  /// instrumented kernel and instrumented device functions for moving
  /// the value state register and the instrumentation flat scratch register
  bool OnlyKernelNeedsPrologue{false};

  // If we found a VGPR to store the kernel arguments or found an AGPR/SGPR pair
  // to store the kernel arguments and the stack scratch reg, then all slot
  // indexes get the same value, and there's only a single interval which covers
  // the entirety of all functions involved in the lifted representation
  if (TRI.isVGPR(Kernel.second->getRegInfo(), FixedValueStateRegLocation) ||
      (TRI.isAGPR(Kernel.second->getRegInfo(), FixedValueStateRegLocation) &&
       FixedInstrumentationStackFlatScratchLocation != 0)) {
    for (const auto &[FuncSymbol, MF] : LR.functions()) {
      auto &Segments = ValueStateRegAndFlatScratchIntervals.insert({MF, {}})
                           .first->getSecond();
      Segments.emplace_back(FunctionsSlotIndexes.at(MF)->getZeroIndex(),
                            FunctionsSlotIndexes.at(MF)->getLastIndex(),
                            FixedValueStateRegLocation,
                            FixedInstrumentationStackFlatScratchLocation);
    }
    OnlyKernelNeedsPrologue = true;
  } else {
    // If not, we'll have to shuffle the value state reg and flat scratch
    // registers' locations around for each function involved
    for (const auto &[_, MF] : LR.functions()) {
      // Pick the highest numbered VGPR to hold the value state
      // If the SGPR is not fixed, pick the highest numbered SGPR
      // TODO: is there a more informed way to do initialize this?
      // TODO: if an argument is passed specifying to keep the register
      // usage of the kernel the same as before, these needs to be initialized
      // to the last available SGPR/VGPR/AGPR
      // The current location of the state value register; 0 means the state
      // value reg has been spilled to the instrumentation stack
      llvm::MCRegister StateValueRegCurrentLocation =
          *(llvm::AMDGPU::VGPR_32RegClass.end() - 1);
      // The current location of the instrumentation stack flat scratch register
      // Value of zero for this variable means the value state register is
      // in a VGPR, hence there's no need to keep an SGPR pair occupied
      llvm::MCRegister InstStackFSCurrentLocation =
          FixedInstrumentationStackFlatScratchLocation != 0
              ? FixedInstrumentationStackFlatScratchLocation
              : llvm::MCRegister(*(llvm::AMDGPU::SGPR_128RegClass.end() - 1));
      auto &MRI = MF->getRegInfo();
      // llvm::SlotIndex is used to create intervals that keep track of the
      // locations of the value state/flat scratch registers
      // Marks the beginning of the current interval we are in this loop
      llvm::SlotIndex CurrentIntervalBegin =
          FunctionsSlotIndexes.at(MF)->getZeroIndex();
      // Create an
      auto &Segments = ValueStateRegAndFlatScratchIntervals.insert({MF, {}})
                           .first->getSecond();
      for (const auto &MBB : *MF) {
        for (const auto &MI : MBB)
          // Checks whether the current MI is part of the application kernel
          // and not created by the tool writer inside the mutator function
          if (LR.getHSAInstrOfMachineInstr(MI) != nullptr) {
            {
              // TODO: Recalculate the live-ins after applying the mutator
              auto &InstrLiveRegs = *LR.getLiveInPhysRegsOfMachineInstr(MI);
              // Q: Do we have to relocate the state value register?
              // A:
              // 1. If we have spilled the state value reg and this instruction
              // will require a hook to be inserted. In this instance, since
              // the hook will have to load the value state register anyway, we
              // try and see if after loading it, we can keep it in an unused
              // VPGR/AGPR. If not, then the hook prologue will load the
              // value register, and the epilogue will spill it back
              // 2. Where we keep the value state register is going to be used.
              // In this case, we need to move the value state register
              // someplace else.
              // Note that if the value state is spilled, we will keep it
              // in private memory until the next hook is encountered, or we
              // run out of SGPRs to keep the FS register, so we try to
              // see if we can move it back to a VGPR
              bool TryRelocatingValueStateReg =
                  (StateValueRegCurrentLocation == 0 &&
                   MIToHookMap.contains(&MI)) ||
                  !InstrLiveRegs.available(MF->getRegInfo(),
                                           StateValueRegCurrentLocation);
              // Do we have to relocate the stack flat scratch register?
              bool MustRelocateStackReg =
                  InstStackFSCurrentLocation != 0 &&
                  !InstrLiveRegs.available(MF->getRegInfo(),
                                           InstStackFSCurrentLocation);
              // If we have to relocate something, then create a new interval
              // for it;
              // Note that reg scavenging might conclude that the values remain
              // where they are, and that's okay
              if (TryRelocatingValueStateReg || MustRelocateStackReg) {
                auto InstrIndex =
                    FunctionsSlotIndexes.at(MF)->getInstructionIndex(MI, true);
                Segments.emplace_back(CurrentIntervalBegin, InstrIndex,
                                      StateValueRegCurrentLocation,
                                      InstStackFSCurrentLocation);
                CurrentIntervalBegin = InstrIndex;
              }

              if (TryRelocatingValueStateReg) {
                // Find the next highest VGPR that is available
                // TODO: Limit this to the amount user requests
                bool ScavengedAVGPR{false};
                for (llvm::MCRegister Reg :
                     reverse(llvm::AMDGPU::VGPR_32RegClass)) {
                  if (MRI.isAllocatable(Reg) &&
                      InstrLiveRegs.available(MF->getRegInfo(), Reg)) {
                    StateValueRegCurrentLocation = Reg;
                    ScavengedAVGPR = true;
                    break;
                  }
                }
                // If we weren't able to scavenge a VGPR, scavenge an AGPR
                if (!ScavengedAVGPR) {
                  for (llvm::MCRegister Reg :
                       reverse(llvm::AMDGPU::AGPR_32RegClass)) {
                    if (MRI.isAllocatable(Reg) &&
                        InstrLiveRegs.available(MF->getRegInfo(), Reg)) {
                      StateValueRegCurrentLocation = Reg;
                      ScavengedAVGPR = true;
                      break;
                    }
                  }
                }
                // If we weren't able to scavenge anything, then we have to
                // spill the kernel arguments; We mark this by setting the
                // current location of the kernel arg to zero
                if (!ScavengedAVGPR) {
                  StateValueRegCurrentLocation = 0;
                }
              }
              // We will have to relocate the flat scratch ptr if we had to
              // move the kernel args to an AGPR or spill it, and we didn't
              // have it in the SGPRs
              if (MustRelocateStackReg ||
                  (TryRelocatingValueStateReg &&
                   !TRI.isVGPR(MRI, StateValueRegCurrentLocation) &&
                   InstStackFSCurrentLocation == 0)) {
                bool ScavengedSGPR{false};
                for (llvm::MCRegister Reg :
                     reverse(llvm::AMDGPU::SGPR_64RegClass)) {
                  if (MRI.isAllocatable(Reg) &&
                      InstrLiveRegs.available(MF->getRegInfo(), Reg)) {
                    InstStackFSCurrentLocation = Reg;
                    ScavengedSGPR = true;
                    break;
                  }
                }
                // If no SGPRs were able to be scavenged, and we only got here
                // because we only needed to change the scratch ptr location
                // without changing the location of the kernel arguments, we
                // check if we can move the kernel arguments to the VGPRs
                if (!ScavengedSGPR && MustRelocateStackReg) {
                  bool ScavengedVGPR{false};
                  for (llvm::MCRegister Reg :
                       reverse(llvm::AMDGPU::VGPR_32RegClass)) {
                    if (MRI.isAllocatable(Reg) &&
                        InstrLiveRegs.available(MF->getRegInfo(), Reg)) {
                      StateValueRegCurrentLocation = Reg;
                      InstStackFSCurrentLocation = 0;
                      ScavengedVGPR = true;
                      break;
                    }
                  }
                } else {
                  // Otherwise, if we got here because the kernel arguments
                  // had to be spilled into an AGPR or the instrumentation
                  // stack, we don't have enough registers for
                  // instrumentation, so throw a fatal error
                  // TODO: Make this an llvm::Error
                  llvm::report_fatal_error(
                      "Unable to scavenge enough registers for prologue");
                }
              }
            }
          }
      }
    }
  }
}

bool StateValueLocationIntervalsPass::runOnMachineFunction(
    llvm::MachineFunction &MF) {
  return false;
}
void StateValueLocationIntervalsPass::getAnalysisUsage(
    llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

std::pair<llvm::MCRegister, llvm::MCRegister> StateValueLocationIntervalsPass::
    findFixedRegisterLocationsToStoreInstrValueVGPRAndInstrFlatScratchReg()
        const {
  auto TRI = *reinterpret_cast<const llvm::SIRegisterInfo *>(
      Kernel.second->getSubtarget().getRegisterInfo());
  // Tentative place to hold an SGPR pair for the instrumentation stack
  // flat scratch
  llvm::MCRegister FixedLocationForInstrumentationFlatScratch{0};

  // Find the next VGPR available to hold the kernel args
  llvm::MCRegister FixedLocationForKernelArgs =
      TRI.findUnusedRegister(Kernel.second->getRegInfo(),
                             &llvm::AMDGPU::VGPR_32RegClass, *Kernel.second);
  // If not find the next free AGPR to hold the kernel args
  if (FixedLocationForKernelArgs == 0) {
    FixedLocationForKernelArgs =
        TRI.findUnusedRegister(Kernel.second->getRegInfo(),
                               &llvm::AMDGPU::AGPR_32RegClass, *Kernel.second);
    // Even if we were able to scavenge an AGPR, just to be safe, we
    // keep the flat scratch pointing to the bottom of the instrumentation stack
    // in case of an emergency spill of a VGPR to read back the AGPR
    // This shouldn't be necessary in GFX90A+ where AGPRs can be used as
    // normal VGPRs
    FixedLocationForInstrumentationFlatScratch =
        TRI.findUnusedRegister(Kernel.second->getRegInfo(),
                               &llvm::AMDGPU::SGPR_64RegClass, *Kernel.second);
  }
  return {FixedLocationForKernelArgs,
          FixedLocationForInstrumentationFlatScratch};
}

const InstrumentationStateValueSegment *
StateValueLocationIntervalsPass::getValueSegmentForInstr(
    llvm::MachineFunction &MF, llvm::MachineInstr &MI) const {
  if (!ValueStateRegAndFlatScratchIntervals.contains(&MF))
    return nullptr;
  if (!FunctionsSlotIndexes.contains(&MF)) {
    return nullptr;
  }
  auto &Segments = ValueStateRegAndFlatScratchIntervals.at(&MF);
  auto MISlot = FunctionsSlotIndexes.at(&MF)->getInstructionIndex(MI, false);
  for (auto &Segment : Segments) {
    if (Segment.contains(MISlot))
      return &Segment;
  }
  return nullptr;
}

} // namespace luthier