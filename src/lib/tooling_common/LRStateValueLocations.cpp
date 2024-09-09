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
#include "tooling_common/LRStateValueLocations.hpp"
#include "common/Error.hpp"
#include <GCNSubtarget.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <luthier/LRCallgraph.h>
#include <luthier/LRRegisterLiveness.h>

#include <utility>

namespace luthier {

/// Scavenges \p NumRegs registers with class \p RC available in \p MRI
/// Availability means the register is available in \p MRI that is not in
/// \p AccessedPhysicalRegsNotInLiveIns and not in \p LiveInRegs
/// \param [in] MRI the \c llvm::MachineRegisterInfo of the function being
/// scavenged
/// \param [in] RC the \c llvm::TargetRegisterClass of the register(s) to be
//// scavenged
/// \param [in] AccessedPhysicalRegsNotInLiveIns a set of physical registers
/// that are accessed by hooks (i.e. read from/written to) but are not part of
/// the Live-in registers in the hook insertion points (e.g. if the tool
/// wants to track the value of a register)
/// \param [in] LiveInRegs a set of physical registers that are live at the
/// instruction where the register scavenging is taking place
/// \param [in] NumRegs the number of registers to be scavenged
/// \param [out] ScavengedRegs the registers scavenged by the function
/// \return \c true if the function has succeeded in scavenging the number of
/// requested registers, \c false otherwise
bool scavengeFreeRegister(
    llvm::MachineRegisterInfo &MRI, const llvm::TargetRegisterClass &RC,
    const llvm::LivePhysRegs &AccessedPhysicalRegsNotInLiveIns,
    const llvm::LivePhysRegs &LiveInRegs, int NumRegs,
    llvm::SmallVectorImpl<llvm::MCRegister> &ScavengedRegs) {
  int NumRegsFound = 0;
  for (llvm::MCRegister Reg : reverse(RC)) {
    if (MRI.isAllocatable(Reg) &&
        AccessedPhysicalRegsNotInLiveIns.available(MRI, Reg) &&
        LiveInRegs.available(MRI, Reg)) {
      ScavengedRegs.push_back(Reg);
      NumRegsFound++;
      if (NumRegsFound == NumRegs)
        return true;
    }
  }
  return false;
}

llvm::MCRegister
scavengeFreeRegister(const llvm::MachineRegisterInfo &MRI,
                     const llvm::TargetRegisterClass &RC,
                     const llvm::LivePhysRegs &AccessedPhysicalRegsNotInLiveIns,
                     const llvm::LivePhysRegs &LiveInRegs) {
  for (llvm::MCRegister Reg : reverse(RC)) {
    if (MRI.isAllocatable(Reg) &&
        AccessedPhysicalRegsNotInLiveIns.available(MRI, Reg) &&
        LiveInRegs.available(MRI, Reg)) {
      return Reg;
    }
  }
  return {};
}

std::pair<llvm::MCRegister, bool> findVGPRLocationForHook(
    const llvm::MachineInstr *HookMI, StateValueStorage &SVS,
    const llvm::LivePhysRegs &HookLiveRegs,
    const llvm::LivePhysRegs &AccessedPhysicalRegsNotInLiveIns) {
  bool IsStateValueInVGPR = llvm::isa<VGPRValueStorage>(&SVS);
  // Find a VGPR location to load the state value if not already in a VGPR
  llvm::MCRegister VGPRLocation{0};
  bool ClobbersAppRegister{false};
  if (IsStateValueInVGPR)
    VGPRLocation = llvm::dyn_cast<VGPRValueStorage>(&SVS)->StorageVGPR;
  else {
    auto *InstrumentedMF = HookMI->getParent()->getParent();
    // Scavenge a dead VGPR to hold the state value
    VGPRLocation = scavengeFreeRegister(
        InstrumentedMF->getRegInfo(), llvm::AMDGPU::VGPR_32RegClass,
        AccessedPhysicalRegsNotInLiveIns, HookLiveRegs);
    // If a dead VGPR is not scavenged, then pick one that's not in
    // accessed Physical registers of the hook
    if (VGPRLocation == 0) {
      ClobbersAppRegister = true;
      auto &InstrumentedMFRI = InstrumentedMF->getRegInfo();
      for (llvm::MCRegister Reg : llvm::AMDGPU::VGPR_32RegClass) {
        if (InstrumentedMFRI.isAllocatable(Reg) &&
            AccessedPhysicalRegsNotInLiveIns.available(InstrumentedMFRI, Reg)) {
          VGPRLocation = Reg;
          break;
        }
      }
      // If we didn't find anything, just pick V0
      if (VGPRLocation == 0) {
        VGPRLocation = llvm::AMDGPU::VGPR0;
      }
    }
  }
  return {VGPRLocation, ClobbersAppRegister};
}

LRStateValueLocations::LRStateValueLocations(
    const luthier::LiftedRepresentation &LR, hsa::LoadedCodeObject LCO,
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &MIToHookMap,
    const llvm::LivePhysRegs &HooksAccessedPhysicalRegistersNotInLiveIns,
    const luthier::LRRegisterLiveness &RegLiveness,
    PreKernelEmissionDescriptor &PKInfo)
    : LR(LR), LCO(std::move(LCO)), MIToHookMap(MIToHookMap),
      HooksAccessedPhysicalRegistersNotInLiveIns(
          HooksAccessedPhysicalRegistersNotInLiveIns),
      RegLiveness(RegLiveness), PKInfo(PKInfo) {}

bool allocateUnusedRegister(
    llvm::ArrayRef<llvm::MachineFunction *> RelatedFunctions,
    const llvm::TargetRegisterClass *RC,
    const llvm::LivePhysRegs &PhysicalRegsUsed, unsigned int NumRegs,
    llvm::SmallVectorImpl<llvm::MCRegister> &Regs) {
  unsigned int NumRegFound = 0;

  for (llvm::MCRegister Reg : *RC) {
    bool IsUnused =
        llvm::all_of(RelatedFunctions, [&](llvm::MachineFunction *MF) {
          auto &MRI = MF->getRegInfo();
          return MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
                 PhysicalRegsUsed.available(MRI, Reg);
        });
    if (IsUnused) {
      Regs.push_back(Reg);
      NumRegFound++;
      if (NumRegFound == NumRegs)
        return true;
    }
  }
  return false;
}

llvm::MCRegister
allocateUnusedRegister(llvm::ArrayRef<llvm::MachineFunction *> RelatedFunctions,
                       const llvm::TargetRegisterClass *RC,
                       const llvm::LivePhysRegs &AccessedPhysRegsNotInLiveIns) {
  for (llvm::MCRegister Reg : *RC) {
    bool IsUnused =
        llvm::all_of(RelatedFunctions, [&](llvm::MachineFunction *MF) {
          auto &MRI = MF->getRegInfo();
          bool IsUnusedInMF = MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg);
          llvm::outs() << MF->getSubtarget().getRegisterInfo()->getName(Reg)
                       << "\n";
          llvm::outs() << "Is allocatable: " << MRI.isAllocatable(Reg) << "\n";
          llvm::outs() << "Is phys reg used: " << MRI.isPhysRegUsed(Reg)
                       << "\n";
          if (!AccessedPhysRegsNotInLiveIns.empty())
            IsUnusedInMF = IsUnusedInMF &&
                           AccessedPhysRegsNotInLiveIns.available(MRI, Reg);
          return IsUnusedInMF;
        });
    if (IsUnused) {
      return Reg;
    }
  }
  return {};
}

std::shared_ptr<StateValueStorage>
LRStateValueLocations::findFixedStateValueStorageLocation(
    llvm::ArrayRef<llvm::MachineFunction *> RelatedFunctions) const {
  // Tentative place to hold an SGPR pair for the instrumentation stack
  // flat scratch
  llvm::SmallVector<llvm::MCRegister, 3> FSLocation{};

  // Find the next VGPR available to hold the value state
  llvm::MCRegister ValueStateFixedLocation =
      allocateUnusedRegister(RelatedFunctions, &llvm::AMDGPU::VGPR_32RegClass,
                             HooksAccessedPhysicalRegistersNotInLiveIns);
  // If not find two free AGPRs; one to hold the value state, the other as
  // a temp register
  if (ValueStateFixedLocation == 0) {
    llvm::SmallVector<llvm::MCRegister, 2> ScavengedAGPRs;
    bool Scavenged = allocateUnusedRegister(
        RelatedFunctions, &llvm::AMDGPU::AGPR_32RegClass,
        HooksAccessedPhysicalRegistersNotInLiveIns, 2, ScavengedAGPRs);
    if (Scavenged)
      return std::make_shared<TwoAGPRValueStorage>(ScavengedAGPRs[0],
                                                   ScavengedAGPRs[1]);
    else if (ScavengedAGPRs.size() == 1) {
      ValueStateFixedLocation = ScavengedAGPRs[0];
    }
    // Even if we were able to scavenge an AGPR, just to be safe, we
    // keep the flat scratch pointing to the bottom of the instrumentation
    // stack in case of an emergency spill of a VGPR to read back the AGPR
    // This shouldn't be necessary in GFX90A+ where AGPRs can be used as
    // normal VGPRs
    Scavenged = allocateUnusedRegister(
        RelatedFunctions, &llvm::AMDGPU::SGPR_32RegClass,
        HooksAccessedPhysicalRegistersNotInLiveIns, 3, FSLocation);
    if (ValueStateFixedLocation != 0 && Scavenged)
      return std::make_shared<AGPRWithThreeSGPRSValueStorage>(
          ValueStateFixedLocation, FSLocation[0], FSLocation[1], FSLocation[2]);
    else if (FSLocation.size() == 3) {
      return std::make_shared<SpilledWithThreeSGPRsValueStorage>(
          FSLocation[0], FSLocation[1], FSLocation[2]);
    } else
      return nullptr;
  } else {
    return std::make_shared<VGPRValueStorage>(ValueStateFixedLocation);
  }
}

const StateValueStorageSegment *
LRStateValueLocations::getValueSegmentForInstr(llvm::MachineInstr &MI) const {
  auto *MF = MI.getParent()->getParent();
  if (!ValueStateRegAndFlatScratchIntervals.contains(MF))
    return nullptr;
  if (!FunctionsSlotIndexes.contains(&MF->getFunction())) {
    return nullptr;
  }
  auto &Segments = ValueStateRegAndFlatScratchIntervals.at(MF);
  auto MISlot = FunctionsSlotIndexes.at(&MF->getFunction())
                    ->getInstructionIndex(MI, false);
  for (auto &Segment : Segments) {
    if (Segment.contains(MISlot))
      return &Segment;
  }
  return nullptr;
}

const InsertionPointStateValueDescriptor &
LRStateValueLocations::getStateValueDescriptorOfHookInsertionPoint(
    const llvm::MachineInstr &MI) const {
  return HookMIToValueStateInterval.at(&MI);
}

llvm::Expected<std::unique_ptr<LRStateValueLocations>>
LRStateValueLocations::create(
    const LiftedRepresentation &LR, const hsa::LoadedCodeObject &LCO,
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *> &MIToHookMap,
    const llvm::LivePhysRegs &HooksAccessedPhysicalRegistersNotInLiveIns,
    const LRRegisterLiveness &RegLiveness,
    PreKernelEmissionDescriptor &PKInfo) {
  std::unique_ptr<LRStateValueLocations> Out(new LRStateValueLocations(
      LR, LCO, MIToHookMap, HooksAccessedPhysicalRegistersNotInLiveIns,
      RegLiveness, PKInfo));
  LUTHIER_RETURN_ON_ERROR(Out->calculateStateValueLocations());
  return Out;
}

llvm::Error LRStateValueLocations::calculateStateValueLocations() {
  // Populate the slot indexes for each instruction of both kernels and
  // device functions in the LCO being processed
  llvm::SmallVector<llvm::MachineFunction *, 4> MFs;
  for (const auto &[FuncSymbol, MF] : LR.functions()) {
    if (FuncSymbol->getLoadedCodeObject() == LCO) {
      auto &SlotIndexes = FunctionsSlotIndexes
                              .insert({&MF->getFunction(),
                                       std::unique_ptr<llvm::SlotIndexes>()})
                              .first->getSecond();
      SlotIndexes = std::make_unique<llvm::SlotIndexes>(*MF);
      MFs.push_back(MF);
    }
  }
  // Try to find a fixed location to store the state value
  auto StateValueFixedLocation = findFixedStateValueStorageLocation(MFs);

  if (StateValueFixedLocation != nullptr) {
    for (const auto &MF : MFs) {
      auto &Segments = ValueStateRegAndFlatScratchIntervals.insert({MF, {}})
                           .first->getSecond();
      for (const auto &MBB : *MF) {
        Segments.emplace_back(FunctionsSlotIndexes.at(&MF->getFunction())
                                  ->getInstructionIndex(MBB.front()),
                              FunctionsSlotIndexes.at(&MF->getFunction())
                                  ->getInstructionIndex(MBB.back()),
                              StateValueFixedLocation);
      }
    }
    for (const auto &[InsertionPointMI, HookFunction] : MIToHookMap) {
      auto *HookLiveRegs =
          RegLiveness.getLiveInPhysRegsOfMachineInstr(*InsertionPointMI);
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(HookLiveRegs != nullptr));
      auto [VGPRLocation, ClobbersAppReg] = findVGPRLocationForHook(
          InsertionPointMI, *StateValueFixedLocation, *HookLiveRegs,
          HooksAccessedPhysicalRegistersNotInLiveIns);
      auto *InsertionPointFunction =
          &InsertionPointMI->getParent()->getParent()->getFunction();

      const auto &InsertionPointMBB = *InsertionPointMI->getParent();

      HookMIToValueStateInterval.insert(
          {InsertionPointMI,
           InsertionPointStateValueDescriptor{
               VGPRLocation,
               ClobbersAppReg,
               {FunctionsSlotIndexes.at(InsertionPointFunction)
                    ->getInstructionIndex(InsertionPointMBB.front()),
                FunctionsSlotIndexes.at(InsertionPointFunction)
                    ->getInstructionIndex(InsertionPointMBB.front()),
                StateValueFixedLocation}}});
    }
    PKInfo.OnlyKernelNeedsPreKernel = true;
  } else {
    // If not, we'll have to shuffle the value state reg and flat scratch
    // registers' locations around for each function involved
    for (const auto &MF : MFs) {
      auto &MRI = MF->getRegInfo();
      // Pick the highest numbered VGPR not accessed by the Hooks
      // to hold the value state
      // TODO: is there a more informed way to do initialize this?
      // TODO: if an argument is passed specifying to keep the register
      // usage of the kernel the same as before, these needs to be initialized
      // to the last available SGPR/VGPR/AGPR

      auto FirstMILiveIns =
          RegLiveness.getLiveInPhysRegsOfMachineInstr(*MF->begin()->begin());
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(FirstMILiveIns != nullptr));

      // The current location of the state value register
      std::shared_ptr<StateValueStorage> SVS =
          std::make_shared<VGPRValueStorage>(scavengeFreeRegister(
              MRI, llvm::AMDGPU::VGPR_32RegClass,
              HooksAccessedPhysicalRegistersNotInLiveIns, *FirstMILiveIns));

      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
          llvm::dyn_cast<VGPRValueStorage>(SVS.get())->StorageVGPR != 0));

      // llvm::SlotIndex is used to create intervals that keep track of the
      // locations of the value state/flat scratch registers
      // Marks the beginning of the current interval we are in this loop
      llvm::SlotIndex CurrentIntervalBegin =
          FunctionsSlotIndexes.at(&MF->getFunction())->getZeroIndex();
      // Create a segment for the current interval
      auto &Segments = ValueStateRegAndFlatScratchIntervals.insert({MF, {}})
                           .first->getSecond();
      // A set of hook insertion points that fall into the current interval
      llvm::SmallDenseSet<const llvm::MachineInstr *, 4>
          HookInsertionPointsInCurrentSegment{};
      for (const auto &MBB : *MF) {
        for (const auto &MI : MBB) {
          if (MIToHookMap.contains(&MI))
            HookInsertionPointsInCurrentSegment.insert(&MI);
          auto *InstrLiveRegs = RegLiveness.getLiveInPhysRegsOfMachineInstr(MI);
          LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(InstrLiveRegs != nullptr));
          // Q: Do we have to relocate the state value register?
          // A:
          // 1. If we have spilled the state value reg and this instruction
          // will require a hook to be inserted. In this instance, since
          // the hook will have to load the value state register anyway, we
          // try and see if after loading it, we can keep it in an unused
          // VPGR/AGPR.
          // 2. Where we keep the value state register is going to be used.
          // In this case, we need to move the value state register
          // someplace else.
          // Note that if the value state is spilled, we will keep it
          // in private memory until the next hook is encountered, or we
          // run out of SGPRs to keep the FS register, so we try to
          // see if we can move it back to a VGPR
          bool TryRelocatingValueStateReg =
              llvm::isa<SpilledWithThreeSGPRsValueStorage>(SVS.get()) &&
              MIToHookMap.contains(&MI);
          bool MustRelocateStateValue =
              SVS->getStateValueStorageReg() != 0 &&
              !InstrLiveRegs->available(MF->getRegInfo(),
                                        SVS->getStateValueStorageReg());
          if (auto *TwoAGPRSVS =
                  llvm::dyn_cast<TwoAGPRValueStorage>(SVS.get())) {
            if (!InstrLiveRegs->available(MF->getRegInfo(),
                                          TwoAGPRSVS->TempAGPR))
              MustRelocateStateValue = true;
          }
          if (auto *AGPRandSGPRSVS =
                  llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(SVS.get())) {
            if (!InstrLiveRegs->available(MF->getRegInfo(),
                                          AGPRandSGPRSVS->FlatScratchSGPRLow) ||
                !InstrLiveRegs->available(MF->getRegInfo(),
                                          AGPRandSGPRSVS->FlatScratchSGPRHigh))
              MustRelocateStateValue = true;
          }

          if (auto *TwoSGPRSVS =
                  llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(SVS.get())) {
            if (!InstrLiveRegs->available(MF->getRegInfo(),
                                          TwoSGPRSVS->FlatScratchSGPRLow) ||
                !InstrLiveRegs->available(MF->getRegInfo(),
                                          TwoSGPRSVS->FlatScratchSGPRHigh))
              MustRelocateStateValue = true;
          }
          // If we have to relocate something, then create a new interval
          // for it;
          // Note that reg scavenging might conclude that the values remain
          // where they are, and that's okay
          // Also create a new interval if we reach the beginning of a MBB
          if (MI.getIterator() == MBB.begin() || TryRelocatingValueStateReg ||
              MustRelocateStateValue) {
            auto InstrIndex = FunctionsSlotIndexes.at(&MF->getFunction())
                                  ->getInstructionIndex(MI, true);
            Segments.emplace_back(CurrentIntervalBegin, InstrIndex, SVS);
            for (const auto &HookMI : HookInsertionPointsInCurrentSegment) {
              auto *HookLiveRegs =
                  RegLiveness.getLiveInPhysRegsOfMachineInstr(*HookMI);
              auto [HookSVGPR, ClobbersAppReg] = findVGPRLocationForHook(
                  HookMI, *StateValueFixedLocation, *HookLiveRegs,
                  HooksAccessedPhysicalRegistersNotInLiveIns);
              HookMIToValueStateInterval.insert(
                  {HookMI,
                   {HookSVGPR,
                    ClobbersAppReg,
                    {CurrentIntervalBegin, InstrIndex, SVS}}});
            }
            HookInsertionPointsInCurrentSegment.clear();
            CurrentIntervalBegin = InstrIndex;
          }

          if (TryRelocatingValueStateReg || MustRelocateStateValue) {
            bool WasRelocationSuccessful{false};

            // Find the next highest VGPR that is available
            // TODO: Limit this to the amount user requests
            llvm::MCRegister StateValueRegStorage = scavengeFreeRegister(
                MRI, llvm::AMDGPU::VGPR_32RegClass,
                HooksAccessedPhysicalRegistersNotInLiveIns, *InstrLiveRegs);

            bool Scavenged = StateValueRegStorage != 0;
            // If we weren't able to scavenge a VGPR, scavenge an AGPR
            if (Scavenged) {
              SVS = std::make_shared<VGPRValueStorage>(StateValueRegStorage);
              WasRelocationSuccessful = true;
            } else {
              llvm::SmallVector<llvm::MCRegister, 2> TwoScavengedAGPRs;
              Scavenged = scavengeFreeRegister(
                  MRI, llvm::AMDGPU::AGPR_32RegClass,
                  HooksAccessedPhysicalRegistersNotInLiveIns, *InstrLiveRegs, 2,
                  TwoScavengedAGPRs);
              if (Scavenged) {
                SVS = std::make_shared<TwoAGPRValueStorage>(
                    TwoScavengedAGPRs[0], TwoScavengedAGPRs[1]);
                WasRelocationSuccessful = true;
              } else {
                llvm::SmallVector<llvm::MCRegister, 3> ThreeScavengedSGPRs;
                Scavenged = scavengeFreeRegister(
                    MRI, llvm::AMDGPU::SGPR_32RegClass,
                    HooksAccessedPhysicalRegistersNotInLiveIns, *InstrLiveRegs,
                    2, ThreeScavengedSGPRs);
                if (!Scavenged)
                  WasRelocationSuccessful = false;
                else {
                  if (TwoScavengedAGPRs.size() == 1) {
                    SVS = std::make_shared<AGPRWithThreeSGPRSValueStorage>(
                        TwoScavengedAGPRs[0], ThreeScavengedSGPRs[0],
                        ThreeScavengedSGPRs[1],
                        ThreeScavengedSGPRs[2]
                        );
                  } else {
                    SVS = std::make_shared<SpilledWithThreeSGPRsValueStorage>(
                        ThreeScavengedSGPRs[0], ThreeScavengedSGPRs[1],
                        ThreeScavengedSGPRs[2]);
                  }
                  WasRelocationSuccessful = true;
                }
              }
            }
            LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(WasRelocationSuccessful ||
                                                      !MustRelocateStateValue));
          }
        }
      }
    }
  }
  return llvm::Error::success();
}

} // namespace luthier