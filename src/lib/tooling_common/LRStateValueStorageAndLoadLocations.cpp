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
#include "tooling_common/LRStateValueStorageAndLoadLocations.hpp"
#include "common/Error.hpp"
#include <GCNSubtarget.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <luthier/LRCallgraph.h>
#include <luthier/LRRegisterLiveness.h>

#include <utility>

namespace luthier {

/// Scavenges \p NumRegs registers with class \p RC available in \p MRI
/// Availability means a register is allocatable and not in \p MRI and
/// is not in \p AccessedPhysicalRegsNotInLiveIns and not in \p LiveInRegs
/// \param [in] MRI the \c llvm::MachineRegisterInfo of the function being
/// scavenged
/// \param [in] RC the \c llvm::TargetRegisterClass of the register(s) to be
//// scavenged
/// \param [in] AccessedPhysicalRegsNotInLiveIns a set of physical registers
/// that are accessed by injected payloads of the instrumentation module but
/// at the point of access are not part of the Live-in registers of the
/// instrumentation points
/// \param [in] LiveInRegs a set of physical registers that are live at the
/// app instruction where the register scavenging is taking place
/// \param [in] NumRegs the number of registers to be scavenged
/// \param [out] ScavengedRegs the registers scavenged by the function
/// \return \c true if the function has succeeded in scavenging the number of
/// requested registers, \c false otherwise
static bool
scavengeFreeRegister(llvm::MachineRegisterInfo &MRI,
                     const llvm::TargetRegisterClass &RC,
                     const llvm::LivePhysRegs &AccessedPhysicalRegsNotInLiveIns,
                     const llvm::LivePhysRegs &LiveInRegs, int NumRegs,
                     llvm::SmallVectorImpl<llvm::MCRegister> &ScavengedRegs) {
  int NumRegsFound = 0;
  for (llvm::MCRegister Reg : reverse(RC)) {
    if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
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

/// Scavenges \p NumRegs registers with class \p RC available in \p MRI
/// Availability means a register is allocatable and not used in \p MRI and
/// is not in \p AccessedPhysicalRegsNotInLiveIns and not in \p LiveInRegs
/// \param MRI the \c llvm::MachineRegisterInfo of the function being
/// scavenged
/// \param RC the \c llvm::TargetRegisterClass of the register to be
//// scavenged
/// \param AccessedPhysicalRegsNotInLiveIns a set of physical registers
/// that are accessed by injected payloads of the instrumentation module but
/// at the point of access are not part of the Live-in registers of the
/// instrumentation points
/// \param LiveInRegs a set of physical registers that are live at the
/// app instruction where the register scavenging is taking place
/// \return the scavenged register if successful, or zero otherwise
static llvm::MCRegister
scavengeFreeRegister(const llvm::MachineRegisterInfo &MRI,
                     const llvm::TargetRegisterClass &RC,
                     const llvm::LivePhysRegs &AccessedPhysicalRegsNotInLiveIns,
                     const llvm::LivePhysRegs &LiveInRegs) {
  for (llvm::MCRegister Reg : reverse(RC)) {
    if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
        AccessedPhysicalRegsNotInLiveIns.available(MRI, Reg) &&
        LiveInRegs.available(MRI, Reg)) {
      return Reg;
    }
  }
  return {};
}

/// Scavenges \p NumRegs register of class \p RC that are unused across
/// all \p RelatedFunctions and are not in \p AccessedPhysRegsNotInLiveIns
/// \param [in] Functions the functions being scavenged for a free register
/// \param [in] RC the register class of the registers being scavenged
/// \param [in] AccessedPhysRegsNotInLiveIns a set of physical registers
/// accessed by the injected payloads that are not in the live-in set of their
/// injected payload at the point of access
/// \param [in] NumRegs number of registers to be scavenged
/// \param [out] Regs the set of registers that were scavenged
/// \return \c true if the function was able to scavenge the requested number
/// of registers, \c false otherwise
static bool
scavengeFreeRegister(llvm::ArrayRef<llvm::MachineFunction *> Functions,
                     const llvm::TargetRegisterClass *RC,
                     const llvm::LivePhysRegs &AccessedPhysRegsNotInLiveIns,
                     unsigned int NumRegs,
                     llvm::SmallVectorImpl<llvm::MCRegister> &Regs) {
  unsigned int NumRegFound = 0;

  for (llvm::MCRegister Reg : *RC) {
    bool IsUnused = llvm::all_of(Functions, [&](llvm::MachineFunction *MF) {
      auto &MRI = MF->getRegInfo();
      return MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
             AccessedPhysRegsNotInLiveIns.available(MRI, Reg);
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
scavengeFreeRegister(llvm::ArrayRef<llvm::MachineFunction *> RelatedFunctions,
                     const llvm::TargetRegisterClass *RC,
                     const llvm::LivePhysRegs &AccessedPhysRegsNotInLiveIns) {
  for (llvm::MCRegister Reg : *RC) {
    bool IsUnused =
        llvm::all_of(RelatedFunctions, [&](llvm::MachineFunction *MF) {
          auto &MRI = MF->getRegInfo();
          bool IsUnusedInMF = MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg);
          //          llvm::outs() <<
          //          MF->getSubtarget().getRegisterInfo()->getName(Reg)
          //                       << "\n";
          //          llvm::outs() << "Is allocatable: " <<
          //          MRI.isAllocatable(Reg) << "\n"; llvm::outs() << "Is phys
          //          reg used: " << MRI.isPhysRegUsed(Reg)
          //                       << "\n";
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

/// Selects a VGPR to load the state value array into for use for the
/// injected payload of \p InstPoint
/// \param InstPoint instrumentation point for which we are selecting a VGPR
/// to load the state value array into
/// \param SVS the state value array storage at the location of \p InstPoint
/// \param InstPointLiveRegs a set of physical registers that are live before
/// the \p InstPoint
/// \param AccessedPhysicalRegsNotInLiveIns a set of physical registers
/// accessed in injected payloads that aren't in the live-ins set of their
/// instrumentation point at the point of access
/// \return a pair, with the first element indicating the VGPR selected, and
/// the second element indicating whether the selected VGPR will clobber a
/// live register of the app and needs preserving
static std::pair<llvm::MCRegister, bool>
selectVGPRLoadLocationForInjectedPayload(
    const llvm::MachineInstr &InstPoint, StateValueArrayStorage &SVS,
    const llvm::LivePhysRegs &InstPointLiveRegs,
    const llvm::LivePhysRegs &AccessedPhysicalRegsNotInLiveIns) {
  bool IsStateValueInVGPR = llvm::isa<VGPRStateValueArrayStorage>(&SVS);
  llvm::MCRegister VGPRLocation{0};
  bool ClobbersAppRegister{false};
  // if the state value array already in a VGPR, then select the same VGPR
  // to be the load destination
  if (IsStateValueInVGPR)
    VGPRLocation =
        llvm::dyn_cast<VGPRStateValueArrayStorage>(&SVS)->StorageVGPR;
  else {
    auto &InstrumentedMF = *InstPoint.getParent()->getParent();
    // Scavenge a dead VGPR to hold the state value
    VGPRLocation = scavengeFreeRegister(
        InstrumentedMF.getRegInfo(), llvm::AMDGPU::VGPR_32RegClass,
        AccessedPhysicalRegsNotInLiveIns, InstPointLiveRegs);
    // If a dead VGPR is not scavenged, then pick one that's not in
    // accessed Physical registers of the hook
    if (VGPRLocation == 0) {
      ClobbersAppRegister = true;
      auto &InstrumentedMFRI = InstrumentedMF.getRegInfo();
      for (llvm::MCRegister Reg : llvm::AMDGPU::VGPR_32RegClass) {
        if (InstrumentedMFRI.isPhysRegUsed(Reg) &&
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

LRStateValueStorageAndLoadLocations::LRStateValueStorageAndLoadLocations(
    const luthier::LiftedRepresentation &LR, hsa::LoadedCodeObject LCO,
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
        &InstPointToInjectedPayloadMap,
    const llvm::LivePhysRegs &AccessedPhysicalRegistersNotInLiveIns,
    const luthier::LRRegisterLiveness &RegLiveness,
    PreKernelEmissionDescriptor &PKInfo)
    : LR(LR), LCO(std::move(LCO)),
      InstPointToInjectedPayloadMap(InstPointToInjectedPayloadMap),
      AccessedPhysicalRegistersNotInLiveIns(
          AccessedPhysicalRegistersNotInLiveIns),
      RegLiveness(RegLiveness), PKInfo(PKInfo) {}

std::shared_ptr<StateValueArrayStorage>
LRStateValueStorageAndLoadLocations::findFixedStateValueArrayStorage(
    llvm::ArrayRef<llvm::MachineFunction *> RelatedFunctions) const {
  // Tentative place to hold three SGPRs for emergency spill
  llvm::SmallVector<llvm::MCRegister, 3> FSLocation{};

  // Find the next VGPR available to hold the value state
  llvm::MCRegister ValueStateFixedLocation =
      scavengeFreeRegister(RelatedFunctions, &llvm::AMDGPU::VGPR_32RegClass,
                           AccessedPhysicalRegistersNotInLiveIns);
  // If not find two free AGPRs; one to hold the value state, the other as
  // a temp register
  if (ValueStateFixedLocation == 0) {
    llvm::SmallVector<llvm::MCRegister, 2> ScavengedAGPRs;
    bool Scavenged = scavengeFreeRegister(
        RelatedFunctions, &llvm::AMDGPU::AGPR_32RegClass,
        AccessedPhysicalRegistersNotInLiveIns, 2, ScavengedAGPRs);
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
    Scavenged = scavengeFreeRegister(
        RelatedFunctions, &llvm::AMDGPU::SGPR_32RegClass,
        AccessedPhysicalRegistersNotInLiveIns, 3, FSLocation);
    if (ValueStateFixedLocation != 0 && Scavenged)
      return std::make_shared<AGPRWithThreeSGPRSValueStorage>(
          ValueStateFixedLocation, FSLocation[0], FSLocation[1], FSLocation[2]);
    else if (FSLocation.size() == 3) {
      return std::make_shared<SpilledWithThreeSGPRsValueStorage>(
          FSLocation[0], FSLocation[1], FSLocation[2]);
    } else
      return nullptr;
  } else {
    return std::make_shared<VGPRStateValueArrayStorage>(
        ValueStateFixedLocation);
  }
}

llvm::ArrayRef<StateValueStorageSegment>
LRStateValueStorageAndLoadLocations::getStorageIntervalsOfBasicBlock(
    const llvm::MachineBasicBlock &MBB) const {
  auto *MF = MBB.getParent();
  auto It = StateValueStorageIntervals.find(MF);
  if (It == StateValueStorageIntervals.end())
    return {};
  else
    return It->second;
}

const InstPointSVALoadPlan *
LRStateValueStorageAndLoadLocations::getStateValueArrayLoadPlanForInstPoint(
    const llvm::MachineInstr &MI) const {
  auto It = InstPointSVSLoadPlans.find(&MI);
  if (It == InstPointSVSLoadPlans.end())
    return nullptr;
  else
    return &It->second;
}

llvm::Expected<std::unique_ptr<LRStateValueStorageAndLoadLocations>>
LRStateValueStorageAndLoadLocations::create(
    const LiftedRepresentation &LR, const hsa::LoadedCodeObject &LCO,
    const llvm::DenseMap<llvm::MachineInstr *, llvm::Function *>
        &InstPointToInjectedPayloadMap,
    const llvm::LivePhysRegs &AccessedPhysicalRegistersNotInLiveIns,
    const LRRegisterLiveness &RegLiveness,
    PreKernelEmissionDescriptor &PKInfo) {
  std::unique_ptr<LRStateValueStorageAndLoadLocations> Out(
      new LRStateValueStorageAndLoadLocations(
          LR, LCO, InstPointToInjectedPayloadMap,
          AccessedPhysicalRegistersNotInLiveIns, RegLiveness, PKInfo));
  LUTHIER_RETURN_ON_ERROR(
      Out->calculateStateValueArrayStorageAndLoadLocations());
  return Out;
}

llvm::Error LRStateValueStorageAndLoadLocations::
    calculateStateValueArrayStorageAndLoadLocations() {
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
  auto StateValueFixedLocation = findFixedStateValueArrayStorage(MFs);

  if (StateValueFixedLocation != nullptr) {
    // If a fixed location was found, then all MBB intervals inside all MFs
    // will get the fixed state value location
    for (const auto &MF : MFs) {
      auto &Segments =
          StateValueStorageIntervals.insert({MF, {}}).first->getSecond();
      for (const auto &MBB : *MF) {
        Segments.emplace_back(FunctionsSlotIndexes.at(&MF->getFunction())
                                  ->getInstructionIndex(MBB.front()),
                              FunctionsSlotIndexes.at(&MF->getFunction())
                                  ->getInstructionIndex(MBB.back()),
                              StateValueFixedLocation);
      }
    }
    for (const auto &[InsertionPointMI, HookFunction] :
         InstPointToInjectedPayloadMap) {
      auto *HookLiveRegs =
          RegLiveness.getLiveInPhysRegsOfMachineInstr(*InsertionPointMI);
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(HookLiveRegs != nullptr, ""));
      auto [VGPRLocation, ClobbersAppReg] =
          selectVGPRLoadLocationForInjectedPayload(
              *InsertionPointMI, *StateValueFixedLocation, *HookLiveRegs,
              AccessedPhysicalRegistersNotInLiveIns);
      auto *InsertionPointFunction =
          &InsertionPointMI->getParent()->getParent()->getFunction();

      const auto &InsertionPointMBB = *InsertionPointMI->getParent();

      InstPointSVSLoadPlans.insert(
          {InsertionPointMI, InstPointSVALoadPlan{VGPRLocation, ClobbersAppReg,
                                                  *StateValueFixedLocation}});
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
      LUTHIER_RETURN_ON_ERROR(
          LUTHIER_ERROR_CHECK(FirstMILiveIns != nullptr, ""));

      // The current location of the state value register
      std::shared_ptr<StateValueArrayStorage> SVS =
          std::make_shared<VGPRStateValueArrayStorage>(scavengeFreeRegister(
              MRI, llvm::AMDGPU::VGPR_32RegClass,
              AccessedPhysicalRegistersNotInLiveIns, *FirstMILiveIns));

      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
          llvm::dyn_cast<VGPRStateValueArrayStorage>(SVS.get())->StorageVGPR !=
              0,
          ""));

      // llvm::SlotIndex is used to create intervals that keep track of the
      // locations of the value state/flat scratch registers
      // Marks the beginning of the current interval we are in this loop
      llvm::SlotIndex CurrentIntervalBegin =
          FunctionsSlotIndexes.at(&MF->getFunction())->getZeroIndex();
      // Create a segment for the current interval
      auto &Segments =
          StateValueStorageIntervals.insert({MF, {}}).first->getSecond();
      // A set of hook insertion points that fall into the current interval
      llvm::SmallDenseSet<const llvm::MachineInstr *, 4>
          HookInsertionPointsInCurrentSegment{};
      for (const auto &MBB : *MF) {
        for (const auto &MI : MBB) {
          if (InstPointToInjectedPayloadMap.contains(&MI))
            HookInsertionPointsInCurrentSegment.insert(&MI);
          auto *InstrLiveRegs = RegLiveness.getLiveInPhysRegsOfMachineInstr(MI);
          LUTHIER_RETURN_ON_ERROR(
              LUTHIER_ERROR_CHECK(InstrLiveRegs != nullptr, ""));
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
              InstPointToInjectedPayloadMap.contains(&MI);
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
                  llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(
                      SVS.get())) {
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
              auto [HookSVGPR, ClobbersAppReg] =
                  selectVGPRLoadLocationForInjectedPayload(
                      *HookMI, *SVS, *HookLiveRegs,
                      AccessedPhysicalRegistersNotInLiveIns);
              InstPointSVSLoadPlans.insert(
                  {HookMI, {HookSVGPR, ClobbersAppReg, *SVS}});
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
                AccessedPhysicalRegistersNotInLiveIns, *InstrLiveRegs);

            bool Scavenged = StateValueRegStorage != 0;
            // If we weren't able to scavenge a VGPR, scavenge an AGPR
            if (Scavenged) {
              SVS = std::make_shared<VGPRStateValueArrayStorage>(
                  StateValueRegStorage);
              WasRelocationSuccessful = true;
            } else {
              llvm::SmallVector<llvm::MCRegister, 2> TwoScavengedAGPRs;
              Scavenged =
                  scavengeFreeRegister(MRI, llvm::AMDGPU::AGPR_32RegClass,
                                       AccessedPhysicalRegistersNotInLiveIns,
                                       *InstrLiveRegs, 2, TwoScavengedAGPRs);
              if (Scavenged) {
                SVS = std::make_shared<TwoAGPRValueStorage>(
                    TwoScavengedAGPRs[0], TwoScavengedAGPRs[1]);
                WasRelocationSuccessful = true;
              } else {
                llvm::SmallVector<llvm::MCRegister, 3> ThreeScavengedSGPRs;
                Scavenged = scavengeFreeRegister(
                    MRI, llvm::AMDGPU::SGPR_32RegClass,
                    AccessedPhysicalRegistersNotInLiveIns, *InstrLiveRegs, 3,
                    ThreeScavengedSGPRs);
                if (!Scavenged)
                  WasRelocationSuccessful = false;
                else {
                  if (TwoScavengedAGPRs.size() == 1) {
                    SVS = std::make_shared<AGPRWithThreeSGPRSValueStorage>(
                        TwoScavengedAGPRs[0], ThreeScavengedSGPRs[0],
                        ThreeScavengedSGPRs[1], ThreeScavengedSGPRs[2]);
                  } else {
                    SVS = std::make_shared<SpilledWithThreeSGPRsValueStorage>(
                        ThreeScavengedSGPRs[0], ThreeScavengedSGPRs[1],
                        ThreeScavengedSGPRs[2]);
                  }
                  WasRelocationSuccessful = true;
                }
              }
            }
            LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
                WasRelocationSuccessful || !MustRelocateStateValue, ""));
          }
        }
      }
    }
  }
  return llvm::Error::success();
}

} // namespace luthier