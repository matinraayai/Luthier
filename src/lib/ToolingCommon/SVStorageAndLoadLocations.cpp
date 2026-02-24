//===-- StateValueLocationAndLoadLocations.cpp ----------------------------===//
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
/// This file implements the State Value Location Intervals Pass.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/SVStorageAndLoadLocations.h"
#include "luthier/Common/LuthierError.h"
#include "luthier/Tooling/IModuleIRGeneratorPass.h"
#include "luthier/Tooling/LRCallgraph.h"
#include "luthier/Tooling/MMISlotIndexesAnalysis.h"
#include "luthier/Tooling/SlotIndexes.h"
#include "luthier/Tooling/IPPredicatedCFG.h"
#include "luthier/Tooling/PhysRegsNotInLiveInsAnalysis.h"
#include "luthier/Tooling/StateValueArrayStorage.h"
#include "luthier/Tooling/IPVectorRegLiveness.h"
#include "luthier/Tooling/WrapperAnalysisPasses.h"
#include <GCNSubtarget.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>

#include <utility>
#include <iostream>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lr-state-value-storage-and-load"

namespace luthier {

/// Finds \p NumRegs registers with class \p RC available in \p MRI
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
static void
findAvailableRegister(const llvm::MachineRegisterInfo &MRI,
                     const llvm::TargetRegisterClass &RC,
                     const llvm::LivePhysRegs &AccessedPhysicalRegsNotInLiveIns,
                     const llvm::LiveRegUnits &LiveInRegs, int NumRegs,
                     llvm::SmallVectorImpl<llvm::MCRegister> &ScavengedRegs) {
  int NumRegsFound = 0;
  for (llvm::MCRegister Reg : reverse(RC)) {
    if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
        AccessedPhysicalRegsNotInLiveIns.available(MRI, Reg) &&
        LiveInRegs.available(Reg)) {
      ScavengedRegs.push_back(Reg);
      NumRegsFound++;
      if (NumRegsFound == NumRegs)
        return;
    }
  }
}

/// Searches for \p NumRegs registers with class \p RC available in \p MRI
/// Availability means a register is allocatable and not used in \p MRI and
/// is not in \p AccessedPhysicalRegsNotInLiveIns and not in \p LiveInRegs
/// \param MRI the \c llvm::MachineRegisterInfo of the function being
/// searches
/// \param RC the \c llvm::TargetRegisterClass of the register to be
//// searched
/// \param AccessedPhysicalRegsNotInLiveIns a set of physical registers
/// that are accessed by injected payloads of the instrumentation module but
/// at the point of access are not part of the Live-in registers of the
/// instrumentation points
/// \param LiveInRegs a set of physical registers that are live at the
/// app instruction where the register scavenging is taking place
/// \return the available register if successful, or zero otherwise
static llvm::MCRegister
findAvailableRegister(const llvm::MachineRegisterInfo &MRI,
                     const llvm::TargetRegisterClass &RC,
                     const llvm::LivePhysRegs &AccessedPhysicalRegsNotInLiveIns,
                     const llvm::LiveRegUnits &LiveInRegs) {
  for (llvm::MCRegister Reg : reverse(RC)) {
    if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
        AccessedPhysicalRegsNotInLiveIns.available(MRI, Reg) &&
        LiveInRegs.available(Reg)) {
      return Reg;
    }
  }
  return {};
}

/// Finds \p NumRegs register of class \p RC that are unused across
/// all \p RelatedFunctions and are not in \p AccessedPhysRegsNotInLiveIns
/// \param [in] Functions the functions being searched for a free register
/// \param [in] RC the register class of the registers being searched
/// \param [in] AccessedPhysRegsNotInLiveIns a set of physical registers
/// accessed by the injected payloads that are not in the live-in set of their
/// injected payload at the point of access
/// \param [in] NumRegs number of available registers needed
/// \param [out] Regs the set of registers that were available
static void
findAvailableRegister(llvm::ArrayRef<const PredicatedMachineFunction *> Functions,
                     const llvm::TargetRegisterClass *RC,
                     const llvm::LivePhysRegs &AccessedPhysRegsNotInLiveIns,
                     unsigned int NumRegs,
                     llvm::SmallVectorImpl<llvm::MCRegister> &Regs) {
  unsigned int NumRegFound = 0;

  for (llvm::MCRegister Reg : *RC) {
    bool IsUnused = llvm::all_of(Functions, [&](const PredicatedMachineFunction *PMF) {
      auto &MF = PMF->getMF();
      auto &MRI = MF.getRegInfo();

      LLVM_DEBUG(auto TRI = MF.getSubtarget().getRegisterInfo();
                 llvm::dbgs() << "Trying to find register "
                              << llvm::printReg(Reg, TRI) << "...\n";
                 llvm::dbgs()
                 << "Is reg allocatable? " << MRI.isAllocatable(Reg) << ".\n";
                 llvm::dbgs()
                 << "Is not used? " << !MRI.isPhysRegUsed(Reg) << ".\n";
                 llvm::dbgs()
                 << "Is not in accessed phys regs not in live-ins? "
                 << AccessedPhysRegsNotInLiveIns.available(MRI, Reg) << ".\n";);

      return MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
             AccessedPhysRegsNotInLiveIns.available(MRI, Reg);
    });
    if (IsUnused) {
      Regs.push_back(Reg);
      NumRegFound++;
      if (NumRegFound == NumRegs) {
        LLVM_DEBUG(llvm::dbgs() << "Found " << NumRegFound
                                << " registers; Search was a success!\n";);
        return;
      }
    }
  }
}

llvm::MCRegister
findAvailableRegister(llvm::ArrayRef<const PredicatedMachineFunction *> RelatedFunctions,
                     const llvm::TargetRegisterClass *RC,
                     const llvm::LivePhysRegs &AccessedPhysRegsNotInLiveIns) {
  for (llvm::MCRegister Reg : *RC) {
    bool IsUnused =
        llvm::all_of(RelatedFunctions, [&](const PredicatedMachineFunction *PMF) {
          auto &MF = PMF->getMF();
          auto &MRI = MF.getRegInfo();
          bool IsUnusedInMF = MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg);
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

bool getMILevelLiveIns(const llvm::MachineInstr& TargetMI, const PredicatedMachineBasicBlock& PMBB, const IPVectorRegLiveness& RegLiveness, llvm::LiveRegUnits& LRU){
  LRU.clear();
  RegLiveness.addLiveOuts(PMBB, LRU);
  if(PMBB.getExecMaskValue() == PredicatedMachineBasicBlock::ZeroOrOne){
    for(const llvm::MachineInstr& MI : llvm::reverse(PMBB)){
      LRU.stepBackward(MI);
      if(&MI == &TargetMI) return true;
    }
  }
  else{
    RegLiveness.addLiveIns(PMBB, LRU);
    return true;
  }
  
  return false;
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
/// \param ScavengeDeadAVGPRs if \c true then it will try to scavenge a dead
/// A/VGPR that is not used at the instrumentation point; This flag is only
/// set when the state value array storage is fixed
/// \return a pair, with the first element indicating the VGPR selected, and
/// the second element indicating whether the selected VGPR will clobber a
/// live register of the app and needs preserving
static std::pair<llvm::MCRegister, bool>
selectVGPRLoadLocationForInjectedPayload(
    const llvm::MachineInstr &InstPoint, StateValueArrayStorage &SVS,
    const llvm::LiveRegUnits &InstPointLiveRegs,
    const llvm::LivePhysRegs &AccessedPhysicalRegsNotInLiveIns,
    bool ScavengeDeadAVGPRs) {
  llvm::MCRegister AVGPRLocation{0};
  bool ClobbersAppRegister{false};
  // if the state value array already in a VGPR, then select the same VGPR
  // to be the load destination
  if (!SVS.requiresLoadAndStoreBeforeUse())
    AVGPRLocation = SVS.getStateValueStorageReg();
  else {
    if (!ScavengeDeadAVGPRs) {
      AVGPRLocation = llvm::AMDGPU::VGPR0;
      ClobbersAppRegister = true;
    } else {
      auto &InstrumentedMF = *(InstPoint.getParent()->getParent());
      // Scavenge a dead VGPR to hold the state value array
      AVGPRLocation = findAvailableRegister(
          InstrumentedMF.getRegInfo(), llvm::AMDGPU::VGPR_32RegClass,
          AccessedPhysicalRegsNotInLiveIns, InstPointLiveRegs);
      // Scavenge a dead AGPR to hold the state value array if no VGPR is
      // found
      if (AVGPRLocation == 0)
        AVGPRLocation = findAvailableRegister(
            InstrumentedMF.getRegInfo(), llvm::AMDGPU::AGPR_32RegClass,
            AccessedPhysicalRegsNotInLiveIns, InstPointLiveRegs);
      if (AVGPRLocation == 0) {
        ClobbersAppRegister = true;
        auto &InstrumentedMFRI = InstrumentedMF.getRegInfo();
        for (llvm::MCRegister Reg : llvm::AMDGPU::VGPR_32RegClass) {
          if (InstrumentedMFRI.isPhysRegUsed(Reg) &&
              AccessedPhysicalRegsNotInLiveIns.available(InstrumentedMFRI,
                                                         Reg)) {
            AVGPRLocation = Reg;
            break;
          }
        }
        // If we didn't find anything, just pick V0
        if (AVGPRLocation == 0) {
          AVGPRLocation = llvm::AMDGPU::VGPR0;
        }
      }
    }
  }
  return {AVGPRLocation, ClobbersAppRegister};
}

/// \brief Tries to find a fixed location for storing the state value array
/// \details The order of searching for the storage location is as follows:
/// 1. Find an unused VGPR. This is the ideal scenario, as no further action
/// is required in the prologue/epilogue of an injected payload to load/store
/// the state value array\n
/// 2. If no unused VGPRs are found, then this routine will find the next
/// unused AGPR. This usually comes at no cost to the occupancy, as the app
/// will get the same amount of AGPRs as it gets VGPRs. In gfx90A-, since
/// AGPRs cannot be used directly by vector instructions and have to be moved
/// to a VGPR, a single application VGPR must be spilled. Preference is
/// given to finding another free AGPR to act as a spill slot. If no other
/// free AGPR is found, then three free SGPRs must be found to spill the
/// app's VGPR into an emergency spill slot in the instrumentation stack.\n
/// 3. If no unused V/AGPRs are found in the kernel or a free AGPR is found
/// but allocation of the spill registers is unsuccessful on gfx90A-,
/// then as a last resort, this function tries to find three free SGPRs
/// that can be used to spill an app's VGPR onto the stack, and load the
/// state value array from the stack
/// TODO: This function must take an argument indicating whether the tool
/// writer wants to respect the original kernel's granulated register usage
/// or not.
static std::shared_ptr<StateValueArrayStorage> findFixedStateValueArrayStorage(
    llvm::ArrayRef<const PredicatedMachineFunction *> RelatedFunctions,
    llvm::ArrayRef<StateValueArrayStorage::StorageKind> SupportedStorage,
    int MaxAGPRsUsedByAllStorage, int MaxSGPRsUsedByAllStorage,
    const llvm::LivePhysRegs &AccessedPhysicalRegistersNotInLiveIns) {
  // Find the next VGPR available to hold the value state array
  llvm::MCRegister StateValueArrayFixedVGPRLocation =
      findAvailableRegister(RelatedFunctions, &llvm::AMDGPU::VGPR_32RegClass,
                           AccessedPhysicalRegistersNotInLiveIns);
  // If we failed to find a free VGPR, we then have to scavenge for all
  // possible SGPRs and AGPRs that can be used in storing the state value
  // array
  if (StateValueArrayFixedVGPRLocation == 0) {
    llvm::SmallVector<llvm::MCRegister, 3> SGPRsScavenged;
    llvm::SmallVector<llvm::MCRegister, 2> AGPRsScavenged;
    // Scavenge the maximum number of AGPRs used by all storage schemes
    findAvailableRegister(RelatedFunctions, &llvm::AMDGPU::AGPR_32RegClass,
                         AccessedPhysicalRegistersNotInLiveIns,
                         MaxAGPRsUsedByAllStorage, AGPRsScavenged);
    // Scavenge the maximum number of SGPRs used by all storage schemes
    findAvailableRegister(RelatedFunctions, &llvm::AMDGPU::SGPR_32RegClass,
                         AccessedPhysicalRegistersNotInLiveIns,
                         MaxSGPRsUsedByAllStorage, SGPRsScavenged);

    LLVM_DEBUG(

        llvm::dbgs()
            << "Number of AGPRs available for fixed location SVA storage: "
            << AGPRsScavenged.size() << "\n";
        llvm::dbgs()
        << "Number of SGPRs available for fixed location SVA storage: "
        << SGPRsScavenged.size() << "\n";

    );

    // Loop over all possible supported storage schemes and select the best
    // preferred one which we can use
    for (const auto &StorageScheme : SupportedStorage) {
      if (StorageScheme == StateValueArrayStorage::SVS_SINGLE_VGPR)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "Evaluating fixed " << StorageScheme
                              << " storage scheme.\n";);
      int NumAGPRsUsedByStorage =
          StateValueArrayStorage::getNumAGPRsUsed(StorageScheme);
      int NumSGPRsUsedByStorage =
          StateValueArrayStorage::getNumSGPRsUsed(StorageScheme);
      LLVM_DEBUG(llvm::dbgs() << "Number of ARGPs required by the scheme: "
                              << NumAGPRsUsedByStorage << "\n";
                 llvm::dbgs() << "Number of SGPRs required by the scheme: "
                              << NumSGPRsUsedByStorage << "\n";);
      if (NumSGPRsUsedByStorage <= SGPRsScavenged.size() &&
          NumAGPRsUsedByStorage <= AGPRsScavenged.size()) {
        LLVM_DEBUG(llvm::dbgs() << "Found a suitable fixed storage scheme!\n";);
        auto Out = StateValueArrayStorage::createSVAStorage(
            {}, AGPRsScavenged, SGPRsScavenged, StorageScheme);
        if (Out.takeError()) {
          return nullptr;
        }
        return std::move(*Out);
      }
    }
    // If we made it out of the loop, we weren't able to find a fixed location
    // for the state value array, so we return nullptr
    return nullptr;
  } else
    return std::make_shared<VGPRStateValueArrayStorage>(
        StateValueArrayFixedVGPRLocation);
}

static std::shared_ptr<StateValueArrayStorage> findStateValueArrayStorageAtMI(
    const llvm::MachineRegisterInfo &MRI, const llvm::LiveRegUnits &MILiveIns,
    const llvm::LivePhysRegs &AccessedPhysicalRegistersNotInLiveIns,
    llvm::ArrayRef<StateValueArrayStorage::StorageKind> SupportedStorage,
    int MaxAGPRsUsedByAllStorage, int MaxSGPRsUsedByAllStorage) {
  // Find the next VGPR available to hold the value state array
  llvm::MCRegister StateValueArrayVGPRLocation =
      findAvailableRegister(MRI, llvm::AMDGPU::VGPR_32RegClass,
                           AccessedPhysicalRegistersNotInLiveIns, MILiveIns);
  // If we failed to find a free VGPR, we then have to scavenge for all
  // possible SGPRs and AGPRs that can be used in storing the state value
  // array
  if (StateValueArrayVGPRLocation == 0) {
    llvm::SmallVector<llvm::MCRegister, 3> SGPRsScavenged;
    llvm::SmallVector<llvm::MCRegister, 2> AGPRsScavenged;
    // Scavenge the maximum number of AGPRs used by all storage schemes
    findAvailableRegister(MRI, llvm::AMDGPU::AGPR_32RegClass,
                         AccessedPhysicalRegistersNotInLiveIns, MILiveIns,
                         MaxAGPRsUsedByAllStorage, AGPRsScavenged);

    // Scavenge the maximum number of SGPRs used by all storage schemes
    findAvailableRegister(MRI, llvm::AMDGPU::SGPR_32RegClass,
                         AccessedPhysicalRegistersNotInLiveIns, MILiveIns,
                         MaxSGPRsUsedByAllStorage, SGPRsScavenged);

    LLVM_DEBUG(

        llvm::dbgs() << "Number of AGPRs scavenged for location SVA storage: "
                     << AGPRsScavenged.size() << "\n";
        llvm::dbgs() << "Number of SGPRs scavenged for location SVA storage: "
                     << SGPRsScavenged.size() << "\n";

    );

    // Loop over all possible supported storage schemes and select the best
    // preferred one which we can use
    for (const auto &StorageScheme : SupportedStorage) {
      if (StorageScheme == StateValueArrayStorage::SVS_SINGLE_VGPR)
        continue;
      int NumAGPRsUsedByStorage =
          StateValueArrayStorage::getNumAGPRsUsed(StorageScheme);
      int NumSGPRsUsedByStorage =
          StateValueArrayStorage::getNumSGPRsUsed(StorageScheme);
      if (NumSGPRsUsedByStorage <= SGPRsScavenged.size() &&
          NumAGPRsUsedByStorage <= AGPRsScavenged.size()) {
        auto Out = StateValueArrayStorage::createSVAStorage(
            {}, AGPRsScavenged, SGPRsScavenged, StorageScheme);
        if (Out.takeError()) {
          return nullptr;
        }
        return std::move(*Out);
      }
    }
    // If we made it out of the loop, we weren't able to find a location
    // for the state value array, so we return nullptr
    return nullptr;
  } else
    return std::make_shared<VGPRStateValueArrayStorage>(
        StateValueArrayVGPRLocation);
}

llvm::ArrayRef<StateValueStorageSegment>
SVStorageAndLoadLocations::getStorageIntervals(
    const PredicatedMachineBasicBlock &MBB) const {
  auto It = StateValueStorageIntervals.find(&MBB);
  if (It == StateValueStorageIntervals.end())
    return {};
  else
    return It->second;
}

const InstPointSVALoadPlan *
SVStorageAndLoadLocations::getStateValueArrayLoadPlanForInstPoint(
    const llvm::MachineInstr &MI) const {
  auto It = InstPointSVSLoadPlans.find(&MI);
  if (It == InstPointSVSLoadPlans.end())
    return nullptr;
  else
    return &It->second;
}
// FIXME: AccessedPhysicalRegistersNotInLiveIns is not valid anymore and should change, as should IPIP
llvm::Error SVStorageAndLoadLocations::calculate(
    const llvm::MachineModuleInfo &TargetMMI,  llvm::Module &TargetM,
    const MMISlotIndexesAnalysis::Result &SlotIndexes,
    const IPVectorRegLiveness &RegLiveness,
    const InjectedPayloadAndInstPoint &IPIP, 
    llvm::LivePhysRegs &AccessedPhysicalRegistersNotInLiveIns,
    const IPPredicatedCFG &IPCFG,
    llvm::FunctionAnalysisManager& FAM){
  llvm::SmallVector<const PredicatedMachineFunction*, 4> MFs;
  // We need FAM to get MF, MAM doesn't work
  for (auto &F : TargetM) {
    llvm::MachineFunction &MF =
          FAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();
    MFs.push_back(&IPCFG.at(MF));
    
  }

  // Early exit if no MF is present in the LCO of LR
  if (MFs.empty())
    return llvm::Error::success();
  // Get all the possible state value array storage for the sub-target being
  // used and check if we have at least only one method for storage
  const auto &ST = MFs[0]->getMF().getSubtarget<llvm::GCNSubtarget>();
  const auto TRI = ST.getRegisterInfo();
  AccessedPhysicalRegistersNotInLiveIns.init(*TRI);
  llvm::SmallVector<StateValueArrayStorage::StorageKind, 6> SupportedStorage;
  getSupportedSVAStorageList(ST, SupportedStorage);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      !SupportedStorage.empty(),
      llvm::formatv("Failed to find compatible state value array storage "
                    "for ST {0}, CPU {1}.",
                    ST.getTargetTriple().str(), ST.getCPU())));
  // Query the maximum number of SGPRs and AGPRs in all storage methods;
  // This saves us time during register scavenging
  int MaxNumAGPRsUsedByAllStorage = 0;
  int MaxNumSGPRsUsedByAllStorage = 0;
  for (const auto &StorageScheme : SupportedStorage) {
    int MaxNumAGPRsUsedByStorage =
        StateValueArrayStorage::getNumAGPRsUsed(StorageScheme);
    if (MaxNumAGPRsUsedByStorage > MaxNumAGPRsUsedByAllStorage)
      MaxNumAGPRsUsedByAllStorage = MaxNumAGPRsUsedByStorage;
    int MaxNumSGPRsUsedByStorage =
        StateValueArrayStorage::getNumSGPRsUsed(StorageScheme);
    if (MaxNumSGPRsUsedByStorage > MaxNumSGPRsUsedByAllStorage)
      MaxNumSGPRsUsedByAllStorage = MaxNumSGPRsUsedByStorage;
  }

  // Try to find a fixed location to store the state value array
  auto StateValueFixedLocation = findFixedStateValueArrayStorage(
      MFs, SupportedStorage, MaxNumAGPRsUsedByAllStorage,
      MaxNumSGPRsUsedByAllStorage, AccessedPhysicalRegistersNotInLiveIns);
  // TODO: Add this information to metadata
  if (StateValueFixedLocation != nullptr) {
    // If a fixed location was found, then all MBB intervals inside all MFs
    // will get the fixed state value location
    // Also in a fixed storage case, there is no need to emit any kind of
    // preamble code to any device functions involved inside the lifted
    // representation
    for (const auto *MF : MFs) {
      for (const auto &LBB : *MF) {
        for (const auto &PBB : LBB){
          auto &Segments =
            StateValueStorageIntervals
                .insert({&PBB, llvm::SmallVector<StateValueStorageSegment>{}})
                .first->getSecond();
        const auto& SI = SlotIndexes.at(*MF);
        Segments.emplace_back(SI.getMBBStartIdx(&PBB),
                              SI.getMBBEndIdx(&PBB),
                              StateValueFixedLocation);
        }
      }
    }
    for (const auto &[InsertionPointMI, HookFunction] : IPIP.mi_payload()) {
      llvm::LiveRegUnits HookLiveRegs(*TRI);
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          getMILevelLiveIns(*InsertionPointMI, IPCFG.getPredMBB(*InsertionPointMI), RegLiveness, HookLiveRegs),
          llvm::formatv(
              "Failed to get the Live Physical register set for MI {0}.",
              *InsertionPointMI)));
      auto [VGPRLocation, ClobbersAppReg] =
          selectVGPRLoadLocationForInjectedPayload(
              *InsertionPointMI, *StateValueFixedLocation, HookLiveRegs,
              AccessedPhysicalRegistersNotInLiveIns, true);

      InstPointSVSLoadPlans.insert(
          {InsertionPointMI, InstPointSVALoadPlan{VGPRLocation, ClobbersAppReg,
                                                  *StateValueFixedLocation}});
    }
  } else {
    // If not, we'll have to shuffle between possible state value array
    // storage schemes
    for (const auto *MF : MFs) {
      auto &MRI = MF->getMF().getRegInfo();
      // Pick the highest numbered VGPR not accessed by the Hooks
      // to hold the value state
      // TODO: is there a more informed way to do initialize this?
      // TODO: if an argument is passed specifying to keep the register
      // usage of the kernel the same as before, these needs to be initialized
      // to the last available SGPR/VGPR/AGPR
      const auto& FirstPMBB = MF->front().front();
      // May change to LPR
      llvm::LiveRegUnits FirstMILiveIns{*TRI};
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          getMILevelLiveIns(*FirstPMBB.getFirstNonDebugInstr(), FirstPMBB, RegLiveness, FirstMILiveIns),
          llvm::formatv( "Failed to obtain the live physical regs for MI {0}.",
                        *FirstPMBB.getFirstNonDebugInstr())));

      // The current location of the state value register
      std::shared_ptr<StateValueArrayStorage> SVS =
          findStateValueArrayStorageAtMI(
              MRI, FirstMILiveIns, AccessedPhysicalRegistersNotInLiveIns,
              SupportedStorage, MaxNumAGPRsUsedByAllStorage,
              MaxNumSGPRsUsedByAllStorage);

      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          SVS != nullptr,
          llvm::formatv("Failed to get a state value array storage for MI {0}.",
                        *FirstPMBB.getFirstNonDebugInstr())));

      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          llvm::isa<VGPRStateValueArrayStorage>(SVS.get()) ||
              llvm::isa<SingleAGPRStateValueArrayStorage>(SVS.get()),
          "The entry SVS must be stored in a VGPR or an AGPR."));

      // A set of hook insertion points that fall into the current interval
      llvm::SmallDenseSet<const llvm::MachineInstr *, 4>
          HookInsertionPointsInCurrentSegment{};
      for (const auto &LBB : *MF) {
        for(const auto &PBB : LBB){
          // Marks the beginning of the current interval we are in this loop
          SlotIndex CurrentIntervalBegin =
              SlotIndexes.at(*MF).getMBBStartIdx(&PBB);

          auto &CurrentMBBSegments =
              StateValueStorageIntervals.insert({&PBB, {}}).first->getSecond();
          for (const auto &MI : PBB) {
            if (IPIP.contains(MI))
              HookInsertionPointsInCurrentSegment.insert(&MI);
            llvm::LiveRegUnits InstrLiveRegs{*TRI};
            
            LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
                getMILevelLiveIns(MI, PBB, RegLiveness, InstrLiveRegs),
                llvm::formatv(
                    "Failed to get the live physical register set for MI {0}.",
                    MI)));
            // - If we have spilled the state value reg and this instruction
            // will require a hook to be inserted, then we try to relocate the
            // SVS. In this instance, since the hook will have to load the value
            // state register anyway, we try and see if after loading it, we can
            // store it in a V/AGPR.
            // - If the SVS registers are going to be used, we must relocate
            // the SVS.
            // - Otherwise, we keep the SVS in its place.
            bool TryRelocatingValueStateReg =
                SVS->getStateValueStorageReg() == 0 && IPIP.contains(MI);
            llvm::SmallVector<llvm::MCRegister, 4> SVSRegs;
            SVS->getAllStorageRegisters(SVSRegs);
            bool MustRelocateStateValue =
                llvm::any_of(SVSRegs, [&](llvm::MCRegister Reg) {
                  return !InstrLiveRegs.available(Reg);
                });
            // If we have to relocate something, then create a new interval
            // for it;
            // Note that reg scavenging might conclude that the values remain
            // where they are, and that's okay
            // Also create a new interval if we reach the end of a MBB
            if (&MI == &PBB.back() || TryRelocatingValueStateReg ||
                MustRelocateStateValue) {
              auto NextIndex = &MI == &PBB.back()
                                  ? SlotIndexes.at(*MF).getMBBEndIdx(&PBB)
                                  : SlotIndexes.at(*MF).getInstructionIndex(MI);
              CurrentMBBSegments.emplace_back(CurrentIntervalBegin, NextIndex,
                                              SVS);
              for (const auto &HookMI : HookInsertionPointsInCurrentSegment) {
                llvm::LiveRegUnits HookLiveRegs{*TRI};
                LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
                getMILevelLiveIns(*HookMI, PBB, RegLiveness, HookLiveRegs),
                llvm::formatv(
                    "Failed to get the live physical register set for MI {0}.",
                    *HookMI)));
                auto [HookSVGPR, ClobbersAppReg] =
                    selectVGPRLoadLocationForInjectedPayload(
                        *HookMI, *SVS, HookLiveRegs,
                        AccessedPhysicalRegistersNotInLiveIns, false);
                InstPointSVSLoadPlans.insert(
                    {HookMI, {HookSVGPR, ClobbersAppReg, *SVS}});
              }
              HookInsertionPointsInCurrentSegment.clear();
              CurrentIntervalBegin = NextIndex;
            }
            if (TryRelocatingValueStateReg || MustRelocateStateValue) {
              SVS = findStateValueArrayStorageAtMI(
                  MRI, FirstMILiveIns, AccessedPhysicalRegistersNotInLiveIns,
                  SupportedStorage, MaxNumAGPRsUsedByAllStorage,
                  MaxNumSGPRsUsedByAllStorage);
              LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
                  SVS != nullptr, "Failed to relocate the SVA storage."));
            }
          }
        }
      }
    }
  }
  return llvm::Error::success();
}

llvm::AnalysisKey LRStateValueStorageAndLoadLocationsAnalysis::Key;

LRStateValueStorageAndLoadLocationsAnalysis::Result
LRStateValueStorageAndLoadLocationsAnalysis::run(
    llvm::Module &TargetModule, llvm::ModuleAnalysisManager &TargetMAM) {
  SVStorageAndLoadLocations Out;
  // auto &IModuleAndPMRes = TargetMAM.getResult<IModulePMAnalysis>(TargetModule);
  // auto &IModule = IModuleAndPMRes.getModule();
  // auto &IMAM = IModuleAndPMRes.getMAM();
  llvm::LivePhysRegs LPR{};
  auto Err = Out.calculate(
      TargetMAM.getCachedResult<llvm::MachineModuleAnalysis>(TargetModule)
          ->getMMI(),
      TargetModule, TargetMAM.getResult<MMISlotIndexesAnalysis>(TargetModule),
      *TargetMAM.getCachedResult<IPVectorRegLivenessAnalysis>(TargetModule),
      /**IMAM.getCachedResult<InjectedPayloadAndInstPointAnalysis>(IModule)*/ {},
      // IMAM.getResult<PhysRegsNotInLiveInsAnalysis>(IModule)
      //     .getPhysRegsNotInLiveIns(),
      LPR,
      TargetMAM.getResult<IPPredCFGAnalysis>(TargetModule).getVecCFG(),
      TargetMAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(TargetModule).getManager());
  if (Err)
    TargetModule.getContext().emitError(llvm::toString(std::move(Err)));

  return Out;
}

llvm::PreservedAnalyses LRStateValueStorageAndLoadLocationsPrinterPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM){
  const auto &MMI = MAM.getResult<llvm::MachineModuleAnalysis>(M).getMMI();
  const auto &SVS = MAM.getResult<LRStateValueStorageAndLoadLocationsAnalysis>(M);
  const auto &IPCFG = MAM.getResult<IPPredCFGAnalysis>(M).getVecCFG();
  const auto &ModuleSI = MAM.getResult<MMISlotIndexesAnalysis>(M);
  int Indent = 2;
  for(const auto& F : M){
    const auto& MF = IPCFG.at(*(MMI.getMachineFunction(F)));
    const auto& MFSI = ModuleSI.at(MF);
    const auto &ST = MF.getMF().getSubtarget();
    const auto *TII = ST.getInstrInfo();
    const auto *TRI = ST.getRegisterInfo();
    for(const auto& LMBB : MF){
      for(const auto& PMBB : LMBB){
        auto Segments = SVS.getStorageIntervals(PMBB);
        const auto* NextSegment = Segments.begin();
        OS << "Load Plan for segment [";
        NextSegment->begin().print(OS);
        OS << ", ";
        NextSegment->end().print(OS);
        OS << ") -> ";
        NextSegment->getSVS().print(OS);
        OS << "\n";
        ++NextSegment;
        if(!PMBB.empty()){
          for(const auto& MI : PMBB){
            SlotIndex MIIdx = MFSI.getInstructionIndex(MI);
            if (NextSegment && NextSegment != Segments.end() && NextSegment->contains(MIIdx)) {
              OS << "Load Plan for segment [";
              NextSegment->begin().print(OS);
              OS << ", ";
              NextSegment->end().print(OS);
              OS << ") -> ";
              NextSegment->getSVS().print(OS);
              OS << "\n";
              ++NextSegment;
            }
            MI.print(OS.indent(Indent + 2), true, false, false, true, TII);
          }
        }
      }
    }
  }
  return llvm::PreservedAnalyses::all();
}
} // namespace luthier