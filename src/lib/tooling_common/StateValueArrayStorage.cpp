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
#include "tooling_common/MIRConvenience.hpp"
#include "tooling_common/StateValueArraySpecs.hpp"
#include <llvm/CodeGen/MachineInstrBuilder.h>

namespace luthier {

static const llvm::DenseMap<StateValueArrayStorage::StorageKind, int>
    NumVGPRsUsedBySVS{
        {StateValueArrayStorage::SVS_SINGLE_VGPR, 1},
        {StateValueArrayStorage::SVS_ONE_AGPR_post_gfx908, 0},
        {StateValueArrayStorage::SVS_TWO_AGPRs_pre_gfx908, 0},
        {StateValueArrayStorage::SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908,
         0},
        {StateValueArrayStorage::SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs, 0},
        {StateValueArrayStorage::SVS_SPILLED_WITH_ONE_SGPR_architected_fs, 0}};

static const llvm::DenseMap<StateValueArrayStorage::StorageKind, int>
    NumAGPRsUsedBySVS{
        {StateValueArrayStorage::SVS_SINGLE_VGPR, 0},
        {StateValueArrayStorage::SVS_ONE_AGPR_post_gfx908, 1},
        {StateValueArrayStorage::SVS_TWO_AGPRs_pre_gfx908, 2},
        {StateValueArrayStorage::SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908,
         1},
        {StateValueArrayStorage::SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs, 0},
        {StateValueArrayStorage::SVS_SPILLED_WITH_ONE_SGPR_architected_fs, 0}};

static const llvm::DenseMap<StateValueArrayStorage::StorageKind, int>
    NumSGPRsUsedBySVS{
        {StateValueArrayStorage::SVS_SINGLE_VGPR, 0},
        {StateValueArrayStorage::SVS_ONE_AGPR_post_gfx908, 0},
        {StateValueArrayStorage::SVS_TWO_AGPRs_pre_gfx908, 0},
        {StateValueArrayStorage::SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908,
         3},
        {StateValueArrayStorage::SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs, 3},
        {StateValueArrayStorage::SVS_SPILLED_WITH_ONE_SGPR_architected_fs, 1}};

int StateValueArrayStorage::getNumVGPRsUsed(
    StateValueArrayStorage::StorageKind Kind) {
  return NumVGPRsUsedBySVS.at(Kind);
}

int StateValueArrayStorage::getNumAGPRsUsed(
    StateValueArrayStorage::StorageKind Kind) {
  return NumAGPRsUsedBySVS.at(Kind);
}

int StateValueArrayStorage::getNumSGPRsUsed(
    StateValueArrayStorage::StorageKind Kind) {
  return NumSGPRsUsedBySVS.at(Kind);
}

static const llvm::DenseMap<StateValueArrayStorage::StorageKind,
                            std::function<bool(const llvm::GCNSubtarget &)>>
    StorageSTCompatibility{
        {StateValueArrayStorage::SVS_SINGLE_VGPR,
         [](const llvm::GCNSubtarget &) { return true; }},
        {StateValueArrayStorage::SVS_ONE_AGPR_post_gfx908,
         [](const llvm::GCNSubtarget &ST) { return ST.hasGFX90AInsts(); }},
        {StateValueArrayStorage::SVS_TWO_AGPRs_pre_gfx908,
         [](const llvm::GCNSubtarget &ST) { return !ST.hasGFX90AInsts(); }},
        {StateValueArrayStorage::SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908,
         [](const llvm::GCNSubtarget &ST) { return !ST.hasGFX90AInsts(); }},
        {StateValueArrayStorage::SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs,
         [](const llvm::GCNSubtarget &ST) {
           return !ST.flatScratchIsArchitected();
         }},
        {StateValueArrayStorage::SVS_SPILLED_WITH_ONE_SGPR_architected_fs,
         [](const llvm::GCNSubtarget &ST) {
           return ST.flatScratchIsArchitected();
         }}};

bool StateValueArrayStorage::isSupportedOnSubTarget(
    StateValueArrayStorage::StorageKind Kind, const llvm::GCNSubtarget &ST) {
  return StorageSTCompatibility.at(Kind)(ST);
}

llvm::Expected<std::unique_ptr<StateValueArrayStorage>>
StateValueArrayStorage::createSVAStorage(
    llvm::ArrayRef<llvm::MCRegister> VGPRs,
    llvm::ArrayRef<llvm::MCRegister> AGPRs,
    llvm::ArrayRef<llvm::MCRegister> SGPRs,
    StateValueArrayStorage::StorageKind Scheme) {
  switch (Scheme) {
  case SVS_SINGLE_VGPR:
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        VGPRs.size() >= 1,
        "Insufficient number of VGPRs for single VGPR SVA storage."));
    return std::make_unique<VGPRStateValueArrayStorage>(VGPRs[0]);
  case SVS_ONE_AGPR_post_gfx908:
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        AGPRs.size() >= 1,
        "Insufficient number of AGPRs for single AGPR SVA storage."));
    return std::make_unique<SingleAGPRStateValueArrayStorage>(AGPRs[0]);
  case SVS_TWO_AGPRs_pre_gfx908:
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        AGPRs.size() >= 2,
        "Insufficient number of AGPRs for two AGPR SVA storage."));
    return std::make_unique<TwoAGPRValueStorage>(AGPRs[0], AGPRs[1]);
  case SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908:
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        AGPRs.size() >= 1, "Insufficient number of AGPRs for single AGPR with "
                           "three SGPR SVA storage."));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        SGPRs.size() >= 3, "Insufficient number of AGPRs for single AGPR with "
                           "three SGPR SVA storage."));
    return std::make_unique<AGPRWithThreeSGPRSValueStorage>(AGPRs[0], SGPRs[0],
                                                            SGPRs[1], SGPRs[2]);
  case SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs:
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        SGPRs.size() >= 3, "Insufficient number of AGPRs for spilled with "
                           "three SGPR SVA storage."));
    return std::make_unique<SpilledWithThreeSGPRsValueStorage>(
        SGPRs[0], SGPRs[1], SGPRs[2]);
  case SVS_SPILLED_WITH_ONE_SGPR_architected_fs:
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        SGPRs.size() >= 1, "Insufficient number of SGPRs for spilled with "
                           "single SGPR SVA storage."));
    return std::make_unique<SpilledWithOneSGPRsValueStorage>(SGPRs[0]);
  }
  llvm_unreachable("Invalid SVA storage Enum value.");
}

//===----------------------------------------------------------------------===//
// VGPRStateValueArrayStorage Switch logic
//===----------------------------------------------------------------------===//

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                const VGPRStateValueArrayStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS) {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Do a move on the active lanes
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcSVS.StorageVGPR,
                               TargetSVS.StorageVGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Do a move on the inactive lanes
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcSVS.StorageVGPR,
                               TargetSVS.StorageVGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const VGPRStateValueArrayStorage &SrcSVS,
                    const SingleAGPRStateValueArrayStorage &TargetSVS) {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Do a move on the active lanes
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcSVS.StorageVGPR,
                               TargetSVS.StorageAGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Do a move on the inactive lanes
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcSVS.StorageVGPR,
                               TargetSVS.StorageAGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                const VGPRStateValueArrayStorage &SrcSVS,
                                const TwoAGPRValueStorage &TargetSVS) {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Do a move on the active lanes
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcSVS.StorageVGPR,
                               TargetSVS.StorageAGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Do a move on the inactive lanes
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcSVS.StorageVGPR,
                               TargetSVS.StorageAGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const VGPRStateValueArrayStorage &SrcSVS,
                    const AGPRWithThreeSGPRSValueStorage &TargetSVS) {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Do a move on the active lanes
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcSVS.StorageVGPR,
                               TargetSVS.StorageAGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Do a move on the inactive lanes
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcSVS.StorageVGPR,
                               TargetSVS.StorageAGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const VGPRStateValueArrayStorage &SrcSVS,
                    const SpilledWithThreeSGPRsValueStorage &TargetSVS) {
  // Read FS_lo, FS_hi and SGPR32 into their storage SGPRs
  for (const auto &[PhysReg, SVSSaveSGPR] :
       {std::pair{llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh},
        {llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow},
        {llvm::AMDGPU::SGPR32, TargetSVS.EmergencyVGPRSpillSlotOffset}}) {
    auto StoreSlot =
        stateValueArray::getInstrumentationStackFrameLaneIdStoreSlot(PhysReg);
    emitMoveFromVGPRLaneToSGPR(MI, SrcSVS.StorageVGPR, SVSSaveSGPR, StoreSlot,
                               false);
  }

  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Temporarily swap the FS Hi and FS Lo of the app with the storage to
        // spill the SVA
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     TargetSVS.FlatScratchSGPRHigh);
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     TargetSVS.FlatScratchSGPRLow);
        // Spill the SVA on the active lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            SrcSVS.StorageVGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill the SVA on the inactive lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            SrcSVS.StorageVGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // swap the FS Hi and FS Lo of the app back to its correct place
        emitSGPRSwap(InsertionPointMBB.end(), TargetSVS.FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);
        emitSGPRSwap(InsertionPointMBB.end(), TargetSVS.FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const VGPRStateValueArrayStorage &SrcSVS,
                    const SpilledWithOneSGPRsValueStorage &TargetSVS) {
  // Store the instrumentation stack pointer
  auto StoreSlot = stateValueArray::getInstrumentationStackFrameLaneIdStoreSlot(
      llvm::AMDGPU::SGPR32);
  emitMoveFromVGPRLaneToSGPR(MI, SrcSVS.StorageVGPR,
                             TargetSVS.EmergencyVGPRSpillSlotOffset, StoreSlot,
                             false);
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Spill the SVA on the active lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            SrcSVS.StorageVGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill the SVA on the inactive lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            SrcSVS.StorageVGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}

//===----------------------------------------------------------------------===//
// SingleAGPRStateValueArrayStorage Switch logic
//===----------------------------------------------------------------------===//

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                const SingleAGPRStateValueArrayStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Do a move on the active lanes
    emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                           TargetSVS.StorageVGPR, false);
    // Flip the exec mask
    emitExecMaskFlip(InsertionPointMBB.end());
    // Do a move on the inactive lanes
    emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                           TargetSVS.StorageVGPR, true);
    // Flip the exec mask back
    emitExecMaskFlip(InsertionPointMBB.end());
  });
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const SingleAGPRStateValueArrayStorage &SrcSVS,
                    const SingleAGPRStateValueArrayStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Do a move on the active lanes
    emitMoveFromVGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                           TargetSVS.StorageAGPR, false);
    // Flip the exec mask
    emitExecMaskFlip(InsertionPointMBB.end());
    // Do a move on the inactive lanes
    emitMoveFromVGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                           TargetSVS.StorageAGPR, true);
    // Flip the exec mask back
    emitExecMaskFlip(InsertionPointMBB.end());
  });
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const SingleAGPRStateValueArrayStorage &SrcSVS,
                    const SpilledWithThreeSGPRsValueStorage &TargetSVS) {
  const auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
  for (const auto &[PhysReg, SVSSaveSGPR] :
       {std::pair{llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh},
        {llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow},
        {llvm::AMDGPU::SGPR32, TargetSVS.EmergencyVGPRSpillSlotOffset}}) {
    auto StoreSlot =
        stateValueArray::getInstrumentationStackFrameLaneIdStoreSlot(PhysReg);
    emitMoveFromVGPRLaneToSGPR(MI, SrcSVS.StorageAGPR, SVSSaveSGPR, StoreSlot,
                               false);
  }

  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Temporarily swap the FS Hi and FS Lo of the app with the storage to
        // spill the SVA
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     TargetSVS.FlatScratchSGPRHigh);

        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     TargetSVS.FlatScratchSGPRLow);

        // Spill the SVA on the active lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            SrcSVS.StorageAGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill the SVA on the inactive register
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            SrcSVS.StorageAGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // swap the FS Hi and FS Lo of the app back to its correct place
        emitSGPRSwap(InsertionPointMBB.end(), TargetSVS.FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);
        emitSGPRSwap(InsertionPointMBB.end(), TargetSVS.FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
      });
  // Wait on all memory operations
  emitWaitCnt(NextIPoint);
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const SingleAGPRStateValueArrayStorage &SrcSVS,
                    const SpilledWithOneSGPRsValueStorage &TargetSVS) {
  // Move the instrumentation stack pointer to its destination
  auto StoreSlot = stateValueArray::getInstrumentationStackFrameLaneIdStoreSlot(
      llvm::AMDGPU::SGPR32);
  emitMoveFromVGPRLaneToSGPR(MI, SrcSVS.StorageAGPR,
                             TargetSVS.EmergencyVGPRSpillSlotOffset, StoreSlot,
                             false);

  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Spill the SVA on the active register
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            SrcSVS.StorageAGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill the SVA on the inactive register
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            SrcSVS.StorageAGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}

//===----------------------------------------------------------------------===//
// TwoAGPRValueStorage Switch logic
//===----------------------------------------------------------------------===//

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                const TwoAGPRValueStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS) {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Do a move on the active lanes
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               TargetSVS.StorageVGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Do a move on the inactive lanes
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               TargetSVS.StorageVGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                const TwoAGPRValueStorage &SrcSVS,
                                const TwoAGPRValueStorage &TargetSVS) {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Spill V0 on the active lanes to the SrcSVS AGPR storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               SrcSVS.TempAGPR, true);
        // Move the SVS from its storage to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, true);
        // Move the SVS from V0 to its final storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());

        // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               SrcSVS.TempAGPR, true);
        // Move the SVS from its storage to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, true);
        // Move the SVS from V0 to its final storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
      });
};

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI, const TwoAGPRValueStorage &SrcSVS,
                    const AGPRWithThreeSGPRSValueStorage &TargetSVS) {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Spill V0 on the active lanes to the SrcSVS AGPR storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               SrcSVS.TempAGPR, true);
        // Move the SVS from its storage to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, true);
        // Move the SVS from V0 to its final storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());

        // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               SrcSVS.TempAGPR, true);
        // Move the SVS from its storage to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, true);
        // Move the SVS from V0 to its final storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
      });
};

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI, const TwoAGPRValueStorage &SrcSVS,
                    const SpilledWithThreeSGPRsValueStorage &TargetSVS) {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Spill V0 on the active lanes to the SrcSVS temp AGPR
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               SrcSVS.TempAGPR, true);
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());

        // Spill V0 on the active lanes to the SrcSVS temp AGPR
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               SrcSVS.TempAGPR, true);
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, true);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
      });

  const auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
  for (const auto &[PhysReg, SVSSaveSGPR] :
       {std::pair{llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh},
        {llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow},
        {llvm::AMDGPU::SGPR32, TargetSVS.EmergencyVGPRSpillSlotOffset}}) {
    auto StoreSlot =
        stateValueArray::getInstrumentationStackFrameLaneIdStoreSlot(PhysReg);
    emitMoveFromVGPRLaneToSGPR(NextIPoint, llvm::AMDGPU::VGPR0, SVSSaveSGPR,
                               StoreSlot, false);
  }

  NextIPoint = createSCCSafeSequenceOfMIs(
      NextIPoint, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                      const llvm::TargetInstrInfo &TII) {
        // Temporarily swap the FS Hi and FS Lo of the app with the storage to
        // spill the SVA
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     TargetSVS.FlatScratchSGPRHigh);
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     TargetSVS.FlatScratchSGPRLow);
        // Spill the SVA on the active lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill the SVA on the inactive register
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // swap the FS Hi and FS Lo of the app back to its correct place
        emitSGPRSwap(InsertionPointMBB.end(), TargetSVS.FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);

        emitSGPRSwap(InsertionPointMBB.end(), TargetSVS.FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);

  // Swap the V0 with Temp AGPR
  (void)createSCCSafeSequenceOfMIs(
      NextIPoint, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                      const llvm::TargetInstrInfo &TII) {
        // Read V0 back from the SrcSVS AGPR storage
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.TempAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Read V0 back from the SrcSVS AGPR temp
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.TempAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

//===----------------------------------------------------------------------===//
// AGPRWithThreeSGPRSValueStorage Switch logic
//===----------------------------------------------------------------------===//

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                const AGPRWithThreeSGPRSValueStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS) {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Do a move on the active lanes
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               TargetSVS.StorageVGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Do a move on the inactive lanes
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               TargetSVS.StorageVGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                const AGPRWithThreeSGPRSValueStorage &SrcSVS,
                                const TwoAGPRValueStorage &TargetSVS) {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Spill V0 on the active lanes to the TargetSVS AGPR temp
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.TempAGPR, false);
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Write V0 to TargetSVS storage AGPR
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Read V0 back from the TargetSVS AGPR temp
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), TargetSVS.TempAGPR,
                               llvm::AMDGPU::VGPR0, true);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());

        // Spill V0 on the active lanes to the TargetSVS AGPR temp
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.TempAGPR, false);
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Write V0 to TargetSVS storage AGPR
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Read V0 back from the TargetSVS AGPR temp
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), TargetSVS.TempAGPR,
                               llvm::AMDGPU::VGPR0, true);
        // Flip the exec mask again
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const AGPRWithThreeSGPRSValueStorage &SrcSVS,
                    const AGPRWithThreeSGPRSValueStorage &TargetSVS) {
  // Move the SGPRs first
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRHigh,
                         TargetSVS.FlatScratchSGPRHigh, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRLow,
                         TargetSVS.FlatScratchSGPRLow, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.EmergencyVGPRSpillSlotOffset,
                         TargetSVS.EmergencyVGPRSpillSlotOffset, true);

  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Temporarily swap the FS Hi and FS Lo of the app with the storage to
        // spill the SVA
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     TargetSVS.FlatScratchSGPRHigh);
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     TargetSVS.FlatScratchSGPRLow);
        // Spill V0 on the active lanes to the emergency spill slot
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0, true);
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Read the SVS from V0 into its target storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Restore V0's original value
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0, true);
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Read the SVS from V0 into its target storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Restore V0's original value
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // swap the FS Hi and FS Lo of the app back
        emitSGPRSwap(InsertionPointMBB.end(), TargetSVS.FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);
        emitSGPRSwap(InsertionPointMBB.end(), TargetSVS.FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
      });
  emitWaitCnt(NextIPoint);
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const AGPRWithThreeSGPRSValueStorage &SrcSVS,
                    const SpilledWithThreeSGPRsValueStorage &TargetSVS) {
  // Move the SGPRs first
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRHigh,
                         TargetSVS.FlatScratchSGPRHigh, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRLow,
                         TargetSVS.FlatScratchSGPRLow, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.EmergencyVGPRSpillSlotOffset,
                         TargetSVS.EmergencyVGPRSpillSlotOffset, true);

  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Temporarily swap the FS Hi and FS Lo of the app with the storage to
        // spill the SVA
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     TargetSVS.FlatScratchSGPRHigh);

        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     TargetSVS.FlatScratchSGPRLow);
        // Spill V0 on the active lanes to the emergency spill slot
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Spill the SVS to the stack
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Restore V0's original value
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Spill the SVS to the stack
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Restore V0's original value
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // swap the FS Hi and FS Lo of the app back
        emitSGPRSwap(InsertionPointMBB.end(), TargetSVS.FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);

        emitSGPRSwap(InsertionPointMBB.end(), TargetSVS.FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
      });
  emitWaitCnt(NextIPoint);
}

//===----------------------------------------------------------------------===//
// SpilledWithThreeSGPRsValueStorage Switch logic
//===----------------------------------------------------------------------===//

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                const SpilledWithThreeSGPRsValueStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS) {

  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Temporarily swap the FS Hi and FS Lo of the app with the storage to
        // spill the SVA
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     SrcSVS.FlatScratchSGPRHigh);

        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     SrcSVS.FlatScratchSGPRLow);

        // Load the SVS from the stack on the active lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            TargetSVS.StorageVGPR);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());

        // Load the SVS from the stack on the inactive lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            TargetSVS.StorageVGPR);

        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());

        // swap the FS Hi and FS Lo of the app back
        emitSGPRSwap(InsertionPointMBB.end(), SrcSVS.FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);

        emitSGPRSwap(InsertionPointMBB.end(), SrcSVS.FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
      });
  emitWaitCnt(NextIPoint);
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const SpilledWithThreeSGPRsValueStorage &SrcSVS,
                    const SingleAGPRStateValueArrayStorage &TargetSVS) {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Temporarily swap the FS Hi and FS Lo of the app with the storage to
        // spill the SVA
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     SrcSVS.FlatScratchSGPRHigh);

        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     SrcSVS.FlatScratchSGPRLow);

        // Load the SVS from the stack on the active lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            TargetSVS.StorageAGPR);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());

        // Load the SVS from the stack on the inactive lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            TargetSVS.StorageAGPR);

        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());

        // swap the FS Hi and FS Lo of the app back
        emitSGPRSwap(InsertionPointMBB.end(), SrcSVS.FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);

        emitSGPRSwap(InsertionPointMBB.end(), SrcSVS.FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
      });
  emitWaitCnt(NextIPoint);
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                const SpilledWithThreeSGPRsValueStorage &SrcSVS,
                                const TwoAGPRValueStorage &TargetSVS) {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Temporarily swap the FS Hi and FS Lo of the app with the storage to
        // spill the SVA
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     SrcSVS.FlatScratchSGPRHigh);

        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     SrcSVS.FlatScratchSGPRLow);

        // Move V0 to the TargetSVS's temp AGPR
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.TempAGPR, true);
        // Load the SVS to V0
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0);
        emitWaitCnt(InsertionPointMBB.end());
        // Move V0 to the target AGPR storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Restore V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), TargetSVS.TempAGPR,
                               llvm::AMDGPU::VGPR0, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill V0 on the inactive lanes
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.TempAGPR, true);
        // Load the SVS to V0
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0);
        emitWaitCnt(InsertionPointMBB.end());
        // Move V0 to the target AGPR storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Restore V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), TargetSVS.TempAGPR,
                               llvm::AMDGPU::VGPR0, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());

        // swap the FS Hi and FS Lo of the app back
        emitSGPRSwap(InsertionPointMBB.end(), SrcSVS.FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);

        emitSGPRSwap(InsertionPointMBB.end(), SrcSVS.FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
      });
  emitWaitCnt(NextIPoint);
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const SpilledWithThreeSGPRsValueStorage &SrcSVS,
                    const AGPRWithThreeSGPRSValueStorage &TargetSVS) {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Temporarily swap the FS Hi and FS Lo of the app with the storage to
        // spill the SVA
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     SrcSVS.FlatScratchSGPRHigh);

        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     SrcSVS.FlatScratchSGPRLow);

        // Spill V0 on the active lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Load the SVS to V0
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0);
        emitWaitCnt(InsertionPointMBB.end());
        // Move V0 to the target AGPR storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Restore V0
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill V0 on the inactive lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Load the SVS to V0
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0);
        emitWaitCnt(InsertionPointMBB.end());
        // Move V0 to the target AGPR storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Restore V0
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            llvm::AMDGPU::VGPR0);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // swap the FS Hi and FS Lo of the app back
        emitSGPRSwap(InsertionPointMBB.end(), SrcSVS.FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);

        emitSGPRSwap(InsertionPointMBB.end(), SrcSVS.FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
      });
  emitWaitCnt(NextIPoint);
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const SpilledWithThreeSGPRsValueStorage &SrcSVS,
                    const SpilledWithThreeSGPRsValueStorage &TargetSVS) {
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRHigh,
                         TargetSVS.FlatScratchSGPRHigh, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRLow,
                         TargetSVS.FlatScratchSGPRLow, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.EmergencyVGPRSpillSlotOffset,
                         TargetSVS.EmergencyVGPRSpillSlotOffset, true);
}

//===----------------------------------------------------------------------===//
// SpilledWithOneSGPRsValueStorage Switch logic
//===----------------------------------------------------------------------===//

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                const SpilledWithOneSGPRsValueStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS) {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Load the SVS on the active lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            TargetSVS.StorageVGPR);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // Load the SVS on the inactive lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            TargetSVS.StorageVGPR);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  emitWaitCnt(NextIPoint);
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const SpilledWithOneSGPRsValueStorage &SrcSVS,
                    const SingleAGPRStateValueArrayStorage &TargetSVS) {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Load the SVS on the active lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            TargetSVS.StorageAGPR);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // Load the SVS on the inactive lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.EmergencyVGPRSpillSlotOffset,
            TargetSVS.StorageAGPR);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  emitWaitCnt(NextIPoint);
}

static void
emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                    const SpilledWithOneSGPRsValueStorage &SrcSVS,
                    const SpilledWithOneSGPRsValueStorage &TargetSVS) {
  emitMoveFromSGPRToSGPR(MI, SrcSVS.EmergencyVGPRSpillSlotOffset,
                         TargetSVS.EmergencyVGPRSpillSlotOffset, true);
}

void VGPRStateValueArrayStorage::emitCodeToSwitchSVS(
    llvm::MachineInstr &MI, StateValueArrayStorage &TargetSVS) const {
  if (auto *TargetVgprStorage =
          llvm::dyn_cast<VGPRStateValueArrayStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetVgprStorage);
  } else if (auto *TargetAgprStorage =
                 llvm::dyn_cast<SingleAGPRStateValueArrayStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetAgprStorage);
  } else if (auto *TargetTwoAgprStorage =
                 llvm::dyn_cast<TwoAGPRValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetTwoAgprStorage);
  } else if (auto *TargetAgprWith3SgprStorage =
                 llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetAgprWith3SgprStorage);
  } else if (auto *Target3SgprStorage =
                 llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *Target3SgprStorage);
  } else if (auto *TargetSgprStorage =
                 llvm::dyn_cast<SpilledWithOneSGPRsValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetSgprStorage);
  }
  llvm_unreachable("Invalid SVS passed.");
}

void SingleAGPRStateValueArrayStorage::emitCodeToSwitchSVS(
    llvm::MachineInstr &MI, StateValueArrayStorage &TargetSVS) const {
  if (auto *TargetVgprStorage =
          llvm::dyn_cast<VGPRStateValueArrayStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetVgprStorage);
  } else if (auto *TargetAgprStorage =
                 llvm::dyn_cast<SingleAGPRStateValueArrayStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetAgprStorage);
  } else if (auto *Target3SgprStorage =
                 llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *Target3SgprStorage);
  } else if (auto *TargetSgprStorage =
                 llvm::dyn_cast<SpilledWithOneSGPRsValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetSgprStorage);
  }
  llvm_unreachable("Invalid SVS passed.");
}

void TwoAGPRValueStorage::emitCodeToLoadSVA(llvm::MachineInstr &MI,
                                            llvm::MCRegister DestVGPR) const {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Spill the Dest VGPR in the active lanes to the temp AGPR
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), DestVGPR, TempAGPR);
        // Copy the state value from AGPR to the dest VGPR in the active lanes
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), StorageAGPR, DestVGPR,
                               false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill the Dest VGPR in the remaining non-active lanes to the temp
        // AGPR
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), DestVGPR, TempAGPR);
        // Copy the state value from AGPR to the dest VGPR in the active lanes
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), StorageAGPR, DestVGPR,
                               true);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

void TwoAGPRValueStorage::emitCodeToStoreSVA(llvm::MachineInstr &MI,
                                             llvm::MCRegister SrcVGPR) const {
  (void)createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock
                                               &InsertionPointMBB,
                                           const llvm::TargetInstrInfo &TII) {
    // Spill the Src VGPR in the active lanes to the storage AGPR
    emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcVGPR, StorageAGPR, true);
    // Restore the temp AGPR contents into the src VGPR in the active lanes
    emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), TempAGPR, SrcVGPR, false);
    // Flip the exec mask
    emitExecMaskFlip(InsertionPointMBB.end());
    // Spill the Src VGPR in the inactive lanes to the storage AGPR
    emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcVGPR, StorageAGPR, true);
    // Restore the temp AGPR contents into the src VGPR in the active lanes
    emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), TempAGPR, SrcVGPR, true);
    // Flip the exec mask to its original value
    emitExecMaskFlip(InsertionPointMBB.end());
  });
}

void TwoAGPRValueStorage::emitCodeToSwitchSVS(
    llvm::MachineInstr &MI, StateValueArrayStorage &TargetSVS) const {
  if (auto *TargetVgprStorage =
          llvm::dyn_cast<VGPRStateValueArrayStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetVgprStorage);
  } else if (auto *TargetTwoAgprStorage =
                 llvm::dyn_cast<TwoAGPRValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetTwoAgprStorage);
  } else if (auto *TargetAgprWith3SgprStorage =
                 llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetAgprWith3SgprStorage);
  } else if (auto *Target3SgprStorage =
                 llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *Target3SgprStorage);
  }
  llvm_unreachable("Invalid SVS passed.");
}

void AGPRWithThreeSGPRSValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// FS swap
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     FlatScratchSGPRLow);
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     FlatScratchSGPRHigh);
        /// Spill the DestVGPR to the emergency spill slot in the active lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, DestVGPR,
            true);
        /// Restore the state value array from the storage AGPR to the dest VGPR
        /// in the active lanes
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), StorageAGPR, DestVGPR,
                               false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Spill the DestVGPR to the emergency spill slot in the active lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, DestVGPR,
            true);
        /// Restore the state value array from the storage AGPR to the dest VGPR
        /// in the active lanes
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), StorageAGPR, DestVGPR,
                               true);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}

void AGPRWithThreeSGPRSValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  auto NextIPoint = createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// FS swap
    emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                 FlatScratchSGPRLow);
    emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                 FlatScratchSGPRHigh);
    /// Move the SVS from the SrcVGPR to the storage AGPR
    emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcVGPR, StorageAGPR, true);

    /// Load the app VGPR to the SrcVGPR
    emitLoadFromEmergencyVGPRScratchSpillLocation(
        InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, SrcVGPR);
    // Flip the exec mask
    emitExecMaskFlip(InsertionPointMBB.end());

    /// Move the SVS from the SrcVGPR to the storage AGPR in the inactive lanes
    emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcVGPR, StorageAGPR, true);

    /// Load the app VGPR to the SrcVGPR
    emitLoadFromEmergencyVGPRScratchSpillLocation(
        InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, SrcVGPR);
    // Flip the exec mask to its original value
    emitExecMaskFlip(InsertionPointMBB.end());
  });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}
void AGPRWithThreeSGPRSValueStorage::emitCodeToSwitchSVS(
    llvm::MachineInstr &MI, StateValueArrayStorage &TargetSVS) const {
  if (auto *TargetVgprStorage =
          llvm::dyn_cast<VGPRStateValueArrayStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetVgprStorage);
  } else if (auto *TargetTwoAgprStorage =
                 llvm::dyn_cast<TwoAGPRValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetTwoAgprStorage);
  } else if (auto *TargetAgprWith3SgprStorage =
                 llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetAgprWith3SgprStorage);
  } else if (auto *Target3SgprStorage =
                 llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *Target3SgprStorage);
  }
  llvm_unreachable("Invalid SVS passed.");
}

void SpilledWithThreeSGPRsValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// FS swap
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     FlatScratchSGPRLow);
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     FlatScratchSGPRHigh);
        /// Spill the DestVGPR to the emergency spill slot in the active lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, DestVGPR,
            true);
        /// Restore the state value array from its fixed storage to the dest
        /// VGPR in the active lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, DestVGPR);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Spill the DestVGPR to the emergency spill slot in the inactive lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, DestVGPR,
            true);
        /// Restore the state value array from its fixed storage to the dest
        /// VGPR in the active lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, DestVGPR);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}

void SpilledWithThreeSGPRsValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// FS swap
        emitSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_LO, FlatScratchSGPRLow);
        emitSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_HI, FlatScratchSGPRHigh);
        /// Spill the Src to the emergency spill slot in the active lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, SrcVGPR,
            true);
        /// Restore the app VGPR from its fixed storage to the src VGPR
        /// in the active lanes
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, SrcVGPR);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Spill the Src to the emergency spill slot in the inactive lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, SrcVGPR,
            true);
        /// Restore the app VGPR from its fixed storage to the src VGPR
        /// in the active lanes
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, SrcVGPR);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}
void SpilledWithThreeSGPRsValueStorage::emitCodeToSwitchSVS(
    llvm::MachineInstr &MI, StateValueArrayStorage &TargetSVS) const {
  if (auto *TargetVgprStorage =
          llvm::dyn_cast<VGPRStateValueArrayStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetVgprStorage);
  } else if (auto *TargetAgprStorage =
                 llvm::dyn_cast<SingleAGPRStateValueArrayStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetAgprStorage);
  } else if (auto *TargetTwoAgprStorage =
                 llvm::dyn_cast<TwoAGPRValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetTwoAgprStorage);
  } else if (auto *TargetAgprWith3SgprStorage =
                 llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetAgprWith3SgprStorage);
  } else if (auto *Target3SgprStorage =
                 llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *Target3SgprStorage);
  }
  llvm_unreachable("Invalid SVS passed.");
}

void SpilledWithOneSGPRsValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// Spill the DestVGPR to the emergency spill slot in the active lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, DestVGPR,
            true);
        /// Load the SVS
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, DestVGPR);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Spill the DestVGPR to the emergency spill slot in the inactive lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, DestVGPR,
            true);
        /// Load the SVS
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, DestVGPR);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}

void SpilledWithOneSGPRsValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// Spill the Src to the emergency spill slot in the active lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, SrcVGPR,
            false);
        /// Restore the app VGPR from its fixed storage to the src VGPR
        /// in the active lanes
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, SrcVGPR);
        /// flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Spill the Src to the emergency spill slot in the inactive lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, SrcVGPR,
            false);
        /// Restore the app VGPR from its fixed storage to the src VGPR
        /// in the active lanes
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), EmergencyVGPRSpillSlotOffset, SrcVGPR);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  emitWaitCnt(NextIPoint);
}
void SpilledWithOneSGPRsValueStorage::emitCodeToSwitchSVS(
    llvm::MachineInstr &MI, StateValueArrayStorage &TargetSVS) const {
  if (auto *TargetVgprStorage =
          llvm::dyn_cast<VGPRStateValueArrayStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetVgprStorage);
  } else if (auto *TargetAgprStorage =
                 llvm::dyn_cast<SingleAGPRStateValueArrayStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetAgprStorage);
  } else if (auto *TargetSgprStorage =
                 llvm::dyn_cast<SpilledWithOneSGPRsValueStorage>(this)) {
    luthier::emitCodeToSwitchSVS(MI, *this, *TargetSgprStorage);
  }
  llvm_unreachable("Invalid SVS passed.");
}

void getSupportedSVAStorageList(
    const llvm::GCNSubtarget &ST,
    llvm::SmallVectorImpl<StateValueArrayStorage::StorageKind>
        &SupportedStorageKinds) {
  /// Single VGPR storage is always supported and the most preferred
  SupportedStorageKinds.push_back(StateValueArrayStorage::SVS_SINGLE_VGPR);
  /// Other storage types are listed here based on preference
  for (auto SK :
       {StateValueArrayStorage::SVS_SINGLE_VGPR,
        StateValueArrayStorage::SVS_ONE_AGPR_post_gfx908,
        StateValueArrayStorage::SVS_TWO_AGPRs_pre_gfx908,
        StateValueArrayStorage::SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908,
        StateValueArrayStorage::SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs,
        StateValueArrayStorage::SVS_SPILLED_WITH_ONE_SGPR_architected_fs}) {
    if (StateValueArrayStorage::isSupportedOnSubTarget(SK, ST))
      SupportedStorageKinds.push_back(SK);
  };
}

} // namespace luthier