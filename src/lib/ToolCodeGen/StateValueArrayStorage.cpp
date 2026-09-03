//===-- StateValueArrayStorage.cpp ----------------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
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
#include "luthier/ToolCodeGen/StateValueArrayStorage.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/ToolCodeGen/MIRConvenience.h"
#include "luthier/ToolCodeGen/StateValueArraySpecs.h"
#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <SIRegisterInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/IR/Module.h>

namespace luthier {

static const llvm::DenseMap<StateValueArrayStorage::StorageKind, int>
    NumVGPRsUsedBySVS{
        {StateValueArrayStorage::SVS_SINGLE_VGPR, 1},
        {StateValueArrayStorage::SVS_TWO_AGPRs, 0},
        {StateValueArrayStorage::SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908,
         0},
        {StateValueArrayStorage::SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs, 0},
        {StateValueArrayStorage::SVS_SPILLED_WITH_ONE_SGPR_architected_fs, 0}};

static const llvm::DenseMap<StateValueArrayStorage::StorageKind, int>
    NumAGPRsUsedBySVS{
        {StateValueArrayStorage::SVS_SINGLE_VGPR, 0},
        {StateValueArrayStorage::SVS_TWO_AGPRs, 2},
        {StateValueArrayStorage::SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908,
         1},
        {StateValueArrayStorage::SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs, 0},
        {StateValueArrayStorage::SVS_SPILLED_WITH_ONE_SGPR_architected_fs, 0}};

static const llvm::DenseMap<StateValueArrayStorage::StorageKind, int>
    NumSGPRsUsedBySVS{
        {StateValueArrayStorage::SVS_SINGLE_VGPR, 0},
        {StateValueArrayStorage::SVS_TWO_AGPRs, 0},
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
        {StateValueArrayStorage::SVS_TWO_AGPRs,
         [](const llvm::GCNSubtarget &ST) { return ST.hasMAIInsts(); }},
        {StateValueArrayStorage::SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908,
         [](const llvm::GCNSubtarget &ST) {
           return ST.hasMAIInsts() && !ST.hasGFX90AInsts();
         }},
        {StateValueArrayStorage::SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs,
         [](const llvm::GCNSubtarget &ST) {
           return !ST.hasArchitectedFlatScratch();
         }},
        {StateValueArrayStorage::SVS_SPILLED_WITH_ONE_SGPR_architected_fs,
         [](const llvm::GCNSubtarget &ST) {
           return ST.hasArchitectedFlatScratch();
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
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        VGPRs.size() >= 1,
        "Insufficient number of VGPRs for single VGPR SVA storage."));
    return std::make_unique<VGPRStateValueArrayStorage>(VGPRs[0]);
  case SVS_TWO_AGPRs:
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        AGPRs.size() >= 2,
        "Insufficient number of AGPRs for two AGPR SVA storage."));
    return std::make_unique<TwoAGPRValueStorage>(AGPRs[0], AGPRs[1]);
  case SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908:
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        AGPRs.size() >= 1, "Insufficient number of AGPRs for single AGPR with "
                           "three SGPR SVA storage."));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        SGPRs.size() >= 3, "Insufficient number of AGPRs for single AGPR with "
                           "three SGPR SVA storage."));
    return std::make_unique<AGPRWithThreeSGPRSValueStorage>(AGPRs[0], SGPRs[0],
                                                            SGPRs[1], SGPRs[2]);
  case SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs:
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        SGPRs.size() >= 3, "Insufficient number of AGPRs for spilled with "
                           "three SGPR SVA storage."));
    return std::make_unique<SpilledWithThreeSGPRsValueStorage>(
        SGPRs[0], SGPRs[1], SGPRs[2]);
  case SVS_SPILLED_WITH_ONE_SGPR_architected_fs:
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        SGPRs.size() >= 1, "Insufficient number of SGPRs for spilled with "
                           "single SGPR SVA storage."));
    return std::make_unique<SpilledWithOneSGPRsValueStorage>(SGPRs[0]);
  }
  llvm_unreachable("Invalid SVA storage Enum value.");
}

static void
loadStackPointerFromSVALanes(llvm::MachineBasicBlock::iterator Iter,
                             llvm::MCRegister SrcVGPR,
                             llvm::MCRegister StackPointer,
                             const StateValueArraySpecs &Specs);

static void loadFlatScratchFromSVALanes(llvm::MachineBasicBlock::iterator Iter,
                                        llvm::MCRegister SrcVGPR,
                                        llvm::MCRegister FSLo,
                                        llvm::MCRegister FSHi,
                                        const StateValueArraySpecs &Specs,
                                        const char *Context);

//===----------------------------------------------------------------------===//
// VGPRStateValueArrayStorage Switch logic
//===----------------------------------------------------------------------===//

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const VGPRStateValueArrayStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Do a move on the active lanes
        emitMoveFromVGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageVGPR,
                               TargetSVS.StorageVGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Do a move on the inactive lanes
        emitMoveFromVGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageVGPR,
                               TargetSVS.StorageVGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const VGPRStateValueArrayStorage &SrcSVS,
                                const TwoAGPRValueStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
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

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const VGPRStateValueArrayStorage &SrcSVS,
                                const AGPRWithThreeSGPRSValueStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
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
emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                    const VGPRStateValueArrayStorage &SrcSVS,
                    const SpilledWithThreeSGPRsValueStorage &TargetSVS,
                    const StateValueArraySpecs &Specs) {
  // Do SCC-uniform work here
  loadFlatScratchFromSVALanes(MI, SrcSVS.StorageVGPR,
                              TargetSVS.FlatScratchSGPRLow,
                              TargetSVS.FlatScratchSGPRHigh, Specs,
                              "emitCodeToSwitchSVS(VGPR->SpilledWithThree)");
  loadStackPointerFromSVALanes(MI, SrcSVS.StorageVGPR, TargetSVS.StackPointer,
                               Specs);

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
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            SrcSVS.StorageVGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill the SVA on the inactive lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
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
emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                    const VGPRStateValueArrayStorage &SrcSVS,
                    const SpilledWithOneSGPRsValueStorage &TargetSVS,
                    const StateValueArraySpecs &Specs) {
  // Do SCC-uniform work here
  emitMoveFromVGPRLaneToSGPR(MI, SrcSVS.StorageVGPR,
                             TargetSVS.StackPointer,
                             Specs.getStackPointerStoreLane(), false);
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Spill the SVA on the active lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            SrcSVS.StorageVGPR, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill the SVA on the inactive lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            SrcSVS.StorageVGPR, true);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}

//===----------------------------------------------------------------------===//
// TwoAGPRValueStorage Switch logic
//===----------------------------------------------------------------------===//

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const TwoAGPRValueStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
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

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const TwoAGPRValueStorage &SrcSVS,
                                const TwoAGPRValueStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
  const auto &ST = MI->getMF()->getSubtarget<llvm::GCNSubtarget>();
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        if (ST.hasGFX90AInsts()) {
          // V_ACCVGPR_MOV_B32 is available for GFX90A and later
          auto EmitAGPRMove = [&](bool KillSource) {
            (void)llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(),
                                llvm::DebugLoc(),
                                TII.get(llvm::AMDGPU::V_ACCVGPR_MOV_B32),
                                TargetSVS.StorageAGPR)
                .addReg(SrcSVS.StorageAGPR, llvm::getKillRegState(KillSource));
          };
          EmitAGPRMove(/*KillSource=*/false);
          emitExecMaskFlip(InsertionPointMBB.end());
          EmitAGPRMove(/*KillSource=*/true);
          emitExecMaskFlip(InsertionPointMBB.end());
        } else {
          // gfx908 (AGPRs exist, but only V_ACCVGPR_READ/WRITE via a VGPR).

          // Active lanes.
          emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                                 SrcSVS.TempAGPR, /*KillSource=*/false);
          emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                                 llvm::AMDGPU::VGPR0, /*KillSource=*/false);
          emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                                 TargetSVS.StorageAGPR, /*KillSource=*/false);
          emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.TempAGPR,
                                 llvm::AMDGPU::VGPR0, /*KillSource=*/false);

          emitExecMaskFlip(InsertionPointMBB.end());

          // Inactive lanes.
          emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                                 SrcSVS.TempAGPR, /*KillSource=*/false);
          emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                                 llvm::AMDGPU::VGPR0, /*KillSource=*/true);
          emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                                 TargetSVS.StorageAGPR, /*KillSource=*/false);
          emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.TempAGPR,
                                 llvm::AMDGPU::VGPR0, /*KillSource=*/true);

          emitExecMaskFlip(InsertionPointMBB.end());
        }
      });
};

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const TwoAGPRValueStorage &SrcSVS,
                                const AGPRWithThreeSGPRSValueStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
  const auto &ST = MI->getMF()->getSubtarget<llvm::GCNSubtarget>();
  (void)createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        if (ST.hasGFX90AInsts()) {
          auto EmitAGPRMove = [&](bool KillSource) {
            (void)llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(),
                                llvm::DebugLoc(),
                                TII.get(llvm::AMDGPU::V_ACCVGPR_MOV_B32),
                                TargetSVS.StorageAGPR)
                .addReg(SrcSVS.StorageAGPR, llvm::getKillRegState(KillSource));
          };
          EmitAGPRMove(/*KillSource=*/false);
          emitExecMaskFlip(InsertionPointMBB.end());
          EmitAGPRMove(/*KillSource=*/true);
          emitExecMaskFlip(InsertionPointMBB.end());
        } else {

          // Active lanes.
          emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                                 SrcSVS.TempAGPR, /*KillSource=*/false);
          emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                                 llvm::AMDGPU::VGPR0, /*KillSource=*/false);
          emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                                 TargetSVS.StorageAGPR, /*KillSource=*/false);
          emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.TempAGPR,
                                 llvm::AMDGPU::VGPR0, /*KillSource=*/false);

          emitExecMaskFlip(InsertionPointMBB.end());

          // Inactive lanes.
          emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                                 SrcSVS.TempAGPR, /*KillSource=*/false);
          emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                                 llvm::AMDGPU::VGPR0, /*KillSource=*/true);
          emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                                 TargetSVS.StorageAGPR, /*KillSource=*/false);
          emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.TempAGPR,
                                 llvm::AMDGPU::VGPR0, /*KillSource=*/true);

          emitExecMaskFlip(InsertionPointMBB.end());
        }
      });
};

static void
emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                    const TwoAGPRValueStorage &SrcSVS,
                    const SpilledWithThreeSGPRsValueStorage &TargetSVS,
                    const StateValueArraySpecs &Specs) {
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
  loadFlatScratchFromSVALanes(NextIPoint, llvm::AMDGPU::VGPR0,
                              TargetSVS.FlatScratchSGPRLow,
                              TargetSVS.FlatScratchSGPRHigh, Specs,
                              "emitCodeToSwitchSVS(TwoAGPR->SpilledWithThree)");
  loadStackPointerFromSVALanes(NextIPoint, llvm::AMDGPU::VGPR0,
                               TargetSVS.StackPointer, Specs);

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
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            llvm::AMDGPU::VGPR0, false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill the SVA on the inactive register
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
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
        // Read V0 (active lanes) back from SrcSVS.TempAGPR.
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.TempAGPR,
                               llvm::AMDGPU::VGPR0, /*KillSource=*/false);
        // Flip the exec mask.
        emitExecMaskFlip(InsertionPointMBB.end());
        // Read V0 (inactive lanes) back from SrcSVS.TempAGPR;
        // last use — kill.
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.TempAGPR,
                               llvm::AMDGPU::VGPR0, /*KillSource=*/true);
        // Flip the exec mask back.
        emitExecMaskFlip(InsertionPointMBB.end());
      });
}

//===----------------------------------------------------------------------===//
// AGPRWithThreeSGPRSValueStorage Switch logic
//===----------------------------------------------------------------------===//

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const AGPRWithThreeSGPRSValueStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
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

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const AGPRWithThreeSGPRSValueStorage &SrcSVS,
                                const TwoAGPRValueStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
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

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const AGPRWithThreeSGPRSValueStorage &SrcSVS,
                                const AGPRWithThreeSGPRSValueStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
  // Move the SGPRs first
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRHigh,
                         TargetSVS.FlatScratchSGPRHigh, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRLow,
                         TargetSVS.FlatScratchSGPRLow, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.StackPointer,
                         TargetSVS.StackPointer, true);

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
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            llvm::AMDGPU::VGPR0, true);
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Read the SVS from V0 into its target storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Restore V0's original value
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            llvm::AMDGPU::VGPR0);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            llvm::AMDGPU::VGPR0, true);
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Read the SVS from V0 into its target storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Restore V0's original value
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
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
emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                    const AGPRWithThreeSGPRSValueStorage &SrcSVS,
                    const SpilledWithThreeSGPRsValueStorage &TargetSVS,
                    const StateValueArraySpecs &Specs) {
  // Move the SGPRs first
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRHigh,
                         TargetSVS.FlatScratchSGPRHigh, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRLow,
                         TargetSVS.FlatScratchSGPRLow, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.StackPointer,
                         TargetSVS.StackPointer, true);

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
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Spill the SVS to the stack
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Restore V0's original value
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            llvm::AMDGPU::VGPR0);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Read the SrcSVS AGPR to V0
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), SrcSVS.StorageAGPR,
                               llvm::AMDGPU::VGPR0, false);
        // Spill the SVS to the stack
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Restore V0's original value
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), TargetSVS.StackPointer,
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

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const SpilledWithThreeSGPRsValueStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {

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
            InsertionPointMBB.end(), SrcSVS.StackPointer,
            TargetSVS.StorageVGPR);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());

        // Load the SVS from the stack on the inactive lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.StackPointer,
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

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const SpilledWithThreeSGPRsValueStorage &SrcSVS,
                                const TwoAGPRValueStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
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
            InsertionPointMBB.end(), SrcSVS.StackPointer,
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
            InsertionPointMBB.end(), SrcSVS.StackPointer,
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

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const SpilledWithThreeSGPRsValueStorage &SrcSVS,
                                const AGPRWithThreeSGPRSValueStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
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
            InsertionPointMBB.end(), SrcSVS.StackPointer,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Load the SVS to V0
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.StackPointer,
            llvm::AMDGPU::VGPR0);
        emitWaitCnt(InsertionPointMBB.end());
        // Move V0 to the target AGPR storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Restore V0
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.StackPointer,
            llvm::AMDGPU::VGPR0);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // Spill V0 on the inactive lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.StackPointer,
            llvm::AMDGPU::VGPR0, true);
        emitWaitCnt(InsertionPointMBB.end());
        // Load the SVS to V0
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.StackPointer,
            llvm::AMDGPU::VGPR0);
        emitWaitCnt(InsertionPointMBB.end());
        // Move V0 to the target AGPR storage
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), llvm::AMDGPU::VGPR0,
                               TargetSVS.StorageAGPR, true);
        // Restore V0
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.StackPointer,
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
emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                    const SpilledWithThreeSGPRsValueStorage &SrcSVS,
                    const SpilledWithThreeSGPRsValueStorage &TargetSVS,
                    const StateValueArraySpecs &Specs) {
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRHigh,
                         TargetSVS.FlatScratchSGPRHigh, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.FlatScratchSGPRLow,
                         TargetSVS.FlatScratchSGPRLow, true);
  emitMoveFromSGPRToSGPR(MI, SrcSVS.StackPointer,
                         TargetSVS.StackPointer, true);
}

//===----------------------------------------------------------------------===//
// SpilledWithOneSGPRsValueStorage Switch logic
//===----------------------------------------------------------------------===//

static void emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                                const SpilledWithOneSGPRsValueStorage &SrcSVS,
                                const VGPRStateValueArrayStorage &TargetSVS,
                                const StateValueArraySpecs &Specs) {
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        // Load the SVS on the active lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.StackPointer,
            TargetSVS.StorageVGPR);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
        // Load the SVS on the inactive lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), SrcSVS.StackPointer,
            TargetSVS.StorageVGPR);
        // Flip the exec mask back
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  emitWaitCnt(NextIPoint);
}

static void
emitCodeToSwitchSVS(llvm::MachineBasicBlock::iterator &MI,
                    const SpilledWithOneSGPRsValueStorage &SrcSVS,
                    const SpilledWithOneSGPRsValueStorage &TargetSVS,
                    const StateValueArraySpecs &Specs) {
  emitMoveFromSGPRToSGPR(MI, SrcSVS.StackPointer,
                         TargetSVS.StackPointer, true);
}

void VGPRStateValueArrayStorage::emitCodeToSwitchSVS(
    llvm::MachineBasicBlock::iterator MI,
    const StateValueArrayStorage &TargetSVS,
    const StateValueArraySpecs &Specs) const {
  if (auto *Tgt = llvm::dyn_cast<VGPRStateValueArrayStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<TwoAGPRValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<SpilledWithOneSGPRsValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  llvm_unreachable("Invalid SVS passed.");
}

bool VGPRStateValueArrayStorage::operator==(
    const StateValueArrayStorage &LHS) const {
  if (auto *LHSCast = llvm::dyn_cast<VGPRStateValueArrayStorage>(&LHS)) {
    return this->StorageVGPR == LHSCast->StorageVGPR;
  } else
    return false;
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
    llvm::MachineBasicBlock::iterator MI,
    const StateValueArrayStorage &TargetSVS,
    const StateValueArraySpecs &Specs) const {
  if (auto *Tgt = llvm::dyn_cast<VGPRStateValueArrayStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<TwoAGPRValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  llvm_unreachable("Invalid SVS passed.");
}

bool TwoAGPRValueStorage::operator==(const StateValueArrayStorage &LHS) const {
  if (auto *LHSCast = llvm::dyn_cast<TwoAGPRValueStorage>(&LHS)) {
    return (this->StorageAGPR == LHSCast->StorageAGPR) &&
           (this->TempAGPR == LHSCast->TempAGPR);
  } else
    return false;
}

void AGPRWithThreeSGPRSValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  assert(!MI.getMF()
              ->getSubtarget<llvm::GCNSubtarget>()
              .hasArchitectedFlatScratch() &&
         "target with architected flat scratch is using "
         "AGPRWithThreeSGPRSValueStorage");
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// Swap FS_LO/HI with the thread-FS copies so subsequent scratch
        /// ops address the instrumentation's private segment, not the app's.
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     FlatScratchSGPRLow);
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     FlatScratchSGPRHigh);
        /// Spill the DestVGPR to the emergency spill slot in the active lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, DestVGPR,
            /*KillSource=*/false);
        /// Restore the state value array from the storage AGPR to the dest VGPR
        /// in the active lanes
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), StorageAGPR, DestVGPR,
                               /*KillSource=*/false);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Spill the DestVGPR to the emergency spill slot in the inactive
        /// lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, DestVGPR,
            /*KillSource=*/true);
        /// Restore the state value array from the storage AGPR to the dest VGPR
        /// in the inactive lanes; last read of StorageAGPR — kill.
        emitMoveFromAGPRToVGPR(InsertionPointMBB.end(), StorageAGPR, DestVGPR,
                               /*KillSource=*/true);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Swap FS_LO/HI back so the app's FLAT_SCR is restored before
        /// the injected payload starts executing.
        emitSGPRSwap(InsertionPointMBB.end(), FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
        emitSGPRSwap(InsertionPointMBB.end(), FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}

void AGPRWithThreeSGPRSValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  assert(!MI.getMF()
              ->getSubtarget<llvm::GCNSubtarget>()
              .hasArchitectedFlatScratch() &&
         "target with architected flat scratch is using "
         "AGPRWithThreeSGPRSValueStorage");
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// Swap FS_LO/HI with the thread-FS copies so subsequent scratch
        /// ops address the instrumentation's private segment, not the app's.
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     FlatScratchSGPRLow);
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     FlatScratchSGPRHigh);
        /// Move the SVS from the SrcVGPR back to the storage AGPR
        /// (active lanes).
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcVGPR, StorageAGPR,
                               /*KillSource=*/false);

        /// Load the app VGPR to the SrcVGPR (redefs SrcVGPR active lanes)
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, SrcVGPR);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());

        /// Move the SVS from the SrcVGPR back to the storage AGPR in the
        /// inactive lanes; last read of SrcVGPR SVA content — kill.
        emitMoveFromVGPRToAGPR(InsertionPointMBB.end(), SrcVGPR, StorageAGPR,
                               /*KillSource=*/true);

        /// Load the app VGPR to the SrcVGPR
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, SrcVGPR);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Swap FS_LO/HI back so the app's FLAT_SCR is restored on return.
        emitSGPRSwap(InsertionPointMBB.end(), FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
        emitSGPRSwap(InsertionPointMBB.end(), FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}
void AGPRWithThreeSGPRSValueStorage::emitCodeToSwitchSVS(
    llvm::MachineBasicBlock::iterator MI,
    const StateValueArrayStorage &TargetSVS,
    const StateValueArraySpecs &Specs) const {
  if (auto *Tgt = llvm::dyn_cast<VGPRStateValueArrayStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<TwoAGPRValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  llvm_unreachable("Invalid SVS passed.");
}

bool AGPRWithThreeSGPRSValueStorage::operator==(
    const StateValueArrayStorage &LHS) const {
  if (auto *LHSCast = llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(&LHS)) {
    return (this->StorageAGPR == LHSCast->StorageAGPR) &&
           (this->StackPointer ==
            LHSCast->StackPointer) &&
           (this->FlatScratchSGPRHigh == LHSCast->FlatScratchSGPRHigh) &&
           (this->FlatScratchSGPRLow == LHSCast->FlatScratchSGPRLow);
  } else
    return false;
}

void SpilledWithThreeSGPRsValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  assert(!MI.getMF()
              ->getSubtarget<llvm::GCNSubtarget>()
              .hasArchitectedFlatScratch() &&
         "target with architected flat scratch is using "
         "SpilledWithThreeSGPRsValueStorage");
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// Swap FS_LO/HI with the thread-FS copies so subsequent scratch
        /// ops address the instrumentation's private segment, not the app's.
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     FlatScratchSGPRLow);
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     FlatScratchSGPRHigh);
        /// Spill the DestVGPR to the emergency spill slot in the active
        /// lanes.
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, DestVGPR,
            /*KillSource=*/false);
        /// Restore the state value array from its fixed storage to the dest
        /// VGPR in the active lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, DestVGPR);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Spill the DestVGPR to the emergency spill slot in the inactive
        /// lanes.
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, DestVGPR,
            /*KillSource=*/true);
        /// Restore the state value array from its fixed storage to the dest
        /// VGPR in the inactive lanes
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, DestVGPR);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Swap FS_LO/HI back so the app's FLAT_SCR is restored before
        /// the injected payload starts executing.
        emitSGPRSwap(InsertionPointMBB.end(), FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
        emitSGPRSwap(InsertionPointMBB.end(), FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}

void SpilledWithThreeSGPRsValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  assert(!MI.getMF()
              ->getSubtarget<llvm::GCNSubtarget>()
              .hasArchitectedFlatScratch() &&
         "target with architected flat scratch is using "
         "SpilledWithThreeSGPRsValueStorage");
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// Swap FS_LO/HI with the thread-FS copies (inside the SCC-safe
        /// sequence; the previous version used `MI` as the insertion point,
        /// which landed these swaps BEFORE the SCC save and could clobber
        /// SCC).
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                     FlatScratchSGPRLow);
        emitSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                     FlatScratchSGPRHigh);
        /// Spill the Src (SVA) to the SVS emergency slot on the active
        /// lanes. KillSource=false — SrcVGPR is read again on the
        /// inactive-lanes spill below before being redefed by the load.
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, SrcVGPR,
            /*KillSource=*/false);
        /// Restore the app VGPR from its fixed storage to the src VGPR
        /// in the active lanes
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, SrcVGPR);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Spill the Src (SVA) to the SVS emergency slot on the inactive
        /// lanes; last read of SrcVGPR SVA content — kill.
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, SrcVGPR,
            /*KillSource=*/true);
        /// Restore the app VGPR from its fixed storage to the src VGPR
        /// in the inactive lanes
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, SrcVGPR);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Swap FS_LO/HI back so the app's FLAT_SCR is restored on return.
        emitSGPRSwap(InsertionPointMBB.end(), FlatScratchSGPRLow,
                     llvm::AMDGPU::FLAT_SCR_LO);
        emitSGPRSwap(InsertionPointMBB.end(), FlatScratchSGPRHigh,
                     llvm::AMDGPU::FLAT_SCR_HI);
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}
void SpilledWithThreeSGPRsValueStorage::emitCodeToSwitchSVS(
    llvm::MachineBasicBlock::iterator MI,
    const StateValueArrayStorage &TargetSVS,
    const StateValueArraySpecs &Specs) const {
  if (auto *Tgt = llvm::dyn_cast<VGPRStateValueArrayStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<TwoAGPRValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<AGPRWithThreeSGPRSValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  llvm_unreachable("Invalid SVS passed.");
}

bool SpilledWithThreeSGPRsValueStorage::operator==(
    const StateValueArrayStorage &LHS) const {
  if (auto *LHSCast = llvm::dyn_cast<SpilledWithThreeSGPRsValueStorage>(&LHS)) {
    return (this->StackPointer ==
            LHSCast->StackPointer) &&
           (this->FlatScratchSGPRHigh == LHSCast->FlatScratchSGPRHigh) &&
           (this->FlatScratchSGPRLow == LHSCast->FlatScratchSGPRLow);
  } else
    return false;
}

void SpilledWithOneSGPRsValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  assert(MI.getMF()
             ->getSubtarget<llvm::GCNSubtarget>()
             .hasArchitectedFlatScratch() &&
         "target without architected flat scratch is using "
         "SpilledWithOneSGPRsValueStorage");
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// Spill the DestVGPR to the emergency spill slot in the active
        /// lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, DestVGPR,
            /*KillSource=*/false);
        /// Load the SVS
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, DestVGPR);
        // Flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Spill the DestVGPR to the emergency spill slot in the inactive
        /// lanes
        emitStoreToEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, DestVGPR,
            /*KillSource=*/true);
        /// Load the SVS
        emitLoadFromEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, DestVGPR);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  // Wait on the memory operation to complete
  emitWaitCnt(NextIPoint);
}

void SpilledWithOneSGPRsValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  assert(MI.getMF()
             ->getSubtarget<llvm::GCNSubtarget>()
             .hasArchitectedFlatScratch() &&
         "target without architected flat scratch is using "
         "SpilledWithOneSGPRsValueStorage");
  auto NextIPoint = createSCCSafeSequenceOfMIs(
      MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
              const llvm::TargetInstrInfo &TII) {
        /// Spill the Src to the emergency spill slot in the active lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, SrcVGPR,
            false);
        /// Restore the app VGPR from its fixed storage to the src VGPR
        /// in the active lanes
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, SrcVGPR);
        /// flip the exec mask
        emitExecMaskFlip(InsertionPointMBB.end());
        /// Spill the Src to the emergency spill slot in the inactive lanes
        emitStoreToEmergencySVSScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, SrcVGPR,
            false);
        /// Restore the app VGPR from its fixed storage to the src VGPR
        /// in the active lanes
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            InsertionPointMBB.end(), StackPointer, SrcVGPR);
        // Flip the exec mask to its original value
        emitExecMaskFlip(InsertionPointMBB.end());
      });
  emitWaitCnt(NextIPoint);
}
void SpilledWithOneSGPRsValueStorage::emitCodeToSwitchSVS(
    llvm::MachineBasicBlock::iterator MI,
    const StateValueArrayStorage &TargetSVS,
    const StateValueArraySpecs &Specs) const {
  if (auto *Tgt = llvm::dyn_cast<VGPRStateValueArrayStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  if (auto *Tgt = llvm::dyn_cast<SpilledWithOneSGPRsValueStorage>(&TargetSVS))
    return luthier::emitCodeToSwitchSVS(MI, *this, *Tgt, Specs);
  llvm_unreachable("Invalid SVS passed.");
}
bool SpilledWithOneSGPRsValueStorage::operator==(
    const StateValueArrayStorage &LHS) const {
  if (auto *LHSCast = llvm::dyn_cast<SpilledWithOneSGPRsValueStorage>(&LHS)) {
    return (this->StackPointer ==
            LHSCast->StackPointer);
  } else
    return false;
}

//===----------------------------------------------------------------------===//
// Partial-callgraph V0-courier handoff protocol
//===----------------------------------------------------------------------===//
//
// The protocol shuttles the SVA through VGPR0 cross trace functions
//
//   Caller (handOffSVA @ call/indirect-branch MI):
//     * Spill VGPR0 (all lanes) to the SVS's emergency slot.
//     * Load the SVA (from this scheme) into VGPR0 (all lanes).
//     * Perform the call. Callee sees V0 == SVA.
//
//   Callee (pickOffSVA @ device-function entry MI):
//     * Store VGPR0 (SVA) into the entry-block SVS storage (all lanes).
//     * Restore VGPR0 (all lanes) from the SVS's emergency slot.

void VGPRStateValueArrayStorage::handOffSVA(
    llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
    const llvm::GCNSubtarget &ST) const {
  llvm::MachineBasicBlock::iterator Iter = MI.getIterator();
  const bool NeedsFSInstall = !ST.hasArchitectedFlatScratch();
  std::optional<uint8_t> FSSaveLane;
  auto FSLoLane = Specs.findArgumentLane(FLAT_SCRATCH);
  if (NeedsFSInstall) {
    FSSaveLane = Specs.getScratchSpillLane();
    if (!FSSaveLane || FSLoLane == Specs.argument_lane_end())
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_GENERIC_ERROR(
          "VGPRStateValueArrayStorage::handOffSVA on absolute-FS target "
          "needs both ScratchSpillLane and a FLAT_SCRATCH SA lane in the "
          "SVA layout to trampoline FLAT_SCR through."));
  }
  // 1. Stash SGPR0 in the SVS's FramePointerRegSpillLane so we can
  //    scavenge SGPR0 as the SADDR for the SP-relative scratch ops.
  emitMoveFromSGPRToVGPRLane(Iter, llvm::AMDGPU::SGPR0, StorageVGPR,
                             Specs.getFramePointerRegSpillLane(), false);
  // 2. Load the instrumentation SP from the SVS's StackPointerStoreLane
  //    into SGPR0.
  emitMoveFromVGPRLaneToSGPR(Iter, StorageVGPR, llvm::AMDGPU::SGPR0,
                             Specs.getStackPointerStoreLane(), false);
  // 3+4. Spill V0 (all lanes) → [SGPR0-8] and copy SVA → V0 (all lanes).
  llvm::MachineBasicBlock::iterator Next =
      createSCCSafeSequenceOfMIs(Iter, [&](llvm::MachineBasicBlock &IPMBB,
                                           const llvm::TargetInstrInfo &TII) {
        if (NeedsFSInstall) {
          emitMoveFromSGPRToVGPRLane(IPMBB, llvm::AMDGPU::FLAT_SCR_LO,
                                     StorageVGPR, *FSSaveLane, false);
          emitMoveFromSGPRToVGPRLane(IPMBB, llvm::AMDGPU::FLAT_SCR_HI,
                                     StorageVGPR, *FSSaveLane + 1, false);
          emitMoveFromVGPRLaneToSGPR(IPMBB, StorageVGPR,
                                     llvm::AMDGPU::FLAT_SCR_LO,
                                     FSLoLane->second, false);
          emitMoveFromVGPRLaneToSGPR(IPMBB, StorageVGPR,
                                     llvm::AMDGPU::FLAT_SCR_HI,
                                     FSLoLane->second + 1, false);
        }
        // Active lanes.
        emitStoreToEmergencyVGPRScratchSpillLocation(
            IPMBB, llvm::AMDGPU::SGPR0, llvm::AMDGPU::VGPR0, false);
        emitMoveFromVGPRToVGPR(IPMBB, StorageVGPR, llvm::AMDGPU::VGPR0,
                               false);
        emitExecMaskFlip(IPMBB);
        // Inactive lanes.
        emitStoreToEmergencyVGPRScratchSpillLocation(
            IPMBB, llvm::AMDGPU::SGPR0, llvm::AMDGPU::VGPR0, false);
        emitMoveFromVGPRToVGPR(IPMBB, StorageVGPR, llvm::AMDGPU::VGPR0,
                               false);
        emitExecMaskFlip(IPMBB);
        // Restore app's FLAT_SCR pair from the ScratchSpillLane spill.
        if (NeedsFSInstall) {
          emitMoveFromVGPRLaneToSGPR(IPMBB, StorageVGPR,
                                     llvm::AMDGPU::FLAT_SCR_LO, *FSSaveLane,
                                     false);
          emitMoveFromVGPRLaneToSGPR(IPMBB, StorageVGPR,
                                     llvm::AMDGPU::FLAT_SCR_HI,
                                     *FSSaveLane + 1, false);
        }
      });
  emitWaitCnt(Next);
  // 5. Restore SGPR0 from the SVS's FramePointerRegSpillLane.
  emitMoveFromVGPRLaneToSGPR(Next, StorageVGPR, llvm::AMDGPU::SGPR0,
                             Specs.getFramePointerRegSpillLane(), false);
}

void VGPRStateValueArrayStorage::pickOffSVA(
    llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
    const llvm::GCNSubtarget &ST) const {
  const bool NeedsFSInstall = !ST.hasArchitectedFlatScratch();
  std::optional<uint8_t> FSSaveLane;
  auto FSLoLane = Specs.findArgumentLane(FLAT_SCRATCH);
  if (NeedsFSInstall) {
    FSSaveLane = Specs.getScratchSpillLane();
    if (!FSSaveLane || FSLoLane == Specs.argument_lane_end())
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_GENERIC_ERROR(
          "VGPRStateValueArrayStorage::pickOffSVA on absolute-FS target "
          "needs both ScratchSpillLane and a FLAT_SCRATCH SA lane in the "
          "SVA layout to trampoline FLAT_SCR through."));
  }
  // 1. Copy V0 (which arrives holding the SVA) → StorageVGPR (all lanes).
  //    After this, StorageVGPR holds the SVA and V0 still holds the SVA.
  llvm::MachineBasicBlock::iterator AfterMove = createSCCSafeSequenceOfMIs(
      MI.getIterator(),
      [&](llvm::MachineBasicBlock &IPMBB, const llvm::TargetInstrInfo &TII) {
        emitMoveFromVGPRToVGPR(IPMBB, llvm::AMDGPU::VGPR0, StorageVGPR, false);
        emitExecMaskFlip(IPMBB);
        emitMoveFromVGPRToVGPR(IPMBB, llvm::AMDGPU::VGPR0, StorageVGPR, false);
        emitExecMaskFlip(IPMBB);
      });
  // 2. Stash SGPR0 in the SVS's FramePointerRegSpillLane of StorageVGPR.
  emitMoveFromSGPRToVGPRLane(AfterMove, llvm::AMDGPU::SGPR0, StorageVGPR,
                             Specs.getFramePointerRegSpillLane(), false);
  // 3. Load the instrumentation SP from StorageVGPR's StackPointerStoreLane.
  emitMoveFromVGPRLaneToSGPR(AfterMove, StorageVGPR, llvm::AMDGPU::SGPR0,
                             Specs.getStackPointerStoreLane(), false);
  // 4. Restore V0 (all lanes) from [SGPR0-8] (caller's SP-8 emergency slot).
  llvm::MachineBasicBlock::iterator AfterLoad = createSCCSafeSequenceOfMIs(
      AfterMove,
      [&](llvm::MachineBasicBlock &IPMBB, const llvm::TargetInstrInfo &TII) {
        if (NeedsFSInstall) {
          emitMoveFromSGPRToVGPRLane(IPMBB, llvm::AMDGPU::FLAT_SCR_LO,
                                     StorageVGPR, *FSSaveLane, false);
          emitMoveFromSGPRToVGPRLane(IPMBB, llvm::AMDGPU::FLAT_SCR_HI,
                                     StorageVGPR, *FSSaveLane + 1, false);
          emitMoveFromVGPRLaneToSGPR(IPMBB, StorageVGPR,
                                     llvm::AMDGPU::FLAT_SCR_LO,
                                     FSLoLane->second, false);
          emitMoveFromVGPRLaneToSGPR(IPMBB, StorageVGPR,
                                     llvm::AMDGPU::FLAT_SCR_HI,
                                     FSLoLane->second + 1, false);
        }
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            IPMBB, llvm::AMDGPU::SGPR0, llvm::AMDGPU::VGPR0);
        emitExecMaskFlip(IPMBB);
        emitLoadFromEmergencyVGPRScratchSpillLocation(
            IPMBB, llvm::AMDGPU::SGPR0, llvm::AMDGPU::VGPR0);
        emitExecMaskFlip(IPMBB);
        if (NeedsFSInstall) {
          emitMoveFromVGPRLaneToSGPR(IPMBB, StorageVGPR,
                                     llvm::AMDGPU::FLAT_SCR_LO, *FSSaveLane,
                                     false);
          emitMoveFromVGPRLaneToSGPR(IPMBB, StorageVGPR,
                                     llvm::AMDGPU::FLAT_SCR_HI,
                                     *FSSaveLane + 1, false);
        }
      });
  emitWaitCnt(AfterLoad);
  // 5. Restore SGPR0 from StorageVGPR's FramePointerRegSpillLane.
  emitMoveFromVGPRLaneToSGPR(AfterLoad, StorageVGPR, llvm::AMDGPU::SGPR0,
                             Specs.getFramePointerRegSpillLane(), false);
}

// ---- SVA-lane reads for scheme SGPR bootstrap -------------------------------

/// Read the instrumentation SP from \p SrcVGPR 's
/// \c StackPointerStoreLane into \p StackPointer .
static void
loadStackPointerFromSVALanes(llvm::MachineBasicBlock::iterator Iter,
                             llvm::MCRegister SrcVGPR,
                             llvm::MCRegister StackPointer,
                             const StateValueArraySpecs &Specs) {
  emitMoveFromVGPRLaneToSGPR(Iter, SrcVGPR, StackPointer,
                             Specs.getStackPointerStoreLane(),
                             /*KillSource=*/false);
}

/// Read the wave FS_LO / FS_HI from \p SrcVGPR 's \c FLAT_SCRATCH SA lanes
/// into \p FSLo / \p FSHi.
static void loadFlatScratchFromSVALanes(llvm::MachineBasicBlock::iterator Iter,
                                        llvm::MCRegister SrcVGPR,
                                        llvm::MCRegister FSLo,
                                        llvm::MCRegister FSHi,
                                        const StateValueArraySpecs &Specs,
                                        const char *Context) {
  auto FSLoLane = Specs.findArgumentLane(FLAT_SCRATCH);
  if (FSLoLane == Specs.argument_lane_end())
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "{0}: SVA layout has no FLAT_SCRATCH argument lane; cannot load "
        "FS_LO / FS_HI shadow SGPRs.",
        Context)));
  if (StateValueArraySpecs::getArgumentLaneSize(FLAT_SCRATCH) != 2)
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "{0}: FLAT_SCRATCH SA is expected to span exactly 2 SVA lanes; "
        "layout reports {1}. Aborting to avoid reading a mis-aligned "
        "FS_HI.",
        Context,
        StateValueArraySpecs::getArgumentLaneSize(FLAT_SCRATCH))));
  const uint8_t Start = FSLoLane->second;
  emitMoveFromVGPRLaneToSGPR(Iter, SrcVGPR, FSLo, Start,
                             /*KillSource=*/false);
  emitMoveFromVGPRLaneToSGPR(Iter, SrcVGPR, FSHi, Start + 1,
                             /*KillSource=*/false);
}

void TwoAGPRValueStorage::handOffSVA(llvm::MachineInstr &MI,
                                     const StateValueArraySpecs &,
                                     const llvm::GCNSubtarget &) const {
  emitCodeToLoadSVA(MI, llvm::AMDGPU::VGPR0);
}

void TwoAGPRValueStorage::pickOffSVA(llvm::MachineInstr &MI,
                                     const StateValueArraySpecs &,
                                     const llvm::GCNSubtarget &) const {
  emitCodeToStoreSVA(MI, llvm::AMDGPU::VGPR0);
}

void AGPRWithThreeSGPRSValueStorage::handOffSVA(
    llvm::MachineInstr &MI, const StateValueArraySpecs &,
    const llvm::GCNSubtarget &) const {
  emitCodeToLoadSVA(MI, llvm::AMDGPU::VGPR0);
}

void AGPRWithThreeSGPRSValueStorage::pickOffSVA(
    llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
    const llvm::GCNSubtarget &) const {
  llvm::MachineBasicBlock::iterator Iter = MI.getIterator();
  loadStackPointerFromSVALanes(Iter, llvm::AMDGPU::VGPR0, StackPointer, Specs);
  loadFlatScratchFromSVALanes(Iter, llvm::AMDGPU::VGPR0, FlatScratchSGPRLow,
                              FlatScratchSGPRHigh, Specs,
                              "AGPRWithThreeSGPRSValueStorage::pickOffSVA");
  emitCodeToStoreSVA(MI, llvm::AMDGPU::VGPR0);
}

void SpilledWithThreeSGPRsValueStorage::handOffSVA(
    llvm::MachineInstr &MI, const StateValueArraySpecs &,
    const llvm::GCNSubtarget &) const {
  emitCodeToLoadSVA(MI, llvm::AMDGPU::VGPR0);
}

void SpilledWithThreeSGPRsValueStorage::pickOffSVA(
    llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
    const llvm::GCNSubtarget &) const {
  llvm::MachineBasicBlock::iterator Iter = MI.getIterator();
  loadStackPointerFromSVALanes(Iter, llvm::AMDGPU::VGPR0, StackPointer, Specs);
  loadFlatScratchFromSVALanes(Iter, llvm::AMDGPU::VGPR0, FlatScratchSGPRLow,
                              FlatScratchSGPRHigh, Specs,
                              "SpilledWithThreeSGPRsValueStorage::pickOffSVA");
  emitCodeToStoreSVA(MI, llvm::AMDGPU::VGPR0);
}

void SpilledWithOneSGPRsValueStorage::handOffSVA(
    llvm::MachineInstr &MI, const StateValueArraySpecs &,
    const llvm::GCNSubtarget &) const {
  emitCodeToLoadSVA(MI, llvm::AMDGPU::VGPR0);
}

void SpilledWithOneSGPRsValueStorage::pickOffSVA(
    llvm::MachineInstr &MI, const StateValueArraySpecs &Specs,
    const llvm::GCNSubtarget &) const {
  llvm::MachineBasicBlock::iterator Iter = MI.getIterator();
  loadStackPointerFromSVALanes(Iter, llvm::AMDGPU::VGPR0, StackPointer, Specs);
  emitCodeToStoreSVA(MI, llvm::AMDGPU::VGPR0);
}

void getSupportedSVAStorageList(
    const llvm::GCNSubtarget &ST,
    llvm::SmallVectorImpl<StateValueArrayStorage::StorageKind>
        &SupportedStorageKinds) {
  // Storage kinds are appended in preference order (lower-indexed entries
  // are preferred more). SVS_SINGLE_VGPR is always supported and most
  // preferred, so it heads the list and falls through the subtarget filter
  // unconditionally.
  for (auto SK :
       {StateValueArrayStorage::SVS_SINGLE_VGPR,
        StateValueArrayStorage::SVS_TWO_AGPRs,
        StateValueArrayStorage::SVS_SINGLE_AGPR_WITH_THREE_SGPRS_pre_gfx908,
        StateValueArrayStorage::SVS_SPILLED_WITH_THREE_SGPRS_absolute_fs,
        StateValueArrayStorage::SVS_SPILLED_WITH_ONE_SGPR_architected_fs}) {
    if (StateValueArrayStorage::isSupportedOnSubTarget(SK, ST))
      SupportedStorageKinds.push_back(SK);
  }
}

} // namespace luthier