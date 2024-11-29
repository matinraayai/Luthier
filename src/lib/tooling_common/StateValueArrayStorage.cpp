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

/// Swaps the value between \p ScrSGPR and \p DestSGPR by inserting 3
/// <tt>S_XOR_B32</tt>s before \p InsertionPoint
static void buildSGPRSwap(llvm::MachineBasicBlock::iterator InsertionPoint,
                          llvm::MCRegister SrcSGPR, llvm::MCRegister DestSGPR) {
  auto &MBB = *InsertionPoint->getParent();
  const auto *TII = MBB.getParent()->getSubtarget().getInstrInfo();
  llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::S_XOR_B32), SrcSGPR)
      .addReg(SrcSGPR)
      .addReg(DestSGPR);
  llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::S_XOR_B32), DestSGPR)
      .addReg(SrcSGPR)
      .addReg(DestSGPR, llvm::RegState::Kill);
  llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                TII->get(llvm::AMDGPU::S_XOR_B32), SrcSGPR)
      .addReg(SrcSGPR)
      .addReg(DestSGPR);
}

/// Generates a set of MBBs that ensures the \c SCC bit does not get clobbered
/// due to the sequence of instructions built by \p MIBuilder before the
/// insertion point \p MI
/// This is a common pattern used when loading and storing the state value
/// array that allows flipping the exec mask without clobbering the
/// \c SCC bit and not requiring temporary registers
static void createSCCSafeSequenceOfMIs(
    llvm::MachineInstr &MI,
    const std::function<void(llvm::MachineBasicBlock &,
                             const llvm::TargetInstrInfo &)> &MIBuilder) {
  auto &MBB = *MI.getParent();
  auto &MF = *MBB.getParent();
  const auto &TII = *MF.getSubtarget().getInstrInfo();
  // First we add an SCC1 branch before the MI
  auto Builder = llvm::BuildMI(MBB, MI, llvm::DebugLoc(),
                               TII.get(llvm::AMDGPU::S_CBRANCH_SCC1));
  // We then split MBB into two at the newly inserted branch instruction;
  // The MBB is the entry block while the newly created block is the exit
  // block of this code snippet
  llvm::MachineBasicBlock &ExitBlock = *MBB.splitAt(*Builder);
  MBB.removeSuccessor(&ExitBlock);
  // Create the SCC0MBB, which will house the code for when SCC=0
  // It comes right after the entry block because the CBRANCH is not taken
  auto *SCC0MBB = MF.CreateMachineBasicBlock();
  MF.insert(ExitBlock.getIterator(), SCC0MBB);
  MBB.addSuccessor(SCC0MBB);
  SCC0MBB->addSuccessor(&ExitBlock);
  // Create the SCC1MBB, which will house the code for when SCC=1
  // It comes after SCC0MBB, and falls right through to the exit block
  auto *SCC1MBB = MF.CreateMachineBasicBlock();
  MF.insert(ExitBlock.getIterator(), SCC1MBB);
  MBB.addSuccessor(SCC1MBB);
  SCC1MBB->addSuccessor(&ExitBlock);
  // Make the S_CBRANCH_SCC1 instruction jump to the SCC1MBB branch
  Builder.addMBB(SCC1MBB);
  // Now that we've created the basic blocks, and we've implicitly saved the
  // SCC value by branching, we can now safely carry out operations that
  // clobber the SCC bit
  for (auto *SCMBB : {SCC0MBB, SCC1MBB}) {
    // Insert the user-defined instructions
    MIBuilder(*SCMBB, TII);
    // If this is the SCC0 block, we need to set SCC to zero.
    // We also need to do an unconditional branch to the exit block
    if (SCMBB == SCC0MBB) {
      llvm::BuildMI(*SCMBB, SCMBB->end(), llvm::DebugLoc(),
                    TII.get(llvm::AMDGPU::S_CMP_EQ_I32))
          .addImm(0)
          .addImm(1);
      llvm::BuildMI(*SCMBB, SCMBB->end(), llvm::DebugLoc(),
                    TII.get(llvm::AMDGPU::S_BRANCH))
          .addMBB(&ExitBlock);
    } else {
      // If this is the SCC1 block, we need to set SCC to one.
      llvm::BuildMI(*SCMBB, SCMBB->end(), llvm::DebugLoc(),
                    TII.get(llvm::AMDGPU::S_CMP_EQ_I32))
          .addImm(0)
          .addImm(0);
    }
  }
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                VGPRStateValueArrayStorage &SrcSVS,
                                VGPRStateValueArrayStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Do a move on the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_MOV_B32_e32), TargetSVS.StorageVGPR)
        .addReg(SrcSVS.StorageVGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Do a move on the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_MOV_B32_e32), TargetSVS.StorageVGPR)
        .addReg(SrcSVS.StorageVGPR, llvm::RegState::Kill);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                VGPRStateValueArrayStorage &SrcSVS,
                                SingleAGPRStateValueArrayStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Do a move on the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(SrcSVS.StorageVGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Do a move on the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(SrcSVS.StorageVGPR, llvm::RegState::Kill);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                VGPRStateValueArrayStorage &SrcSVS,
                                TwoAGPRValueStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Do a move on the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(SrcSVS.StorageVGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Do a move on the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(SrcSVS.StorageVGPR, llvm::RegState::Kill);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                VGPRStateValueArrayStorage &SrcSVS,
                                AGPRWithThreeSGPRSValueStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Do a move on the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(SrcSVS.StorageVGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Do a move on the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(SrcSVS.StorageVGPR, llvm::RegState::Kill);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                VGPRStateValueArrayStorage &SrcSVS,
                                SpilledWithThreeSGPRsValueStorage &TargetSVS) {
  const auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
  for (const auto &[PhysReg, SVSSaveSGPR] :
       {std::pair{llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh},
        {llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow},
        {llvm::AMDGPU::SGPR32, TargetSVS.EmergencyVGPRSpillSlotOffset}}) {
    auto StoreSlot =
        stateValueArray::getInstrumentationStackFrameLaneIdStoreSlot(PhysReg);
    LUTHIER_REPORT_FATAL_ON_ERROR(StoreSlot.takeError());
    llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_READLANE_B32), SVSSaveSGPR)
        .addReg(SrcSVS.StorageVGPR)
        .addImm(*StoreSlot);
  }
  // Temporarily swap the FS Hi and FS Lo of the app with the storage to
  // spill the SVA
  buildSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh);

  buildSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow);

  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill the SVA on the active register
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcSVS.StorageVGPR)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill the SVA on the inactive register
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcSVS.StorageVGPR)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });

  // swap the FS Hi and FS Lo of the app back to its correct place
  buildSGPRSwap(MI, TargetSVS.FlatScratchSGPRHigh, llvm::AMDGPU::FLAT_SCR_HI);

  buildSGPRSwap(MI, TargetSVS.FlatScratchSGPRLow, llvm::AMDGPU::FLAT_SCR_LO);
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                VGPRStateValueArrayStorage &SrcSVS,
                                SpilledWithOneSGPRsValueStorage &TargetSVS) {
  const auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
  // Spill the
  auto StoreSlot = stateValueArray::getInstrumentationStackFrameLaneIdStoreSlot(
      llvm::AMDGPU::SGPR32);
  LUTHIER_REPORT_FATAL_ON_ERROR(StoreSlot.takeError());
  llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_READLANE_B32),
                TargetSVS.EmergencyVGPRSpillSlotOffset)
      .addReg(SrcSVS.StorageVGPR)
      .addImm(*StoreSlot);

  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill the SVA on the active register
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcSVS.StorageVGPR)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill the SVA on the inactive register
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcSVS.StorageVGPR)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                SingleAGPRStateValueArrayStorage &SrcSVS,
                                VGPRStateValueArrayStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Do a move on the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  TargetSVS.StorageVGPR)
        .addReg(SrcSVS.StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Do a move on the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  TargetSVS.StorageVGPR)
        .addReg(SrcSVS.StorageAGPR);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                SingleAGPRStateValueArrayStorage &SrcSVS,
                                SingleAGPRStateValueArrayStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Do a move on the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_MOV_B32_e32), TargetSVS.StorageAGPR)
        .addReg(SrcSVS.StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Do a move on the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_MOV_B32_e32), TargetSVS.StorageAGPR)
        .addReg(SrcSVS.StorageAGPR);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                SingleAGPRStateValueArrayStorage &SrcSVS,
                                SpilledWithThreeSGPRsValueStorage &TargetSVS) {
  const auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
  for (const auto &[PhysReg, SVSSaveSGPR] :
       {std::pair{llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh},
        {llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow},
        {llvm::AMDGPU::SGPR32, TargetSVS.EmergencyVGPRSpillSlotOffset}}) {
    auto StoreSlot =
        stateValueArray::getInstrumentationStackFrameLaneIdStoreSlot(PhysReg);
    LUTHIER_REPORT_FATAL_ON_ERROR(StoreSlot.takeError());
    llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_READLANE_B32), SVSSaveSGPR)
        .addReg(SrcSVS.StorageAGPR)
        .addImm(*StoreSlot);
  }
  // Temporarily swap the FS Hi and FS Lo of the app with the storage to
  // spill the SVA
  buildSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh);

  buildSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow);

  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill the SVA on the active register
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcSVS.StorageAGPR)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill the SVA on the inactive register
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcSVS.StorageAGPR)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });

  // swap the FS Hi and FS Lo of the app back to its correct place
  buildSGPRSwap(MI, TargetSVS.FlatScratchSGPRHigh, llvm::AMDGPU::FLAT_SCR_HI);

  buildSGPRSwap(MI, TargetSVS.FlatScratchSGPRLow, llvm::AMDGPU::FLAT_SCR_LO);
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                SingleAGPRStateValueArrayStorage &SrcSVS,
                                SpilledWithOneSGPRsValueStorage &TargetSVS) {
  const auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
  // Spill the
  auto StoreSlot = stateValueArray::getInstrumentationStackFrameLaneIdStoreSlot(
      llvm::AMDGPU::SGPR32);
  LUTHIER_REPORT_FATAL_ON_ERROR(StoreSlot.takeError());
  llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_READLANE_B32),
                TargetSVS.EmergencyVGPRSpillSlotOffset)
      .addReg(SrcSVS.StorageAGPR)
      .addImm(*StoreSlot);

  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill the SVA on the active register
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcSVS.StorageAGPR)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill the SVA on the inactive register
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcSVS.StorageAGPR)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                TwoAGPRValueStorage &SrcSVS,
                                VGPRStateValueArrayStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Do a move on the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  TargetSVS.StorageVGPR)
        .addReg(SrcSVS.StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Do a move on the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  TargetSVS.StorageVGPR)
        .addReg(SrcSVS.StorageAGPR);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                TwoAGPRValueStorage &SrcSVS,
                                TwoAGPRValueStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill V0 on the active lanes to the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  SrcSVS.TempAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    // Write V0 to TargetSVS storage AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read V0 back from the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.TempAGPR);

    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  SrcSVS.TempAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    // Write V0 to TargetSVS storage AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read V0 back from the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.TempAGPR);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
};

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                TwoAGPRValueStorage &SrcSVS,
                                AGPRWithThreeSGPRSValueStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill V0 on the active lanes to the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  SrcSVS.TempAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    // Write V0 to TargetSVS storage AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read V0 back from the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.TempAGPR);

    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  SrcSVS.TempAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    // Write V0 to TargetSVS storage AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read V0 back from the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.TempAGPR);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
};

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                TwoAGPRValueStorage &SrcSVS,
                                SpilledWithThreeSGPRsValueStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill V0 on the active lanes to the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  SrcSVS.TempAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  SrcSVS.TempAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });

  const auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();
  for (const auto &[PhysReg, SVSSaveSGPR] :
       {std::pair{llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh},
        {llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow},
        {llvm::AMDGPU::SGPR32, TargetSVS.EmergencyVGPRSpillSlotOffset}}) {
    auto StoreSlot =
        stateValueArray::getInstrumentationStackFrameLaneIdStoreSlot(PhysReg);
    LUTHIER_REPORT_FATAL_ON_ERROR(StoreSlot.takeError());
    llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_READLANE_B32), SVSSaveSGPR)
        .addReg(llvm::AMDGPU::VGPR0)
        .addImm(*StoreSlot);
  }
  // Temporarily swap the FS Hi and FS Lo of the app with the storage to
  // spill the SVA
  buildSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh);

  buildSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow);

  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill the SVA on the active register
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill the SVA on the inactive register
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });

  // swap the FS Hi and FS Lo of the app back to its correct place
  buildSGPRSwap(MI, TargetSVS.FlatScratchSGPRHigh, llvm::AMDGPU::FLAT_SCR_HI);

  buildSGPRSwap(MI, TargetSVS.FlatScratchSGPRLow, llvm::AMDGPU::FLAT_SCR_LO);

  // Swap the V0 with Temp AGPR
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Read V0 back from the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.TempAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Read V0 back from the SrcSVS AGPR temp
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.TempAGPR);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                AGPRWithThreeSGPRSValueStorage &SrcSVS,
                                VGPRStateValueArrayStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Do a move on the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  TargetSVS.StorageVGPR)
        .addReg(SrcSVS.StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Do a move on the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageVGPR)
        .addReg(SrcSVS.StorageAGPR, llvm::RegState::Kill);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                AGPRWithThreeSGPRSValueStorage &SrcSVS,
                                TwoAGPRValueStorage &TargetSVS) {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill V0 on the active lanes to the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.TempAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    // Write V0 to TargetSVS storage AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read V0 back from the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.TempAGPR);

    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.TempAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    // Write V0 to TargetSVS storage AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    // Read V0 back from the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.TempAGPR);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                AGPRWithThreeSGPRSValueStorage &SrcSVS,
                                AGPRWithThreeSGPRSValueStorage &TargetSVS) {
  // Move the SGPRs first
  const auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();

  llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32), TargetSVS.FlatScratchSGPRHigh)
      .addReg(SrcSVS.FlatScratchSGPRHigh);

  llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32), TargetSVS.FlatScratchSGPRLow)
      .addReg(SrcSVS.FlatScratchSGPRLow);

  llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32),
                TargetSVS.EmergencyVGPRSpillSlotOffset)
      .addReg(SrcSVS.EmergencyVGPRSpillSlotOffset);

  // Temporarily swap the FS Hi and FS Lo of the app with the storage to
  // spill the SVA
  buildSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh);

  buildSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow);

  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill V0 on the active lanes to the emergency spill slot
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR),
                  llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);

    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64),
                  TargetSVS.StorageAGPR)
        .addReg(llvm::AMDGPU::VGPR0);
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR),
                  llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });
  // swap the FS Hi and FS Lo of the app back
  buildSGPRSwap(MI, TargetSVS.FlatScratchSGPRHigh, llvm::AMDGPU::FLAT_SCR_HI);

  buildSGPRSwap(MI, TargetSVS.FlatScratchSGPRLow, llvm::AMDGPU::FLAT_SCR_LO);
}


static void emitCodeToSwitchSVS(llvm::MachineInstr &MI,
                                AGPRWithThreeSGPRSValueStorage &SrcSVS,
                                SpilledWithThreeSGPRsValueStorage &TargetSVS) {
  // Move the SGPRs first
  const auto &TII = *MI.getMF()->getSubtarget().getInstrInfo();

  llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32), TargetSVS.FlatScratchSGPRHigh)
      .addReg(SrcSVS.FlatScratchSGPRHigh);

  llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32), TargetSVS.FlatScratchSGPRLow)
      .addReg(SrcSVS.FlatScratchSGPRLow);

  llvm::BuildMI(*MI.getParent(), MI, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32),
                TargetSVS.EmergencyVGPRSpillSlotOffset)
      .addReg(SrcSVS.EmergencyVGPRSpillSlotOffset);

  // Temporarily swap the FS Hi and FS Lo of the app with the storage to
  // spill the SVA
  buildSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_HI, TargetSVS.FlatScratchSGPRHigh);

  buildSGPRSwap(MI, llvm::AMDGPU::FLAT_SCR_LO, TargetSVS.FlatScratchSGPRLow);

  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill V0 on the active lanes to the emergency spill slot
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    // Spill V0 to the emergency SVS spill slot
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR),
                  llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill V0 on the inactive lanes to the SrcSVS AGPR storage
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    // Read the SrcSVS AGPR to V0
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64),
                  llvm::AMDGPU::VGPR0)
        .addReg(SrcSVS.StorageAGPR);
    // Spill V0 to the emergency SVS spill slot
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR),
                  llvm::AMDGPU::VGPR0)
        .addReg(TargetSVS.EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    // Flip the exec mask back
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });
  // swap the FS Hi and FS Lo of the app back
  buildSGPRSwap(MI, TargetSVS.FlatScratchSGPRHigh, llvm::AMDGPU::FLAT_SCR_HI);

  buildSGPRSwap(MI, TargetSVS.FlatScratchSGPRLow, llvm::AMDGPU::FLAT_SCR_LO);
}


void VGPRStateValueArrayStorage::emitCodeToSwitchSVS(
    llvm::MachineInstr &MI, StateValueArrayStorage &TargetSVS) const {}

void TwoAGPRValueStorage::emitCodeToLoadSVA(llvm::MachineInstr &MI,
                                            llvm::MCRegister DestVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill the Dest VGPR in the active lanes to the temp AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64), TempAGPR)
        .addReg(DestVGPR);
    // Copy the state value from AGPR to the dest VGPR in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), DestVGPR)
        .addReg(StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill the Dest VGPR in the remaining non-active lanes to the temp
    // AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64), TempAGPR)
        .addReg(DestVGPR);
    // Copy the state value from AGPR to the dest VGPR in the non-active
    // lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), DestVGPR)
        .addReg(StorageAGPR, llvm::RegState::Kill);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

void TwoAGPRValueStorage::emitCodeToStoreSVA(llvm::MachineInstr &MI,
                                             llvm::MCRegister SrcVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    // Spill the Src VGPR in the active lanes to the storage AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64), StorageAGPR)
        .addReg(SrcVGPR);
    // Restore the temp AGPR contents into the src VGPR in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), SrcVGPR)
        .addReg(TempAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Spill the Src VGPR in the inactive lanes to the storage AGPR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_WRITE_B32_e64), StorageAGPR)
        .addReg(SrcVGPR);
    // Restore the temp AGPR contents into the src VGPR in inactive laness
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), SrcVGPR)
        .addReg(TempAGPR, llvm::RegState::Kill);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
  });
}

void AGPRWithThreeSGPRSValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// FS swap
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                  FlatScratchSGPRLow);
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                  FlatScratchSGPRHigh);
    /// Spill the DestVGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    /// Restore the state value array from the storage AGPR to the dest VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), DestVGPR)
        .addReg(StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    /// Spill the DestVGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DestVGPR, llvm::RegState::Kill)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    /// Restore the state value array from the storage AGPR to the dest VPGR
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), DestVGPR)
        .addReg(StorageAGPR);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });
}

void AGPRWithThreeSGPRSValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// FS swap
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                  FlatScratchSGPRLow);
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                  FlatScratchSGPRHigh);
    /// Spill the Src VGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    /// Restore the app VGPR from the storage AGPR in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), SrcVGPR)
        .addReg(StorageAGPR);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);

    /// Spill the Src to the emergency spill slot in the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    /// Restore the app VGPR from the storage AGPR in the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_ACCVGPR_READ_B32_e64), SrcVGPR)
        .addReg(StorageAGPR);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });
}

void SpilledWithThreeSGPRsValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// FS swap
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                  FlatScratchSGPRLow);
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                  FlatScratchSGPRHigh);
    /// Spill the DestVGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    /// Restore the state value array from its fixed storage to the dest VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    /// Spill the DestVGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DestVGPR, llvm::RegState::Kill)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    /// Restore the state value array from its fixed storage to the dest VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });
}

void SpilledWithThreeSGPRsValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// FS swap
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_LO,
                  FlatScratchSGPRLow);
    buildSGPRSwap(InsertionPointMBB.end(), llvm::AMDGPU::FLAT_SCR_HI,
                  FlatScratchSGPRHigh);
    /// Spill the Src to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    /// Restore the app VGPR from its fixed storage to the src VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);

    /// Spill the Src to the emergency spill slot in the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    /// Restore the app VGPR from its fixed storage to the src VGPR
    /// in the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });
}

void SpilledWithOneSGPRsValueStorage::emitCodeToLoadSVA(
    llvm::MachineInstr &MI, llvm::MCRegister DestVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// Spill the DestVGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    /// Restore the state value array from its fixed storage to the dest VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    /// Spill the DestVGPR to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(DestVGPR, llvm::RegState::Kill)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    /// Restore the state value array from its fixed storage to the dest VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), DestVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });
}
void SpilledWithOneSGPRsValueStorage::emitCodeToStoreSVA(
    llvm::MachineInstr &MI, llvm::MCRegister SrcVGPR) const {
  createSCCSafeSequenceOfMIs(MI, [&](llvm::MachineBasicBlock &InsertionPointMBB,
                                     const llvm::TargetInstrInfo &TII) {
    /// Spill the Src to the emergency spill slot in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    /// Restore the app VGPR from its fixed storage to the src VGPR
    /// in the active lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);

    /// Spill the Src to the emergency spill slot in the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_STORE_DWORD_SADDR))
        .addReg(SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-8)
        .addImm(0);
    /// Restore the app VGPR from its fixed storage to the src VGPR
    /// in the inactive lanes
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::SCRATCH_LOAD_DWORD_SADDR), SrcVGPR)
        .addReg(EmergencyVGPRSpillSlotOffset)
        .addImm(-4)
        .addImm(0);
    // Flip the exec mask to its original value
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_NOT_B64), llvm::AMDGPU::EXEC)
        .addReg(llvm::AMDGPU::EXEC, llvm::RegState::Kill);
    // Wait on the memory operation to complete
    llvm::BuildMI(InsertionPointMBB, InsertionPointMBB.end(), llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::S_WAITCNT))
        .addImm(0);
  });
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