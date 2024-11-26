//===-- PreKernelEmitter.cpp ----------------------------------------------===//
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
/// This file implements the Pre-kernel emitter.
//===----------------------------------------------------------------------===//
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "tooling_common/LRStateValueStorageAndLoadLocations.hpp"
#include "tooling_common/PrePostAmbleEmitter.hpp"
#include "tooling_common/StateValueArraySpecs.hpp"
#include <GCNSubtarget.h>
#include <SIMachineFunctionInfo.h>

namespace luthier {

/// A set of \c LiftedKernelArgumentManager::KernelArgumentType that have
/// a \c llvm::AMDGPUFunctionArgInfo::PreloadedValue equivalent (i.e.
/// can/must be preloaded into register values)
static const llvm::SmallDenseMap<KernelArgumentType,
                                 llvm::AMDGPUFunctionArgInfo::PreloadedValue>
    ToLLVMRegArgMap{
        {PRIVATE_SEGMENT_BUFFER,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::PRIVATE_SEGMENT_BUFFER},
        {DISPATCH_PTR,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::DISPATCH_PTR},
        {QUEUE_PTR, llvm::AMDGPUFunctionArgInfo::PreloadedValue::QUEUE_PTR},
        {KERNARG_SEGMENT_PTR,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::KERNARG_SEGMENT_PTR},
        {DISPATCH_ID, llvm::AMDGPUFunctionArgInfo::PreloadedValue::DISPATCH_ID},
        {FLAT_SCRATCH,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::FLAT_SCRATCH_INIT},
        {PRIVATE_SEGMENT_SIZE,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::PRIVATE_SEGMENT_SIZE},
        {PRIVATE_SEGMENT_WAVE_BYTE_OFFSET,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::
             PRIVATE_SEGMENT_WAVE_BYTE_OFFSET},
        {WORK_ITEM_X,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::WORKITEM_ID_X},
        {WORK_ITEM_Y,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::WORKITEM_ID_Y},
        {WORK_ITEM_Z,
         llvm::AMDGPUFunctionArgInfo::PreloadedValue::WORKITEM_ID_Z},
    };

static llvm::MCRegister
getArgReg(const llvm::MachineFunction &MF,
          llvm::AMDGPUFunctionArgInfo::PreloadedValue ArgReg) {
  return MF.getInfo<llvm::SIMachineFunctionInfo>()->getPreloadedReg(ArgReg);
}

static bool doesLRRequireSVA(const LiftedRepresentation &LR,
                             const hsa::LoadedCodeObject &LCO,
                             const FunctionPreambleDescriptor &PKInfo) {
  for (const auto &[FuncSymbol, MF] : LR.functions()) {
    if (MF->getFunction().getCallingConv() ==
        llvm::CallingConv::AMDGPU_KERNEL) {
      auto &SVAKernelInfo = PKInfo.Kernels.at(MF);
      if (SVAKernelInfo.usesSVA()) {
        return true;
      }
    } else {
      auto &SVADeviceFuncInfo = PKInfo.DeviceFunctions.at(MF);
      if (SVADeviceFuncInfo.UsesStateValueArray) {
        return true;
      }
    }
  }
  return false;
}

/// \returns the kernel arguments of the \p MF and the SGPR it is positioned
/// in
static llvm::DenseMap<llvm::AMDGPUFunctionArgInfo::PreloadedValue,
                      llvm::MCRegister>
getSGPRArgumentPositions(const llvm::MachineFunction &MF) {
  llvm::DenseMap<llvm::AMDGPUFunctionArgInfo::PreloadedValue, llvm::MCRegister>
      SGPRKernelArguments;

  std::initializer_list<llvm::AMDGPUFunctionArgInfo::PreloadedValue> SGPRArgs{
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::PRIVATE_SEGMENT_BUFFER,
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::DISPATCH_PTR,
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::QUEUE_PTR,
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::KERNARG_SEGMENT_PTR,
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::DISPATCH_ID,
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::FLAT_SCRATCH_INIT,
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::PRIVATE_SEGMENT_SIZE,
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::
          PRIVATE_SEGMENT_WAVE_BYTE_OFFSET,
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::WORKITEM_ID_X,
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::WORKITEM_ID_Y,
      llvm::AMDGPUFunctionArgInfo::PreloadedValue::WORKITEM_ID_Z};

  for (const auto &PreloadVal : SGPRArgs) {
    SGPRKernelArguments.insert({PreloadVal, getArgReg(MF, PreloadVal)});
  }
  return SGPRKernelArguments;
}

#define DEFINE_ENABLE_SGPR_ARGUMENT_WITHOUT_TRI(                               \
    KernArgName, AMDGPUKernArgEnum, MFIEnableFunc)                             \
  static void enable##KernArgName(llvm::SIMachineFunctionInfo &MFI) {          \
    if (!MFI.getPreloadedReg(AMDGPUKernArgEnum)) {                             \
      MFI.MFIEnableFunc();                                                     \
    }                                                                          \
  }

#define DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(KernArgName, AMDGPUKernArgEnum,   \
                                             MFIEnableFunc)                    \
  static void enable##KernArgName(llvm::SIMachineFunctionInfo &MFI,            \
                                  const llvm::SIRegisterInfo &TRI) {           \
    if (!MFI.getPreloadedReg(AMDGPUKernArgEnum)) {                             \
      MFI.MFIEnableFunc(TRI);                                                  \
    }                                                                          \
  }

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(
    PrivateSegmentBuffer, llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER,
    addPrivateSegmentBuffer);

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(
    FlatScratchInit, llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT,
    addFlatScratchInit);

DEFINE_ENABLE_SGPR_ARGUMENT_WITHOUT_TRI(
    PrivateSegmentWaveOffset,
    llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET,
    addPrivateSegmentWaveByteOffset);

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(
    KernArg, llvm::AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR,
    addKernargSegmentPtr)

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(DispatchID,
                                     llvm::AMDGPUFunctionArgInfo::DISPATCH_ID,
                                     addDispatchID)

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(DispatchPtr,
                                     llvm::AMDGPUFunctionArgInfo::DISPATCH_PTR,
                                     addDispatchPtr)

DEFINE_ENABLE_SGPR_ARGUMENT_WITH_TRI(QueuePtr,
                                     llvm::AMDGPUFunctionArgInfo::QUEUE_PTR,
                                     addQueuePtr)

static llvm::Error
emitCodeToSetupScratch(llvm::MachineInstr &EntryInstr,
                       llvm::MCRegister SVSStorageVGPR,
                       const hsa::md::Kernel::Metadata &KernelMD) {
  auto &MF = *EntryInstr.getMF();
  const auto &TII = *MF.getSubtarget().getInstrInfo();
  auto &MFI = *MF.getInfo<llvm::SIMachineFunctionInfo>();
  // First make a copy of S0 and S1/ FS_lo FS_hi in the state value
  // register
  auto SGPR0SpillSlot =
      stateValueArray::getFrameSpillSlotLaneId(llvm::AMDGPU::SGPR0);
  LUTHIER_RETURN_ON_ERROR(SGPR0SpillSlot.takeError());

  auto SGPR1SpillSlot =
      stateValueArray::getFrameSpillSlotLaneId(llvm::AMDGPU::SGPR1);
  LUTHIER_RETURN_ON_ERROR(SGPR1SpillSlot.takeError());

  auto SGPRFlatScrLoSpillSlot =
      stateValueArray::getFrameSpillSlotLaneId(llvm::AMDGPU::FLAT_SCR_LO);
  LUTHIER_RETURN_ON_ERROR(SGPRFlatScrLoSpillSlot.takeError());

  auto SGPRFlatScrHiSpillSlot =
      stateValueArray::getFrameSpillSlotLaneId(llvm::AMDGPU::FLAT_SCR_HI);
  LUTHIER_RETURN_ON_ERROR(SGPRFlatScrHiSpillSlot.takeError());

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(MFI.getPreloadedReg(
                  llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
              0, llvm::AMDGPU::sub0)
      .addImm(*SGPR0SpillSlot)
      .addReg(SVSStorageVGPR);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(MFI.getPreloadedReg(
                  llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
              0, llvm::AMDGPU::sub1)
      .addImm(*SGPR1SpillSlot)
      .addReg(SVSStorageVGPR);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(
          MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
          0, llvm::AMDGPU::sub0)
      .addImm(*SGPRFlatScrLoSpillSlot)
      .addReg(SVSStorageVGPR);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(
          MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
          0, llvm::AMDGPU::sub1)
      .addImm(*SGPRFlatScrHiSpillSlot)
      .addReg(SVSStorageVGPR);

  // Add the PSWO to SGPR0/its carry to SGPR1
  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_ADD_U32))
      .addReg(MFI.getPreloadedReg(
                  llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
              llvm::RegState::Define, llvm::AMDGPU::sub0)
      .addReg(MFI.getPreloadedReg(
                  llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
              llvm::RegState::Kill, llvm::AMDGPU::sub0)
      .addReg(MFI.getPreloadedReg(
          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET));

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_ADDC_U32))
      .addReg(MFI.getPreloadedReg(
                  llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
              llvm::RegState::Define, llvm::AMDGPU::sub1)
      .addReg(MFI.getPreloadedReg(
                  llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
              llvm::RegState::Kill, llvm::AMDGPU::sub1)
      .addImm(0);
  // Add the PSWO to FS_init_lo/its carry to FS_init_hi
  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_ADD_U32))
      .addReg(
          MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
          llvm::RegState::Define, llvm::AMDGPU::sub0)
      .addReg(
          MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
          llvm::RegState::Kill, llvm::AMDGPU::sub0)
      .addReg(MFI.getPreloadedReg(
          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET));
  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_ADDC_U32))
      .addReg(
          MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
          llvm::RegState::Define, llvm::AMDGPU::sub1)
      .addReg(
          MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
          llvm::RegState::Kill, llvm::AMDGPU::sub1)
      .addImm(0);

  unsigned int InstrumentationStackStart{0};
  if (KernelMD.UsesDynamicStack)
    llvm_unreachable("Not implemented");
  else {
    InstrumentationStackStart = KernelMD.PrivateSegmentFixedSize;
  }
  // Set s32 to be the maximum amount of stack requested by the hook
  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32), llvm::AMDGPU::SGPR32)
      .addImm(InstrumentationStackStart);
  // Set s33 to be zero
  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_MOV_B32), llvm::AMDGPU::SGPR33)
      .addImm(0);

  // Store frame registers in their slots
  for (const auto &[PhysReg, StoreSlot] :
       stateValueArray::getFrameStoreSlots()) {
    llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
        .addReg(PhysReg)
        .addImm(StoreSlot)
        .addReg(SVSStorageVGPR);
  }

  // Restore S0, S1, FS_init_lo, and FS_init_hi
  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_READLANE_B32), llvm::AMDGPU::SGPR0)
      .addReg(SVSStorageVGPR)
      .addImm(*SGPR0SpillSlot);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_READLANE_B32), llvm::AMDGPU::SGPR1)
      .addReg(SVSStorageVGPR)
      .addImm(*SGPR1SpillSlot);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_READLANE_B32))
      .addReg(
          MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
          llvm::RegState::Define, llvm::AMDGPU::sub0)
      .addReg(SVSStorageVGPR)
      .addImm(*SGPRFlatScrLoSpillSlot);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_READLANE_B32))
      .addReg(
          MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
          llvm::RegState::Define, llvm::AMDGPU::sub1)
      .addReg(SVSStorageVGPR)
      .addImm(*SGPRFlatScrHiSpillSlot);

  return llvm::Error::success();
}

static llvm::Error
emitCodeToStoreSGPRKernelArg(llvm::MachineInstr &InsertionPoint,
                             llvm::MCRegister SrcSGPR, llvm::MCRegister SVSVGPR,
                             int SpillSlotStart, int NumSlots,
                             bool KillAfterUse) {
  const auto &TRI = *InsertionPoint.getMF()->getSubtarget().getRegisterInfo();
  const auto &TII = *InsertionPoint.getMF()->getSubtarget().getInstrInfo();
  size_t Size = TRI.getRegSizeInBits(*TRI.getPhysRegBaseClass(SrcSGPR));
  auto &InsertionPointMBB = *InsertionPoint.getParent();
  if (Size == 32) {
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        NumSlots == 1, "Mismatch between number of SGPRs in the argument and "
                       "save slot lanes."));
    llvm::BuildMI(InsertionPointMBB, InsertionPoint, llvm::DebugLoc(),
                  TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSVGPR)
        .addReg(KillAfterUse ? llvm::RegState::Kill : 0)
        .addImm(SpillSlotStart)
        .addReg(SVSVGPR);
  } else {
    size_t NumChannels = Size / 32;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        NumSlots == NumChannels,
        "Mismatch between number of SGPRs in the argument and "
        "save slot lanes."));
    for (int i = 0; i < NumSlots; i++) {
      auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i);
      llvm::BuildMI(InsertionPointMBB, InsertionPoint, llvm::DebugLoc(),
                    TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSVGPR)
          .addReg(SrcSGPR, KillAfterUse ? llvm::RegState::Kill : 0, SubIdx)
          .addImm(SpillSlotStart)
          .addReg(SVSVGPR);
    }
  }
  return llvm::Error::success();
}

static void emitCodeToReturnSGPRArgsToOriginalPlace(
    const llvm::DenseMap<llvm::AMDGPUFunctionArgInfo::PreloadedValue,
                         llvm::MCRegister> &OriginalKernelArguments,
    llvm::MachineInstr &InsertionPoint) {
  auto &MF = *InsertionPoint.getMF();
  auto &MBB = *InsertionPoint.getParent();
  const auto &TRI = *MF.getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
  const auto &TII = *MF.getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
  for (const auto &[KernArg, OriginalArgReg] : OriginalKernelArguments) {
    size_t Size =
        TRI.getRegSizeInBits(*TRI.getPhysRegBaseClass(OriginalArgReg));
    if (Size == 32) {
      llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                    TII.get(llvm::AMDGPU::S_MOV_B32), OriginalArgReg)
          .addReg(getArgReg(MF, KernArg), llvm::RegState::Kill);
    } else if (Size == 64) {
      llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                    TII.get(llvm::AMDGPU::S_MOV_B64), OriginalArgReg)
          .addReg(getArgReg(MF, KernArg), llvm::RegState::Kill);
    } else {
      size_t NumChannels = Size / 32;
      for (int i = 0; i < NumChannels / 2; i++) {
        auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i, 2);
        llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::S_MOV_B64))
            .addReg(OriginalArgReg, llvm::RegState::Define, SubIdx)
            .addReg(getArgReg(MF, KernArg), llvm::RegState::Kill, SubIdx);
      }
    }
  }
}

llvm::Error PrePostAmbleEmitter::emit() {
  // First we need to figure out if we need to set up the state value array
  // at all
  auto LCO = SVLocations.getLCO();

  bool MustSetupSVA = doesLRRequireSVA(LR, LCO, PKInfo);

  if (MustSetupSVA) {
    for (auto &[FuncSymbol, MF] : LR.functions()) {
      if (MF->getFunction().getCallingConv() ==
              llvm::CallingConv::AMDGPU_KERNEL &&
          FuncSymbol->getLoadedCodeObject() == LCO) {
        auto &SVAInfo = PKInfo.Kernels.at(MF);
        auto EntryInstr = MF->begin()->begin();
        auto &EntryInstrSVS =
            SVLocations.getStorageIntervalsOfBasicBlock(*MF->begin())[0];
        auto &MFI = *MF->getInfo<llvm::SIMachineFunctionInfo>();
        auto &TRI = *MF->getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
        auto *TII = MF->getSubtarget<llvm::GCNSubtarget>().getInstrInfo();

        // Get the original position of all the reg arguments before they
        // are changed
        auto OriginalSGPRArgLocs = getSGPRArgumentPositions(*MF);
        // Check if the SVA requires access to scratch or stack
        bool RequiresAccessToScratch =
            SVAInfo.RequiresScratchAndStackSetup ||
            SVAInfo.RequestedKernelArguments.contains(
                KERNEL_PRIVATE_SEGMENT_BUFFER) ||
            SVAInfo.RequestedKernelArguments.contains(FLAT_SCRATCH);
        // If SVA requires access to scratch, then we have to enable it
        if (RequiresAccessToScratch) {
          // Enable the PSB if not already enabled
          enablePrivateSegmentBuffer(MFI, TRI);
          // Enable Flat scratch init if not already enabled
          enableFlatScratchInit(MFI, TRI);
          // Check if Private buffer wave segment offset was enabled; if not,
          // enable it
          enablePrivateSegmentWaveOffset(MFI);
        }

        // If access to kernarg buffer was requested, enable it
        if (SVAInfo.RequestedKernelArguments.contains(KERNARG_SEGMENT_PTR)) {
          enableKernArg(MFI, TRI);
        }

        // If access to dispatch id was requested, enable it
        if (SVAInfo.RequestedKernelArguments.contains(DISPATCH_ID)) {
          enableDispatchID(MFI, TRI);
        }

        // If access to dispatch ptr was requested, enable it
        if (SVAInfo.RequestedKernelArguments.contains(DISPATCH_PTR)) {
          enableDispatchPtr(MFI, TRI);
        }
        // If access to queue ptr was requested, enable it
        if (SVAInfo.RequestedKernelArguments.contains(QUEUE_PTR)) {
          enableQueuePtr(MFI, TRI);
        }

        // Emit the kernel preamble: ===========================================
        llvm::MCRegister SVSStorageReg =
            EntryInstrSVS.getSVS().getStateValueStorageReg();

        // If stack access was requested, then emit code to save it into
        // the SVS storage V/AGPR
        if (RequiresAccessToScratch) {
          LUTHIER_RETURN_ON_ERROR(emitCodeToSetupScratch(
              *EntryInstr, SVSStorageReg,
              llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(FuncSymbol)
                  ->getKernelMetadata()));
        }
        // Emit code to store the rest of the requested SGPR kernel arguments
        for (const auto &[KernArg, PreloadValue] :
             {std::pair{KERNARG_SEGMENT_PTR,
                        llvm::AMDGPUFunctionArgInfo::KERNARG_SEGMENT_PTR},
              {DISPATCH_ID, llvm::AMDGPUFunctionArgInfo::DISPATCH_ID},
              {DISPATCH_PTR, llvm::AMDGPUFunctionArgInfo::DISPATCH_PTR},
              {QUEUE_PTR, llvm::AMDGPUFunctionArgInfo::QUEUE_PTR}}) {
          if (SVAInfo.RequestedKernelArguments.contains(KernArg)) {
            auto StoreSlotBegin =
                stateValueArray::getKernelArgumentLaneIdStoreSlotBeginForWave64(
                    KernArg);
            LUTHIER_RETURN_ON_ERROR(StoreSlotBegin.takeError());
            auto StoreSlotSize =
                stateValueArray::getKernelArgumentStoreSlotSizeForWave64(
                    KernArg);
            LUTHIER_RETURN_ON_ERROR(StoreSlotSize.takeError());
            LUTHIER_RETURN_ON_ERROR(emitCodeToStoreSGPRKernelArg(
                *EntryInstr, getArgReg(*MF, PreloadValue), SVSStorageReg,
                *StoreSlotBegin, *StoreSlotSize,
                OriginalSGPRArgLocs.contains(PreloadValue)));
          }
        }
        // Put every SGPR argument back in its place
        emitCodeToReturnSGPRArgsToOriginalPlace(OriginalSGPRArgLocs,
                                                *EntryInstr);
      }
    }
  }
  return llvm::Error::success();
}

FunctionPreambleDescriptor::FunctionPreambleDescriptor(LiftedRepresentation &LR)
    : LR(LR) {
  for (const auto &[FuncSymbol, MF] : LR.functions()) {
    if (MF->getFunction().getCallingConv() ==
        llvm::CallingConv::AMDGPU_KERNEL) {
      Kernels.insert({MF, {}});
    } else {
      DeviceFunctions.insert({MF, {}});
    }
  }
}
} // namespace luthier