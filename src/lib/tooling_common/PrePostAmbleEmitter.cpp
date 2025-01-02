//===-- PrePostAmbleEmitter.hpp -------------------------------------------===//
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
/// This file implements the Pre and post amble emitter, as well as the
/// \c FunctionPreambleDescriptor and its analysis pass.
//===----------------------------------------------------------------------===//
#include "tooling_common/PrePostAmbleEmitter.hpp"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include "tooling_common/SVStorageAndLoadLocations.hpp"
#include "tooling_common/StateValueArraySpecs.hpp"
#include "tooling_common/WrapperAnalysisPasses.hpp"
#include <GCNSubtarget.h>
#include <SIMachineFunctionInfo.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-pre-post-amble-emitter"

namespace luthier {

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
    auto Reg = getArgReg(MF, PreloadVal);
    if (Reg)
      SGPRKernelArguments.insert({PreloadVal, Reg});
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
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();
  auto &MFI = *MF.getInfo<llvm::SIMachineFunctionInfo>();
  // First make a copy of S0 and S1/ FS_lo FS_hi in the state value
  // register
  auto SGPR0SpillSlot =
      stateValueArray::getFrameSpillSlotLaneId(llvm::AMDGPU::SGPR0);

  auto SGPR1SpillSlot =
      stateValueArray::getFrameSpillSlotLaneId(llvm::AMDGPU::SGPR1);

  auto SGPRFlatScrLoSpillSlot =
      stateValueArray::getFrameSpillSlotLaneId(llvm::AMDGPU::FLAT_SCR_LO);

  auto SGPRFlatScrHiSpillSlot =
      stateValueArray::getFrameSpillSlotLaneId(llvm::AMDGPU::FLAT_SCR_HI);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(TRI.getSubReg(
          MFI.getPreloadedReg(
              llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
          llvm::AMDGPU::sub0))
      .addImm(SGPR0SpillSlot)
      .addReg(SVSStorageVGPR);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(TRI.getSubReg(
          MFI.getPreloadedReg(
              llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
          llvm::AMDGPU::sub1))
      .addImm(SGPR1SpillSlot)
      .addReg(SVSStorageVGPR);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(TRI.getSubReg(
          MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
          llvm::AMDGPU::sub0))
      .addImm(SGPRFlatScrLoSpillSlot)
      .addReg(SVSStorageVGPR);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageVGPR)
      .addReg(TRI.getSubReg(
          MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
          llvm::AMDGPU::sub1))
      .addImm(SGPRFlatScrHiSpillSlot)
      .addReg(SVSStorageVGPR);

  // Add the PSWO to SGPR0/its carry to SGPR1
  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_ADD_U32))
      .addReg(TRI.getSubReg(
                  MFI.getPreloadedReg(
                      llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                  llvm::AMDGPU::sub0),
              llvm::RegState::Define)
      .addReg(TRI.getSubReg(
                  MFI.getPreloadedReg(
                      llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                  llvm::AMDGPU::sub0),
              llvm::RegState::Kill)
      .addReg(MFI.getPreloadedReg(
          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET));

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_ADDC_U32))
      .addReg(TRI.getSubReg(
                  MFI.getPreloadedReg(
                      llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                  llvm::AMDGPU::sub1),
              llvm::RegState::Define)
      .addReg(TRI.getSubReg(
                  MFI.getPreloadedReg(
                      llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                  llvm::AMDGPU::sub1),
              llvm::RegState::Kill)
      .addImm(0);
  // Add the PSWO to FS_init_lo/its carry to FS_init_hi
  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_ADD_U32))
      .addReg(TRI.getSubReg(MFI.getPreloadedReg(
                                llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                            llvm::AMDGPU::sub0),
              llvm::RegState::Define)
      .addReg(TRI.getSubReg(MFI.getPreloadedReg(
                                llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                            llvm::AMDGPU::sub0),
              llvm::RegState::Kill)
      .addReg(MFI.getPreloadedReg(
          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET));
  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::S_ADDC_U32))
      .addReg(TRI.getSubReg(MFI.getPreloadedReg(
                                llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                            llvm::AMDGPU::sub1),
              llvm::RegState::Define)
      .addReg(TRI.getSubReg(MFI.getPreloadedReg(
                                llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                            llvm::AMDGPU::sub1),
              llvm::RegState::Kill)
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
      .addImm(SGPR0SpillSlot);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_READLANE_B32), llvm::AMDGPU::SGPR1)
      .addReg(SVSStorageVGPR)
      .addImm(SGPR1SpillSlot);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_READLANE_B32))
      .addReg(TRI.getSubReg(MFI.getPreloadedReg(
                                llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                            llvm::AMDGPU::sub0),
              llvm::RegState::Define)
      .addReg(SVSStorageVGPR)
      .addImm(SGPRFlatScrLoSpillSlot);

  llvm::BuildMI(MF.front(), EntryInstr, llvm::DebugLoc(),
                TII.get(llvm::AMDGPU::V_READLANE_B32))
      .addReg(TRI.getSubReg(MFI.getPreloadedReg(
                                llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                            llvm::AMDGPU::sub1),
              llvm::RegState::Define)
      .addReg(SVSStorageVGPR)
      .addImm(SGPRFlatScrHiSpillSlot);

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
          .addReg(TRI.getSubReg(SrcSGPR, SubIdx),
                  KillAfterUse ? llvm::RegState::Kill : 0)
          .addImm(SpillSlotStart + i)
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
    llvm::MCRegister ModifiedArgReg = getArgReg(MF, KernArg);
    size_t Size =
        TRI.getRegSizeInBits(*TRI.getPhysRegBaseClass(OriginalArgReg));
    if (OriginalArgReg != ModifiedArgReg) {
      if (Size == 32) {
        llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::S_MOV_B32), OriginalArgReg)
            .addReg(ModifiedArgReg, llvm::RegState::Kill);
      } else if (Size == 64) {
        llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                      TII.get(llvm::AMDGPU::S_MOV_B64), OriginalArgReg)
            .addReg(ModifiedArgReg, llvm::RegState::Kill);
      } else {
        size_t NumChannels = Size / 32;
        for (int i = 0; i < NumChannels / 2; i++) {
          auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i, 2);
          llvm::BuildMI(MBB, InsertionPoint, llvm::DebugLoc(),
                        TII.get(llvm::AMDGPU::S_MOV_B64))
              .addReg(TRI.getSubReg(OriginalArgReg, SubIdx),
                      llvm::RegState::Define)
              .addReg(TRI.getSubReg(ModifiedArgReg, SubIdx),
                      llvm::RegState::Kill);
        }
      }
    }
  }
}

llvm::AnalysisKey FunctionPreambleDescriptorAnalysis::Key;

FunctionPreambleDescriptorAnalysis::Result
FunctionPreambleDescriptorAnalysis::run(
    llvm::Module &TargetModule, llvm::ModuleAnalysisManager &TargetMAM) {

  return {TargetMAM.getCachedResult<llvm::MachineModuleAnalysis>(TargetModule)
              ->getMMI(),
          TargetModule};
}

llvm::PreservedAnalyses
PrePostAmbleEmitter::run(llvm::Module &TargetModule,
                         llvm::ModuleAnalysisManager &TargetMAM) {
  auto T1 = std::chrono::high_resolution_clock::now();
  const auto &PKInfo =
      *TargetMAM.getCachedResult<FunctionPreambleDescriptorAnalysis>(
          TargetModule);

  auto &LR =
      TargetMAM.getResult<LiftedRepresentationAnalysis>(TargetModule).getLR();

  auto LCO = TargetMAM.getResult<LoadedCodeObjectAnalysis>(TargetModule);

  auto &SVLocations =
      *TargetMAM.getCachedResult<LRStateValueStorageAndLoadLocationsAnalysis>(
          TargetModule);

  // First we need to figure out if we need to set up the state value array
  // at all

  bool MustSetupSVA = doesLRRequireSVA(LR, LCO, PKInfo);

  if (MustSetupSVA) {
    LLVM_DEBUG(llvm::dbgs() << "Have to setup the SVA.\n");
    for (auto &[FuncSymbol, MF] : LR.functions()) {
      if (MF->getFunction().getCallingConv() ==
              llvm::CallingConv::AMDGPU_KERNEL &&
          FuncSymbol->getLoadedCodeObject() == LCO) {
        auto &SVAInfo = PKInfo.Kernels.at(MF);
        auto EntryInstr = MF->begin()->begin();
        auto &EntryInstrSVS = SVLocations.getStorageIntervals(MF->front())[0];
        auto &MFI = *MF->getInfo<llvm::SIMachineFunctionInfo>();
        auto &TRI = *MF->getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
        auto *TII = MF->getSubtarget<llvm::GCNSubtarget>().getInstrInfo();

        // Get the original position of all the reg arguments before they
        // are changed
        auto OriginalSGPRArgLocs = getSGPRArgumentPositions(*MF);

        LLVM_DEBUG(
            llvm::dbgs() << "Original Positions of the kernel args:\n";
            llvm::dbgs() << "[ "; llvm::interleave(
                OriginalSGPRArgLocs.begin(), OriginalSGPRArgLocs.end(),
                [&](std::pair<llvm::AMDGPUFunctionArgInfo::PreloadedValue,
                              llvm::MCRegister>
                        Args) {
                  llvm::dbgs() << Args.first << ": "
                               << llvm::printReg(Args.second, &TRI);
                },
                [&]() { llvm::dbgs() << ", "; });
            llvm::dbgs() << "]\n";

        );

        // Check if the SVA requires access to scratch or stack
        bool RequiresAccessToScratch =
            SVAInfo.RequiresScratchAndStackSetup ||
            SVAInfo.RequestedKernelArguments.contains(
                WAVEFRONT_PRIVATE_SEGMENT_BUFFER) ||
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
          if (auto Err = emitCodeToSetupScratch(
                  *EntryInstr, SVSStorageReg,
                  llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(FuncSymbol)
                      ->getKernelMetadata())) {
            TargetModule.getContext().emitError(toString(std::move(Err)));
            return llvm::PreservedAnalyses::all();
          }
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
            if (auto Err = StoreSlotBegin.takeError()) {
              TargetModule.getContext().emitError(toString(std::move(Err)));
              return llvm::PreservedAnalyses::all();
            }
            auto StoreSlotSize =
                stateValueArray::getKernelArgumentStoreSlotSizeForWave64(
                    KernArg);
            if (auto Err = StoreSlotSize.takeError()) {
              TargetModule.getContext().emitError(toString(std::move(Err)));
              return llvm::PreservedAnalyses::all();
            }
            if (auto Err = emitCodeToStoreSGPRKernelArg(
                    *EntryInstr, getArgReg(*MF, PreloadValue), SVSStorageReg,
                    *StoreSlotBegin, *StoreSlotSize,
                    OriginalSGPRArgLocs.contains(PreloadValue))) {
              TargetModule.getContext().emitError(toString(std::move(Err)));
              return llvm::PreservedAnalyses::all();
            }
          }
        }
        // Add code
        if (SVAInfo.RequestedKernelArguments.contains(HIDDEN_KERNARG_OFFSET)) {
          llvm::outs() << "emitting code to store the hidden arg offset.\n";
          auto &KernArgs =
              llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(FuncSymbol)
                  ->getKernelMetadata()
                  .Args;

          LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
              KernArgs.has_value(), "Attempted to access the hidden arguments "
                                    "of a kernel without any arguments."));
          uint32_t HiddenOffset = [&]() {
            for (const auto &Arg : *KernArgs) {
              if (Arg.ValueKind >= hsa::md::ValueKind::HiddenArgKindBegin &&
                  Arg.ValueKind <= hsa::md::ValueKind::HiddenArgKindEnd) {
                return Arg.Offset;
              }
            }
            return uint32_t{0};
          }();
          auto StoreLane =
              stateValueArray::getKernelArgumentLaneIdStoreSlotBeginForWave64(
                  HIDDEN_KERNARG_OFFSET);
          LUTHIER_REPORT_FATAL_ON_ERROR(StoreLane.takeError());

          llvm::BuildMI(*EntryInstr->getParent(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_WRITELANE_B32), SVSStorageReg)
              .addImm(HiddenOffset)
              .addImm(*StoreLane)
              .addReg(SVSStorageReg);

          EntryInstr->getMF()->print(llvm::outs());
        }


        // Put every SGPR argument back in its place
        emitCodeToReturnSGPRArgsToOriginalPlace(OriginalSGPRArgLocs,
                                                *EntryInstr);
      }

      auto &SlotIndexes =
          TargetMAM.getCachedResult<MMISlotIndexesAnalysis>(TargetModule)
              ->at(*MF);

      // Now we need to emit code that juggles the SVS between different
      // storage schemes
      const auto &TII = *MF->getSubtarget().getInstrInfo();
      for (auto &MBB : *MF) {
        const auto MBBIntervals = SVLocations.getStorageIntervals(MBB);
        for (unsigned int I = 0; I < MBBIntervals.size() - 1; I++) {
          auto &CurMBBInterval = MBBIntervals[I];
          auto &NextMBBInterval = MBBIntervals[I + 1];
          if (CurMBBInterval.getSVS() != NextMBBInterval.getSVS()) {
            auto InsertionMI =
                SlotIndexes.getInstructionFromIndex(NextMBBInterval.begin());
            CurMBBInterval.getSVS().emitCodeToSwitchSVS(
                *InsertionMI, NextMBBInterval.getSVS());
          }
        }
        // Analyze the branch at the end of this block (if exists)
        llvm::MachineBasicBlock *TBB;
        llvm::MachineBasicBlock *FBB;
        llvm::MachineBasicBlock *NewTBB{nullptr};
        llvm::MachineBasicBlock *NewFBB{nullptr};
        llvm::SmallVector<llvm::MachineOperand, 4> Cond;
        bool Fail = TII.analyzeBranch(MBB, TBB, FBB, Cond, false);
        if (Fail)
          continue;
        llvm::SmallVector<
            std::pair<llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>, 4>
            OldToNewSuccessorsList;
        for (llvm::MachineBasicBlock *SuccessorMBB : MBB.successors()) {
          auto &SuccessorIntervalBegin =
              SVLocations.getStorageIntervals(*SuccessorMBB).front();
          // If the successor and the end of this MBB don't have the same
          // storage, we need to emit switch code in between them
          if (SuccessorIntervalBegin.getSVS() != MBBIntervals.back().getSVS()) {
            if (TBB == SuccessorMBB) {
              // Create a new basic block and insert it at the end
              NewTBB = MF->CreateMachineBasicBlock();
              MF->insert(MF->end(), NewTBB);
              // Insert an unconditional branch to the old FBB
              TII.insertUnconditionalBranch(*NewTBB, TBB, llvm::DebugLoc());
              NewTBB->addSuccessor(TBB);
              // Emit the SVS switch code before the branch
              MBBIntervals.back().getSVS().emitCodeToSwitchSVS(
                  NewTBB->front(), SuccessorIntervalBegin.getSVS());
              OldToNewSuccessorsList.emplace_back(TBB, NewTBB);
            } else if (FBB == SuccessorMBB) {
              // Create a new basic block and insert it at the end
              NewFBB = MF->CreateMachineBasicBlock();
              MF->insert(MF->end(), NewFBB);
              // Insert an unconditional branch to the old FBB
              TII.insertUnconditionalBranch(*NewFBB, FBB, llvm::DebugLoc());
              NewFBB->addSuccessor(FBB);
              // Emit the SVS switch code before the branch
              MBBIntervals.back().getSVS().emitCodeToSwitchSVS(
                  NewFBB->front(), SuccessorIntervalBegin.getSVS());
              OldToNewSuccessorsList.emplace_back(TBB, NewTBB);
            } else {
              // This is a fallthrough block; We insert the SVS code inside
              // an MBB between the two blocks
              llvm::MachineBasicBlock *NewFallthrough =
                  MF->CreateMachineBasicBlock();
              MF->insert(MBB.getNextNode()->getIterator(), NewFallthrough);
              NewFallthrough->addSuccessor(SuccessorMBB);
              OldToNewSuccessorsList.emplace_back(SuccessorMBB, NewFallthrough);
            }
          }
        }
        // Insert the new branch if needed
        if ((NewFBB != nullptr && FBB != nullptr) ||
            (NewTBB != nullptr && TBB != nullptr)) {
          auto DebugLoc = MBB.getFirstInstrTerminator()->getDebugLoc();
          TII.removeBranch(MBB);
          TII.insertBranch(MBB, NewTBB, NewFBB, Cond, DebugLoc);
          for (const auto &[OldSuccessor, NewSuccessor] :
               OldToNewSuccessorsList) {
            MBB.removeSuccessor(OldSuccessor);
            MBB.addSuccessor(NewSuccessor);
          }
        }
      }
    }
  }
  auto T2 = std::chrono::high_resolution_clock::now();
  llvm::outs()
      << "Time to Run Pre-PostAmble Pass: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(T2 - T1).count()
      << "ms.\n";
  return llvm::PreservedAnalyses::all();
}

FunctionPreambleDescriptor::FunctionPreambleDescriptor(
    const llvm::MachineModuleInfo &TargetMMI,
    const llvm::Module &TargetModule) {
  for (const auto &F : TargetModule) {
    auto *MF = TargetMMI.getMachineFunction(F);
    if (!MF)
      continue;
    if (MF->getFunction().getCallingConv() ==
        llvm::CallingConv::AMDGPU_KERNEL) {
      Kernels.insert({MF, {}});
    } else {
      DeviceFunctions.insert({MF, {}});
    }
  }
}
} // namespace luthier