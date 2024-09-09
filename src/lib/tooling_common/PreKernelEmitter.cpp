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
#include "tooling_common/PreKernelEmitter.hpp"
#include "luthier/Intrinsic/IntrinsicProcessor.h"
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
        {FLAT_SCRATCH_INIT,
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

const static llvm::SmallDenseMap<llvm::MCRegister, unsigned int>
    ValueRegisterSpillSlots{
        {llvm::AMDGPU::SGPR0, 0},      {llvm::AMDGPU::SGPR1, 1},
        {llvm::AMDGPU::SGPR2, 2},      {llvm::AMDGPU::SGPR3, 3},
        {llvm::AMDGPU::SGPR32, 4},     {llvm::AMDGPU::FLAT_SCR_LO, 5},
        {llvm::AMDGPU::FLAT_SCR_HI, 6}};

const static llvm::SmallDenseMap<llvm::MCRegister, unsigned int>
    ValueRegisterInstrumentationSlots{
        {llvm::AMDGPU::SGPR0, 7},       {llvm::AMDGPU::SGPR1, 8},
        {llvm::AMDGPU::SGPR2, 9},       {llvm::AMDGPU::SGPR3, 10},
        {llvm::AMDGPU::SGPR32, 11},     {llvm::AMDGPU::FLAT_SCR_LO, 12},
        {llvm::AMDGPU::FLAT_SCR_HI, 13}};

llvm::MCRegister getArgReg(llvm::MachineFunction &MF,
                           KernelArgumentType ArgReg) {
  return MF.getInfo<llvm::SIMachineFunctionInfo>()->getPreloadedReg(
      ToLLVMRegArgMap.at(ArgReg));
}

llvm::Error PreKernelEmitter::emitPreKernel() {
  if (PKInfo.DoesNeedPreKernel) {
    auto LCO = SVLocations.getLCO();
    for (auto &[FuncSymbol, MF] : LR.functions()) {
      if (MF->getFunction().getCallingConv() ==
              llvm::CallingConv::AMDGPU_KERNEL &&
          FuncSymbol->getLoadedCodeObject() == LCO) {
        auto EntryInstr = MF->begin()->begin();
        auto *EntryInstrSVS = SVLocations.getValueSegmentForInstr(*EntryInstr);

        if (PKInfo.EnableScratchAndStoreStackInfo) {
          auto &MFI = *MF->getInfo<llvm::SIMachineFunctionInfo>();
          auto &TRI = *MF->getSubtarget<llvm::GCNSubtarget>().getRegisterInfo();
          auto *TII = MF->getSubtarget<llvm::GCNSubtarget>().getInstrInfo();
          // Get the original position of all the reg arguments
          llvm::DenseMap<KernelArgumentType, llvm::MCRegister>
              OriginalArgLocs{};

          for (const auto &[RegArg, PreloadVal] : ToLLVMRegArgMap) {
            OriginalArgLocs.insert({RegArg, getArgReg(*MF, RegArg)});
          }

          // We have to enable the scratch explicitly if not already enabled
          // Check if the PSB was enabled in the original kernel; If not
          // enabled, enable it
          auto OriginalPSBReg = MFI.getPreloadedReg(
              llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);
          if (OriginalPSBReg == 0) {
            MFI.addPrivateSegmentBuffer(TRI);
          }
          // Check if Flat scratch init was enabled in the original kernel;
          // If not, enable it
          auto FSIReg = MFI.getPreloadedReg(
              llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT);
          if (FSIReg == 0) {
            MFI.addFlatScratchInit(TRI);
          }
          // Check if Private buffer wave segment offset was enabled; if not,
          // enable it
          auto OriginalPSWOReg = MFI.getPreloadedReg(
              llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);
          if (OriginalPSWOReg == 0) {
            MFI.addPrivateSegmentWaveByteOffset();
          }

          // Emit the pre-kernel: ==============================================
          llvm::MCRegister ValueStateLocation =
              EntryInstrSVS->getSVS().getStateValueStorageReg();

          // First make a copy of S0 and S1/ FS_lo FS_hi in the state value
          // register
          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                        ValueStateLocation)
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                      0, llvm::AMDGPU::sub0)
              .addImm(ValueRegisterSpillSlots.at(llvm::AMDGPU::SGPR0))
              .addReg(ValueStateLocation);

          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                        ValueStateLocation)
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                      0, llvm::AMDGPU::sub1)
              .addImm(ValueRegisterSpillSlots.at(llvm::AMDGPU::SGPR1))
              .addReg(ValueStateLocation);

          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                        ValueStateLocation)
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                      0, llvm::AMDGPU::sub0)
              .addImm(ValueRegisterSpillSlots.at(llvm::AMDGPU::FLAT_SCR_LO))
              .addReg(ValueStateLocation);

          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                        ValueStateLocation)
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                      0, llvm::AMDGPU::sub1)
              .addImm(ValueRegisterSpillSlots.at(llvm::AMDGPU::FLAT_SCR_HI))
              .addReg(ValueStateLocation);

          // Add the PSWO to SGPR0/its carry to SGPR1
          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::S_ADD_U32))
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                      llvm::RegState::Define, llvm::AMDGPU::sub0)
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                      llvm::RegState::Kill, llvm::AMDGPU::sub0)
              .addReg(
                  MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::
                                          PRIVATE_SEGMENT_WAVE_BYTE_OFFSET));
          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::S_ADDC_U32))
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                      llvm::RegState::Define, llvm::AMDGPU::sub1)
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER),
                      llvm::RegState::Kill, llvm::AMDGPU::sub1)
              .addImm(0);
          // Add the PSWO to FS_init_lo/its carry to FS_init_hi
          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::S_ADD_U32))
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                      llvm::RegState::Define, llvm::AMDGPU::sub0)
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                      llvm::RegState::Kill, llvm::AMDGPU::sub0)
              .addReg(
                  MFI.getPreloadedReg(llvm::AMDGPUFunctionArgInfo::
                                          PRIVATE_SEGMENT_WAVE_BYTE_OFFSET));
          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::S_ADDC_U32))
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                      llvm::RegState::Define, llvm::AMDGPU::sub1)
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                      llvm::RegState::Kill, llvm::AMDGPU::sub1)
              .addImm(0);

          // Set s32 to be the maximum amount of stack requested by the hook
          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::S_MOV_B32), llvm::AMDGPU::SGPR32)
              .addImm(PKInfo.AmountOfScratchRequested);

          // Store everything in their slots

          for (const auto &[PhysReg, StoreSlot] :
               ValueRegisterInstrumentationSlots) {
            llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                          TII->get(llvm::AMDGPU::V_WRITELANE_B32),
                          ValueStateLocation)
                .addReg(PhysReg)
                .addImm(ValueRegisterInstrumentationSlots.at(PhysReg))
                .addReg(ValueStateLocation);
          }

          // Restore S0, S1, FS_init_lo, and FS_init_hi

          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_READLANE_B32),
                        llvm::AMDGPU::SGPR0)
              .addReg(ValueStateLocation)
              .addImm(ValueRegisterSpillSlots.at(llvm::AMDGPU::SGPR0));

          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_READLANE_B32),
                        llvm::AMDGPU::SGPR1)
              .addReg(ValueStateLocation)
              .addImm(ValueRegisterSpillSlots.at(llvm::AMDGPU::SGPR1));

          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_READLANE_B32))
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                      llvm::RegState::Define, llvm::AMDGPU::sub0)
              .addReg(ValueStateLocation)
              .addImm(ValueRegisterSpillSlots.at(llvm::AMDGPU::FLAT_SCR_LO));

          llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                        TII->get(llvm::AMDGPU::V_READLANE_B32))
              .addReg(MFI.getPreloadedReg(
                          llvm::AMDGPUFunctionArgInfo::FLAT_SCRATCH_INIT),
                      llvm::RegState::Define, llvm::AMDGPU::sub1)
              .addReg(ValueStateLocation)
              .addImm(ValueRegisterSpillSlots.at(llvm::AMDGPU::FLAT_SCR_HI));

          // Put everything back in their original place
          for (const auto &[KernArg, RegVal] : ToLLVMRegArgMap) {
            auto OriginalArgLoc = OriginalArgLocs.at(KernArg);
            if (OriginalArgLoc != 0 &&
                OriginalArgLoc != getArgReg(*MF, KernArg)) {
              int Size = TRI.getRegSizeInBits(
                  *TRI.getPhysRegBaseClass(OriginalArgLoc));
              if (Size == 32) {
                llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                              TII->get(llvm::AMDGPU::S_MOV_B32))
                    .addReg(OriginalArgLoc, llvm::RegState::Define)
                    .addReg(getArgReg(*MF, KernArg), llvm::RegState::Kill);
              } else {
                size_t NumChannels = Size / 32;
                for (int i = 0; i < NumChannels; i++) {
                  auto SubIdx = llvm::SIRegisterInfo::getSubRegFromChannel(i);
                  llvm::BuildMI(MF->front(), EntryInstr, llvm::DebugLoc(),
                                TII->get(llvm::AMDGPU::S_MOV_B32))
                      .addReg(OriginalArgLoc, llvm::RegState::Define, SubIdx)
                      .addReg(getArgReg(*MF, KernArg), llvm::RegState::Kill,
                              SubIdx);
                }
              }
            }
          }
        }
      }
    }
  }
  return llvm::Error::success();
}

} // namespace luthier