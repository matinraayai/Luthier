//===-- StateValueArraySpecs.cpp ------------------------------------------===//
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
/// This file implements functions used to query the state value array
/// specs.
//===----------------------------------------------------------------------===//
#include "tooling_common/StateValueArraySpecs.hpp"
#include "common/Error.hpp"
#include <SIMachineFunctionInfo.h>
#include <llvm/ADT/DenseMap.h>

namespace luthier::stateValueArray {

// TODO: Investigate replacing these static maps with frozen constexpr maps

/// A mapping between the application registers that need to be spilled before
/// the instrumentation frame is loaded, and their spill lane IDs in the state
/// value array
const static llvm::SmallDenseMap<llvm::MCRegister, unsigned short, 8>
    FrameSpillSlots{
        {llvm::AMDGPU::SGPR0, 0},       {llvm::AMDGPU::SGPR1, 1},
        {llvm::AMDGPU::SGPR2, 2},       {llvm::AMDGPU::SGPR3, 3},
        {llvm::AMDGPU::SGPR32, 4},      {llvm::AMDGPU::SGPR33, 5},
        {llvm::AMDGPU::FLAT_SCR_LO, 6}, {llvm::AMDGPU::FLAT_SCR_HI, 7}};

/// A mapping between stack frame registers of the instrumentation function
/// and the lane IDs of where they will be stored in the state value array
const static llvm::SmallDenseMap<llvm::MCRegister, unsigned short, 8>
    InstrumentationStackFrameStoreSlots{
        {llvm::AMDGPU::SGPR0, 8},       {llvm::AMDGPU::SGPR1, 9},
        {llvm::AMDGPU::SGPR2, 10},      {llvm::AMDGPU::SGPR3, 11},
        {llvm::AMDGPU::SGPR32, 12},     {llvm::AMDGPU::FLAT_SCR_LO, 13},
        {llvm::AMDGPU::FLAT_SCR_HI, 14}};

/// A mapping between the kernel arguments and the lane ID of where they
/// will be stored in the state value array, as well as
/// Intended for use for when the kernel's wavefront size is 64
const static llvm::SmallDenseMap<KernelArgumentType, std::pair<short, short>,
                                 32>
    WaveFront64KernelArgumentStoreSlots{
        {KERNARG_SEGMENT_PTR, {15, 2}},
        {HIDDEN_KERNARG_OFFSET, {17, 1}},
        {USER_KERNARG_OFFSET, {18, 1}},
        {DISPATCH_ID, {19, 2}},
        {FLAT_SCRATCH_INIT, {21, 2}},
        {PRIVATE_SEGMENT_WAVE_BYTE_OFFSET, {23, 1}},
        {DISPATCH_PTR, {24, 2}},
        {QUEUE_PTR, {26, 2}},
        {PRIVATE_SEGMENT_SIZE, {28, 1}},
        {GLOBAL_OFFSET_X, {29, 1}},
        {GLOBAL_OFFSET_Y, {30, 1}},
        {GLOBAL_OFFSET_Z, {31, 1}},
        {PRINT_BUFFER, {32, 2}},
        {HOSTCALL_BUFFER, {34, 2}},
        {DEFAULT_QUEUE, {36, 2}},
        {COMPLETION_ACTION, {38, 2}},
        {MULTIGRID_SYNC, {40, 2}},
        {BLOCK_COUNT_X, {42, 1}},
        {BLOCK_COUNT_Y, {43, 1}},
        {BLOCK_COUNT_Z, {44, 1}},
        {GROUP_SIZE_X, {45, 1}},
        {GROUP_SIZE_Y, {46, 1}},
        {GROUP_SIZE_Z, {47, 1}},
        {REMAINDER_X, {48, 1}},
        {REMAINDER_Y, {49, 1}},
        {REMAINDER_Z, {50, 1}},
        {HEAP_V1, {51, 1}},
        {DYNAMIC_LDS_SIZE, {52, 1}},
        {PRIVATE_BASE, {53, 2}},
        {SHARED_BASE, {55, 2}}};

// TODO: Add wave32 state value array

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

bool isFrameSpillSlot(llvm::MCRegister Reg) {
  return FrameSpillSlots.contains(Reg);
}

llvm::Expected<unsigned short> getFrameSpillSlotLaneId(llvm::MCRegister Reg) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      FrameSpillSlots.contains(Reg),
      "MC Reg {0} is not in the state value array frame spill slots.",
      llvm::printReg(Reg)));
  return FrameSpillSlots.at(Reg);
}

llvm::Expected<unsigned short>
getInstrumentationStackFrameLaneIdStoreSlot(llvm::MCRegister Reg) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      InstrumentationStackFrameStoreSlots.contains(Reg),
      "MC Reg {0} is not in the state value array frame store slots."));
  return InstrumentationStackFrameStoreSlots.at(Reg);
}

llvm::Expected<unsigned short>
getKernelArgumentLaneIdStoreSlotBeginForWave64(KernelArgumentType Arg) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      WaveFront64KernelArgumentStoreSlots.contains(Arg),
      "Arg enum {0} does not have an entry in the wave64 state value array."));
  return WaveFront64KernelArgumentStoreSlots.at(Arg).first;
}

llvm::Expected<unsigned short>
getKernelArgumentStoreSlotSizeForWave64(KernelArgumentType Arg) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      WaveFront64KernelArgumentStoreSlots.contains(Arg),
      "Arg enum {0} does not have an entry in the wave64 state value array."));
  return WaveFront64KernelArgumentStoreSlots.at(Arg).second;
}

}; // namespace luthier::stateValueArray