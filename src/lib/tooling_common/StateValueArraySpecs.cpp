//===-- StateValueArraySpecs.cpp ------------------------------------------===//
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
/// This file implements functions used to query the state value array
/// specs.
//===----------------------------------------------------------------------===//
#include "tooling_common/StateValueArraySpecs.hpp"
#include "common/Error.hpp"
#include <SIMachineFunctionInfo.h>
#include <llvm/ADT/DenseMap.h>
#include <luthier/ErrorCheck.h>

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
        {PRIVATE_SEGMENT_WAVE_BYTE_OFFSET, {21, 1}},
        {DISPATCH_PTR, {22, 2}},
        {QUEUE_PTR, {24, 2}},
        {WORK_ITEM_PRIVATE_SEGMENT_SIZE, {26, 1}},
        {GLOBAL_OFFSET_X, {27, 1}},
        {GLOBAL_OFFSET_Y, {28, 1}},
        {GLOBAL_OFFSET_Z, {29, 1}},
        {PRINT_BUFFER, {30, 2}},
        {HOSTCALL_BUFFER, {32, 2}},
        {DEFAULT_QUEUE, {34, 2}},
        {COMPLETION_ACTION, {36, 2}},
        {MULTIGRID_SYNC, {38, 2}},
        {BLOCK_COUNT_X, {40, 1}},
        {BLOCK_COUNT_Y, {41, 1}},
        {BLOCK_COUNT_Z, {42, 1}},
        {GROUP_SIZE_X, {43, 1}},
        {GROUP_SIZE_Y, {44, 1}},
        {GROUP_SIZE_Z, {45, 1}},
        {REMAINDER_X, {46, 1}},
        {REMAINDER_Y, {47, 1}},
        {REMAINDER_Z, {58, 1}},
        {HEAP_V1, {49, 1}},
        {DYNAMIC_LDS_SIZE, {50, 1}},
        {PRIVATE_BASE, {51, 2}},
        {SHARED_BASE, {53, 2}}};

// TODO: Add wave32 state value array


bool isFrameSpillSlot(llvm::MCRegister Reg) {
  return FrameSpillSlots.contains(Reg);
}

llvm::iterator_range<
    llvm::SmallDenseMap<llvm::MCRegister, unsigned short, 8>::const_iterator>
getFrameSpillSlots() {
  return llvm::make_range(FrameSpillSlots.begin(), FrameSpillSlots.end());
}

unsigned short getFrameSpillSlotLaneId(llvm::MCRegister Reg) {
  return FrameSpillSlots.at(Reg);
}

unsigned short
getInstrumentationStackFrameLaneIdStoreSlot(llvm::MCRegister Reg) {
  return InstrumentationStackFrameStoreSlots.at(Reg);
}

llvm::iterator_range<
    llvm::SmallDenseMap<llvm::MCRegister, unsigned short, 8>::const_iterator>
getFrameStoreSlots() {
  return llvm::make_range(InstrumentationStackFrameStoreSlots.begin(),
                          InstrumentationStackFrameStoreSlots.end());
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