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
/// Implements the \c StateValueArraySpecsAnalysis class.
//===----------------------------------------------------------------------===//
#include <SIMachineFunctionInfo.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/FormatVariadic.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/Instrumentation/StateValueArraySpecsAnalysis.h>

namespace luthier {

llvm::AnalysisKey StateValueArraySpecsAnalysis::Key;

StateValueArraySpecsAnalysis::Result::Result(const llvm::GCNSubtarget &STI)
    : FrameSpillSlots(std::move([&] -> decltype(FrameSpillSlots) {
        if (!STI.flatScratchIsArchitected()) {
          return {
              {llvm::AMDGPU::SGPR0, 0},       {llvm::AMDGPU::SGPR1, 1},
              {llvm::AMDGPU::SGPR2, 2},       {llvm::AMDGPU::SGPR3, 3},
              {llvm::AMDGPU::SGPR32, 4},      {llvm::AMDGPU::SGPR33, 5},
              {llvm::AMDGPU::FLAT_SCR_LO, 6}, {llvm::AMDGPU::FLAT_SCR_HI, 7}};
        } else {
          return {{llvm::AMDGPU::SGPR32, 0}, {llvm::AMDGPU::SGPR33, 1}};
        };
      }())),
      InstrumentationStackFrameStoreSlots(

          std::move([&] -> decltype(InstrumentationStackFrameStoreSlots) {
            if (!STI.flatScratchIsArchitected()) {
              return {{llvm::AMDGPU::SGPR0, 8},
                      {llvm::AMDGPU::SGPR1, 9},
                      {llvm::AMDGPU::SGPR2, 10},
                      {llvm::AMDGPU::SGPR3, 11},
                      {llvm::AMDGPU::SGPR32, 12},
                      {llvm::AMDGPU::FLAT_SCR_LO, 13},
                      {llvm::AMDGPU::FLAT_SCR_HI, 14}};
            } else {
              return {{llvm::AMDGPU::SGPR32, 2}};
            }
          }())),
      KernelArgumentStoreSlots(
      std::move([&] -> decltype(KernelArgumentStoreSlots) {
        if (!STI.flatScratchIsArchitected()) {
          return {{KERNARG_SEGMENT_PTR, {15, 2}},
           {HIDDEN_KERNARG_OFFSET, {17, 1}},
           {USER_KERNARG_OFFSET, {18, 1}},
           {DISPATCH_ID, {19, 2}},
           {PRIVATE_SEGMENT_WAVE_BYTE_OFFSET, {21, 1}},
           {DISPATCH_PTR, {22, 2}},
           {QUEUE_PTR, {24, 2}},
           {WORK_ITEM_PRIVATE_SEGMENT_SIZE, {26, 1}}};
        }
        else {
          return {{KERNARG_SEGMENT_PTR, {3, 2}},
           {HIDDEN_KERNARG_OFFSET, {5, 1}},
           {USER_KERNARG_OFFSET, {6, 1}},
           {DISPATCH_ID, {7, 2}},
           {PRIVATE_SEGMENT_WAVE_BYTE_OFFSET, {9, 1}},
           {DISPATCH_PTR, {10, 2}},
           {QUEUE_PTR, {12, 2}},
           {WORK_ITEM_PRIVATE_SEGMENT_SIZE, {14, 1}}};
        }
      }())) {};

bool StateValueArraySpecsAnalysis::Result::isFrameSpillSlot(
    llvm::MCRegister Reg) const {
  return FrameSpillSlots.contains(Reg);
}

llvm::iterator_range<
    llvm::SmallDenseMap<llvm::MCRegister, unsigned short, 8>::const_iterator>
StateValueArraySpecsAnalysis::Result::getFrameSpillSlots() const {
  return llvm::make_range(FrameSpillSlots.begin(), FrameSpillSlots.end());
}

unsigned short StateValueArraySpecsAnalysis::Result::getFrameSpillSlotLaneId(
    llvm::MCRegister Reg) const {
  return FrameSpillSlots.at(Reg);
}

unsigned short StateValueArraySpecsAnalysis::Result::
    getInstrumentationStackFrameLaneIdStoreSlot(llvm::MCRegister Reg) const {
  return InstrumentationStackFrameStoreSlots.at(Reg);
}

llvm::iterator_range<
    llvm::SmallDenseMap<llvm::MCRegister, unsigned short, 8>::const_iterator>
StateValueArraySpecsAnalysis::Result::getFrameStoreSlots() const {
  return llvm::make_range(InstrumentationStackFrameStoreSlots.begin(),
                          InstrumentationStackFrameStoreSlots.end());
}

llvm::Expected<unsigned short> StateValueArraySpecsAnalysis::Result::
    getKernelArgumentLaneIdStoreSlotBeginForWave64(
        KernelArgumentType Arg) const {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      KernelArgumentStoreSlots.contains(Arg),
      llvm::formatv("Arg enum {0} does not have an entry in the wave64 state "
                    "value array.",
                    Arg)));
  return KernelArgumentStoreSlots.at(Arg).first;
}

llvm::Expected<unsigned short>
StateValueArraySpecsAnalysis::Result::getKernelArgumentStoreSlotSizeForWave64(
    KernelArgumentType Arg) const {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      KernelArgumentStoreSlots.contains(Arg),
      llvm::formatv("Arg enum {0} does not have an entry in the wave64 state "
                    "value array.",
                    Arg)));
  return KernelArgumentStoreSlots.at(Arg).second;
}

}; // namespace luthier