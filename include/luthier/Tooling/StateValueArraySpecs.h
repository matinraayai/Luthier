//===-- StateValueArraySpecs.h ----------------------------------*- C++ -*-===//
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
/// This file provides functions used to query specifications of the state
/// value array (e.g. frame spill slots, where the kernel arguments are
/// stored, etc).
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOLING_STATE_VALUE_ARRAY_SPECS_H
#define LUTHIER_TOOLING_STATE_VALUE_ARRAY_SPECS_H
#include "luthier/Intrinsic/IntrinsicProcessor.h"
#include <llvm/Support/Error.h>

namespace luthier::stateValueArray {

/// \return \c true if \p Reg belongs to a spill slot on the state value array,
/// \c false otherwise
bool isFrameSpillSlot(llvm::MCRegister Reg);

llvm::iterator_range<
    llvm::SmallDenseMap<llvm::MCRegister, unsigned short, 8>::const_iterator>
getFrameSpillSlots();

/// \param Reg SGPRs that clobber the frame of an AMD GPU device function with
/// the C-calling convention, i.e. s0, s1, s2, s3, s32, s33, FS_LO, and FS_HI
/// \return the lane ID in the state value array where the SGPR is spilled, or
/// 255 if the register doesn't get clobbered by a device function's stack frame
unsigned short getFrameSpillSlotLaneId(llvm::MCRegister Reg);

unsigned short
getInstrumentationStackFrameLaneIdStoreSlot(llvm::MCRegister Reg);

llvm::iterator_range<
    llvm::SmallDenseMap<llvm::MCRegister, unsigned short, 8>::const_iterator>
getFrameStoreSlots();

llvm::Expected<unsigned short>
getKernelArgumentLaneIdStoreSlotBeginForWave64(KernelArgumentType Arg);

llvm::Expected<unsigned short>
getKernelArgumentStoreSlotSizeForWave64(KernelArgumentType Arg);

} // namespace luthier::stateValueArray

#endif