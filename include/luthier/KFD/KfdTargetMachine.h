//===-- KfdTargetMachine.h --------------------------------------*- C++ -*-===//
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
/// Builds the \c llvm::TargetMachine a kernel must be lifted against, using only
/// the KFD driver.
///
/// \par Why this exists next to the HSA one
/// \c HSATool::buildTargetMachineForKD gets there from the kernel descriptor's
/// owning HSA agent: pointer info gives the agent, the agent gives an ISA, the
/// ISA gives triple, CPU and features. In an application that drives KFD
/// directly there is no agent for that pointer -- and no HSA at all, because
/// such an application holds the DRM virtual address space for its GPUs and the
/// kernel permits only one per GPU per process, so \c hsa_init fails there.
/// Measured both orders: the application's \c ACQUIRE_VM then \c hsa_init gives
/// \c HSA_STATUS_ERROR_OUT_OF_RESOURCES, and the reverse makes the application's
/// \c ACQUIRE_VM fail with \c EBUSY.
///
/// So the same three facts are recovered from the driver instead. All of them
/// are available; none of them needs a runtime.
///
/// The facts it is built from come from \c IsaInfo.h; see there for how each
/// one is obtained without a runtime.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_KFD_TARGET_MACHINE_H
#define LUTHIER_KFD_KFD_TARGET_MACHINE_H
#include "luthier/KFD/IsaInfo.h"

#include <llvm/Support/AMDHSAKernelDescriptor.h>
#include <llvm/Support/Error.h>
#include <llvm/Target/TargetMachine.h>

#include <cstdint>
#include <memory>
#include <string>

namespace luthier {

/// \brief Build the \c TargetMachine for a kernel dispatched on \p GpuId.
///
/// The KFD counterpart of \c HSATool::buildTargetMachineForKD, and it folds in
/// the same two per-kernel facts from \p KD: the wavefront size and the CU/WGP
/// execution mode live in the kernel descriptor rather than in the ISA, and the
/// lifted MIR depends on the subtarget reflecting both.
///
/// \note The AMDGPU target must already be registered with LLVM.
[[nodiscard]] llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
buildTargetMachineForKfdDispatch(uint32_t GpuId,
                                 const llvm::amdhsa::kernel_descriptor_t &KD);

} // namespace luthier

#endif // LUTHIER_KFD_KFD_TARGET_MACHINE_H
