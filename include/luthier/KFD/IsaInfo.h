//===-- IsaInfo.h - a GPU's instruction set, from KFD -----------*- C++ -*-===//
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
/// Names a GPU's instruction set using only the KFD driver.
///
/// \par Why not HSA
/// \c HSATool::buildTargetMachineForKD gets this from the kernel descriptor's
/// owning HSA agent. In an application that drives KFD directly there is no agent
/// for that pointer -- and no HSA at all, because such an application holds the
/// DRM virtual address space for its GPUs and the kernel permits only one per GPU
/// per process, so \c hsa_init fails there. Measured both orders: the
/// application's \c ACQUIRE_VM then \c hsa_init gives
/// \c HSA_STATUS_ERROR_OUT_OF_RESOURCES, and the reverse makes the application's
/// \c ACQUIRE_VM fail with \c EBUSY.
///
/// \par Where each fact comes from
/// | fact | source |
/// | --- | --- |
/// | triple | constant, \c amdgcn-amd-amdhsa |
/// | CPU | \c gfx_target_version in the node's sysfs \c properties |
/// | sramecc | bit 26 (\c SRAM_EDCSupport) of the same file's \c capability word |
/// | xnack | \c AMDKFD_IOC_SET_XNACK_MODE with \c xnack_enabled \c = \c -1, which queries rather than sets |
/// | wavefront size, CU/WGP mode | the kernel descriptor, exactly as on the HSA path |
///
/// Verified on gfx908: this produces \c gfx908 with \c +sramecc and \c -xnack,
/// which is what \c rocminfo and the canonical MI100 target agree on.
///
/// \par Why the version is turned into a name by lookup rather than by formula
/// \c gfx_target_version encodes a (major, minor, stepping) triple, and the
/// obvious formula -- major in decimal, then minor and stepping as hex digits --
/// does reproduce every name we have checked, \c gfx90a included. It is still not
/// what this uses. LLVM keeps the authoritative list, and asking it has two
/// properties a formula does not: a chip LLVM has never heard of fails loudly
/// rather than producing a plausible CPU string that nothing accepts, and the
/// answer cannot drift from the LLVM the tool is actually compiled against.
///
/// Kept apart from \c KfdTargetMachine.h, which turns these facts into an
/// \c llvm::TargetMachine, so that asking what the driver says costs a caller
/// nothing but LLVM's target parser.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_ISA_INFO_H
#define LUTHIER_KFD_ISA_INFO_H
#include <llvm/Support/Error.h>

#include <cstdint>
#include <string>

namespace luthier::kfd {

/// \brief What the driver says about a GPU's instruction set.
///
/// Deliberately the driver's answer and nothing more: whether a chip \e supports
/// sramecc or xnack at all is LLVM's question, not KFD's, and mixing the two here
/// would make it impossible to tell "the chip has it and it is off" from "the
/// chip does not have it".
struct IsaInfo {
  unsigned Major{0};
  unsigned Minor{0};
  unsigned Stepping{0};

  /// From the \c capability word's \c SRAM_EDCSupport bit. Meaningful only for
  /// chips that have sramecc.
  bool SrameccEnabled{false};

  /// From the driver's current xnack mode for this process. Meaningful only for
  /// chips that have xnack.
  bool XnackEnabled{false};
};

/// \brief Ask the driver about the GPU with this \c gpu_id.
///
/// \note \p GpuId is the identifier ioctls use, not a topology node index. See
/// \c Topology.h.
[[nodiscard]] llvm::Expected<IsaInfo> queryIsaInfo(uint32_t GpuId);

/// \brief The LLVM AMDGPU CPU name for an ISA version triple.
///
/// Found by asking LLVM for its own list of valid architectures and matching on
/// version, so an unrecognised chip is an error rather than a guess.
[[nodiscard]] llvm::Expected<std::string>
archNameForIsaVersion(unsigned Major, unsigned Minor, unsigned Stepping);

} // namespace luthier::kfd

#endif // LUTHIER_KFD_ISA_INFO_H
