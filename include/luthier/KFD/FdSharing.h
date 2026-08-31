//===-- FdSharing.h - one DRM file per GPU, shared -------------*- C++ -*-===//
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
/// Makes every party in the process use one DRM file per GPU.
///
/// \par The problem
/// A GPU's virtual address space is claimed by opening its DRM render node and
/// calling \c AMDKFD_IOC_ACQUIRE_VM, and the kernel permits \b one such claim per
/// GPU per process. An application that drives KFD directly makes that claim, and
/// so does \c hsa_init -- with its own descriptor. Whichever goes second fails:
/// measured, application-first gives \c hsa_init
/// \c HSA_STATUS_ERROR_OUT_OF_RESOURCES, and the reverse makes the application's
/// \c ACQUIRE_VM fail with \c EBUSY. Instrumenting a KFD application needs HSA in
/// the process, so this has to be solved rather than avoided.
///
/// \par The fix
/// The kernel refuses a second address \e space, not a second \e call: an
/// \c ACQUIRE_VM naming the DRM file that is already bound returns 0. So if every
/// party ends up holding the same file, every claim succeeds. This redirects
/// later opens of a render node onto the descriptor already bound for that GPU.
///
/// \par Which descriptor, and why the obvious answer is wrong
/// Not the first one opened. hsakmt opens the render node, hands it to libdrm
/// which opens its \e own, closes the first, and calls \c ACQUIRE_VM with
/// libdrm's (\c libhsakmt/src/fmm.c:2329-2335). Measured: the first open returned
/// fd 6 while \c ACQUIRE_VM bound fd 8. Sharing the first descriptor therefore
/// shares a file the driver never bound -- and because both render nodes' first
/// opens reused the same descriptor number, the second GPU was handed the first
/// GPU's file, which presented as \c hsa_init failing rather than as anything
/// about descriptors.
///
/// The authoritative descriptor is the one \c ACQUIRE_VM carries, which
/// \c luthier::kfd::recordGpuDrmFd already captures. This module only decides who
/// gets a copy of it.
///
/// \par Ordering
/// The application must claim first. Its own opens pass through untouched, its
/// \c ACQUIRE_VM records the authoritative descriptor, and only then does HSA's
/// open get redirected. Done the other way round the application is the party
/// that fails, and tinygrad does not guard that call
/// (\c tinygrad/runtime/ops_amd.py:724). Failing on our side is recoverable;
/// failing on theirs is a crash in someone else's program.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_FD_SHARING_H
#define LUTHIER_KFD_FD_SHARING_H

#include <cstdint>

namespace luthier::kfd {

/// \brief Whether \p Path names a DRM render node.
[[nodiscard]] bool isRenderNodePath(const char *Path);

/// \brief The \c gpu_id whose render node is at \p Path.
///
/// Derived from the node's \c drm_render_minor in the KFD topology, which is the
/// same place \c renderNodeForGpuId reads it from -- this is that walk inverted.
///
/// \return the \c gpu_id, or 0 when no GPU node reports this path. 0 is never a
/// GPU in KFD's topology, so it doubles as "not found".
[[nodiscard]] uint32_t gpuIdForRenderNodePath(const char *Path);

/// \brief A descriptor onto the DRM file already bound for the GPU at \p Path.
///
/// \return a fresh descriptor the caller owns, or -1 when nothing is bound for
/// that GPU yet -- in which case the caller must let the real \c open through, so
/// that whoever is opening becomes the party whose descriptor gets bound.
///
/// \note A duplicate, never the recorded number itself. The caller will
/// \c close() what it believes is its own descriptor, and closing the tracker's
/// would unbind the address space for everyone.
[[nodiscard]] int borrowBoundRenderNodeFd(const char *Path);

/// \brief Whether redirection is switched on.
///
/// Off unless a tool asks for it. The wrapper is preloaded into applications that
/// did not ask for any of this, and handing one of them a descriptor it did not
/// open is a change in behaviour rather than an observation -- so it happens only
/// when something in the process actually needs HSA alongside the application.
[[nodiscard]] bool isFdSharingEnabled();

/// \brief Switch redirection on. Called by a tool before it initializes HSA.
void enableFdSharing();

} // namespace luthier::kfd

extern "C" {

/// \brief C-linkage form of \c luthier::kfd::enableFdSharing, for a tool that
/// lives in a different module from the wrapper and resolves it at run time.
void luthierKfdEnableFdSharing();

/// \brief C-linkage form of \c luthier::kfd::isFdSharingEnabled.
int luthierKfdFdSharingEnabled();
}

#endif // LUTHIER_KFD_FD_SHARING_H
