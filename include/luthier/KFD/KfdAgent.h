//===-- KfdAgent.h - a gpu_id's HSA agent -----------------------*- C++ -*-===//
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
/// Finds the HSA agent for a GPU the driver named by \c gpu_id.
///
/// \par Why this is needed
/// Loading an instrumented kernel needs an \c hsa_agent_t.
/// \c InstrumentedKernelLoaderAndLauncher normally recovers one from the original
/// kernel descriptor with \c hsa_amd_pointer_info, which works when HSA allocated
/// the memory the descriptor sits in. In an application that drives KFD directly,
/// it did not: the descriptor is in a driver allocation HSA has never heard of,
/// pointer info reports \c HSA_EXT_POINTER_TYPE_UNKNOWN, and the loader has
/// nothing to target. What we do have is the \c gpu_id of the queue the dispatch
/// arrived on.
///
/// \par The hop that is easy to get wrong
/// There is no HSA attribute that returns a \c gpu_id. Both of the ones that look
/// like they should -- \c HSA_AGENT_INFO_NODE and
/// \c HSA_AMD_AGENT_INFO_DRIVER_NODE_ID -- return the KFD topology \e node
/// \e index (\c amd_gpu_agent.cpp:1401 and \c :1509, both \c node_id()). A node
/// index is a small dense counter that also covers CPUs; a \c gpu_id is a large
/// opaque number. So the bridge is agent -> node index -> \c gpu_id, with the
/// last step read from sysfs.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_KFD_AGENT_H
#define LUTHIER_KFD_KFD_AGENT_H
#include "luthier/HSA/ApiTable.h"

#include <llvm/Support/Error.h>

#include <cstdint>
#include <hsa/hsa.h>

namespace luthier::kfd {

/// \brief The HSA agent for the GPU the driver calls \p GpuId.
///
/// \param CoreApi used to enumerate agents; must come from a captured snapshot
/// rather than the live table, so this does not re-enter a tool's own wrappers.
/// \return the agent, or an \c llvm::Error naming \p GpuId when no HSA agent maps
/// to it -- which happens legitimately when HSA has not been initialized, or when
/// it was restricted to a subset of the machine's GPUs.
[[nodiscard]] llvm::Expected<hsa_agent_t>
agentForGpuId(const hsa::ApiTableContainer<::CoreApiTable> &CoreApi,
              uint32_t GpuId);

} // namespace luthier::kfd

#endif // LUTHIER_KFD_KFD_AGENT_H
