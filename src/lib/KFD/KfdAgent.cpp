//===-- KfdAgent.cpp - a gpu_id's HSA agent -------------------------------===//
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
/// Implements \c luthier/KFD/KfdAgent.h. See that header for why the bridge goes
/// through the topology node index rather than straight from an HSA attribute.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/KfdAgent.h"

#include "luthier/Common/GenericLuthierError.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/HsaError.h"
#include "luthier/KFD/Topology.h"
#include "luthier/LLVM/streams.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>

#include <hsa/hsa_ext_amd.h>

#define DEBUG_TYPE "luthier-kfd-agent"

namespace luthier::kfd {

llvm::Expected<hsa_agent_t>
agentForGpuId(const hsa::ApiTableContainer<::CoreApiTable> &CoreApi,
              uint32_t GpuId) {
  if (GpuId == 0)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "gpu_id 0 is a CPU node in KFD's topology, so no GPU agent can "
        "correspond to it.");

  llvm::SmallVector<hsa_agent_t, 8> Agents;
  LUTHIER_RETURN_ON_ERROR(
      hsa::getAllAgentsWithDeviceType<HSA_DEVICE_TYPE_GPU>(CoreApi, Agents));

  for (const hsa_agent_t Agent : Agents) {
    uint32_t Node = 0;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
        CoreApi.callFunction<hsa_agent_get_info>(
            Agent,
            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
            &Node),
        "Failed to read an agent's KFD topology node index."));

    // The node index is not a gpu_id; sysfs is what relates them.
    std::optional<uint32_t> Found = gpuIdForTopologyNode(Node);
    if (Found && *Found == GpuId) {
      LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                     "[KfdAgent] gpu_id {0} is topology node {1}, agent {2:x}\n",
                     GpuId, Node, Agent.handle));
      return Agent;
    }
  }

  return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
      "No HSA agent corresponds to KFD gpu_id {0}. Either HSA is not "
      "initialized in this process, or it was restricted to a subset of the "
      "machine's GPUs -- note that a restriction like ROCR_VISIBLE_DEVICES "
      "hides agents from HSA while leaving the device visible to the driver, so "
      "the dispatch can be observed on a GPU no agent names. {1} GPU agent(s) "
      "were examined.",
      GpuId, Agents.size()));
}

} // namespace luthier::kfd
