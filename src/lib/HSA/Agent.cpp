//===-- Agent.cpp ---------------------------------------------------------===//
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
/// \file Agent.cpp
/// Implements a set of commonly used functionality around the \c hsa_agent_t
/// handle in HSA.
//===----------------------------------------------------------------------===//
#include "luthier/HSA/Agent.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSA/HsaError.h"
#include "luthier/HSA/Region.h"
#include <llvm/Support/FormatVariadic.h>

namespace luthier::hsa {

llvm::Error getAllAgents(const ApiTableContainer<::CoreApiTable> &CoreApi,
                         llvm::SmallVectorImpl<hsa_agent_t> &Agents) {
  return iterateAgents(CoreApi, [&](hsa_agent_t Agent) -> llvm::Error {
    Agents.emplace_back(Agent);
    return llvm::Error::success();
  });
}

template <typename T>
static llvm::Expected<T>
agentGetInfoT(const ApiTableContainer<::CoreApiTable> &CoreApi,
              hsa_agent_t Agent, hsa_agent_info_t Attr) {
  T Val{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_agent_get_info>(Agent, Attr, &Val),
      llvm::formatv("hsa_agent_get_info({0}) failed for agent {1:x}", Attr,
                    Agent.handle)));
  return Val;
}

llvm::Expected<std::string>
agentGetName(const ApiTableContainer<::CoreApiTable> &CoreApi,
             hsa_agent_t Agent) {
  char Buf[64]{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_agent_get_info>(Agent, HSA_AGENT_INFO_NAME, Buf),
      llvm::formatv("Failed to get the name of agent {0:x}", Agent.handle)));
  return std::string(Buf);
}

llvm::Expected<std::string>
agentGetVendorName(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_agent_t Agent) {
  char Buf[64]{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_agent_get_info>(Agent,
                                               HSA_AGENT_INFO_VENDOR_NAME, Buf),
      llvm::formatv("Failed to get the vendor name of agent {0:x}",
                    Agent.handle)));
  return std::string(Buf);
}

llvm::Expected<hsa_device_type_t>
agentGetDeviceType(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_agent_t Agent) {
  return agentGetInfoT<hsa_device_type_t>(CoreApi, Agent,
                                          HSA_AGENT_INFO_DEVICE);
}

llvm::Expected<uint32_t>
agentGetQueueMinSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_agent_t Agent) {
  return agentGetInfoT<uint32_t>(CoreApi, Agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE);
}

llvm::Expected<uint32_t>
agentGetQueueMaxSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_agent_t Agent) {
  return agentGetInfoT<uint32_t>(CoreApi, Agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE);
}

llvm::Expected<hsa_queue_type32_t>
agentGetQueueType(const ApiTableContainer<::CoreApiTable> &CoreApi,
                  hsa_agent_t Agent) {
  return agentGetInfoT<hsa_queue_type32_t>(CoreApi, Agent,
                                           HSA_AGENT_INFO_QUEUE_TYPE);
}

llvm::Expected<hsa_agent_feature_t>
agentGetFeature(const ApiTableContainer<::CoreApiTable> &CoreApi,
                hsa_agent_t Agent) {
  return agentGetInfoT<hsa_agent_feature_t>(CoreApi, Agent,
                                            HSA_AGENT_INFO_FEATURE);
}

llvm::Expected<uint16_t>
agentGetVersionMajor(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_agent_t Agent) {
  return agentGetInfoT<uint16_t>(CoreApi, Agent, HSA_AGENT_INFO_VERSION_MAJOR);
}

llvm::Expected<uint16_t>
agentGetVersionMinor(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_agent_t Agent) {
  return agentGetInfoT<uint16_t>(CoreApi, Agent, HSA_AGENT_INFO_VERSION_MINOR);
}

llvm::Error agentGetCaches(const ApiTableContainer<::CoreApiTable> &CoreApi,
                           hsa_agent_t Agent,
                           llvm::SmallVectorImpl<hsa_cache_t> &Caches) {
  return agentIterateCaches(CoreApi, Agent, [&](hsa_cache_t C) -> llvm::Error {
    Caches.push_back(C);
    return llvm::Error::success();
  });
}

llvm::Error agentGetRegions(const ApiTableContainer<::CoreApiTable> &CoreApi,
                            hsa_agent_t Agent,
                            llvm::SmallVectorImpl<hsa_region_t> &Regions) {
  return agentIterateRegions(CoreApi, Agent,
                             [&](hsa_region_t R) -> llvm::Error {
                               Regions.push_back(R);
                               return llvm::Error::success();
                             });
}

llvm::Expected<std::optional<hsa_region_t>>
agentFindKernargRegion(const ApiTableContainer<::CoreApiTable> &CoreApi,
                       hsa_agent_t Agent) {
  return agentFindFirstRegion(CoreApi, Agent,
                              [&](hsa_region_t R) -> llvm::Expected<bool> {
                                return regionIsKernArg(CoreApi, R);
                              });
}

llvm::Expected<std::optional<hsa_region_t>>
agentFindFineGrainedRegion(const ApiTableContainer<::CoreApiTable> &CoreApi,
                           hsa_agent_t Agent) {
  return agentFindFirstRegion(CoreApi, Agent,
                              [&](hsa_region_t R) -> llvm::Expected<bool> {
                                return regionIsFineGrained(CoreApi, R);
                              });
}

llvm::Expected<std::optional<hsa_region_t>>
agentFindCoarseGrainedRegion(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             hsa_agent_t Agent) {
  return agentFindFirstRegion(CoreApi, Agent,
                              [&](hsa_region_t R) -> llvm::Expected<bool> {
                                return regionIsCoarseGrained(CoreApi, R);
                              });
}

llvm::Error
agentGetSupportedISAs(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      const hsa_agent_t Agent,
                      llvm::SmallVectorImpl<hsa_isa_t> &ISAList) {
  auto Iterator = [](hsa_isa_t Isa, void *Data) {
    auto SupportedIsaList =
        static_cast<llvm::SmallVectorImpl<hsa_isa_t> *>(Data);
    SupportedIsaList->emplace_back(Isa);
    return HSA_STATUS_SUCCESS;
  };
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_agent_iterate_isas>(Agent, Iterator, &ISAList),
      llvm::formatv("Failed to iterate over ISAs of Agent {0:x}"));
}

} // namespace luthier::hsa