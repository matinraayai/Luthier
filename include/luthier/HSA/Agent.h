//===-- Agent.h -------------------------------------------------*- C++ -*-===//
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
/// \file Agent.h
/// Defines a set of commonly used functionality for the \c hsa_agent_t handle
/// in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_AGENT_H
#define LUTHIER_HSA_AGENT_H
#include "luthier/HSA/ApiTable.h"
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <optional>
#include <string>

namespace luthier::hsa {

//===----------------------------------------------------------------------===//
// Agent enumeration
//===----------------------------------------------------------------------===//

/// Iterates over all available HSA agents and invokes \p Callback for each.
/// Iteration stops early if the callback returns a non-success error.
/// \sa hsa_iterate_agents
template <typename CBType>
llvm::Error iterateAgents(const ApiTableContainer<::CoreApiTable> &CoreApi,
                          const CBType &Callback) {
  struct CBData {
    const CBType &CB;
    llvm::Error Err;
  } Data{Callback, llvm::Error::success()};

  auto Iterator = [](hsa_agent_t Agent, void *D) -> hsa_status_t {
    auto *Data = static_cast<CBData *>(D);
    Data->Err = Data->CB(Agent);
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t S =
          CoreApi.callFunction<hsa_iterate_agents>(Iterator, &Data);
      S == HSA_STATUS_SUCCESS || S == HSA_STATUS_INFO_BREAK)
    return std::move(Data.Err);

  return LUTHIER_MAKE_HSA_ERROR("Failed to iterate over HSA agents.");
}

/// Finds all agents with the same \c hsa_device_type_t attached to the system
/// and returns it in \p Agents
/// \sa hsa_iterate_agents, hsa_device_type_t
template <hsa_device_type_t TargetDevType>
llvm::Error
getAllAgentsWithDeviceType(const ApiTableContainer<::CoreApiTable> &CoreApi,
                           llvm::SmallVectorImpl<hsa_agent_t> &Agents) {
  auto ReturnGpuAgentsCallback = [](hsa_agent_t Agent, void *Data) {
    auto AgentList = static_cast<llvm::SmallVector<hsa_agent_t> *>(Data);
    hsa_device_type_t DevType = HSA_DEVICE_TYPE_CPU;

    const hsa_status_t Status =
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DevType);

    if (Status != HSA_STATUS_SUCCESS)
      return Status;
    if (DevType == TargetDevType) {
      AgentList->emplace_back(Agent);
    }
    return Status;
  };
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_iterate_agents>(ReturnGpuAgentsCallback,
                                               &Agents),
      "Failed to iterate over all HSA agents attached to the system");
}

/// Queries all available agents and returns them in \p Agents.
/// \sa hsa_iterate_agents
llvm::Error getAllAgents(const ApiTableContainer<::CoreApiTable> &CoreApi,
                         llvm::SmallVectorImpl<hsa_agent_t> &Agents);

//===----------------------------------------------------------------------===//
// Agent info queries
//===----------------------------------------------------------------------===//

/// Queries the name of the \p Agent.
/// \sa hsa_agent_get_info, HSA_AGENT_INFO_NAME
llvm::Expected<std::string>
agentGetName(const ApiTableContainer<::CoreApiTable> &CoreApi,
             hsa_agent_t Agent);

/// Queries the vendor name of the \p Agent.
/// \sa hsa_agent_get_info, HSA_AGENT_INFO_VENDOR_NAME
llvm::Expected<std::string>
agentGetVendorName(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_agent_t Agent);

/// Queries the device type (CPU, GPU, DSP) of the \p Agent.
/// \sa hsa_agent_get_info, HSA_AGENT_INFO_DEVICE
llvm::Expected<hsa_device_type_t>
agentGetDeviceType(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   hsa_agent_t Agent);

/// Queries the minimum queue size of the \p Agent.
/// \sa hsa_agent_get_info, HSA_AGENT_INFO_QUEUE_MIN_SIZE
llvm::Expected<uint32_t>
agentGetQueueMinSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_agent_t Agent);

/// Queries the maximum queue size of the \p Agent.
/// \sa hsa_agent_get_info, HSA_AGENT_INFO_QUEUE_MAX_SIZE
llvm::Expected<uint32_t>
agentGetQueueMaxSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_agent_t Agent);

/// Queries the supported queue type of the \p Agent.
/// \sa hsa_agent_get_info, HSA_AGENT_INFO_QUEUE_TYPE
llvm::Expected<hsa_queue_type32_t>
agentGetQueueType(const ApiTableContainer<::CoreApiTable> &CoreApi,
                  hsa_agent_t Agent);

/// Queries the feature set of the \p Agent (AGENT_DISPATCH or KERNEL_DISPATCH).
/// \sa hsa_agent_get_info, HSA_AGENT_INFO_FEATURE
llvm::Expected<hsa_agent_feature_t>
agentGetFeature(const ApiTableContainer<::CoreApiTable> &CoreApi,
                hsa_agent_t Agent);

/// Queries the version major number of the HSA specification supported by the
/// \p Agent
/// \sa hsa_agent_get_info, HSA_AGENT_INFO_VERSION_MAJOR
llvm::Expected<uint16_t>
agentGetVersionMajor(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_agent_t Agent);

/// Queries the version minor number of the HSA specification supported by the
/// \p Agent
/// \sa hsa_agent_get_info, HSA_AGENT_INFO_VERSION_MINOR
llvm::Expected<uint16_t>
agentGetVersionMinor(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_agent_t Agent);

//===----------------------------------------------------------------------===//
// Agent cache iteration
//===----------------------------------------------------------------------===//

/// Iterates over the caches of the \p Agent and invokes \p Callback.
/// \sa hsa_agent_iterate_caches
template <typename CBType>
llvm::Error agentIterateCaches(const ApiTableContainer<::CoreApiTable> &CoreApi,
                               hsa_agent_t Agent, const CBType &Callback) {
  struct CBData {
    const CBType &CB;
    llvm::Error Err;
  } Data{Callback, llvm::Error::success()};

  auto Iterator = [](hsa_cache_t Cache, void *D) -> hsa_status_t {
    auto *Data = static_cast<CBData *>(D);
    Data->Err = Data->CB(Cache);
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t S = CoreApi.callFunction<hsa_agent_iterate_caches>(
          Agent, Iterator, &Data);
      S == HSA_STATUS_SUCCESS || S == HSA_STATUS_INFO_BREAK)
    return std::move(Data.Err);

  return LUTHIER_MAKE_HSA_ERROR(llvm::formatv(
      "Failed to iterate over caches of agent {0:x}.", Agent.handle));
}

/// Queries all caches of the \p Agent.
/// \sa hsa_agent_iterate_caches
llvm::Error agentGetCaches(const ApiTableContainer<::CoreApiTable> &CoreApi,
                           hsa_agent_t Agent,
                           llvm::SmallVectorImpl<hsa_cache_t> &Caches);

//===----------------------------------------------------------------------===//
// Agent region iteration
//===----------------------------------------------------------------------===//

/// Iterates over the memory regions of the \p Agent and invokes \p Callback.
/// \sa hsa_agent_iterate_regions
template <typename CBType>
llvm::Error
agentIterateRegions(const ApiTableContainer<::CoreApiTable> &CoreApi,
                    hsa_agent_t Agent, const CBType &Callback) {
  struct CBData {
    const CBType &CB;
    llvm::Error Err;
  } Data{Callback, llvm::Error::success()};

  auto Iterator = [](hsa_region_t Region, void *D) -> hsa_status_t {
    auto *Data = static_cast<CBData *>(D);
    Data->Err = Data->CB(Region);
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t S = CoreApi.callFunction<hsa_agent_iterate_regions>(
          Agent, Iterator, &Data);
      S == HSA_STATUS_SUCCESS || S == HSA_STATUS_INFO_BREAK)
    return std::move(Data.Err);

  return LUTHIER_MAKE_HSA_ERROR(llvm::formatv(
      "Failed to iterate over regions of agent {0:x}.", Agent.handle));
}

/// Queries all memory regions of the \p Agent.
/// \sa hsa_agent_iterate_regions
llvm::Error agentGetRegions(const ApiTableContainer<::CoreApiTable> &CoreApi,
                            hsa_agent_t Agent,
                            llvm::SmallVectorImpl<hsa_region_t> &Regions);

/// Iterates over the \c hsa_region_t list of \p Agent and finds and
/// returns the first \c hsa_region_t of the \p Agent that the \p Predicate
/// returns
/// \c true on
template <typename CBType>
llvm::Expected<std::optional<hsa_region_t>>
agentFindFirstRegion(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_agent_t Agent, const CBType &Predicate) {
  struct CallbackDataType {
    const CBType &CB;
    std::optional<hsa_region_t> ISA;
    llvm::Error Err;
  } CBData{Predicate, std::nullopt, llvm::Error::success()};

  auto Iterator = [](const hsa_region_t ISA, void *D) -> hsa_status_t {
    auto *Data = static_cast<CallbackDataType *>(D);
    if (!Data) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    llvm::Expected<bool> Res = Data->CB(ISA);
    Data->Err = Res.takeError();
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    if (*Res) {
      Data->ISA = ISA;
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t Out = CoreApi.callFunction<hsa_agent_iterate_regions>(
          Agent, Iterator, &CBData);
      Out == HSA_STATUS_SUCCESS || Out == HSA_STATUS_INFO_BREAK) {
    LUTHIER_RETURN_ON_ERROR(CBData.Err);
    return CBData.ISA;
  }

  return LUTHIER_MAKE_HSA_ERROR(
      llvm::formatv("Failed to iterate over the ISAs of agent "
                    "{0:x}.",
                    Agent.handle));
}

/// Finds the first kernarg region of the \p Agent.
llvm::Expected<std::optional<hsa_region_t>>
agentFindKernargRegion(const ApiTableContainer<::CoreApiTable> &CoreApi,
                       hsa_agent_t Agent);

/// Finds the first fine-grained global region of the \p Agent.
llvm::Expected<std::optional<hsa_region_t>>
agentFindFineGrainedRegion(const ApiTableContainer<::CoreApiTable> &CoreApi,
                           hsa_agent_t Agent);

/// Finds the first coarse-grained global region of the \p Agent.
llvm::Expected<std::optional<hsa_region_t>>
agentFindCoarseGrainedRegion(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             hsa_agent_t Agent);

//===----------------------------------------------------------------------===//
// Agent ISA iteration
//===----------------------------------------------------------------------===//

/// Queries all the <tt>hsa_isa_t</tt>s supported by the \p Agent
/// \param [in] CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param [in] Agent the \c hsa_agent_t being queried
/// \param [out] ISAList list of ISAs supported by the Agent
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa hsa_agent_iterate_isas
llvm::Error
agentGetSupportedISAs(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_agent_t Agent,
                      llvm::SmallVectorImpl<hsa_isa_t> &ISAList);

/// Iterates over the supported \c hsa_isa_t list of \p Agent and invokes the
/// \p Callback
/// \param CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param Agent the agent being queried
/// \param Callback a callback function invoked for each \c hsa_isa_t of the
/// <tt>Agent</tt>. If the callback doesn't return a success error value, it
/// will halt the iteration, and the error will be returned
/// \return \c llvm::Error indication the success or failure of the operation
template <typename CBType>
llvm::Error agentIterateISAs(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             hsa_agent_t Agent, const CBType &Callback) {
  struct CallbackDataType {
    const CBType &CB;
    llvm::Error Err;
  } CBData{Callback, llvm::Error::success()};

  auto Iterator = [](const hsa_isa_t ISA, void *D) -> hsa_status_t {
    auto *Data = static_cast<CallbackDataType *>(D);
    if (!Data) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    Data->Err = Data->CB(ISA);
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t Out = CoreApi.callFunction<hsa_agent_iterate_isas>(
          Agent, Iterator, &CBData);
      Out == HSA_STATUS_SUCCESS || Out == HSA_STATUS_INFO_BREAK)
    return std::move(CBData.Err);

  return LUTHIER_MAKE_HSA_ERROR(
      llvm::formatv("Failed to iterate over the ISAs of agent "
                    "{0:x}.",
                    Agent.handle));
}

/// Iterates over the supported \c hsa_isa_t list of \p Agent and finds and
/// returns the first \c hsa_isa_t of the \p Agent that the \p Predicate returns
/// \c true on
/// \param Agent the agent being queried
/// \param CoreApi the \c ::CoreApiTable used to dispatch HSA functions
/// \param Predicate a predicate function invoked for each \c hsa_isa_t of the
/// <tt>Agent</tt>. If the callback returns a failure error value, it
/// will halt the iteration, and the error will be returned
/// \return Expects the first \c hsa_isa_t entry found by the predicate; Expects
/// a \c std::nullopt if no ISA entry was found by the predicate
template <typename CBType>
llvm::Expected<std::optional<hsa_isa_t>>
agentFindFirstISA(const ApiTableContainer<::CoreApiTable> &CoreApi,
                  hsa_agent_t Agent, const CBType &Predicate) {
  struct CallbackDataType {
    const CBType &CB;
    std::optional<hsa_isa_t> ISA;
    llvm::Error Err;
  } CBData{Predicate, std::nullopt, llvm::Error::success()};

  auto Iterator = [](const hsa_isa_t ISA, void *D) -> hsa_status_t {
    auto *Data = static_cast<CallbackDataType *>(D);
    if (!Data) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    llvm::Expected<bool> Res = Data->CB(ISA);
    Data->Err = Res.takeError();
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    if (*Res) {
      Data->ISA = ISA;
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t Out = CoreApi.callFunction<hsa_agent_iterate_isas>(
          Agent, Iterator, &CBData);
      Out == HSA_STATUS_SUCCESS || Out == HSA_STATUS_INFO_BREAK) {
    LUTHIER_RETURN_ON_ERROR(CBData.Err);
    return CBData.ISA;
  }

  return LUTHIER_MAKE_HSA_ERROR(
      llvm::formatv("Failed to iterate over the ISAs of agent "
                    "{0:x}.",
                    Agent.handle));
}

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo, for insertion into LLVM-based containers
//===----------------------------------------------------------------------===//

template <> struct llvm::DenseMapInfo<hsa_agent_t> {
  static hsa_agent_t getEmptyKey() {
    return hsa_agent_t(
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getEmptyKey()});
  }

  static hsa_agent_t getTombstoneKey() {
    return hsa_agent_t(
        {DenseMapInfo<decltype(hsa_agent_t::handle)>::getTombstoneKey()});
  }

  static unsigned getHashValue(const hsa_agent_t &Agent) {
    return DenseMapInfo<decltype(hsa_agent_t::handle)>::getHashValue(
        Agent.handle);
  }

  static bool isEqual(const hsa_agent_t &Lhs, const hsa_agent_t &Rhs) {
    return Lhs.handle == Rhs.handle;
  }
}; // namespace llvm

//===----------------------------------------------------------------------===//
// C++ std library function objects for hashing and comparison, for insertion
// into stl container
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<hsa_agent_t> {
  size_t operator()(const hsa_agent_t &Obj) const noexcept {
    return hash<unsigned long>()(Obj.handle);
  }
};

template <> struct equal_to<hsa_agent_t> {
  bool operator()(const hsa_agent_t &Lhs, const hsa_agent_t &Rhs) const {
    return Lhs.handle == Rhs.handle;
  }
};

} // namespace std

#endif