//===-- MemoryPool.cpp ----------------------------------------------------===//
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
/// \file MemoryPool.cpp
/// Implements wrappers for HSA \c hsa_amd_memory_pool_t queries and operations.
//===----------------------------------------------------------------------===//
#include "luthier/HSA/MemoryPool.h"
#include "luthier/HSA/HsaError.h"
#include <llvm/Support/FormatVariadic.h>

namespace luthier::hsa {

namespace {

template <typename T>
llvm::Expected<T>
poolGetInfoT(const ApiTableContainer<::AmdExtTable> &AmdExt,
             hsa_amd_memory_pool_t Pool, hsa_amd_memory_pool_info_t Attr) {
  T Val{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      AmdExt.callFunction<hsa_amd_memory_pool_get_info>(Pool, Attr, &Val),
      llvm::formatv(
          "hsa_amd_memory_pool_get_info({0}) failed for pool {1:x}", Attr,
          Pool.handle)));
  return Val;
}

template <typename T>
llvm::Expected<T> agentPoolGetInfoT(
    const ApiTableContainer<::AmdExtTable> &AmdExt, hsa_agent_t Agent,
    hsa_amd_memory_pool_t Pool, hsa_amd_agent_memory_pool_info_t Attr) {
  T Val{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      AmdExt.callFunction<hsa_amd_agent_memory_pool_get_info>(Agent, Pool, Attr,
                                                              &Val),
      llvm::formatv("hsa_amd_agent_memory_pool_get_info({0}) failed for "
                    "agent {1:x} pool {2:x}",
                    Attr, Agent.handle, Pool.handle)));
  return Val;
}

} // namespace

//===----------------------------------------------------------------------===//
// Memory pool info queries
//===----------------------------------------------------------------------===//

llvm::Expected<hsa_amd_segment_t>
memoryPoolGetSegment(const ApiTableContainer<::AmdExtTable> &AmdExt,
                     hsa_amd_memory_pool_t Pool) {
  return poolGetInfoT<hsa_amd_segment_t>(AmdExt, Pool,
                                         HSA_AMD_MEMORY_POOL_INFO_SEGMENT);
}

llvm::Expected<uint32_t>
memoryPoolGetGlobalFlags(const ApiTableContainer<::AmdExtTable> &AmdExt,
                         hsa_amd_memory_pool_t Pool) {
  return poolGetInfoT<uint32_t>(AmdExt, Pool,
                                HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS);
}

llvm::Expected<size_t>
memoryPoolGetSize(const ApiTableContainer<::AmdExtTable> &AmdExt,
                  hsa_amd_memory_pool_t Pool) {
  return poolGetInfoT<size_t>(AmdExt, Pool, HSA_AMD_MEMORY_POOL_INFO_SIZE);
}

llvm::Expected<bool> memoryPoolGetRuntimeAllocAllowed(
    const ApiTableContainer<::AmdExtTable> &AmdExt,
    hsa_amd_memory_pool_t Pool) {
  return poolGetInfoT<bool>(AmdExt, Pool,
                            HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED);
}

llvm::Expected<size_t> memoryPoolGetRuntimeAllocGranule(
    const ApiTableContainer<::AmdExtTable> &AmdExt,
    hsa_amd_memory_pool_t Pool) {
  return poolGetInfoT<size_t>(AmdExt, Pool,
                              HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE);
}

llvm::Expected<size_t> memoryPoolGetRuntimeAllocRecGranule(
    const ApiTableContainer<::AmdExtTable> &AmdExt,
    hsa_amd_memory_pool_t Pool) {
  return poolGetInfoT<size_t>(
      AmdExt, Pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE);
}

llvm::Expected<size_t> memoryPoolGetRuntimeAllocAlignment(
    const ApiTableContainer<::AmdExtTable> &AmdExt,
    hsa_amd_memory_pool_t Pool) {
  return poolGetInfoT<size_t>(
      AmdExt, Pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT);
}

llvm::Expected<size_t>
memoryPoolGetAllocMaxSize(const ApiTableContainer<::AmdExtTable> &AmdExt,
                          hsa_amd_memory_pool_t Pool) {
  return poolGetInfoT<size_t>(AmdExt, Pool,
                              HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE);
}

llvm::Expected<bool>
memoryPoolIsAccessibleByAll(const ApiTableContainer<::AmdExtTable> &AmdExt,
                            hsa_amd_memory_pool_t Pool) {
  return poolGetInfoT<bool>(AmdExt, Pool,
                            HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL);
}

llvm::Expected<hsa_amd_memory_pool_location_t>
memoryPoolGetLocation(const ApiTableContainer<::AmdExtTable> &AmdExt,
                      hsa_amd_memory_pool_t Pool) {
  return poolGetInfoT<hsa_amd_memory_pool_location_t>(
      AmdExt, Pool, HSA_AMD_MEMORY_POOL_INFO_LOCATION);
}

//===----------------------------------------------------------------------===//
// Memory pool convenience queries
//===----------------------------------------------------------------------===//

llvm::Expected<bool>
memoryPoolIsGlobal(const ApiTableContainer<::AmdExtTable> &AmdExt,
                   hsa_amd_memory_pool_t Pool) {
  llvm::Expected<hsa_amd_segment_t> SegOrErr =
      memoryPoolGetSegment(AmdExt, Pool);
  LUTHIER_RETURN_ON_ERROR(SegOrErr.takeError());
  return *SegOrErr == HSA_AMD_SEGMENT_GLOBAL;
}

static llvm::Expected<bool>
poolHasGlobalFlag(const ApiTableContainer<::AmdExtTable> &AmdExt,
                  hsa_amd_memory_pool_t Pool, uint32_t Flag) {
  llvm::Expected<bool> IsGlobalOrErr = memoryPoolIsGlobal(AmdExt, Pool);
  LUTHIER_RETURN_ON_ERROR(IsGlobalOrErr.takeError());
  if (!*IsGlobalOrErr)
    return false;
  llvm::Expected<uint32_t> FlagsOrErr =
      memoryPoolGetGlobalFlags(AmdExt, Pool);
  LUTHIER_RETURN_ON_ERROR(FlagsOrErr.takeError());
  return (*FlagsOrErr & Flag) != 0;
}

llvm::Expected<bool>
memoryPoolIsFineGrained(const ApiTableContainer<::AmdExtTable> &AmdExt,
                        hsa_amd_memory_pool_t Pool) {
  return poolHasGlobalFlag(AmdExt, Pool,
                           HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED);
}

llvm::Expected<bool>
memoryPoolIsCoarseGrained(const ApiTableContainer<::AmdExtTable> &AmdExt,
                          hsa_amd_memory_pool_t Pool) {
  return poolHasGlobalFlag(AmdExt, Pool,
                           HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED);
}

llvm::Expected<bool>
memoryPoolIsKernArgInit(const ApiTableContainer<::AmdExtTable> &AmdExt,
                        hsa_amd_memory_pool_t Pool) {
  return poolHasGlobalFlag(AmdExt, Pool,
                           HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);
}

//===----------------------------------------------------------------------===//
// Iteration
//===----------------------------------------------------------------------===//

llvm::Error getAllMemoryPoolsOfAgent(
    const ApiTableContainer<::AmdExtTable> &AmdExt, hsa_agent_t Agent,
    llvm::SmallVectorImpl<hsa_amd_memory_pool_t> &Pools) {
  return agentIterateMemoryPools(
      AmdExt, Agent, [&](hsa_amd_memory_pool_t Pool) -> llvm::Error {
        Pools.emplace_back(Pool);
        return llvm::Error::success();
      });
}

namespace {

/// Walks \p Agent 's pools and returns the first allocatable one for which
/// \p IsWantedGranularity holds.
template <typename PredicateType>
llvm::Expected<std::optional<hsa_amd_memory_pool_t>>
findAllocatablePool(const ApiTableContainer<::AmdExtTable> &AmdExt,
                    hsa_agent_t Agent,
                    const PredicateType &IsWantedGranularity) {
  std::optional<hsa_amd_memory_pool_t> Found;
  LUTHIER_RETURN_ON_ERROR(agentIterateMemoryPools(
      AmdExt, Agent, [&](hsa_amd_memory_pool_t Pool) -> llvm::Error {
        if (Found)
          return llvm::Error::success();
        llvm::Expected<bool> WantedOrErr = IsWantedGranularity(AmdExt, Pool);
        LUTHIER_RETURN_ON_ERROR(WantedOrErr.takeError());
        if (!*WantedOrErr)
          return llvm::Error::success();
        llvm::Expected<bool> AllocOrErr =
            memoryPoolGetRuntimeAllocAllowed(AmdExt, Pool);
        LUTHIER_RETURN_ON_ERROR(AllocOrErr.takeError());
        if (*AllocOrErr)
          Found = Pool;
        return llvm::Error::success();
      }));
  return Found;
}

} // namespace

llvm::Expected<std::optional<hsa_amd_memory_pool_t>>
agentFindCoarseGrainedPool(const ApiTableContainer<::AmdExtTable> &AmdExt,
                           hsa_agent_t Agent) {
  return findAllocatablePool(AmdExt, Agent, memoryPoolIsCoarseGrained);
}

llvm::Expected<std::optional<hsa_amd_memory_pool_t>>
agentFindFineGrainedPool(const ApiTableContainer<::AmdExtTable> &AmdExt,
                         hsa_agent_t Agent) {
  return findAllocatablePool(AmdExt, Agent, memoryPoolIsFineGrained);
}

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

llvm::Expected<void *>
memoryPoolAllocate(const ApiTableContainer<::AmdExtTable> &AmdExt,
                   hsa_amd_memory_pool_t Pool, size_t Size, uint32_t Flags) {
  void *Ptr = nullptr;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      AmdExt.callFunction<hsa_amd_memory_pool_allocate>(Pool, Size, Flags,
                                                        &Ptr),
      llvm::formatv("hsa_amd_memory_pool_allocate failed for pool {0:x} "
                    "size {1} flags {2:x}",
                    Pool.handle, Size, Flags)));
  return Ptr;
}

llvm::Error memoryPoolFree(const ApiTableContainer<::AmdExtTable> &AmdExt,
                           void *Ptr) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      AmdExt.callFunction<hsa_amd_memory_pool_free>(Ptr),
      "hsa_amd_memory_pool_free failed.");
}

llvm::Error
agentsAllowAccess(const ApiTableContainer<::AmdExtTable> &AmdExt,
                  llvm::ArrayRef<hsa_agent_t> Agents, const void *Ptr) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      AmdExt.callFunction<hsa_amd_agents_allow_access>(
          Agents.size(), Agents.data(), nullptr, Ptr),
      "hsa_amd_agents_allow_access failed.");
}

//===----------------------------------------------------------------------===//
// Migration & relationships
//===----------------------------------------------------------------------===//

llvm::Expected<bool>
memoryPoolCanMigrate(const ApiTableContainer<::AmdExtTable> &AmdExt,
                     hsa_amd_memory_pool_t SrcPool,
                     hsa_amd_memory_pool_t DstPool) {
  bool Result = false;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      AmdExt.callFunction<hsa_amd_memory_pool_can_migrate>(SrcPool, DstPool,
                                                           &Result),
      llvm::formatv(
          "hsa_amd_memory_pool_can_migrate failed for src {0:x} dst {1:x}",
          SrcPool.handle, DstPool.handle)));
  return Result;
}

llvm::Expected<hsa_amd_memory_pool_access_t>
agentMemoryPoolGetAccess(const ApiTableContainer<::AmdExtTable> &AmdExt,
                         hsa_agent_t Agent, hsa_amd_memory_pool_t Pool) {
  return agentPoolGetInfoT<hsa_amd_memory_pool_access_t>(
      AmdExt, Agent, Pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS);
}

llvm::Expected<uint32_t>
agentMemoryPoolGetNumLinkHops(const ApiTableContainer<::AmdExtTable> &AmdExt,
                              hsa_agent_t Agent, hsa_amd_memory_pool_t Pool) {
  return agentPoolGetInfoT<uint32_t>(
      AmdExt, Agent, Pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS);
}

llvm::Expected<bool>
agentCanAccessMemoryPool(const ApiTableContainer<::AmdExtTable> &AmdExt,
                         hsa_agent_t Agent, hsa_amd_memory_pool_t Pool) {
  llvm::Expected<hsa_amd_memory_pool_access_t> AccessOrErr =
      agentMemoryPoolGetAccess(AmdExt, Agent, Pool);
  LUTHIER_RETURN_ON_ERROR(AccessOrErr.takeError());
  return *AccessOrErr != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
}

} // namespace luthier::hsa
