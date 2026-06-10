//===-- MemoryPool.h --------------------------------------------*- C++ -*-===//
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
/// \file MemoryPool.h
/// Defines a set of commonly used functionality for the
/// \c hsa_amd_memory_pool_t handle in the AMD HSA vendor extension.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_MEMORY_POOL_H
#define LUTHIER_HSA_MEMORY_POOL_H
#include "luthier/HSA/ApiTable.h"
#include "luthier/HSA/HsaError.h"
#include <hsa/hsa_ext_amd.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

//===----------------------------------------------------------------------===//
// Memory pool info queries
//===----------------------------------------------------------------------===//

/// Queries the segment type of the \p Pool.
/// \sa hsa_amd_memory_pool_get_info, HSA_AMD_MEMORY_POOL_INFO_SEGMENT
llvm::Expected<hsa_amd_segment_t>
memoryPoolGetSegment(const ApiTableContainer<::AmdExtTable> &AmdExt,
                     hsa_amd_memory_pool_t Pool);

/// Queries the global flags bit-field of the \p Pool. Only meaningful when the
/// pool's segment is \c HSA_AMD_SEGMENT_GLOBAL.
/// \sa hsa_amd_memory_pool_get_info, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS
llvm::Expected<uint32_t>
memoryPoolGetGlobalFlags(const ApiTableContainer<::AmdExtTable> &AmdExt,
                         hsa_amd_memory_pool_t Pool);

/// Queries the total size (in bytes) of the \p Pool.
/// \sa hsa_amd_memory_pool_get_info, HSA_AMD_MEMORY_POOL_INFO_SIZE
llvm::Expected<size_t>
memoryPoolGetSize(const ApiTableContainer<::AmdExtTable> &AmdExt,
                  hsa_amd_memory_pool_t Pool);

/// Queries whether runtime allocation is allowed from the \p Pool.
/// \sa hsa_amd_memory_pool_get_info,
/// HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED
llvm::Expected<bool>
memoryPoolGetRuntimeAllocAllowed(const ApiTableContainer<::AmdExtTable> &AmdExt,
                                 hsa_amd_memory_pool_t Pool);

/// Queries the minimum allocation granule of the \p Pool. Only valid if
/// runtime allocation is allowed.
/// \sa hsa_amd_memory_pool_get_info,
/// HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE
llvm::Expected<size_t>
memoryPoolGetRuntimeAllocGranule(const ApiTableContainer<::AmdExtTable> &AmdExt,
                                 hsa_amd_memory_pool_t Pool);

/// Queries the recommended allocation granule of the \p Pool. Only valid if
/// runtime allocation is allowed.
/// \sa hsa_amd_memory_pool_get_info,
/// HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE
llvm::Expected<size_t> memoryPoolGetRuntimeAllocRecGranule(
    const ApiTableContainer<::AmdExtTable> &AmdExt, hsa_amd_memory_pool_t Pool);

/// Queries the required alignment for allocations from the \p Pool. Only valid
/// if runtime allocation is allowed.
/// \sa hsa_amd_memory_pool_get_info,
/// HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT
llvm::Expected<size_t> memoryPoolGetRuntimeAllocAlignment(
    const ApiTableContainer<::AmdExtTable> &AmdExt, hsa_amd_memory_pool_t Pool);

/// Queries the maximum aggregate allocation size (in bytes) of the \p Pool.
/// \sa hsa_amd_memory_pool_get_info, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE
llvm::Expected<size_t>
memoryPoolGetAllocMaxSize(const ApiTableContainer<::AmdExtTable> &AmdExt,
                          hsa_amd_memory_pool_t Pool);

/// Queries whether the \p Pool is directly accessible to every agent in the
/// system.
/// \sa hsa_amd_memory_pool_get_info,
/// HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL
llvm::Expected<bool>
memoryPoolIsAccessibleByAll(const ApiTableContainer<::AmdExtTable> &AmdExt,
                            hsa_amd_memory_pool_t Pool);

/// Queries the location (CPU or GPU) of the \p Pool.
/// \sa hsa_amd_memory_pool_get_info, HSA_AMD_MEMORY_POOL_INFO_LOCATION
llvm::Expected<hsa_amd_memory_pool_location_t>
memoryPoolGetLocation(const ApiTableContainer<::AmdExtTable> &AmdExt,
                      hsa_amd_memory_pool_t Pool);

//===----------------------------------------------------------------------===//
// Memory pool convenience queries
//===----------------------------------------------------------------------===//

/// \returns true if the \p Pool resides in the global segment.
llvm::Expected<bool>
memoryPoolIsGlobal(const ApiTableContainer<::AmdExtTable> &AmdExt,
                   hsa_amd_memory_pool_t Pool);

/// \returns true if the \p Pool is global and has the
/// \c HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED flag set.
llvm::Expected<bool>
memoryPoolIsFineGrained(const ApiTableContainer<::AmdExtTable> &AmdExt,
                        hsa_amd_memory_pool_t Pool);

/// \returns true if the \p Pool is global and has the
/// \c HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED flag set.
llvm::Expected<bool>
memoryPoolIsCoarseGrained(const ApiTableContainer<::AmdExtTable> &AmdExt,
                          hsa_amd_memory_pool_t Pool);

/// \returns true if the \p Pool is global and has the
/// \c HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT flag set.
llvm::Expected<bool>
memoryPoolIsKernArgInit(const ApiTableContainer<::AmdExtTable> &AmdExt,
                        hsa_amd_memory_pool_t Pool);

//===----------------------------------------------------------------------===//
// Iteration
//===----------------------------------------------------------------------===//

/// Iterates over the memory pools associated with \p Agent and invokes
/// \p Callback for each. The callback must have signature
/// <tt>llvm::Error(hsa_amd_memory_pool_t)</tt>; returning a non-success error
/// stops iteration.
/// \sa hsa_amd_agent_iterate_memory_pools
template <typename CBType>
llvm::Error
agentIterateMemoryPools(const ApiTableContainer<::AmdExtTable> &AmdExt,
                        hsa_agent_t Agent, const CBType &Callback) {
  struct CBData {
    const CBType &CB;
    llvm::Error Err;
  } Data{Callback, llvm::Error::success()};

  auto Iterator = [](hsa_amd_memory_pool_t Pool, void *D) -> hsa_status_t {
    auto *Cb = static_cast<CBData *>(D);
    Cb->Err = Cb->CB(Pool);
    if (Cb->Err)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  };

  const hsa_status_t S =
      AmdExt.callFunction<hsa_amd_agent_iterate_memory_pools>(Agent, Iterator,
                                                              &Data);
  if (S == HSA_STATUS_SUCCESS || S == HSA_STATUS_INFO_BREAK)
    return std::move(Data.Err);
  // Discard the in-flight callback error (it may already be consumed) and
  // surface the HSA failure instead.
  llvm::consumeError(std::move(Data.Err));
  return LUTHIER_MAKE_HSA_ERROR(
      "Failed to iterate memory pools for the given agent.");
}

/// Collects all memory pools associated with \p Agent into \p Pools.
llvm::Error
getAllMemoryPoolsOfAgent(const ApiTableContainer<::AmdExtTable> &AmdExt,
                         hsa_agent_t Agent,
                         llvm::SmallVectorImpl<hsa_amd_memory_pool_t> &Pools);

//===----------------------------------------------------------------------===//
// Allocation
//===----------------------------------------------------------------------===//

/// Allocate \p Size bytes from the \p Pool with the given AMD \p Flags
/// (bit-field of \c hsa_amd_memory_pool_flag_t values).
/// \sa hsa_amd_memory_pool_allocate
llvm::Expected<void *>
memoryPoolAllocate(const ApiTableContainer<::AmdExtTable> &AmdExt,
                   hsa_amd_memory_pool_t Pool, size_t Size, uint32_t Flags = 0);

/// Free a block of memory previously allocated by \c memoryPoolAllocate.
/// \sa hsa_amd_memory_pool_free
llvm::Error memoryPoolFree(const ApiTableContainer<::AmdExtTable> &AmdExt,
                           void *Ptr);

/// Enable direct access to \p Ptr (previously returned by
/// \c memoryPoolAllocate) from the given \p Agents.
/// \sa hsa_amd_agents_allow_access
llvm::Error agentsAllowAccess(const ApiTableContainer<::AmdExtTable> &AmdExt,
                              llvm::ArrayRef<hsa_agent_t> Agents,
                              const void *Ptr);

//===----------------------------------------------------------------------===//
// Migration & relationships
//===----------------------------------------------------------------------===//

/// Queries whether buffers in \p SrcPool can be relocated to \p DstPool.
/// \sa hsa_amd_memory_pool_can_migrate
llvm::Expected<bool>
memoryPoolCanMigrate(const ApiTableContainer<::AmdExtTable> &AmdExt,
                     hsa_amd_memory_pool_t SrcPool,
                     hsa_amd_memory_pool_t DstPool);

/// Queries the access permission of \p Agent to \p Pool.
/// \sa hsa_amd_agent_memory_pool_get_info,
/// HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS
llvm::Expected<hsa_amd_memory_pool_access_t>
agentMemoryPoolGetAccess(const ApiTableContainer<::AmdExtTable> &AmdExt,
                         hsa_agent_t Agent, hsa_amd_memory_pool_t Pool);

/// Queries the number of link hops between \p Agent and \p Pool.
/// \sa hsa_amd_agent_memory_pool_get_info,
/// HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS
llvm::Expected<uint32_t>
agentMemoryPoolGetNumLinkHops(const ApiTableContainer<::AmdExtTable> &AmdExt,
                              hsa_agent_t Agent, hsa_amd_memory_pool_t Pool);

/// \returns true if \p Agent can directly (with no \c
/// hsa_amd_agents_allow_access call) access \p Pool — i.e., the access
/// attribute is not
/// \c HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED.
llvm::Expected<bool>
agentCanAccessMemoryPool(const ApiTableContainer<::AmdExtTable> &AmdExt,
                         hsa_agent_t Agent, hsa_amd_memory_pool_t Pool);

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo
//===----------------------------------------------------------------------===//

template <> struct llvm::DenseMapInfo<hsa_amd_memory_pool_t> {
  static hsa_amd_memory_pool_t getEmptyKey() {
    return hsa_amd_memory_pool_t(
        {DenseMapInfo<decltype(hsa_amd_memory_pool_t::handle)>::getEmptyKey()});
  }
  static unsigned getHashValue(const hsa_amd_memory_pool_t &P) {
    return DenseMapInfo<decltype(hsa_amd_memory_pool_t::handle)>::getHashValue(
        P.handle);
  }
  static bool isEqual(const hsa_amd_memory_pool_t &Lhs,
                      const hsa_amd_memory_pool_t &Rhs) {
    return Lhs.handle == Rhs.handle;
  }
};

//===----------------------------------------------------------------------===//
// std hash / equal_to
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<hsa_amd_memory_pool_t> {
  size_t operator()(const hsa_amd_memory_pool_t &Obj) const noexcept {
    return hash<unsigned long>()(Obj.handle);
  }
};

template <> struct equal_to<hsa_amd_memory_pool_t> {
  bool operator()(const hsa_amd_memory_pool_t &Lhs,
                  const hsa_amd_memory_pool_t &Rhs) const {
    return Lhs.handle == Rhs.handle;
  }
};

} // namespace std

#endif // LUTHIER_HSA_MEMORY_POOL_H
