//===-- Region.h ------------------------------------------------*- C++ -*-===//
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
/// \file Region.h
/// Defines a set of commonly used functionality for the \c hsa_region_t handle
/// in HSA.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_REGION_H
#define LUTHIER_HSA_REGION_H
#include "luthier/HSA/ApiTable.h"
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <optional>

namespace luthier::hsa {

//===----------------------------------------------------------------------===//
// Region info queries
//===----------------------------------------------------------------------===//

/// Queries the segment type of the \p Region (GLOBAL, GROUP, PRIVATE, etc.)
/// \sa hsa_region_get_info, HSA_REGION_INFO_SEGMENT
llvm::Expected<hsa_region_segment_t>
regionGetSegment(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_region_t Region);

/// Queries the global flags of the \p Region
/// The result is a bit-field of \c hsa_region_global_flag_t values.
/// Only valid if the region's segment is \c HSA_REGION_SEGMENT_GLOBAL.
/// \sa hsa_region_get_info with HSA_REGION_INFO_GLOBAL_FLAGS
llvm::Expected<uint32_t>
regionGetGlobalFlags(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_region_t Region);

/// Queries the total size (in bytes) of the \p Region.
/// \sa hsa_region_get_info with HSA_REGION_INFO_SIZE
llvm::Expected<size_t>
regionGetSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
              hsa_region_t Region);

/// Queries the maximum allocation size from the \p Region.
/// \sa hsa_region_get_info with HSA_REGION_INFO_ALLOC_MAX_SIZE
llvm::Expected<size_t>
regionGetAllocMaxSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_region_t Region);

/// Queries the maximum private segment size per workgroup for the \p Region.
/// Only valid for private segment regions.
/// \sa hsa_region_get_info with
/// HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE
llvm::Expected<uint32_t> regionGetAllocMaxPrivateWorkgroupSize(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_region_t Region);

/// Queries whether runtime allocation is allowed from the \p Region.
/// \sa hsa_region_get_info with HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED
llvm::Expected<bool>
regionGetRuntimeAllocAllowed(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             hsa_region_t Region);

/// Queries the minimum allocation granularity (in bytes) from the \p Region.
/// Only valid if runtime allocation is allowed.
/// \sa hsa_region_get_info with HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE
llvm::Expected<size_t>
regionGetRuntimeAllocGranule(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             hsa_region_t Region);

/// Queries the required allocation alignment (in bytes) from the \p Region.
/// Only valid if runtime allocation is allowed.
/// \sa hsa_region_get_info with HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT
llvm::Expected<size_t>
regionGetRuntimeAllocAlignment(const ApiTableContainer<::CoreApiTable> &CoreApi,
                               hsa_region_t Region);

//===----------------------------------------------------------------------===//
// Region convenience queries
//===----------------------------------------------------------------------===//

/// \returns true if the \p Region is a global segment region.
llvm::Expected<bool>
regionIsGlobal(const ApiTableContainer<::CoreApiTable> &CoreApi,
               hsa_region_t Region);

/// \returns true if the \p Region has the \c HSA_REGION_GLOBAL_FLAG_KERNARG
/// flag set.
llvm::Expected<bool>
regionIsKernArg(const ApiTableContainer<::CoreApiTable> &CoreApi,
                hsa_region_t Region);

/// \returns true if the \p Region has the
/// \c HSA_REGION_GLOBAL_FLAG_FINE_GRAINED flag set.
llvm::Expected<bool>
regionIsFineGrained(const ApiTableContainer<::CoreApiTable> &CoreApi,
                    hsa_region_t Region);

/// \returns true if the \p Region has the
/// \c HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED flag set.
llvm::Expected<bool>
regionIsCoarseGrained(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_region_t Region);

} // namespace luthier::hsa

//===----------------------------------------------------------------------===//
// LLVM DenseMapInfo
//===----------------------------------------------------------------------===//

template <> struct llvm::DenseMapInfo<hsa_region_t> {
  static hsa_region_t getEmptyKey() {
    return hsa_region_t(
        {DenseMapInfo<decltype(hsa_region_t::handle)>::getEmptyKey()});
  }
  static hsa_region_t getTombstoneKey() {
    return hsa_region_t(
        {DenseMapInfo<decltype(hsa_region_t::handle)>::getTombstoneKey()});
  }
  static unsigned getHashValue(const hsa_region_t &R) {
    return DenseMapInfo<decltype(hsa_region_t::handle)>::getHashValue(R.handle);
  }
  static bool isEqual(const hsa_region_t &Lhs, const hsa_region_t &Rhs) {
    return Lhs.handle == Rhs.handle;
  }
};

//===----------------------------------------------------------------------===//
// std hash / equal_to
//===----------------------------------------------------------------------===//

namespace std {

template <> struct hash<hsa_region_t> {
  size_t operator()(const hsa_region_t &Obj) const noexcept {
    return hash<unsigned long>()(Obj.handle);
  }
};

template <> struct equal_to<hsa_region_t> {
  bool operator()(const hsa_region_t &Lhs, const hsa_region_t &Rhs) const {
    return Lhs.handle == Rhs.handle;
  }
};

} // namespace std

#endif // LUTHIER_HSA_REGION_H
