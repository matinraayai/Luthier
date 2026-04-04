//===-- Region.cpp --------------------------------------------------------===//
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
/// \file Region.cpp
/// Implements wrappers for HSA \c hsa_region_t queries.
//===----------------------------------------------------------------------===//
#include "luthier/HSA/Region.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/HsaError.h"
#include <llvm/Support/FormatVariadic.h>

namespace luthier::hsa {

template <typename T>
static llvm::Expected<T>
regionGetInfoT(const ApiTableContainer<::CoreApiTable> &CoreApi,
               hsa_region_t Region, hsa_region_info_t Attr) {
  T Val{};
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_region_get_info>(Region, Attr, &Val),
      llvm::formatv("hsa_region_get_info({0}) failed for region {1:x}", Attr,
                    Region.handle)));
  return Val;
}

llvm::Expected<hsa_region_segment_t>
regionGetSegment(const ApiTableContainer<::CoreApiTable> &CoreApi,
                 hsa_region_t Region) {
  return regionGetInfoT<hsa_region_segment_t>(CoreApi, Region,
                                              HSA_REGION_INFO_SEGMENT);
}

llvm::Expected<uint32_t>
regionGetGlobalFlags(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     hsa_region_t Region) {
  return regionGetInfoT<uint32_t>(CoreApi, Region,
                                  HSA_REGION_INFO_GLOBAL_FLAGS);
}

llvm::Expected<size_t>
regionGetSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
              hsa_region_t Region) {
  return regionGetInfoT<size_t>(CoreApi, Region, HSA_REGION_INFO_SIZE);
}

llvm::Expected<size_t>
regionGetAllocMaxSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_region_t Region) {
  return regionGetInfoT<size_t>(CoreApi, Region,
                                HSA_REGION_INFO_ALLOC_MAX_SIZE);
}

llvm::Expected<uint32_t> regionGetAllocMaxPrivateWorkgroupSize(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_region_t Region) {
  return regionGetInfoT<uint32_t>(
      CoreApi, Region, HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE);
}

llvm::Expected<bool>
regionGetRuntimeAllocAllowed(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             hsa_region_t Region) {
  return regionGetInfoT<bool>(CoreApi, Region,
                              HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED);
}

llvm::Expected<size_t>
regionGetRuntimeAllocGranule(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             hsa_region_t Region) {
  return regionGetInfoT<size_t>(CoreApi, Region,
                                HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE);
}

llvm::Expected<size_t>
regionGetRuntimeAllocAlignment(const ApiTableContainer<::CoreApiTable> &CoreApi,
                               hsa_region_t Region) {
  return regionGetInfoT<size_t>(CoreApi, Region,
                                HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT);
}

llvm::Expected<bool>
regionIsGlobal(const ApiTableContainer<::CoreApiTable> &CoreApi,
               hsa_region_t Region) {
  llvm::Expected<hsa_region_segment_t> SegOrErr =
      regionGetSegment(CoreApi, Region);
  LUTHIER_RETURN_ON_ERROR(SegOrErr.takeError());

  return *SegOrErr == HSA_REGION_SEGMENT_GLOBAL;
}

llvm::Expected<bool>
regionIsKernArg(const ApiTableContainer<::CoreApiTable> &CoreApi,
                hsa_region_t Region) {
  llvm::Expected<bool> IsGlobalOrErr = regionIsGlobal(CoreApi, Region);
  LUTHIER_RETURN_ON_ERROR(IsGlobalOrErr.takeError());

  if (!*IsGlobalOrErr)
    return false;

  llvm::Expected<uint32_t> FlagsOrErr = regionGetGlobalFlags(CoreApi, Region);
  LUTHIER_RETURN_ON_ERROR(FlagsOrErr.takeError());

  return (*FlagsOrErr & HSA_REGION_GLOBAL_FLAG_KERNARG) != 0;
}

llvm::Expected<bool>
regionIsFineGrained(const ApiTableContainer<::CoreApiTable> &CoreApi,
                    hsa_region_t Region) {
  llvm::Expected<bool> IsGlobalOrErr = regionIsGlobal(CoreApi, Region);
  LUTHIER_RETURN_ON_ERROR(IsGlobalOrErr.takeError());

  if (!*IsGlobalOrErr)
    return false;

  llvm::Expected<uint32_t> FlagsOrErr = regionGetGlobalFlags(CoreApi, Region);
  LUTHIER_RETURN_ON_ERROR(FlagsOrErr.takeError());

  return (*FlagsOrErr & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0;
}

llvm::Expected<bool>
regionIsCoarseGrained(const ApiTableContainer<::CoreApiTable> &CoreApi,
                      hsa_region_t Region) {
  llvm::Expected<bool> IsGlobalOrErr = regionIsGlobal(CoreApi, Region);
  LUTHIER_RETURN_ON_ERROR(IsGlobalOrErr.takeError());

  if (!*IsGlobalOrErr)
    return false;

  llvm::Expected<uint32_t> FlagsOrErr = regionGetGlobalFlags(CoreApi, Region);
  LUTHIER_RETURN_ON_ERROR(FlagsOrErr.takeError());

  return (*FlagsOrErr & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0;
}

} // namespace luthier::hsa
