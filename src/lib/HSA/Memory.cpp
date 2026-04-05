//===-- Memory.cpp --------------------------------------------------------===//
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
/// \file Memory.cpp
/// Implements wrappers for HSA core memory allocation and management APIs.
//===----------------------------------------------------------------------===//
#include "luthier/HSA/Memory.h"
#include "luthier/HSA/HsaError.h"
#include <llvm/Support/FormatVariadic.h>

namespace luthier::hsa {

llvm::Expected<void *>
memoryAllocate(const ApiTableContainer<::CoreApiTable> &CoreApi,
               hsa_region_t Region, size_t Size) {
  void *Ptr = nullptr;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_memory_allocate>(Region, Size, &Ptr),
      llvm::formatv("hsa_memory_allocate({0} bytes) failed for region "
                    "{1:x}",
                    Size, Region.handle)));
  return Ptr;
}

llvm::Error memoryFree(const ApiTableContainer<::CoreApiTable> &CoreApi,
                       void *Ptr) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_memory_free>(Ptr),
      llvm::formatv("hsa_memory_free({0:x}) failed", Ptr));
}

llvm::Error memoryCopy(const ApiTableContainer<::CoreApiTable> &CoreApi,
                       void *Dst, const void *Src, size_t Size) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_memory_copy>(Dst, Src, Size),
      llvm::formatv("hsa_memory_copy({0} bytes, {1:x} -> {2:x}) failed", Size,
                    Src, Dst));
}

llvm::Error memoryAssignAgent(const ApiTableContainer<::CoreApiTable> &CoreApi,
                              void *Ptr, hsa_agent_t Agent,
                              hsa_access_permission_t Access) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_memory_assign_agent>(Ptr, Agent, Access),
      llvm::formatv("hsa_memory_assign_agent({0:x}, agent {1:x}) failed", Ptr,
                    Agent.handle));
}

llvm::Error memoryRegister(const ApiTableContainer<::CoreApiTable> &CoreApi,
                           void *Ptr, size_t Size) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_memory_register>(Ptr, Size),
      llvm::formatv("hsa_memory_register({0:x}, {1} bytes) failed", Ptr, Size));
}

llvm::Error memoryDeregister(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             void *Ptr, size_t Size) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_memory_deregister>(Ptr, Size),
      llvm::formatv("hsa_memory_deregister({0:x}, {1} bytes) failed", Ptr,
                    Size));
}

} // namespace luthier::hsa
