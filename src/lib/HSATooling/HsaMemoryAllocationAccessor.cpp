//===-- HsaMemoryAllocationAccessor.cpp -----------------------------------===//
// Copyright 2025-2026 @ Northeastern University Computer Architecture Lab
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
/// \file
/// Implements the \c HsaMemoryAllocationAccessor class.
//===----------------------------------------------------------------------===//
#include "luthier/HSATooling/HsaMemoryAllocationAccessor.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSA/hsa.h"

namespace luthier {

llvm::Expected<MemoryAllocationAccessor::AllocationDescriptor>
HsaMemoryAllocationAccessor::getAllocationDescriptor(
    uint64_t DeviceAddr) const {

  hsa_executable_t Exec;
  /// First check if this address belongs to an HSA executable
  hsa_status_t Status = VenLoaderTable.hsa_ven_amd_loader_query_executable(
      reinterpret_cast<const void *>(DeviceAddr), &Exec);

  switch (Status) {
  case HSA_STATUS_SUCCESS: {
    /// Find the LCO of the device address
    llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;
    LUTHIER_RETURN_ON_ERROR(
        hsa::executableGetLoadedCodeObjects(VenLoaderTable, Exec, LCOs));

    for (const hsa_loaded_code_object_t LCO : LCOs) {
      llvm::ArrayRef<uint8_t> LoadedMemory;
      LUTHIER_RETURN_ON_ERROR(
          hsa::loadedCodeObjectGetLoadedMemory(VenLoaderTable, LCO)
              .moveInto(LoadedMemory));
      const auto LoadedStartAddr =
          reinterpret_cast<uint64_t>(LoadedMemory.data());
      const uint64_t LoadedEndAddr = LoadedStartAddr + LoadedMemory.size();
      if (LoadedStartAddr <= DeviceAddr && DeviceAddr < LoadedEndAddr) {
        /// For now directly use the host copy of the loaded memory managed
        /// by the loader
        llvm::Expected<const uint8_t *> HostCopyBaseAddrOrErr =
            hsa::queryHostAddress(VenLoaderTable, LoadedMemory.data());
        LUTHIER_RETURN_ON_ERROR(HostCopyBaseAddrOrErr.takeError());
        llvm::Expected<object::AMDGCNObjectFile &> ObjFileOrErr =
            COC.getAssociatedObjectFile(LCO);
        LUTHIER_RETURN_ON_ERROR(ObjFileOrErr.takeError());
        return AllocationDescriptor{
            LoadedMemory,
            {*HostCopyBaseAddrOrErr, LoadedMemory.size()},
            &*ObjFileOrErr};
      }
    }
    return LUTHIER_MAKE_HSA_ERROR(
        llvm::formatv("Failed to obtain the loaded code object associated with "
                      "device address {0:x}",
                      DeviceAddr));
  }
  case HSA_STATUS_ERROR_INVALID_ARGUMENT: {
    /// The queried address is not managed by the loader; We have to
    /// directly query it from HSA
    hsa_amd_pointer_info_t PointerInfo{.size = sizeof(hsa_amd_pointer_info_t)};

    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
        AmdExtTable.getTable().callFunction<hsa_amd_pointer_info>(
            reinterpret_cast<void *>(DeviceAddr), &PointerInfo, nullptr,
            nullptr, nullptr),
        llvm::formatv("Failed to get HSA allocation info for address {0:x}.",
                      DeviceAddr)));
    if (PointerInfo.type == HSA_EXT_POINTER_TYPE_UNKNOWN)
      return LUTHIER_MAKE_HSA_ERROR("Pointer is not managed by HSA");
    /// If the allocation already has a host-accessible copy, return it
    if (PointerInfo.hostBaseAddress != nullptr) {
      return AllocationDescriptor{
          llvm::ArrayRef{static_cast<uint8_t *>(PointerInfo.agentBaseAddress),
                         PointerInfo.sizeInBytes},
          llvm::ArrayRef{static_cast<uint8_t *>(PointerInfo.hostBaseAddress),
                         PointerInfo.sizeInBytes},
          nullptr};
    } else {
      /// Otherwise, copy the memory to host, and cache it if not already cached
      auto CacheIt =
          CachedAllocationsHostCopy.find(PointerInfo.agentBaseAddress);
      if (CacheIt == CachedAllocationsHostCopy.end()) {
        std::vector<uint8_t> HostMemory(PointerInfo.sizeInBytes);

        LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
            CoreTable.getTable().callFunction<hsa_memory_copy>(
                HostMemory.data(), PointerInfo.agentBaseAddress,
                PointerInfo.sizeInBytes),
            llvm::formatv(
                "Failed to cache a copy for HSA allocation located at {0:x}",
                PointerInfo.agentBaseAddress)));
        CacheIt =
            CachedAllocationsHostCopy
                .insert({PointerInfo.agentBaseAddress, std::move(HostMemory)})
                .first;
      }
      return AllocationDescriptor{
          llvm::ArrayRef{static_cast<uint8_t *>(PointerInfo.agentBaseAddress),
                         PointerInfo.sizeInBytes},
          CacheIt->second, nullptr};
    }
  }
  default:
    return LUTHIER_MAKE_HSA_ERROR_WITH_STATUS("Failed to query the HSA loader",
                                              Status);
  }
}

} // namespace luthier