//===-- HsaMemoryAllocationAccessor.cpp -----------------------------------===//
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
/// \file
/// Implements the \c HsaMemoryAllocationAccessor class. See its header for the
/// order the sources are asked in and why.
//===----------------------------------------------------------------------===//
#include "luthier/HSATooling/HsaMemoryAllocationAccessor.h"
#include "luthier/LLVM/streams.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSA/hsa.h"
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>

#define DEBUG_TYPE "luthier-hsa-memory-allocation-accessor"

namespace luthier {

llvm::Expected<MemoryAllocationAccessor::AllocationDescriptor>
HsaMemoryAllocationAccessor::askDriverResolver(uint64_t DeviceAddr) const {
  if (DriverResolver == nullptr || !DriverResolver->isAvailable()) {
    LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                   "[HsaMemoryAllocationAccessor] no driver-level allocation "
                   "source in this process; {0:x} is unresolved.\n",
                   DeviceAddr));
    return AllocationDescriptor();
  }

  llvm::Expected<DriverAllocationResolver::Allocation> AllocOrErr =
      DriverResolver->resolve(DeviceAddr);
  LUTHIER_RETURN_ON_ERROR(AllocOrErr.takeError());
  if (AllocOrErr->empty())
    return AllocationDescriptor();

  LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                 "[HsaMemoryAllocationAccessor] {0:x} resolved by the "
                 "driver-level source to the allocation at {1:x} of size "
                 "{2:x}.\n",
                 DeviceAddr,
                 reinterpret_cast<uint64_t>(AllocOrErr->DeviceBase),
                 AllocOrErr->Size));

  /// Built through the shared conversion rather than inline: which base goes in
  /// which slot is the part two accessors must not disagree about, and getting it
  /// wrong is invisible until a source appears whose host view lives elsewhere.
  return DriverOnlyMemoryAllocationAccessor::descriptorFor(*AllocOrErr);
}

llvm::Expected<MemoryAllocationAccessor::AllocationDescriptor>
HsaMemoryAllocationAccessor::getAllocationDescriptor(
    uint64_t DeviceAddr) const {

  /// HSA is not merely absent here but unreadable: the captured API tables were
  /// never filled, and reading one that was not is a fatal error by design. So
  /// this check is what keeps a KFD-only process alive, not an optimization --
  /// see the class comment for why HSA cannot be initialized in such a process.
  if (!isHsaUsable()) {
    LLVM_DEBUG(luthier::dbgs()
               << "[HsaMemoryAllocationAccessor] HSA was never initialized in "
                  "this process; using the driver-level source alone.\n");
    return askDriverResolver(DeviceAddr);
  }

  const auto &VenLoaderTable = VenLoaderSnapshot.getTable();

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
            *reinterpret_cast<const std::byte *>(LoadedMemory.data()),
            *reinterpret_cast<const std::byte *>(*HostCopyBaseAddrOrErr),
            LoadedMemory.size(), &*ObjFileOrErr};
      }
    }
    /// The loader claims the address but no loaded code object covers it. Fall
    /// through to the driver-level source rather than reporting nothing: the
    /// loader's claim is about the executable, not about this exact byte.
    return askDriverResolver(DeviceAddr);
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
    if (PointerInfo.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
      /// HSA does not manage this address, so ask the source that watches the
      /// driver. Reporting an error here instead was the sole outlier among
      /// this interface's implementations, and it made a fallback impossible:
      /// \c InstructionTraces treats an empty descriptor as the normal end of a
      /// disassembly walk and an \c llvm::Error as a reason to abort the whole
      /// analysis, so "I have never seen this address" must not be an error.
      LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                     "[HsaMemoryAllocationAccessor] {0:x} is not managed by "
                     "HSA; asking the driver-level source.\n",
                     DeviceAddr));
      return askDriverResolver(DeviceAddr);
    }
    /// From here HSA has claimed the address, so this is the answer -- even
    /// though it carries no parsed code object. See the class comment for why a
    /// coarser driver-level answer would be worse rather than better.
    /// If the allocation already has a host-accessible copy, return it
    if (PointerInfo.hostBaseAddress != nullptr) {
      return AllocationDescriptor{
          *static_cast<std::byte *>(PointerInfo.agentBaseAddress),
          *static_cast<std::byte *>(PointerInfo.hostBaseAddress),
          PointerInfo.sizeInBytes, nullptr};
    } else {
      /// Otherwise, copy the memory to host, and cache it if not already cached
      auto CacheIt =
          CachedAllocationsHostCopy.find(PointerInfo.agentBaseAddress);
      if (CacheIt == CachedAllocationsHostCopy.end()) {
        std::vector<std::byte> HostMemory(PointerInfo.sizeInBytes);

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
          *static_cast<const std::byte *>(PointerInfo.agentBaseAddress),
          *CacheIt->second.data(), PointerInfo.sizeInBytes, nullptr};
    }
  }
  default:
    return LUTHIER_MAKE_HSA_ERROR_WITH_STATUS("Failed to query the HSA loader",
                                              Status);
  }
}

} // namespace luthier
