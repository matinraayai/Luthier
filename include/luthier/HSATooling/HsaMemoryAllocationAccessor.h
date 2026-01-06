//===-- HsaMemoryAllocationAccessor.h ---------------------------*- C++ -*-===//
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
/// Describes the \c HsaMemoryAllocationAccessor class which implements the
/// \c MemoryAllocationAccessor interface.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_HSA_MEMORY_ALLOCATION_ACCESSOR_H
#define LUTHIER_HSA_TOOLING_HSA_MEMORY_ALLOCATION_ACCESSOR_H
#include "luthier/HSATooling/LoadedCodeObjectCache.h"
#include "luthier/Tooling/MemoryAllocationAccessor.h"

namespace luthier {

/// \brief Implementation of \c MemoryAllocationAccessor interface for the HSA
/// runtime
class HsaMemoryAllocationAccessor : public MemoryAllocationAccessor {

  const hsa::LoadedCodeObjectCache &COC;

  const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreTable;

  const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExtTable;

  const hsa::ExtensionApiTableInfo<HSA_EXTENSION_AMD_LOADER>::TableType
      &VenLoaderTable;

  /// Cache for holding on to the host copy of memory allocations in HSA that
  /// don't have a host-accessible copy
  mutable llvm::SmallDenseMap<void *, std::vector<uint8_t>>
      CachedAllocationsHostCopy;

public:
  [[nodiscard]] llvm::Expected<AllocationDescriptor>
  getAllocationDescriptor(uint64_t DeviceAddr) const override;

  HsaMemoryAllocationAccessor(
      const hsa::LoadedCodeObjectCache &COC,
      const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &CoreTable,
      const rocprofiler::HsaApiTableSnapshot<::AmdExtTable> &AmdExtTable,
      const hsa::ExtensionApiTableInfo<HSA_EXTENSION_AMD_LOADER>::TableType
          &VenLoaderTable)
      : COC(COC), CoreTable(CoreTable), AmdExtTable(AmdExtTable),
        VenLoaderTable(VenLoaderTable) {};

  ~HsaMemoryAllocationAccessor() override = default;
};

} // namespace luthier

#endif