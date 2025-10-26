//===-- ExecutableMemorySegmentAccessor.cpp -------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// Implements the \c ExecutableMemorySegmentAccessor and its subclasses, as
/// well as the \c ExecutableMemorySegmentAccessorAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/tooling/ExecutableMemorySegmentAccessor.h"
#include "luthier/hsa/hsa.h"
#include <llvm/Support/Error.h>

namespace luthier {

llvm::AnalysisKey ExecutableMemorySegmentAccessorAnalysis::Key;

llvm::Expected<ExecutableMemorySegmentAccessor::SegmentDescriptor>
HsaRuntimeExecutableMemorySegmentAccessor::getSegment(
    uint64_t DeviceAddr) const {

  hsa_executable_t Exec;
  /// Check which executable this address belongs to
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      VenLoaderTable.hsa_ven_amd_loader_query_executable(
          reinterpret_cast<const void *>(DeviceAddr), &Exec),
      llvm::formatv(
          "Failed to get the executable associated with address {0:x}.",
          DeviceAddr)));

  /// Find the LCO of the address
  llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;

  LUTHIER_RETURN_ON_ERROR(
      hsa::executableGetLoadedCodeObjects(VenLoaderTable, Exec, LCOs));

  for (hsa_loaded_code_object_t LCO : LCOs) {
    llvm::ArrayRef<uint8_t> LoadedMemory;
    LUTHIER_RETURN_ON_ERROR(
        hsa::loadedCodeObjectGetLoadedMemory(VenLoaderTable, LCO)
            .moveInto(LoadedMemory));
    const auto LoadedStartAddr =
        reinterpret_cast<uint64_t>(LoadedMemory.data());
    const uint64_t LoadedEndAddr = LoadedStartAddr + LoadedMemory.size();
    if (LoadedStartAddr <= DeviceAddr && DeviceAddr < LoadedEndAddr) {

      llvm::Expected<const uint8_t *> HostCopyBaseAddrOrErr =
          hsa::queryHostAddress(VenLoaderTable, LoadedMemory.data());
      LUTHIER_RETURN_ON_ERROR(HostCopyBaseAddrOrErr.takeError());
      llvm::Expected<object::AMDGCNObjectFile &> ObjFileOrErr =
          COC.getAssociatedObjectFile(LCO);
      LUTHIER_RETURN_ON_ERROR(ObjFileOrErr.takeError());
      return SegmentDescriptor{&*ObjFileOrErr,
                               {*HostCopyBaseAddrOrErr, LoadedMemory.size()},
                               LoadedMemory};
    }
  }
  return llvm::make_error<hsa::HsaError>(llvm::formatv(
      "Failed to obtain the segment associated with device address {0:x}",
      DeviceAddr));
}

} // namespace luthier