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
  SegmentDescriptor Out;

  /// Query the pointer's device allocation from HSA
  hsa_amd_pointer_info_t PtrInfo{sizeof(hsa_amd_pointer_info_t)};

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      AmdExtApi.callFunction<hsa_amd_pointer_info>(
          reinterpret_cast<void *>(DeviceAddr), &PtrInfo, nullptr, nullptr,
          nullptr),
      llvm::formatv("Failed to obtain the pointer info for address {0:x} from "
                    "the HSA runtime",
                    DeviceAddr)));
  Out.SegmentOnDevice = {static_cast<uint8_t *>(PtrInfo.agentBaseAddress),
                         PtrInfo.sizeInBytes};
  /// If there is a host-accessible pointer for this allocation return that
  /// instead; otherwise, query the loader API for the host address of the
  /// agent base address of the allocation
  if (PtrInfo.hostBaseAddress) {
    Out.SegmentOnHost = {static_cast<uint8_t *>(PtrInfo.hostBaseAddress),
                         PtrInfo.sizeInBytes};
  } else {
    llvm::Expected<const void *> HostCopyBaseAddrOrErr =
        hsa::queryHostAddress(VenLoaderTable, PtrInfo.agentBaseAddress);
    LUTHIER_RETURN_ON_ERROR(HostCopyBaseAddrOrErr.takeError());
    Out.SegmentOnHost = {static_cast<const uint8_t *>(*HostCopyBaseAddrOrErr),
                         PtrInfo.sizeInBytes};
  }
  return Out;
}

} // namespace luthier