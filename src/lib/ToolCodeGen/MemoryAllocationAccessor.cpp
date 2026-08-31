//===-- MemoryAllocationAccessor.cpp --------------------------------------===//
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
/// Implements the concrete portions of the \c MemoryAllocationAccessor
/// interface, as well as the \c MemoryAllocationAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"

#include "luthier/Common/ErrorCheck.h"

namespace luthier {

llvm::AnalysisKey MemoryAllocationAnalysis::Key;

llvm::Expected<MemoryAllocationAccessor::AllocationDescriptor>
DriverOnlyMemoryAllocationAccessor::getAllocationDescriptor(
    uint64_t DeviceAddr) const {
  if (!hasSource())
    return AllocationDescriptor();

  llvm::Expected<DriverAllocationResolver::Allocation> AllocOrErr =
      Resolver->resolve(DeviceAddr);
  // Propagated, not flattened into "no allocation here": an error means the
  // address is the resolver's and something went wrong obtaining a host view of
  // it, which is a different situation from never having seen the address.
  LUTHIER_RETURN_ON_ERROR(AllocOrErr.takeError());
  if (AllocOrErr->empty())
    return AllocationDescriptor();

  return descriptorFor(*AllocOrErr);
}


}
