//===-- MockLoaderMemoryAccessor.h ------------------------------*- C++ -*-===//
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
/// \file MockLoaderMemoryAccessor.h
/// Describes the \c MockLoaderMemoryAccessor which is the \c
/// MemoryAllocationAccessor implementation for the <tt>AMDGPUMockLoader</tt>.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_TOOL_CODE_GEN_TESTING_COMMON_MOCK_LOADER_MEMORY_ACCESSOR_H
#define LUTHIER_TOOL_CODE_GEN_TESTING_COMMON_MOCK_LOADER_MEMORY_ACCESSOR_H
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"
#include "luthier/ToolCodeGen/MockAMDGPULoader.h"

namespace luthier {

class MockLoaderMemoryAccessor : public MemoryAllocationAccessor {
  const MockAMDGPULoader &Loader;

public:
  explicit MockLoaderMemoryAccessor(const MockAMDGPULoader &loader)
      : Loader(loader) {}

  [[nodiscard]] llvm::Expected<AllocationDescriptor>
  getAllocationDescriptor(uint64_t DeviceAddr) const override;
};

} // namespace luthier

#endif