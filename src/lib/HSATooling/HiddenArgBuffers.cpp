//===-- HiddenArgBuffers.cpp ----------------------------------------------===//
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
#include "luthier/HSATooling/HiddenArgBuffers.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/HSA/MemoryPool.h"

#include <cstring>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier {

//===----------------------------------------------------------------------===//
// DeviceHeapBuffer
//===----------------------------------------------------------------------===//

llvm::Expected<std::unique_ptr<DeviceHeapBuffer>>
DeviceHeapBuffer::create(const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
                         hsa_agent_t Agent) {
  // The heap is hammered with device-scope atomics by every work-item that
  // allocates, so it belongs in memory local to the agent.
  auto PoolOrErr = hsa::agentFindCoarseGrainedPool(AmdExt, Agent);
  LUTHIER_RETURN_ON_ERROR(PoolOrErr.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      PoolOrErr->has_value(),
      llvm::formatv("Agent {0:x} exposes no coarse-grained memory pool to "
                    "back a device heap",
                    Agent.handle)));

  auto AllocOrErr = hsa::memoryPoolAllocate(AmdExt, **PoolOrErr, DeviceHeapSize);
  LUTHIER_RETURN_ON_ERROR(AllocOrErr.takeError());

  const llvm::SmallVector<hsa_agent_t, 1> Agents{Agent};
  if (llvm::Error Err = hsa::agentsAllowAccess(AmdExt, Agents, *AllocOrErr))
    return llvm::joinErrors(std::move(Err),
                            hsa::memoryPoolFree(AmdExt, *AllocOrErr));

  // An all-zero heap is the state the device libraries expect to start from;
  // see the class comment. hsa_amd_memory_pool_allocate makes no guarantee
  // about the contents, so zero it explicitly.
  if (llvm::Error Err = LUTHIER_HSA_CALL_ERROR_CHECK(
          AmdExt.callFunction<hsa_amd_memory_fill>(*AllocOrErr, /*Value=*/0,
                                                   DeviceHeapSize /
                                                       sizeof(uint32_t)),
          llvm::formatv("Failed to zero the {0}-byte device heap for agent "
                        "{1:x}",
                        DeviceHeapSize, Agent.handle)))
    return llvm::joinErrors(std::move(Err),
                            hsa::memoryPoolFree(AmdExt, *AllocOrErr));

  return std::unique_ptr<DeviceHeapBuffer>(
      new DeviceHeapBuffer(AmdExt, *AllocOrErr));
}

DeviceHeapBuffer::~DeviceHeapBuffer() {
  if (Heap == nullptr)
    return;
  llvm::consumeError(hsa::memoryPoolFree(AmdExt, Heap));
  Heap = nullptr;
}

//===----------------------------------------------------------------------===//
// DeviceGridSyncInfo
//===----------------------------------------------------------------------===//

void initializeSingleGridSyncInfo(DeviceGridSyncInfo &Info,
                                  uint32_t NumWorkgroups) {
  std::memset(&Info, 0, sizeof(Info));
  // No other grid takes part, so there is nothing to synchronize against
  // beyond this one and the multi-grid barrier stays unused.
  Info.MultiGridSync = nullptr;
  Info.GridID = 0;
  Info.NumGrids = 1;
  Info.PrevGridSum = 0;
  Info.AllGridSum = NumWorkgroups;
  Info.SingleGridSync = DeviceGridSyncData{0, 0};
  Info.NumWorkgroups = NumWorkgroups;
}

//===----------------------------------------------------------------------===//
// DeviceAqlWrap
//===----------------------------------------------------------------------===//

void initializeCompletionAction(DeviceAqlWrap &Wrap) {
  std::memset(&Wrap, 0, sizeof(Wrap));
  // The kernel this wrapper stands above was launched by the host, so it has
  // no parent to report to and nothing is outstanding against it.
  Wrap.State = DEVICE_AQL_WRAP_DONE;
}

} // namespace luthier
