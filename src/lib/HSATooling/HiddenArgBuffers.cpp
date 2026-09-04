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

namespace {

/// \c heap_t layout constants mirrored from ROCm device-libs
/// \c ockl/src/dm.cl. They describe the structure the device-side allocator
/// reads out of the \c hidden_heap_v1 buffer, which \c DeviceHeapBuffer has to
/// hand over in the state \c __ockl_dm_init_v1 would leave it in.

/// \c NUM_KINDS — the number of block kinds the allocator tracks.
constexpr uint32_t OcklHeapNumKinds = 16;

/// \c NUM_SDATA (\c 1 << SDATA_SHIFT) — how many slabs one level of the slab
/// record array holds, and the value \c num_recordable_slabs is initialized to.
constexpr uint32_t OcklHeapNumSData = 256;

/// Every counter in \c heap_t is a single atomic padded out to a cache line
/// (\c ULONG_PER_CACHE_LINE ulongs), so the per-kind arrays stride by this.
constexpr uint32_t OcklHeapCounterStride = 128;

/// \c heap_t opens with \c start[NUM_KINDS] then
/// \c num_allocated_slabs[NUM_KINDS], both of \c OcklHeapCounterStride-sized
/// entries, so \c num_recordable_slabs starts after two such arrays.
constexpr uint32_t OcklHeapNumRecordableSlabsOffset =
    2 * OcklHeapNumKinds * OcklHeapCounterStride;

static_assert(OcklHeapNumRecordableSlabsOffset +
                      OcklHeapNumKinds * OcklHeapCounterStride <=
                  DeviceHeapSize,
              "num_recordable_slabs runs past the end of the device heap");

} // namespace

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

  // Zero the whole heap first. This matches the clearing loop at the top of
  // \c __ockl_dm_init_v1, which blanks exactly 131072 bytes — the same figure
  // as \c DeviceHeapSize and as ROCclr's own \c HeapBufferSize.
  // hsa_amd_memory_pool_allocate makes no guarantee about the contents.
  if (llvm::Error Err = LUTHIER_HSA_CALL_ERROR_CHECK(
          AmdExt.callFunction<hsa_amd_memory_fill>(*AllocOrErr, /*Value=*/0,
                                                   DeviceHeapSize /
                                                       sizeof(uint32_t)),
          llvm::formatv("Failed to zero the {0}-byte device heap for agent "
                        "{1:x}",
                        DeviceHeapSize, Agent.handle)))
    return llvm::joinErrors(std::move(Err),
                            hsa::memoryPoolFree(AmdExt, *AllocOrErr));

  // Zeroing alone is not enough. ROCclr brings the heap up by launching
  // \c __amd_rocclr_initHeap (a 256-work-item kernel that calls
  // \c __ockl_dm_init_v1) before any device-side malloc can run; Luthier
  // dispatches its own constructor/destructor kernels, so it has to leave the
  // heap in that same state itself.
  //
  // \c heap_t is documented as "all bits 0 is an acceptable state" but
  // \c __ockl_dm_init_v1 still writes \c num_recordable_slabs[k] = NUM_SDATA
  // for each of the \c NUM_KINDS block kinds, and the allocator does not work
  // without it: it looks a slab record up as
  // \c sdata[k][(i - NUM_SDATA) >> SDATA_SHIFT], so an index that never
  // reaches \c NUM_SDATA underflows to a huge value. The resulting
  // \c global_atomic_cmpswap_x2 then lands hundreds of megabytes past the
  // heap, where — with XNACK off — it never retires instead of faulting, and
  // the wave hangs on the following \c s_waitcnt vmcnt(0) with no diagnostic.
  //
  // The remaining fields \c __ockl_dm_init_v1 touches — \c initial_slabs,
  // \c initial_slabs_end and \c initial_slabs_start — are all written from its
  // initial-slab-buffer arguments, which are zero when no such buffer is
  // supplied (ROCclr's \c initial_heap_size_ == 0 path). Luthier supplies
  // none, so the zeroing above already leaves them correct.
  for (uint32_t Kind = 0; Kind < OcklHeapNumKinds; ++Kind) {
    auto *NumRecordableSlabs =
        static_cast<uint8_t *>(*AllocOrErr) + OcklHeapNumRecordableSlabsOffset +
        Kind * OcklHeapCounterStride;
    if (llvm::Error Err = LUTHIER_HSA_CALL_ERROR_CHECK(
            AmdExt.callFunction<hsa_amd_memory_fill>(
                NumRecordableSlabs, /*Value=*/OcklHeapNumSData, /*Count=*/1),
            llvm::formatv("Failed to initialize num_recordable_slabs[{0}] of "
                          "the device heap for agent {1:x}",
                          Kind, Agent.handle)))
      return llvm::joinErrors(std::move(Err),
                              hsa::memoryPoolFree(AmdExt, *AllocOrErr));
  }

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
