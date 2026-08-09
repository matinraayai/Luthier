//===-- HiddenArgBuffers.h --------------------------------------*- C++ -*-===//
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
/// Backing objects for the hidden kernel arguments that need one beyond a
/// plain scalar: the device-side heap, the cooperative-groups grid sync
/// structure, and the device-enqueue completion action.
///
/// Every layout here is an ABI shared with the ROCm device libraries or with
/// the device-enqueue scheduler, so the field order and widths are not
/// Luthier's to choose. Each struct names the definition it mirrors.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_HIDDEN_ARG_BUFFERS_H
#define LUTHIER_HSA_TOOLING_HIDDEN_ARG_BUFFERS_H

#include "luthier/HSA/ApiTable.h"

#include <cstddef>
#include <cstdint>
#include <hsa/hsa.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <type_traits>

namespace luthier {

//===----------------------------------------------------------------------===//
// hidden_heap_v1
//===----------------------------------------------------------------------===//

/// Bytes a device heap allocation must span.
///
/// The device libraries' <tt>heap_t</tt> (\c ockl/src/dm.cl) is the management
/// structure device-side \c malloc keeps its slab bookkeeping in.
/// \c __ockl_dm_init_v1 clears exactly this many bytes of it, which is the
/// device libraries' own statement of the structure's upper bound.
constexpr size_t DeviceHeapSize = 131072;

/// Owns the device memory behind a kernel's \c hidden_heap_v1 argument.
///
/// Device-side \c malloc / \c free / \c new / \c delete reach their slab
/// bookkeeping through this pointer. \c heap_t documents that "all bits 0 is
/// an acceptable state, and the expected initial state", so standing one up
/// is a zero-fill — no device-side initialization kernel is needed.
///
/// What a zeroed heap gives up against a runtime that also runs
/// \c __ockl_dm_init_v1 is only the pre-allocated *initial slabs* that call
/// seeds: with none, the allocator sources every slab through
/// \c __ockl_devmem_request, which is the hostcall device-memory service
/// (\c HOSTCALL_SERVICE_DEVMEM). Allocation therefore still works, at the
/// cost of a hostcall round trip on the first allocation of each size class.
/// A kernel handed this heap must therefore also be handed a hostcall buffer.
class DeviceHeapBuffer {
public:
  /// Allocates and zeroes a heap in memory local to \p Agent.
  static llvm::Expected<std::unique_ptr<DeviceHeapBuffer>>
  create(const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
         hsa_agent_t Agent);

  ~DeviceHeapBuffer();

  DeviceHeapBuffer(const DeviceHeapBuffer &) = delete;
  DeviceHeapBuffer &operator=(const DeviceHeapBuffer &) = delete;

  /// The pointer device code expects in its \c hidden_heap_v1 argument.
  void *getDeviceVisibleAddress() const { return Heap; }

private:
  DeviceHeapBuffer(const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
                   void *Heap)
      : AmdExt(AmdExt), Heap(Heap) {}

  hsa::ApiTableContainer<::AmdExtTable> AmdExt;
  void *Heap{nullptr};
};

//===----------------------------------------------------------------------===//
// hidden_multigrid_sync_arg
//===----------------------------------------------------------------------===//

/// One cooperative-groups barrier's state.
///
/// Mirrors ROCclr's \c amd::Device::MGSyncData.
struct DeviceGridSyncData {
  uint32_t W0;
  uint32_t W1;
};

/// The structure behind a kernel's \c hidden_multigrid_sync_arg argument,
/// which backs <tt>cooperative_groups::this_grid()</tt> and
/// <tt>this_multi_grid()</tt>.
///
/// Mirrors ROCclr's \c amd::Device::MGSyncInfo field for field.
struct DeviceGridSyncInfo {
  /// Barrier state shared by every grid of a multi-device cooperative
  /// launch. Null for a single-grid launch, which synchronizes through
  /// \c SingleGridSync instead.
  DeviceGridSyncData *MultiGridSync;
  /// This grid's index among the grids taking part in the launch.
  uint32_t GridID;
  /// How many grids take part in the launch.
  uint32_t NumGrids;
  /// Total workgroups across the grids that precede this one.
  uint64_t PrevGridSum;
  /// Total workgroups across every grid in the launch.
  uint64_t AllGridSum;
  /// Barrier state for a single-grid launch.
  DeviceGridSyncData SingleGridSync;
  /// Workgroups in this grid — what a grid-wide barrier counts up to.
  uint32_t NumWorkgroups;
};

static_assert(std::is_standard_layout_v<DeviceGridSyncInfo>,
              "the grid sync structure is shared with device code");

/// Fills \p Info in for a launch of a single grid of \p NumWorkgroups
/// workgroups on one device — the only shape Luthier dispatches. The barrier
/// starts unclaimed and the multi-grid pointer stays null, so a
/// <tt>this_grid().sync()</tt> in the kernel resolves against
/// \c SingleGridSync.
void initializeSingleGridSyncInfo(DeviceGridSyncInfo &Info,
                                  uint32_t NumWorkgroups);

//===----------------------------------------------------------------------===//
// hidden_completion_action
//===----------------------------------------------------------------------===//

/// States a device-enqueue AQL wrapper slot can be in.
///
/// Mirrors ROCclr's \c amd::roc::AqlWrapState.
enum DeviceAqlWrapState : uint32_t {
  DEVICE_AQL_WRAP_FREE = 0,
  DEVICE_AQL_WRAP_RESERVED = 1,
  DEVICE_AQL_WRAP_READY = 2,
  DEVICE_AQL_WRAP_MARKER = 3,
  DEVICE_AQL_WRAP_BUSY = 4,
  DEVICE_AQL_WRAP_DONE = 5,
};

/// The structure behind a kernel's \c hidden_completion_action argument: the
/// wrapper a device-enqueued child reports its completion against.
///
/// Mirrors ROCclr's \c amd::roc::AmdAqlWrap field for field.
struct DeviceAqlWrap {
  /// One of \c DeviceAqlWrapState.
  uint32_t State;
  uint32_t EnqueueFlags;
  uint32_t CommandID;
  /// Outstanding child launches; the wrapper is finished when this reaches
  /// zero with \c State set to \c DEVICE_AQL_WRAP_DONE.
  uint32_t ChildCounter;
  uint64_t Completion;
  uint64_t ParentWrap;
  uint64_t WaitList;
  uint32_t WaitNum;
  uint32_t Reserved[5];
  hsa_kernel_dispatch_packet_t Aql;
};

static_assert(std::is_standard_layout_v<DeviceAqlWrap>,
              "the AQL wrapper is shared with device code");

/// Alignment a \c DeviceAqlWrap must be placed at, driven by the AQL packet
/// it embeds.
constexpr size_t DeviceAqlWrapAlignment = 64;

/// Fills \p Wrap in as an already-completed, parentless wrapper — what a
/// kernel launched by the host rather than device-enqueued from another
/// kernel should see above it.
void initializeCompletionAction(DeviceAqlWrap &Wrap);

} // namespace luthier

#endif // LUTHIER_HSA_TOOLING_HIDDEN_ARG_BUFFERS_H
