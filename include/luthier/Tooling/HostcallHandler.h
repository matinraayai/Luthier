//===-- HostcallHandler.h ---------------------------------------*- C++ -*-===//
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
/// Services the hostcall requests kernels Luthier dispatches make.
///
/// A hostcall is a fixed-size request a device wave submits to the host and
/// then blocks on. Device code reaches the request buffer through the
/// \c hidden_hostcall_buffer kernel argument, so a kernel Luthier dispatches
/// itself — a global constructor or destructor, say — gets no hostcall
/// service at all unless Luthier stands one up. Device-side \c printf,
/// \c malloc and \c free are all implemented on top of hostcalls, so without
/// this a constructor that prints hangs waiting for a response that never
/// comes.
///
/// The life-cycle mirrors the one ROCclr implements for HIP streams
/// (<tt>rocclr/device/devhostcall.cpp</tt>), because the device side of the
/// protocol is fixed by the ROCm device libraries:
///
/// \li A \c HostcallListener owns a doorbell signal and a thread waiting on
///     it. Device code bumps the doorbell after pushing a packet.
/// \li A \c HostcallBuffer is allocated per agent out of fine-grained host
///     memory the agent can perform atomics on, initialized, and registered
///     with the listener.
/// \li When the doorbell fires, the listener drains each registered buffer's
///     ready stack and answers every packet on it, which unblocks the waves
///     that submitted them.
///
/// \warning \c HostcallBuffer 's leading fields are a hard ABI shared with
/// the device libraries. Do not reorder or resize them.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_TOOLING_HOSTCALL_HANDLER_H
#define LUTHIER_HSA_TOOLING_HOSTCALL_HANDLER_H

#include "luthier/HSA/ApiTable.h"

#include <atomic>
#include <cstdint>
#include <hsa/hsa.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace luthier {

/// Services the device may request, as the ROCm device libraries number them.
enum HostcallServiceID : uint32_t {
  HOSTCALL_SERVICE_RESERVED = 0,
  /// Call a host function pointer supplied in the payload.
  HOSTCALL_SERVICE_FUNCTION_CALL = 1,
  /// Stream a \c printf message to the host. \sa DevicePrintf.h
  HOSTCALL_SERVICE_PRINTF = 2,
  /// Allocate or release device memory on the wave's behalf.
  HOSTCALL_SERVICE_DEVMEM = 3,
};

/// Values the hostcall doorbell signal takes. The device only ever increments
/// it, so zero is unambiguously the host's request that the listener stop.
enum HostcallSignalValue : hsa_signal_value_t {
  HOSTCALL_SIGNAL_DONE = 0,
  HOSTCALL_SIGNAL_INIT = 1,
};

/// One hostcall packet's payload: eight 64-bit words per lane of a wave. The
/// lanes with meaningful data are the ones set in
/// \c HostcallPacketHeader::ActiveMask.
struct HostcallPayload {
  uint64_t Slots[64][8];
};

/// One hostcall packet's header.
struct HostcallPacketHeader {
  /// Tagged pointer to the next packet in an intrusive stack.
  uint64_t Next;
  /// Which of \c HostcallPayload 's lane slots carry a request.
  uint64_t ActiveMask;
  /// The \c HostcallServiceID the submitting wave is asking for.
  uint32_t Service;
  /// Bit 0 is the READY flag: set by the device when it submits the packet,
  /// cleared by the host once the request has been answered. Clearing it is
  /// what unblocks the wave.
  std::atomic<uint32_t> Control;
};

static_assert(std::is_standard_layout_v<HostcallPacketHeader>,
              "the hostcall packet header is shared with device code");

/// Reassembles the multi-packet messages a service streams.
///
/// A hostcall carries at most seven 64-bit words of content, so a message
/// longer than that arrives as a run of hostcalls. Word 0 of each payload is
/// a descriptor holding a BEGIN flag (bit 0), an END flag (bit 1), the number
/// of content words in this packet (bits 5-7) and a message id (bits 8-63).
/// The host allocates the id when it sees BEGIN and writes it back into the
/// payload so the device can quote it on the packets that follow; the
/// accumulated message is handed to its service when END arrives.
class HostcallMessageAssembler {
public:
  HostcallMessageAssembler() = default;
  HostcallMessageAssembler(const HostcallMessageAssembler &) = delete;
  HostcallMessageAssembler &
  operator=(const HostcallMessageAssembler &) = delete;

  /// Folds one packet's worth of \p Payload into its message, and dispatches
  /// the message to \p Service if this packet ends it. \p Payload is updated
  /// in place with whatever the device is owed back.
  llvm::Error handlePayload(uint32_t Service, uint64_t *Payload);

private:
  struct Message {
    std::vector<uint64_t> Data;
    bool Live{false};
  };

  /// Messages indexed by id; ids are handed out densely and recycled.
  std::vector<Message> Messages;
  /// Ids of messages in \c Messages that have been completed and can be
  /// reused.
  std::vector<uint64_t> FreeIDs;
};

class HostcallBuffer;

/// Host-side state the services need while answering packets, hung off the
/// non-ABI tail of a \c HostcallBuffer so a handler can reach it from the
/// packet alone. Only ever touched by the listener thread, except during
/// teardown once that thread has been joined.
struct HostcallServiceState {
  HostcallServiceState(const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
                       hsa_agent_t Agent, hsa_amd_memory_pool_t DeviceMemoryPool,
                       size_t DeviceMemoryPoolAlignment)
      : AmdExt(AmdExt), Agent(Agent), DeviceMemoryPool(DeviceMemoryPool),
        DeviceMemoryPoolAlignment(DeviceMemoryPoolAlignment) {}

  hsa::ApiTableContainer<::AmdExtTable> AmdExt;
  /// The agent whose waves submit into the owning buffer.
  hsa_agent_t Agent;
  /// Pool that backs \c HOSTCALL_SERVICE_DEVMEM allocations.
  hsa_amd_memory_pool_t DeviceMemoryPool;
  /// Alignment \c DeviceMemoryPool already gives every allocation. Requests
  /// that need more than this are satisfied by over-allocating and aligning
  /// up inside the block.
  size_t DeviceMemoryPoolAlignment;
  /// Every address \c HOSTCALL_SERVICE_DEVMEM has handed out and not been
  /// asked to release, mapped to the pool allocation it lives in — which is
  /// not the same pointer when the request had to be over-aligned. Doubles as
  /// the list teardown reclaims what the kernel leaked from.
  llvm::DenseMap<uint64_t, void *> DeviceAllocations;
  HostcallMessageAssembler Messages;
};

/// The buffer through which device waves submit hostcall requests.
///
/// Packets are referenced by 64-bit tagged pointers: the low bits index the
/// packet arrays, the high bits are a tag bumped on every pop so that the
/// lock-free stacks do not suffer the ABA problem.
///
/// \warning Every field up to and including \c IndexMask is read and written
/// by device code at a fixed offset. Do not reorder, resize, or insert.
class HostcallBuffer {
public:
  /// Lays out the packet arrays inside the allocation this header starts, and
  /// threads every packet onto the free stack. \p NumPackets must be at least
  /// two and must not exceed what \c getRequiredSize was told.
  llvm::Error initialize(uint32_t NumPackets, HostcallServiceState &Services);

  /// Answers every packet currently on the ready stack. Called on the
  /// listener thread.
  void processPackets();

  /// Bytes an allocation must have for a buffer of \p NumPackets packets,
  /// including this header and any padding the packet arrays need.
  static size_t getRequiredSize(uint32_t NumPackets);

  /// Alignment an allocation backing a hostcall buffer must satisfy.
  static size_t getRequiredAlignment();

  /// Points the buffer at the signal the device bumps to wake the listener.
  void setDoorbell(hsa_signal_t Signal) { Doorbell = Signal.handle; }

private:
  HostcallPacketHeader *getHeader(uint64_t TaggedPtr) const {
    return Headers + (TaggedPtr & IndexMask);
  }
  HostcallPayload *getPayload(uint64_t TaggedPtr) const {
    return Payloads + (TaggedPtr & IndexMask);
  }

  /// Answers one lane's request. \p Slot is that lane's eight-word payload,
  /// updated in place with the response.
  void handlePayload(uint32_t Service, uint64_t *Slot);

  /// Implements \c HOSTCALL_SERVICE_DEVMEM: \c Slot[0] is the address to
  /// release, or zero to request an allocation of \c Slot[1] bytes whose
  /// address is returned in \c Slot[0].
  void handleDeviceMemoryRequest(uint64_t *Slot);

  //===-------------------------------------------------------------------===//
  // ABI: shared with the ROCm device libraries. Do not reorder or resize.
  //===-------------------------------------------------------------------===//
  HostcallPacketHeader *Headers;
  HostcallPayload *Payloads;
  /// The doorbell's \c hsa_signal_t handle. Stored untyped because the device
  /// treats it as an opaque 64-bit value.
  uint64_t Doorbell;
  uint64_t FreeStack;
  std::atomic<uint64_t> ReadyStack;
  uint64_t IndexMask;
  //===-------------------------------------------------------------------===//
  // Host-only tail; device code never reads past IndexMask.
  //===-------------------------------------------------------------------===//
  HostcallServiceState *Services;
};

static_assert(std::is_standard_layout_v<HostcallBuffer>,
              "the hostcall buffer is shared with device code");
static_assert(std::atomic<uint64_t>::is_always_lock_free &&
                  sizeof(std::atomic<uint64_t>) == sizeof(uint64_t),
              "the hostcall ready stack must be a plain lock-free 64-bit word "
              "for device code to push onto it");

/// A hostcall buffer plus the allocation and per-agent state behind it.
class HostcallBufferAllocation {
public:
  /// Allocates and initializes a buffer able to serve \p NumWaves waves of
  /// \p Agent concurrently, out of fine-grained host memory \p Agent can
  /// perform atomics on.
  ///
  /// \param NumWaves how many waves may have a hostcall outstanding at once.
  /// A dispatch cannot block on more packets than it has waves in flight, so
  /// sizing this from the dispatch rather than from the agent's peak
  /// occupancy keeps the allocation small — the full-occupancy buffer HIP
  /// allocates per stream runs to tens of megabytes.
  static llvm::Expected<std::unique_ptr<HostcallBufferAllocation>>
  create(const hsa::ApiTableContainer<::CoreApiTable> &CoreApi,
         const hsa::ApiTableContainer<::AmdExtTable> &AmdExt, hsa_agent_t Agent,
         uint32_t NumWaves);

  ~HostcallBufferAllocation();

  HostcallBufferAllocation(const HostcallBufferAllocation &) = delete;
  HostcallBufferAllocation &
  operator=(const HostcallBufferAllocation &) = delete;

  HostcallBuffer &getBuffer() const { return *Buffer; }

  /// The pointer device code expects in its \c hidden_hostcall_buffer
  /// argument.
  void *getDeviceVisibleAddress() const { return Buffer; }

private:
  HostcallBufferAllocation(const hsa::ApiTableContainer<::AmdExtTable> &AmdExt,
                           HostcallBuffer *Buffer,
                           std::unique_ptr<HostcallServiceState> Services)
      : AmdExt(AmdExt), Buffer(Buffer), Services(std::move(Services)) {}

  hsa::ApiTableContainer<::AmdExtTable> AmdExt;
  HostcallBuffer *Buffer{nullptr};
  std::unique_ptr<HostcallServiceState> Services;
};

/// Owns the doorbell signal and the thread that answers hostcall packets.
///
/// One listener can serve any number of buffers. A buffer must be registered
/// before a kernel that can reach it is dispatched, and deregistered before
/// it is freed.
class HostcallListener {
public:
  /// Creates the doorbell signal and starts the listener thread.
  static llvm::Expected<std::unique_ptr<HostcallListener>>
  create(const hsa::ApiTableContainer<::CoreApiTable> &CoreApi);

  /// Stops the thread and destroys the doorbell. Any error tearing the signal
  /// down is swallowed, since a destructor cannot report it; call \c stop
  /// first to see it.
  ~HostcallListener();

  HostcallListener(const HostcallListener &) = delete;
  HostcallListener &operator=(const HostcallListener &) = delete;

  /// Points \p Buffer at this listener's doorbell and starts draining it.
  void addBuffer(HostcallBuffer &Buffer);

  /// Stops draining \p Buffer. Returns without doing anything if it was never
  /// registered.
  void removeBuffer(HostcallBuffer &Buffer);

  /// Asks the listener thread to exit, joins it, and destroys the doorbell.
  /// Idempotent.
  llvm::Error stop();

private:
  explicit HostcallListener(
      const hsa::ApiTableContainer<::CoreApiTable> &CoreApi, hsa_signal_t Doorbell)
      : CoreApi(CoreApi), Doorbell(Doorbell) {}

  /// The listener thread's body: wait on the doorbell, drain every registered
  /// buffer, repeat until the doorbell reads \c HOSTCALL_SIGNAL_DONE.
  void consumePackets();

  hsa::ApiTableContainer<::CoreApiTable> CoreApi;
  hsa_signal_t Doorbell{};
  /// Guards \c Buffers against registration racing with the drain loop.
  std::mutex Mutex;
  llvm::SmallVector<HostcallBuffer *, 2> Buffers;
  std::thread Thread;
  bool Stopped{false};
};

} // namespace luthier

#endif // LUTHIER_HSA_TOOLING_HOSTCALL_HANDLER_H
