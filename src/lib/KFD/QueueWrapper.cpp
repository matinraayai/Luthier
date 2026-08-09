//===-- QueueWrapper.cpp - KFD-level AQL queue interception ---------------===//
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
///
/// \file
/// Implementation of the KFD-level queue wrapper. See
/// \c luthier/KFD/QueueWrapper.h for the design, and
/// \c issue-85-ioctl-intercept-poc/FINDINGS.md for the measurements behind the
/// choices made here.
///
/// \par Two signals, two jobs
/// Confusing these caused every detection bug we hit:
/// \li The application's **write pointer** counts slots it has *claimed*. A
///     packet cannot exist without having been claimed, so this is a safe upper
///     bound on where to look -- and bounding the scan is what keeps the copy
///     loop finite. It is **not** a "finished" count: the application bumps it
///     before writing the packet (measured: caught mid-write in 33 of 6509
///     samples).
/// \li The slot's **header** says whether that slot is finished, because the
///     producer writes the header last. That is the commit test.
///
/// \par The marker is ours to establish
/// The HSA runtime promises its callers that every slot in a new queue starts as
/// \c HSA_PACKET_TYPE_INVALID (see \c hsa.h on \c hsa_queue_create; ROCr does it
/// at \c amd_aql_queue.cpp:122). That is an HSA promise, not a driver or
/// hardware one -- an application that skips HSA gets whatever the allocator
/// left, and zero is a legal packet type. So we write the markers ourselves at
/// queue creation, and put one back after copying each packet.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/QueueWrapper.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <linux/kfd_ioctl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <ctime>
#include <unistd.h>

namespace luthier::kfd {

static constexpr const char *LogPrefix = "[luthier-kfd] ";

//===----------------------------------------------------------------------===//
// AQL packet layout. Fixed by the AQL specification, so these are safe as
// constants -- unlike the queue-descriptor offsets, which belong to an AMD
// struct and are derived with offsetof where they are needed.
//===----------------------------------------------------------------------===//
static constexpr unsigned AqlPacketBytes = 64;
static constexpr unsigned AqlHeaderOffset = 0;
static constexpr unsigned AqlGridSizeXOffset = 12;
static constexpr unsigned PacketTypeInvalid = 1;
static constexpr unsigned PacketTypeKernelDispatch = 2;

static_assert(sizeof(hsa::AqlPacket) == AqlPacketBytes,
              "AQL packets are 64 bytes by specification");

static inline unsigned packetType(uint16_t Header) { return Header & 0xFF; }

/// Read a slot's 2-byte header.
///
/// Acquire ordering pairs with the producer's release store of that header, and
/// is what makes the rest of the packet safe to read afterwards: the producer
/// writes the body first and the header last, so seeing the header means the
/// body is there.
static inline uint16_t loadHeader(volatile unsigned char *Slot) {
  return __atomic_load_n(
      reinterpret_cast<const uint16_t *>(const_cast<const unsigned char *>(Slot)),
      __ATOMIC_ACQUIRE);
}

/// Publish a slot's header.
///
/// Release ordering means every earlier write is visible before this one, which
/// is what stops the GPU seeing a header for a packet whose body has not landed.
static inline void storeHeader(volatile unsigned char *Slot, uint16_t Value) {
  __atomic_store_n(
      reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(Slot)), Value,
      __ATOMIC_RELEASE);
}

//===----------------------------------------------------------------------===//
// Environment switches
//===----------------------------------------------------------------------===//

/// Per-packet logging. Off by default: on a busy queue this is one line per
/// packet, and a runaway once produced 18.6 million lines and a 2.5 GB log. The
/// counts that matter are kept per queue and printed once at teardown instead.
static bool verboseEnabled() {
  static int Cache = -1;
  if (Cache < 0) {
    const char *V = getenv("LUTHIER_VERBOSE");
    Cache = (V && V[0] == '1') ? 1 : 0;
  }
  return Cache != 0;
}

/// Built-in self-test that the edit path is live: zero a dispatch's grid so no
/// work-items launch. A test that verifies an exact value then fails in a
/// specific, predicted way -- which is the difference between a callback that
/// observes and one that controls.
static bool demoZeroGridEnabled() {
  static int Cache = -1;
  if (Cache < 0) {
    const char *V = getenv("LUTHIER_DEMO_ZERO_GRID");
    Cache = (V && V[0] == '1') ? 1 : 0;
  }
  return Cache != 0;
}

//===----------------------------------------------------------------------===//
// The real ioctl
//===----------------------------------------------------------------------===//
using RealIoctlFn = int (*)(int, unsigned long, void *);
static RealIoctlFn RealIoctl = nullptr;

static void ensureRealIoctlResolved() {
  if (RealIoctl == nullptr) {
    // RTLD_NEXT: the next ioctl after us in the search order, i.e. the real
    // one. Without it we would find ourselves and recurse forever.
    RealIoctl = reinterpret_cast<RealIoctlFn>(dlsym(RTLD_NEXT, "ioctl"));
    if (RealIoctl == nullptr) {
      fprintf(stderr, "%sdlsym(RTLD_NEXT, \"ioctl\") failed: %s\n", LogPrefix,
              dlerror());
      abort();
    }
  }
}

/// Identify /dev/kfd by device number rather than descriptor number, since the
/// application chooses its own descriptors.
static bool fdIsKfd(int Fd) {
  static dev_t KfdRdev = 0;
  static bool Cached = false;
  if (!Cached) {
    struct stat St {};
    if (stat("/dev/kfd", &St) == 0 && S_ISCHR(St.st_mode))
      KfdRdev = St.st_rdev;
    else
      fprintf(stderr, "%sstat(\"/dev/kfd\") failed; fd checks will fail\n",
              LogPrefix);
    Cached = true;
  }
  struct stat St {};
  if (fstat(Fd, &St) != 0)
    return false;
  return S_ISCHR(St.st_mode) && St.st_rdev == KfdRdev;
}

//===----------------------------------------------------------------------===//
// Our substitute ring
//===----------------------------------------------------------------------===//

/// mmap rather than malloc: we need page alignment for the GPU registration
/// below, and pages that do not share space with the allocator's bookkeeping.
///
/// \param OutSize receives the **rounded** length. Unmapping needs it: the
/// application's ring size is not page-aligned in general, and munmap with the
/// unrounded figure leaves part of the mapping behind.
static void *allocRingPages(size_t MinSize, size_t *OutSize) {
  static long PageSize = 0;
  if (PageSize == 0)
    PageSize = sysconf(_SC_PAGESIZE);
  size_t Size = ((MinSize + PageSize - 1) / PageSize) * PageSize;
  void *P = mmap(nullptr, Size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (P == MAP_FAILED) {
    fprintf(stderr, "%smmap(%zu) failed: %s\n", LogPrefix, Size,
            strerror(errno));
    abort();
  }
  *OutSize = Size;
  return P;
}

/// The driver rejects a queue whose ring it does not know about, so we must
/// perform the same two-step registration the runtime does. EXECUTABLE turned
/// out to be the load-bearing flag -- without it the GPU faults.
#define LUTHIER_KFD_RING_ALLOC_FLAGS                                           \
  (KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |        \
   KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE |\
   KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED)

/// \param OutHandle receives the driver's handle for the allocation, which is
/// what releaseRing needs to give it back.
static bool registerRingWithGpu(int Fd, void *Va, size_t Size, uint32_t GpuId,
                                uint64_t *OutHandle) {
  struct kfd_ioctl_alloc_memory_of_gpu_args AllocArgs {};
  memset(&AllocArgs, 0, sizeof(AllocArgs));
  AllocArgs.va_addr = reinterpret_cast<__u64>(Va);
  AllocArgs.size = Size;
  AllocArgs.mmap_offset = reinterpret_cast<__u64>(Va);
  AllocArgs.gpu_id = GpuId;
  AllocArgs.flags = LUTHIER_KFD_RING_ALLOC_FLAGS;
  if (RealIoctl(Fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &AllocArgs) != 0) {
    fprintf(stderr, "%sALLOC_MEMORY_OF_GPU(0x%llx) failed: %s\n", LogPrefix,
            static_cast<unsigned long long>(AllocArgs.va_addr),
            strerror(errno));
    return false;
  }

  // Mapping to the queue's own node is sufficient; all-node mapping is not
  // required (the earlier GPU fault was the alloc flags, not the device count).
  __u32 DeviceIds[1] = {GpuId};
  struct kfd_ioctl_map_memory_to_gpu_args MapArgs {};
  memset(&MapArgs, 0, sizeof(MapArgs));
  MapArgs.handle = AllocArgs.handle;
  MapArgs.device_ids_array_ptr = reinterpret_cast<__u64>(DeviceIds);
  MapArgs.n_devices = 1;
  if (RealIoctl(Fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &MapArgs) != 0 ||
      MapArgs.n_success != 1) {
    fprintf(stderr, "%sMAP_MEMORY_TO_GPU(0x%llx) failed: %s (%u/1)\n", LogPrefix,
            static_cast<unsigned long long>(AllocArgs.va_addr), strerror(errno),
            MapArgs.n_success);
    return false;
  }
  *OutHandle = AllocArgs.handle;
  return true;
}

/// Undo registerRingWithGpu and hand the pages back.
///
/// Called when a tracking entry is recycled, not when the queue is destroyed.
/// Deferring it keeps the application's DESTROY_QUEUE free of two extra driver
/// calls, and bounds what is held at any moment to one dead ring per entry.
///
/// Safe by then: the queue is gone, so the GPU is not reading the ring, and the
/// entry's grace period has expired, so neither are we. Errors are reported but
/// not acted on -- the application may already have closed the descriptor, and
/// leaking a ring is better than failing a queue creation over it.
static void releaseRing(int Fd, uint64_t Handle, uint32_t GpuId, void *Pages,
                        size_t PagesBytes) {
  if (Pages == nullptr)
    return;

  __u32 DeviceIds[1] = {GpuId};
  struct kfd_ioctl_unmap_memory_from_gpu_args UnmapArgs {};
  memset(&UnmapArgs, 0, sizeof(UnmapArgs));
  UnmapArgs.handle = Handle;
  UnmapArgs.device_ids_array_ptr = reinterpret_cast<__u64>(DeviceIds);
  UnmapArgs.n_devices = 1;
  if (RealIoctl(Fd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &UnmapArgs) != 0 &&
      verboseEnabled())
    fprintf(stderr, "%sUNMAP_MEMORY_FROM_GPU failed: %s\n", LogPrefix,
            strerror(errno));

  struct kfd_ioctl_free_memory_of_gpu_args FreeArgs {};
  memset(&FreeArgs, 0, sizeof(FreeArgs));
  FreeArgs.handle = Handle;
  if (RealIoctl(Fd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &FreeArgs) != 0 &&
      verboseEnabled())
    fprintf(stderr, "%sFREE_MEMORY_OF_GPU failed: %s\n", LogPrefix,
            strerror(errno));

  munmap(Pages, PagesBytes);
}

//===----------------------------------------------------------------------===//
// Per-queue state
//===----------------------------------------------------------------------===//
namespace {

/// \brief Lifecycle of one tracking-table entry.
///
/// A plain "active" flag is not enough once entries are reused. The poller reads
/// entries without the lock, so "nobody will ever touch this again" is a
/// different statement from "this queue is gone", and reuse depends on the
/// first.
enum SlotState : int {
  /// Never handed out, or handed out for a creation that then failed. Either
  /// way the poller has never looked inside it, so it is reusable at once.
  SlotFree = 0,
  /// Claimed by a creation in progress. Not yet safe to poll: the fields are
  /// still being filled in.
  SlotReserved = 1,
  /// A live wrapped queue. The poller works on exactly these.
  SlotLive = 2,
  /// The queue was destroyed. The poller skips it, but a pass that was already
  /// inside it may still be running, so it cannot be reused immediately.
  SlotDead = 3,
};

struct ForwardedQueue {
  /// The application's own ring. It writes here; after substitution nobody else
  /// reads it but us.
  volatile unsigned char *AppRing;
  /// Our ring, registered with the GPU. The GPU reads here.
  volatile unsigned char *ShimRing;
  /// The application's claim counter. Read-only to us.
  volatile uint64_t *AppWritePointer;
  QueueInfo Info;
  /// Next packet of the **application's** stream to copy. Counts packets, not
  /// slots. This is the number a callback is given, so a tool's packet numbering
  /// matches the application's.
  uint64_t Consumed;
  /// Next slot of **our** ring to write. Counts packets, not slots.
  ///
  /// Equal to \c Consumed today, and separated from it deliberately rather than
  /// because they differ. The two mean different things -- one indexes the
  /// application's stream, the other our output -- and they stop being equal the
  /// moment a callback can emit a different number of packets than it was given.
  /// Keeping one counter for both hid that behind an arithmetic coincidence; the
  /// places that would have to change for 1-to-N are now exactly the places that
  /// mention this field.
  ///
  /// They cannot actually diverge yet, and not just because nothing increments
  /// them differently: the GPU tracks its progress against the application's own
  /// write pointer, which we do not intercept. Emitting a different count means
  /// taking that over too.
  uint64_t Produced;
  uint64_t DispatchCount;
  int State;      ///< one of SlotState
  int Summarized; ///< so the teardown summary prints once
  /// Value of \c PollPass when this queue was destroyed, which is what decides
  /// when the entry may be reused. Only ever touched under \c QueueLock.
  uint64_t DeadAtPass;

  //=== What it takes to give the substitute ring back ======================//
  /// The descriptor CREATE_QUEUE arrived on, needed to undo the registration.
  int Fd;
  /// Driver handle from ALLOC_MEMORY_OF_GPU.
  uint64_t RingHandle;
  /// Start of the mapping, and its **page-rounded** length -- not the ring size
  /// the application asked for, which would under-unmap.
  void *RingPages;
  size_t RingPagesBytes;
};
} // namespace

static constexpr int MaxTrackedQueues = 64;
static ForwardedQueue Queues[MaxTrackedQueues];
static int QueueCount = 0;
static pthread_mutex_t QueueLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_t PollerThread;
static bool PollerStarted = false;

/// Completed passes of the polling thread over the whole table.
///
/// This is the grace-period clock. Reusing an entry is safe once a full pass has
/// finished that began after the queue was marked dead, because any pass still
/// inside the entry must have started before that.
static uint64_t PollPass = 0;

/// Every queue we have ever substituted a ring for. Never decremented, so
/// reclaiming an entry does not erase the evidence that the queue was wrapped.
static uint64_t WrappedQueueTotal = 0;

/// Queues left alone because the thread that created them was running the
/// tool's own code. Counted rather than merely skipped so a test can tell
/// "correctly excluded" from "never created".
static uint64_t ExcludedQueueTotal = 0;

/// Depth of nested tool regions on this thread. See beginToolRegion in the
/// header for why a thread-local is sufficient and what that rests on.
///
/// A depth rather than a flag so that a tool region opened inside another does
/// not end interception when the inner one closes.
static thread_local unsigned ToolRegionDepth = 0;

void beginToolRegion() { ToolRegionDepth++; }

void endToolRegion() {
  if (ToolRegionDepth == 0) {
    // Unbalanced. Saying so is worth a line: silently clamping would hide a
    // missing beginToolRegion, and the failure that causes -- the tool's own
    // queues getting instrumented -- is invisible until something dispatches.
    fprintf(stderr, "%sendToolRegion() with no matching beginToolRegion()\n",
            LogPrefix);
    return;
  }
  ToolRegionDepth--;
}

static bool insideToolRegion() { return ToolRegionDepth != 0; }

uint64_t excludedQueueCount() {
  return __atomic_load_n(&ExcludedQueueTotal, __ATOMIC_ACQUIRE);
}

uint64_t wrappedQueueCount() {
  return __atomic_load_n(&WrappedQueueTotal, __ATOMIC_ACQUIRE);
}

//===----------------------------------------------------------------------===//
// The callback chain
//===----------------------------------------------------------------------===//

/// Fixed array rather than a vector: this is read once per packet on the poller
/// thread, and a preloaded library has no business allocating there.
static detail::CallbackEntry Callbacks[MaxPacketCallbacks];

/// How many entries of \c Callbacks are in use. Read without the lock by the
/// poller; writers serialise on \c QueueLock.
///
/// Only ever grows while callbacks are registered. Removal leaves a hole with a
/// null \c CB, which the walk skips -- so the surviving callbacks keep both
/// their relative order and their handles.
static unsigned CallbackCount = 0;

namespace detail {

void runCallbackChain(const CallbackEntry *Entries, unsigned Count,
                      const QueueInfo &Q, uint64_t PacketIndex,
                      hsa::AqlPacket &Packet) {
  // Downwards: last registered runs first. ROCr does the same, starting at
  // interceptors.size() - 1 (intercept_queue.cpp:375).
  for (unsigned I = Count; I-- > 0;)
    if (Entries[I].CB != nullptr)
      Entries[I].CB(Q, PacketIndex, Packet, Entries[I].UserData);
}

} // namespace detail

void runRegisteredCallbacks(const QueueInfo &Q, uint64_t PacketIndex,
                            hsa::AqlPacket &Packet) {
  detail::runCallbackChain(Callbacks,
                           __atomic_load_n(&CallbackCount, __ATOMIC_ACQUIRE), Q,
                           PacketIndex, Packet);
}

CallbackHandle addPacketCallback(PacketCallback CB, void *UserData) {
  if (CB == nullptr)
    return InvalidCallbackHandle;

  CallbackHandle H = InvalidCallbackHandle;
  pthread_mutex_lock(&QueueLock);

  // Reuse a hole left by a removal before growing, so a tool that repeatedly
  // attaches and detaches does not exhaust the array.
  for (unsigned I = 0; I < CallbackCount; I++) {
    if (Callbacks[I].CB == nullptr) {
      Callbacks[I].UserData = UserData;
      __atomic_store_n(&Callbacks[I].CB, CB, __ATOMIC_RELEASE);
      H = static_cast<CallbackHandle>(I);
      break;
    }
  }

  if (H == InvalidCallbackHandle && CallbackCount < MaxPacketCallbacks) {
    const unsigned I = CallbackCount;
    Callbacks[I].CB = CB;
    Callbacks[I].UserData = UserData;
    // Publish the entry before the count that exposes it. The poller reads the
    // count with acquire, so it never sees a slot it can reach but not read.
    __atomic_store_n(&CallbackCount, I + 1, __ATOMIC_RELEASE);
    H = static_cast<CallbackHandle>(I);
  }

  pthread_mutex_unlock(&QueueLock);

  if (H == InvalidCallbackHandle)
    fprintf(stderr,
            "%sWARNING: %u packet callbacks are already registered; this one "
            "will never be called\n",
            LogPrefix, MaxPacketCallbacks);
  return H;
}

void removePacketCallback(CallbackHandle H) {
  if (H < 0 || static_cast<unsigned>(H) >= MaxPacketCallbacks)
    return;
  pthread_mutex_lock(&QueueLock);
  __atomic_store_n(&Callbacks[H].CB, static_cast<PacketCallback>(nullptr),
                   __ATOMIC_RELEASE);
  Callbacks[H].UserData = nullptr;
  pthread_mutex_unlock(&QueueLock);
}

void setPacketCallback(PacketCallback CB, void *UserData) {
  pthread_mutex_lock(&QueueLock);
  for (unsigned I = 0; I < CallbackCount; I++) {
    __atomic_store_n(&Callbacks[I].CB, static_cast<PacketCallback>(nullptr),
                     __ATOMIC_RELEASE);
    // Clear the user pointer too. Nulling only the function leaves a slot that
    // a later add can recycle while still carrying the previous tool's data --
    // harmless only for as long as every add happens to overwrite both fields.
    Callbacks[I].UserData = nullptr;
  }
  if (CB != nullptr) {
    Callbacks[0].CB = CB;
    Callbacks[0].UserData = UserData;
    __atomic_store_n(&CallbackCount, 1u, __ATOMIC_RELEASE);
  } else {
    __atomic_store_n(&CallbackCount, 0u, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&QueueLock);
}

//===----------------------------------------------------------------------===//
// Copying one packet
//===----------------------------------------------------------------------===//

static void runCallback(ForwardedQueue &Q, uint64_t Index,
                        hsa::AqlPacket &Packet) {
  if (verboseEnabled() &&
      packetType(Packet.Packet.Header) == PacketTypeKernelDispatch)
    fprintf(stderr, "%scallback: kernel dispatch, index=%llu\n", LogPrefix,
            static_cast<unsigned long long>(Index));

  runRegisteredCallbacks(Q.Info, Index, Packet);

  if (demoZeroGridEnabled() &&
      packetType(Packet.Packet.Header) == PacketTypeKernelDispatch) {
    auto *Bytes = reinterpret_cast<unsigned char *>(&Packet);
    uint32_t OldGrid;
    memcpy(&OldGrid, Bytes + AqlGridSizeXOffset, sizeof(OldGrid));
    uint32_t NewGrid = 0;
    memcpy(Bytes + AqlGridSizeXOffset, &NewGrid, sizeof(NewGrid));
    fprintf(stderr, "%sDEMO: grid_size_x %u -> %u (kernel should not run)\n",
            LogPrefix, OldGrid, NewGrid);
  }
}

/// Copy application slot -> our slot, header last.
///
/// The callback is deliberately run on a **staged copy** rather than on our slot
/// directly. Our slot's header is held at INVALID for the whole operation --
/// that is the gate that stops the GPU acting early -- so a callback inspecting
/// the slot would see every packet as INVALID rather than its real type. Staging
/// costs one 64-byte copy and lets the callback see a complete, coherent packet,
/// including the header, which it may also edit.
static void forwardOnePacket(ForwardedQueue &Q, uint64_t Index) {
  // Two indices, not one. The source slot follows the application's stream; the
  // destination slot follows what we have produced. They are equal today -- one
  // packet in, one packet out -- and writing it this way is what makes that a
  // stated property rather than an unexamined coincidence. See Produced.
  const size_t SrcSlot = static_cast<size_t>(Index % Q.Info.SlotCount);
  const size_t DstSlot = static_cast<size_t>(Q.Produced % Q.Info.SlotCount);
  volatile unsigned char *Src = Q.AppRing + SrcSlot * AqlPacketBytes;
  volatile unsigned char *Dst = Q.ShimRing + DstSlot * AqlPacketBytes;

  // Acquire-load the header first: seeing a committed header is what makes the
  // rest of the packet safe to read.
  uint16_t Header = loadHeader(Src + AqlHeaderOffset);

  hsa::AqlPacket Staged;
  memcpy(&Staged, const_cast<const unsigned char *>(Src), AqlPacketBytes);
  Staged.Packet.Header = Header;

  // Close our slot before touching its body, so a partially written packet is
  // never visible to the GPU.
  storeHeader(Dst + AqlHeaderOffset, PacketTypeInvalid);

  runCallback(Q, Index, Staged);

  // Body first, then the header last -- publishing the header is what makes the
  // packet live, so any edit the callback made is already in place by then. The
  // header itself comes from the staged copy, so a callback may change it too.
  memcpy(const_cast<unsigned char *>(Dst) + 2,
         reinterpret_cast<const unsigned char *>(&Staged) + 2,
         AqlPacketBytes - 2);
  storeHeader(Dst + AqlHeaderOffset, Staged.Packet.Header);

  // One packet in, one packet out. The single place that would change if a
  // callback could ever emit a different number.
  Q.Produced++;

  if (packetType(Header) == PacketTypeKernelDispatch)
    Q.DispatchCount++;

  if (verboseEnabled())
    fprintf(stderr, "%sforwarded gpu=%u q=%u idx=%llu slot=%zu type=%u\n",
            LogPrefix, Q.Info.GpuId, Q.Info.QueueId,
            static_cast<unsigned long long>(Index), DstSlot,
            packetType(Header));
}

//===----------------------------------------------------------------------===//
// The polling thread
//===----------------------------------------------------------------------===//
static void *pollerMain(void *) {
  for (;;) {
    pthread_mutex_lock(&QueueLock);
    int N = QueueCount;
    pthread_mutex_unlock(&QueueLock);

    for (int I = 0; I < N; I++) {
      ForwardedQueue &Q = Queues[I];
      // Acquire pairs with the release store that publishes a committed slot,
      // so seeing SlotLive guarantees the rest of the entry is visible. Any
      // other state means the entry's fields are either not ready yet or no
      // longer ours to read.
      if (__atomic_load_n(&Q.State, __ATOMIC_ACQUIRE) != SlotLive)
        continue;

      uint64_t Claimed =
          Q.AppWritePointer
              ? __atomic_load_n(Q.AppWritePointer, __ATOMIC_ACQUIRE)
              : 0;

      // At most one ring per pass, so this loop always returns and the Active
      // check above stays reachable. An unbounded inner loop meant
      // DESTROY_QUEUE could never stop the poller and it read freed memory.
      //
      // The cap is also a throughput ceiling: SlotCount packets per poll
      // period. Measured workloads are far below it, but a producer sustaining
      // more would look like a hang rather than a slowdown.
      uint32_t Budget = Q.Info.SlotCount;
      while (Q.Consumed < Claimed && Budget > 0) {
        Budget--;
        size_t Slot = static_cast<size_t>(Q.Consumed % Q.Info.SlotCount);
        volatile unsigned char *Src = Q.AppRing + Slot * AqlPacketBytes;
        uint16_t Header = loadHeader(Src + AqlHeaderOffset);
        if (packetType(Header) == PacketTypeInvalid)
          break; // claimed, but not finished being written

        forwardOnePacket(Q, Q.Consumed);

        // Re-arm the marker so a reused slot reads "empty" again; otherwise the
        // next lap finds the previous packet's still-valid header and we cannot
        // tell "reused" from "freshly written". ROCr does the same to its own
        // proxy ring (intercept_queue.cpp:384).
        //
        // Safe: the application cannot reuse this slot until the GPU has
        // consumed our copy, and the GPU cannot have consumed it before we just
        // copied it -- so we always erase ahead of the application.
        storeHeader(Src + AqlHeaderOffset, PacketTypeInvalid);
        Q.Consumed++;
      }
    }

    // One full pass is done. Publishing it releases any entry that was marked
    // dead before this pass began -- see reclaimQueueSlot.
    __atomic_add_fetch(&PollPass, 1, __ATOMIC_RELEASE);

    struct timespec Ts = {0, 20 * 1000}; // 20 us
    nanosleep(&Ts, nullptr);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Queue bookkeeping
//===----------------------------------------------------------------------===//
static void summarizeAll();

/// Reuse the entry of a queue that has been destroyed.
///
/// \par Why a grace period rather than a lock
/// The poller walks entries without taking \c QueueLock -- deliberately, since
/// it runs continuously and would otherwise contend with every \c ioctl the
/// application makes. So marking an entry dead does not mean the poller has
/// stopped reading it: a pass that already tested the state and moved on to the
/// ring pointers is still in flight. Overwriting the entry underneath that pass
/// would point it at a freed ring.
///
/// The clock is \c PollPass, the count of completed passes. \c deactivateQueue
/// stores \c SlotDead first and only then records the pass number, so any pass
/// that could still be inside the entry is numbered at or below the recorded
/// one; waiting for one more completed pass is therefore sufficient. We wait for
/// two, which costs about 40 microseconds and removes the need to trust that
/// argument in review.
///
/// \return an index whose entry is now \c SlotReserved, or -1.
static int reclaimQueueSlot() {
  static constexpr uint64_t GracePasses = 2;
  /// 100 tries at 50 us is 5 ms -- two orders of magnitude above the 20 us poll
  /// period, so giving up means the poller has genuinely stopped, not that it
  /// was merely busy.
  static constexpr unsigned MaxAttempts = 100;

  for (unsigned Attempt = 0; Attempt < MaxAttempts; Attempt++) {
    uint64_t Pass = __atomic_load_n(&PollPass, __ATOMIC_ACQUIRE);

    // The lock is what makes the state and its pass number one decision: it is
    // also held across both stores in deactivateQueue, so a half-updated dead
    // entry is never visible here.
    pthread_mutex_lock(&QueueLock);
    for (int I = 0; I < QueueCount; I++) {
      const int St = __atomic_load_n(&Queues[I].State, __ATOMIC_ACQUIRE);
      const bool Reusable =
          St == SlotFree || // never polled, so no grace period is needed
          (St == SlotDead && Pass >= Queues[I].DeadAtPass + GracePasses);
      if (!Reusable)
        continue;
      __atomic_store_n(&Queues[I].State, SlotReserved, __ATOMIC_RELEASE);

      // Take the dead queue's ring with us and clear it from the entry, so the
      // caller cannot mistake it for the new queue's.
      const int OldFd = Queues[I].Fd;
      const uint64_t OldHandle = Queues[I].RingHandle;
      const uint32_t OldGpu = Queues[I].Info.GpuId;
      void *OldPages = Queues[I].RingPages;
      const size_t OldBytes = Queues[I].RingPagesBytes;
      Queues[I].RingPages = nullptr;
      pthread_mutex_unlock(&QueueLock);

      // Outside the lock: this makes two driver calls, and the poller must not
      // be kept waiting behind them. The entry is already reserved, so nothing
      // else can touch it meanwhile.
      releaseRing(OldFd, OldHandle, OldGpu, OldPages, OldBytes);
      return I;
    }
    pthread_mutex_unlock(&QueueLock);

    // Nothing reusable. Without a running poller the clock never advances, so
    // waiting cannot change the answer.
    if (!__atomic_load_n(&PollerStarted, __ATOMIC_ACQUIRE))
      return -1;
    struct timespec Ts = {0, 50 * 1000};
    nanosleep(&Ts, nullptr);
  }
  return -1;
}

/// Claim a tracking slot BEFORE the ring is substituted.
///
/// Order matters. Substituting the ring and then finding we cannot track the
/// queue is the worst outcome available: the GPU would read our ring, nothing
/// would ever fill it, and the application would wait forever on work that can
/// never run -- a silent hang with no error anywhere.
///
/// Prefers a fresh entry and falls back to reusing a dead one. Reuse is not an
/// optimisation: without it a process that creates and destroys queues loses
/// interception permanently once it has created \c MaxTrackedQueues of them over
/// its whole lifetime, even with only one alive at a time.
static int reserveQueueSlot() {
  pthread_mutex_lock(&QueueLock);
  if (QueueCount < MaxTrackedQueues) {
    int Idx = QueueCount++;
    __atomic_store_n(&Queues[Idx].State, SlotReserved, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&QueueLock);
    return Idx;
  }
  pthread_mutex_unlock(&QueueLock);
  return reclaimQueueSlot();
}

/// Give a reserved slot back after a creation that did not happen.
///
/// Without this every failed CREATE_QUEUE would burn an entry permanently:
/// reserved is neither live nor dead, so nothing would ever reclaim it.
static void releaseQueueSlot(int Idx) {
  __atomic_store_n(&Queues[Idx].State, SlotFree, __ATOMIC_RELEASE);
}

static void commitQueueSlot(int Idx, volatile unsigned char *AppRing,
                            volatile unsigned char *ShimRing,
                            volatile uint64_t *AppWritePointer,
                            const QueueInfo &Info, int Fd, uint64_t RingHandle,
                            size_t RingPagesBytes) {
  ForwardedQueue &Q = Queues[Idx];
  Q.AppRing = AppRing;
  Q.ShimRing = ShimRing;
  Q.AppWritePointer = AppWritePointer;
  Q.Info = Info;
  Q.Fd = Fd;
  Q.RingHandle = RingHandle;
  Q.RingPages = const_cast<unsigned char *>(ShimRing);
  Q.RingPagesBytes = RingPagesBytes;
  Q.Consumed = 0;
  Q.Produced = 0;
  Q.DispatchCount = 0;
  Q.Summarized = 0;
  Q.DeadAtPass = 0;

  __atomic_add_fetch(&WrappedQueueTotal, 1, __ATOMIC_RELEASE);

  // Publish last. The poller reads these fields without the lock; its acquire
  // load of State pairs with this store. Until then the entry reads
  // SlotReserved and is skipped, so a reused entry never exposes a mixture of
  // the old queue's pointers and the new one's.
  __atomic_store_n(&Q.State, SlotLive, __ATOMIC_RELEASE);

  pthread_mutex_lock(&QueueLock);
  if (!PollerStarted) {
    PollerStarted = (pthread_create(&PollerThread, nullptr, pollerMain,
                                    nullptr) == 0);
    atexit(summarizeAll);
  }
  pthread_mutex_unlock(&QueueLock);
}

/// One line per queue at teardown, so the numbers worth checking survive with
/// per-packet logging off. Counts are final by then: the GPU cannot complete
/// work we never copied, so an application that waited for its results has
/// necessarily waited for us.
static void summarizeQueue(ForwardedQueue &Q) {
  if (__atomic_exchange_n(&Q.Summarized, 1, __ATOMIC_ACQ_REL))
    return;
  fprintf(stderr, "%ssummary gpu=%u q=%u: forwarded=%llu packets, dispatches=%llu\n",
          LogPrefix, Q.Info.GpuId, Q.Info.QueueId,
          static_cast<unsigned long long>(Q.Consumed),
          static_cast<unsigned long long>(Q.DispatchCount));
}

static void summarizeAll() {
  int N = __atomic_load_n(&QueueCount, __ATOMIC_ACQUIRE);
  for (int I = 0; I < N; I++)
    if (__atomic_load_n(&Queues[I].State, __ATOMIC_ACQUIRE) == SlotLive)
      summarizeQueue(Queues[I]);
}

/// Stop the poller touching a queue's ring before the application frees it.
///
/// Matching on queue id alone is correct, not a simplification: DESTROY_QUEUE
/// passes nothing else, and neither do UPDATE_QUEUE, SET_CU_MASK or
/// GET_QUEUE_WAVE_STATE. The driver resolves a queue from that id alone, which
/// it could only do if the id is unique within the process.
///
/// Only live entries are considered. Dead ones may carry the same id -- the
/// driver reuses queue ids once a queue is gone -- and touching them again would
/// reset a grace period that is already running.
static void deactivateQueue(uint32_t QueueId) {
  pthread_mutex_lock(&QueueLock);
  for (int I = 0; I < QueueCount; I++) {
    if (__atomic_load_n(&Queues[I].State, __ATOMIC_ACQUIRE) != SlotLive)
      continue;
    if (Queues[I].Info.QueueId != QueueId)
      continue;
    summarizeQueue(Queues[I]);

    // Mark dead first, then take the timestamp -- that order is what bounds the
    // grace period, and reclaimQueueSlot explains why. Both stores happen under
    // the lock, which the reclaimer also holds, so it never sees one without
    // the other.
    __atomic_store_n(&Queues[I].State, SlotDead, __ATOMIC_RELEASE);
    Queues[I].DeadAtPass = __atomic_load_n(&PollPass, __ATOMIC_ACQUIRE);
  }
  pthread_mutex_unlock(&QueueLock);
}

//===----------------------------------------------------------------------===//
// ioctl handling
//===----------------------------------------------------------------------===//
// Request decoding is hand-rolled because <sys/ioctl.h> cannot be included
// alongside our own ioctl definition.
static constexpr unsigned ioctlNr(unsigned long Request) {
  return static_cast<unsigned>(Request & 0xFFu);
}
static constexpr unsigned CreateQueueNr = AMDKFD_IOC_CREATE_QUEUE & 0xFFu;
static constexpr unsigned DestroyQueueNr = AMDKFD_IOC_DESTROY_QUEUE & 0xFFu;

static int handleCreateQueue(int Fd, unsigned long Request, void *Arg) {
  auto *Q = static_cast<struct kfd_ioctl_create_queue_args *>(Arg);

  // Only AQL compute queues are wrapped; everything else passes through.
  if (Q->queue_type != KFD_IOC_QUEUE_TYPE_COMPUTE_AQL)
    return RealIoctl(Fd, Request, Arg);

  // The tool's own queues are not the application's. Instrumenting them would
  // feed our dispatches to our own callback.
  if (insideToolRegion()) {
    __atomic_add_fetch(&ExcludedQueueTotal, 1, __ATOMIC_RELEASE);
    if (verboseEnabled())
      fprintf(stderr, "%sleaving AQL queue on gpu=%u unwrapped: the tool "
                      "created it\n",
              LogPrefix, Q->gpu_id);
    return RealIoctl(Fd, Request, Arg);
  }

  int SlotIdx = reserveQueueSlot();
  if (SlotIdx < 0) {
    fprintf(stderr,
            "%sWARNING: %d queues are alive at once, which is the tracking "
            "limit; the queue on gpu=%u was created UNWRAPPED and its packets "
            "will not reach the callback\n",
            LogPrefix, MaxTrackedQueues, Q->gpu_id);
    return RealIoctl(Fd, Request, Arg);
  }

  // Capture the application's addresses before we overwrite anything.
  auto AppRingVa = static_cast<uintptr_t>(Q->ring_base_address);
  auto AppWptrVa = static_cast<uintptr_t>(Q->write_pointer_address);
  uint32_t RingBytes = Q->ring_size;

  size_t ShimRingBytes = 0;
  void *ShimRing = allocRingPages(RingBytes, &ShimRingBytes);
  uint64_t RingHandle = 0;
  uint16_t Invalid = PacketTypeInvalid;
  for (uint32_t Off = 0; Off + AqlPacketBytes <= RingBytes;
       Off += AqlPacketBytes)
    memcpy(static_cast<unsigned char *>(ShimRing) + Off + AqlHeaderOffset,
           &Invalid, sizeof(Invalid));

  if (!registerRingWithGpu(Fd, ShimRing, RingBytes, Q->gpu_id, &RingHandle)) {
    fprintf(stderr, "%sring registration failed; queue created unwrapped\n",
            LogPrefix);
    munmap(ShimRing, ShimRingBytes);
    releaseQueueSlot(SlotIdx);
    return RealIoctl(Fd, Request, Arg);
  }

  // The substitution itself: from here the GPU reads our buffer.
  Q->ring_base_address = static_cast<__u64>(reinterpret_cast<uintptr_t>(ShimRing));

  int Ret = RealIoctl(Fd, Request, Arg);
  int SavedErrno = errno;
  if (Ret != 0) {
    fprintf(stderr, "%sCREATE_QUEUE failed ret=%d errno=%d (%s)\n", LogPrefix,
            Ret, SavedErrno, strerror(SavedErrno));
    // Hand the entry back. The poller never saw it, so this needs no grace
    // period -- and skipping it would burn one entry per failed creation until
    // the table was full of reservations nothing could ever release.
    releaseRing(Fd, RingHandle, Q->gpu_id, ShimRing, ShimRingBytes);
    releaseQueueSlot(SlotIdx);
    errno = SavedErrno; // do not let our logging clobber the caller's errno
    return Ret;
  }

  // Establish the "empty" marker in the application's ring. Safe here: the
  // queue has only just been created, so nothing can have been submitted.
  for (uint32_t Off = 0; Off + AqlPacketBytes <= RingBytes;
       Off += AqlPacketBytes)
    __atomic_store_n(reinterpret_cast<uint16_t *>(AppRingVa + Off +
                                                  AqlHeaderOffset),
                     static_cast<uint16_t>(PacketTypeInvalid),
                     __ATOMIC_RELEASE);

  QueueInfo Info{};
  Info.GpuId = Q->gpu_id;
  Info.QueueId = Q->queue_id;
  Info.RingByteSize = RingBytes;
  Info.SlotCount = RingBytes / AqlPacketBytes;

  fprintf(stderr,
          "%swrapped AQL queue gpu=%u queue_id=%u app_ring=0x%llx "
          "shim_ring=%p slots=%u\n",
          LogPrefix, Info.GpuId, Info.QueueId,
          static_cast<unsigned long long>(AppRingVa), ShimRing,
          Info.SlotCount);

  commitQueueSlot(SlotIdx, reinterpret_cast<volatile unsigned char *>(AppRingVa),
                  static_cast<volatile unsigned char *>(ShimRing),
                  reinterpret_cast<volatile uint64_t *>(AppWptrVa), Info, Fd,
                  RingHandle, ShimRingBytes);
  return Ret;
}

int handleIoctl(int Fd, unsigned long Request, void *Arg) {
  ensureRealIoctlResolved();

  bool IsKfd = fdIsKfd(Fd);
  unsigned Nr = ioctlNr(Request);

  if (IsKfd && Nr == DestroyQueueNr && Arg != nullptr)
    deactivateQueue(*static_cast<const __u32 *>(Arg));

  if (IsKfd && Nr == CreateQueueNr && Arg != nullptr)
    return handleCreateQueue(Fd, Request, Arg);

  return RealIoctl(Fd, Request, Arg);
}

} // namespace luthier::kfd

extern "C" void luthierKfdSetPacketCallback(luthier::kfd::PacketCallback CB,
                                            void *UserData) {
  luthier::kfd::setPacketCallback(CB, UserData);
}

extern "C" int luthierKfdAddPacketCallback(luthier::kfd::PacketCallback CB,
                                           void *UserData) {
  return luthier::kfd::addPacketCallback(CB, UserData);
}

extern "C" void luthierKfdRemovePacketCallback(int Handle) {
  luthier::kfd::removePacketCallback(Handle);
}

extern "C" unsigned long long luthierKfdWrappedQueueCount() {
  return luthier::kfd::wrappedQueueCount();
}

extern "C" unsigned long long luthierKfdExcludedQueueCount() {
  return luthier::kfd::excludedQueueCount();
}

extern "C" void luthierKfdBeginToolRegion() { luthier::kfd::beginToolRegion(); }

extern "C" void luthierKfdEndToolRegion() { luthier::kfd::endToolRegion(); }
