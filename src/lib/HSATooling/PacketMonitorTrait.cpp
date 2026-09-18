//===-- PacketMonitorTrait.h ----------------------------------------------===//
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
/// The non-templated half of the packet monitor: everything whose behaviour
/// does not depend on which tool is doing the monitoring. See
/// \c luthier/HSATooling/PacketMonitorTrait.h for the design, and
/// \c PacketMonitor's own comment for why the split exists.
//===----------------------------------------------------------------------===//
#include "luthier/HSATooling/PacketMonitorTrait.h"

#include "luthier/KFD/FdSharing.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>

#include <dlfcn.h>
#include <linux/kfd_ioctl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#define DEBUG_TYPE "luthier-packet-monitor"

namespace luthier {

namespace {

//===--------------------------------------------------------------------===//
// AQL packet layout. Fixed by the AQL specification, so these are safe as
// constants -- unlike the queue-descriptor offsets, which belong to an AMD
// struct and are derived with offsetof where they are needed.
//===--------------------------------------------------------------------===//
static constexpr unsigned AqlPacketBytes = sizeof(hsa::AqlPacket);
static constexpr unsigned AqlHeaderOffset = 0;
/// Byte offset of \c kernel_object within an AQL kernel-dispatch packet.
static constexpr unsigned AqlKernelObjectOffset =
    offsetof(hsa_kernel_dispatch_packet_t, kernel_object);
static constexpr unsigned PacketTypeInvalid = 1;
static constexpr unsigned PacketTypeKernelDispatch =
    HSA_PACKET_TYPE_KERNEL_DISPATCH;
/// The driver rejects a queue whose ring it does not know about, so we must
/// perform the same two-step registration the runtime does. EXECUTABLE turned
/// out to be the load-bearing flag -- without it the GPU faults.
static constexpr uint32_t RingAllocFlags =
    KFD_IOC_ALLOC_MEM_FLAGS_USERPTR | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
    KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE |
    KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED;

static inline unsigned packetType(uint16_t Header) { return Header & 0xFF; }

/// Read a slot's 2-byte header.
///
/// Acquire ordering pairs with the producer's release store of that header,
/// and is what makes the rest of the packet safe to read afterwards: the
/// producer writes the body first and the header last, so seeing the header
/// means the body is there.
static inline uint16_t loadHeader(volatile unsigned char *Slot) {
  return __atomic_load_n(reinterpret_cast<const uint16_t *>(
                             const_cast<const unsigned char *>(Slot)),
                         __ATOMIC_ACQUIRE);
}

/// Publish a slot's header.
///
/// Release ordering means every earlier write is visible before this one,
/// which is what stops the GPU seeing a header for a packet whose body has
/// not landed.
static inline void storeHeader(volatile unsigned char *Slot, uint16_t Value) {
  __atomic_store_n(
      reinterpret_cast<uint16_t *>(const_cast<unsigned char *>(Slot)), Value,
      __ATOMIC_RELEASE);
}

//===--------------------------------------------------------------------===//
// Per-queue state
//===--------------------------------------------------------------------===//
// ioctl decoding
//
// Hand-rolled because <sys/ioctl.h> cannot be included alongside the driver's
// own request definitions.
//===--------------------------------------------------------------------===//
static constexpr unsigned ioctlNr(unsigned long Request) {
  return static_cast<unsigned>(Request & 0xFFu);
}
static constexpr unsigned CreateQueueNr = AMDKFD_IOC_CREATE_QUEUE & 0xFFu;
static constexpr unsigned DestroyQueueNr = AMDKFD_IOC_DESTROY_QUEUE & 0xFFu;
static constexpr unsigned RuntimeEnableNr = AMDKFD_IOC_RUNTIME_ENABLE & 0xFFu;
static constexpr unsigned CreateEventNr = AMDKFD_IOC_CREATE_EVENT & 0xFFu;

/// Record a successful allocation so an address can later be resolved back to
/// it. Called after the real ioctl, because the handle is an output and a

} // namespace

PacketMonitor::~PacketMonitor() = default;

void PacketMonitor::shutdown() {
  stopPoller();
  summarizeAll();
  releaseAllRings();
}

void PacketMonitor::endProcessWideToolRegion() {
  if (ProcessToolRegionDepth.load() == 0) {
    fprintf(stderr, "%sendProcessWideToolRegion() with no matching begin\n",
            LogPrefix);
    return;
  }
  ProcessToolRegionDepth.fetch_sub(1);
}

PacketMonitor::CallbackHandle
PacketMonitor::addPacketCallback(PacketCallback CB, void *UserData) {
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
    // Publish the entry before the count that exposes it. The poller reads
    // the count with acquire, so it never sees a slot it can reach but not
    // read.
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

void PacketMonitor::removePacketCallback(CallbackHandle H) {
  if (H < 0 || static_cast<unsigned>(H) >= MaxPacketCallbacks)
    return;
  pthread_mutex_lock(&QueueLock);
  __atomic_store_n(&Callbacks[H].CB, static_cast<PacketCallback>(nullptr),
                   __ATOMIC_RELEASE);
  Callbacks[H].UserData = nullptr;
  pthread_mutex_unlock(&QueueLock);
  // Outside the lock: the poller takes it too, so draining while holding it
  // would stop the passes we are waiting for.
  drainPacketCallbacks();
}

void PacketMonitor::setPacketCallback(PacketCallback CB, void *UserData) {
  pthread_mutex_lock(&QueueLock);
  for (unsigned I = 0; I < CallbackCount; I++) {
    __atomic_store_n(&Callbacks[I].CB, static_cast<PacketCallback>(nullptr),
                     __ATOMIC_RELEASE);
    // Clear the user pointer too. Nulling only the function leaves a slot
    // that a later add can recycle while still carrying the previous tool's
    // data -- harmless only for as long as every add happens to overwrite
    // both fields.
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
  // Clearing the chain has the same in-flight hazard as removePacketCallback.
  if (CB == nullptr)
    drainPacketCallbacks();
}

void PacketMonitor::runCallbackChain(const CallbackEntry *Entries,
                                     unsigned Count, const QueueInfo &Q,
                                     uint64_t PacketIndex,
                                     hsa::AqlPacket &Packet) {
  // Downwards: last registered runs first. ROCr does the same, starting at
  // interceptors.size() - 1 (intercept_queue.cpp:375).
  for (unsigned I = Count; I-- > 0;)
    if (Entries[I].CB != nullptr)
      Entries[I].CB(Q, PacketIndex, Packet, Entries[I].UserData);
}

void PacketMonitor::runRegisteredCallbacks(const QueueInfo &Q,
                                           uint64_t PacketIndex,
                                           hsa::AqlPacket &Packet) {
  runCallbackChain(Callbacks, __atomic_load_n(&CallbackCount, __ATOMIC_ACQUIRE),
                   Q, PacketIndex, Packet);
}

int PacketMonitor::handleIoctl(int Fd, unsigned long Request, void *Arg) {
  __atomic_add_fetch(&InterceptedIoctlTotal, 1, __ATOMIC_RELAXED);

  const bool IsKfd = fdIsKfd(Fd);
  const unsigned Nr = ioctlNr(Request);

  if (IsKfd && Nr == DestroyQueueNr && Arg != nullptr)
    deactivateQueue(*static_cast<const __u32 *>(Arg));

  if (IsKfd && Nr == CreateQueueNr && Arg != nullptr)
    return handleCreateQueue(Fd, Request, Arg);

  // Note an event page established before we brought HSA up. Only the window
  // before initialization matters: after it, every CREATE_EVENT in the
  // process
  // -- ours and the application's -- passes event_page_offset as 0 and shares
  // the page hsakmt established. See applicationClaimedEventPage().
  if (IsKfd && Nr == CreateEventNr && Arg != nullptr &&
      !__atomic_load_n(&HsaInitAttempted, __ATOMIC_ACQUIRE)) {
    const int Ret = realIoctl(Fd, Request, Arg);
    if (Ret == 0) {
      __atomic_store_n(&ApplicationClaimedEventPage, true, __ATOMIC_RELEASE);
      LLVM_DEBUG(
          fprintf(stderr,
                  "%screate_event before HSA was initialized: the "
                  "application owns the event page, so ROCr will have to "
                  "busy-wait\n",
                  LogPrefix));
    }
    return Ret;
  }

  // Let a second party enable the runtime even though queues already exist.
  //
  // AMDKFD_IOC_RUNTIME_ENABLE is a per-process call, and the driver documents
  // EEXIST as "user queues already active prior to call". So when an
  // application has created its queues and something else initializes
  // afterwards -- HSA, so that an instrumented kernel can be loaded -- that
  // second call is refused. ROCr treats the refusal as fatal
  // (amd_kfd_driver.cpp, KfdDriver::Init), so hsa_init fails with a generic
  // status naming nothing.
  //
  // Reporting success here is not a lie: EEXIST says the runtime is enabled
  // and the caller is merely late, which is exactly the situation.
  // capabilities_mask is an output, so it is zeroed rather than left as
  // whatever the caller passed -- ROCr reads capabilities through
  // hsaKmtGetRuntimeCapabilities instead, so nothing depends on this one
  // being populated.
  //
  // Only EEXIST is absorbed. EBUSY means a call is genuinely pending and any
  // other failure is a real one, and turning those into success would hide a
  // problem rather than resolve one.
  if (IsKfd && Nr == RuntimeEnableNr && Arg != nullptr) {
    const int Ret = realIoctl(Fd, Request, Arg);
    if (Ret == 0 || errno != EEXIST)
      return Ret;
    auto *R = static_cast<struct kfd_ioctl_runtime_enable_args *>(Arg);
    R->capabilities_mask = 0;
    LLVM_DEBUG(
        fprintf(stderr,
                "%sruntime_enable was refused with EEXIST because queues "
                "already exist; reporting success so a late initializer can "
                "proceed\n",
                LogPrefix));
    errno = 0;
    return 0;
  }

  if (!IsKfd)
    return realIoctl(Fd, Request, Arg);

  // Report failing KFD calls under verbose. Worth having permanently: when a
  // second party initializes in a process that already claimed a per-process
  // driver resource, the symptom is a library returning a generic status code
  // with no indication of which ioctl refused it. This turns that into one
  // line. Failures are not inherently interesting -- userspace probes with
  // ioctls that are expected to fail -- which is why this is verbose-only.
  const int Ret = realIoctl(Fd, Request, Arg);
  if (Ret != 0) {
    const int SavedErrno = errno;
    fprintf(stderr, "%sioctl nr=0x%02x failed: errno=%d (%s)\n", LogPrefix, Nr,
            SavedErrno, strerror(SavedErrno));
    errno = SavedErrno;
  }
  return Ret;
}

llvm::Error PacketMonitor::ensureHsaInitializedInApplication() {
  if (__atomic_load_n(&HsaInitAttempted, __ATOMIC_ACQUIRE))
    return llvm::Error::success();

  pthread_mutex_lock(&HsaInitLock);
  if (__atomic_load_n(&HsaInitAttempted, __ATOMIC_ACQUIRE)) {
    pthread_mutex_unlock(&HsaInitLock);
    return llvm::Error::success();
  }

  llvm::Error Err = llvm::Error::success();
  {
    // Everything HSA does from here creates queues and allocations inside an
    // application that did not ask for them. Without this we would treat the
    // runtime's own queues as the application's and feed our dispatches to
    // our own callback.
    //
    // Process-wide, not the per-thread region: bringing the runtime up
    // creates queues on threads it spawns itself, so a thread-local flag does
    // not cover them. Measured -- with the per-thread region the runtime's
    // queue was wrapped as the process's second queue.
    ProcessWideToolRegion Region(*this);

    // Redirect HSA's render-node opens onto the descriptor the application
    // already had bound. Enabled now rather than at load time because there
    // is nothing to redirect to until the application has claimed a GPU.
    kfd::enableFdSharing();

    // The event page needs no special handling in the ordering we arrange,
    // and this is the fallback for the one where we arrive too late. See
    // applicationClaimedEventPage() for the measurements.
    //
    // Forcing this flag off costs real performance -- it makes every
    // hsa_signal_wait busy-poll instead of sleeping on a KFD event
    // (runtime.cpp:2524 sets g_use_interrupt_wait from it;
    // hsa_ext_amd.cpp:999 then forces every signal to the busy-wait kind) --
    // so it is set only when interrupts genuinely cannot work.
    if (ApplicationClaimedEventPage) {
      fprintf(stderr,
              "%sthe application established the KFD event page before this "
              "tool could initialize HSA, so ROCr is being put on busy-wait "
              "signals (HSA_ENABLE_INTERRUPT=0). This costs performance: "
              "every signal wait spins instead of sleeping. It happens only "
              "when the application held a /dev/kfd descriptor before the "
              "tool attached.\n",
              LogPrefix);
      // Read while the runtime is constructed, which happens inside hsa_init,
      // so setting it here is in time.
      setenv("HSA_ENABLE_INTERRUPT", "0", /*overwrite=*/1);
    }

    // LM_ID_BASE, not RTLD_DEFAULT and not plain dlopen: plain dlopen from
    // here would load the runtime into the auditor's namespace, where its
    // descriptors and the application's are different objects.
    AppHsaHandle =
        dlmopen(LM_ID_BASE, "libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
    if (AppHsaHandle == nullptr) {
      Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
          "dlmopen of libhsa-runtime64.so.1 into the application's linker "
          "namespace failed: {0}. Without the runtime in that namespace an "
          "instrumented kernel cannot be loaded, because it would be loaded "
          "against descriptors the application does not hold.",
          dlerror() ? dlerror() : "no error reported"));
    } else {
      auto *HsaInit = reinterpret_cast<decltype(hsa_init) *>(
          ::dlsym(AppHsaHandle, "hsa_init"));
      if (HsaInit == nullptr) {
        Err = LUTHIER_MAKE_GENERIC_ERROR(
            "libhsa-runtime64.so.1 was loaded into the application's linker "
            "namespace but does not export hsa_init.");
      } else {
        const hsa_status_t St = HsaInit();
        if (St != HSA_STATUS_SUCCESS)
          Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
              "hsa_init failed with status {0} inside an application that "
              "drives the KFD driver. Run with "
              "-debug-only=luthier-packet-monitor: every "
              "failing KFD ioctl is printed with its number and errno, which "
              "is what identifies a per-process driver resource the "
              "application already claimed.",
              static_cast<int>(St)));
      }
    }
  }

  __atomic_store_n(&HsaInitAttempted, true, __ATOMIC_RELEASE);
  pthread_mutex_unlock(&HsaInitLock);
  return Err;
}

bool PacketMonitor::fdIsKfd(int Fd) {
  if (!KfdRdevCached) {
    struct stat St{};
    if (stat("/dev/kfd", &St) == 0 && S_ISCHR(St.st_mode))
      KfdRdev = St.st_rdev;
    else
      fprintf(stderr, "%sstat(\"/dev/kfd\") failed; fd checks will fail\n",
              LogPrefix);
    KfdRdevCached = true;
  }
  struct stat St{};
  if (fstat(Fd, &St) != 0)
    return false;
  return S_ISCHR(St.st_mode) && St.st_rdev == KfdRdev;
}

void *PacketMonitor::allocRingPages(size_t MinSize, size_t *OutSize) {
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

bool PacketMonitor::registerRingWithGpu(int Fd, void *Va, size_t Size,
                                        uint32_t GpuId, uint64_t *OutHandle) {
  struct kfd_ioctl_alloc_memory_of_gpu_args AllocArgs{};
  memset(&AllocArgs, 0, sizeof(AllocArgs));
  AllocArgs.va_addr = reinterpret_cast<__u64>(Va);
  AllocArgs.size = Size;
  AllocArgs.mmap_offset = reinterpret_cast<__u64>(Va);
  AllocArgs.gpu_id = GpuId;
  AllocArgs.flags = RingAllocFlags;
  if (realIoctl(Fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &AllocArgs) != 0) {
    fprintf(stderr, "%sALLOC_MEMORY_OF_GPU(0x%llx) failed: %s\n", LogPrefix,
            static_cast<unsigned long long>(AllocArgs.va_addr),
            strerror(errno));
    return false;
  }

  // Mapping to the queue's own node is sufficient; all-node mapping is not
  // required (the earlier GPU fault was the alloc flags, not the device
  // count).
  __u32 DeviceIds[1] = {GpuId};
  struct kfd_ioctl_map_memory_to_gpu_args MapArgs{};
  memset(&MapArgs, 0, sizeof(MapArgs));
  MapArgs.handle = AllocArgs.handle;
  MapArgs.device_ids_array_ptr = reinterpret_cast<__u64>(DeviceIds);
  MapArgs.n_devices = 1;
  if (realIoctl(Fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &MapArgs) != 0 ||
      MapArgs.n_success != 1) {
    fprintf(stderr, "%sMAP_MEMORY_TO_GPU(0x%llx) failed: %s (%u/1)\n",
            LogPrefix, static_cast<unsigned long long>(AllocArgs.va_addr),
            strerror(errno), MapArgs.n_success);
    return false;
  }
  *OutHandle = AllocArgs.handle;
  return true;
}

void PacketMonitor::releaseRing(int Fd, uint64_t Handle, uint32_t GpuId,
                                void *Pages, size_t PagesBytes) {
  if (Pages == nullptr)
    return;

  __u32 DeviceIds[1] = {GpuId};
  struct kfd_ioctl_unmap_memory_from_gpu_args UnmapArgs{};
  memset(&UnmapArgs, 0, sizeof(UnmapArgs));
  UnmapArgs.handle = Handle;
  UnmapArgs.device_ids_array_ptr = reinterpret_cast<__u64>(DeviceIds);
  UnmapArgs.n_devices = 1;
  if (realIoctl(Fd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &UnmapArgs) != 0)
    LLVM_DEBUG(fprintf(stderr, "%sUNMAP_MEMORY_FROM_GPU failed: %s\n",
                       LogPrefix, strerror(errno)));

  struct kfd_ioctl_free_memory_of_gpu_args FreeArgs{};
  memset(&FreeArgs, 0, sizeof(FreeArgs));
  FreeArgs.handle = Handle;
  if (realIoctl(Fd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &FreeArgs) != 0)
    LLVM_DEBUG(fprintf(stderr, "%sFREE_MEMORY_OF_GPU failed: %s\n", LogPrefix,
                       strerror(errno)));

  munmap(Pages, PagesBytes);
}

void PacketMonitor::releaseAllRings() {
  pthread_mutex_lock(&QueueLock);
  const int N = QueueCount;
  pthread_mutex_unlock(&QueueLock);
  for (int I = 0; I < N; I++) {
    ForwardedQueue &Q = Queues[I];
    void *Pages = Q.RingPages;
    Q.RingPages = nullptr;
    releaseRing(Q.Fd, Q.RingHandle, Q.Info.GpuId, Pages, Q.RingPagesBytes);
  }
}

bool PacketMonitor::insideToolRegion() const {
  return insideThreadToolRegion() || ProcessToolRegionDepth.load() != 0;
}

void PacketMonitor::runCallback(ForwardedQueue &Q, uint64_t Index,
                                hsa::AqlPacket &Packet) {
  runRegisteredCallbacks(Q.Info, Index, Packet);
  deliverDispatchPacket(Q.Info, Index, Packet);
}

void PacketMonitor::forwardOnePacket(ForwardedQueue &Q, uint64_t Index) {
  // Two indices, not one. The source slot follows the application's stream;
  // the destination slot follows what we have produced. They are equal today
  // -- one packet in, one packet out -- and writing it this way is what makes
  // that a stated property rather than an unexamined coincidence. See
  // Produced.
  const size_t SrcSlot = static_cast<size_t>(Index % Q.Info.SlotCount);
  const size_t DstSlot = static_cast<size_t>(Q.Produced % Q.Info.SlotCount);
  volatile unsigned char *Src = Q.AppRing + SrcSlot * AqlPacketBytes;
  volatile unsigned char *Dst = Q.ShimRing + DstSlot * AqlPacketBytes;

  // Acquire-load the header first: seeing a committed header is what makes
  // the rest of the packet safe to read.
  uint16_t Header = loadHeader(Src + AqlHeaderOffset);

  hsa::AqlPacket Staged;
  memcpy(&Staged, const_cast<const unsigned char *>(Src), AqlPacketBytes);
  Staged.Packet.Header = Header;

  // Close our slot before touching its body, so a partially written packet is
  // never visible to the GPU.
  storeHeader(Dst + AqlHeaderOffset, PacketTypeInvalid);

  runCallback(Q, Index, Staged);

  // Body first, then the header last -- publishing the header is what makes
  // the packet live, so any edit the callback made is already in place by
  // then. The header itself comes from the staged copy, so a callback may
  // change it too.
  memcpy(const_cast<unsigned char *>(Dst) + 2,
         reinterpret_cast<const unsigned char *>(&Staged) + 2,
         AqlPacketBytes - 2);
  storeHeader(Dst + AqlHeaderOffset, Staged.Packet.Header);

  // One packet in, one packet out. The single place that would change if a
  // callback could ever emit a different number.
  Q.Produced++;

  if (Staged.asKernelDispatch())
    Q.DispatchCount++;

  LLVM_DEBUG(fprintf(
      stderr, "%sforwarded gpu=%u q=%u idx=%llu slot=%zu type=%u\n", LogPrefix,
      Q.Info.GpuId, Q.Info.QueueId, static_cast<unsigned long long>(Index),
      DstSlot, packetType(Header)));
}

void *PacketMonitor::pollerTrampoline(void *Self) {
  static_cast<PacketMonitor *>(Self)->pollerMain();
  return nullptr;
}

void PacketMonitor::pollerMain() {
  while (!__atomic_load_n(&PollerStopRequested, __ATOMIC_ACQUIRE)) {
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

      uint64_t Claimed = Q.AppWritePointer ? __atomic_load_n(Q.AppWritePointer,
                                                             __ATOMIC_ACQUIRE)
                                           : 0;

      // At most one ring per pass, so this loop always returns and the state
      // check above stays reachable. An unbounded inner loop meant
      // DESTROY_QUEUE could never stop the poller and it read freed memory.
      //
      // Assumption A4: one ring's worth of packets per pass. A producer that
      // sustains more than SlotCount packets per poll period falls
      // permanently behind, and because the application blocks on its own
      // ring filling up, that presents as a hang rather than as a slowdown.
      //
      // Lag is how far the application's stream has run ahead of us. Anything
      // above one ring's worth means it has already overwritten packets we
      // had not copied, and those counts are gone. Recorded here rather than
      // fixed: see ForwardedQueue::OverrunPackets for why.
      if (Claimed > Q.Consumed) {
        const uint64_t Lag = Claimed - Q.Consumed;
        if (Lag > Q.MaxLag)
          Q.MaxLag = Lag;
        if (Lag > Q.Info.SlotCount) {
          const uint64_t Lost = Lag - Q.Info.SlotCount;
          if (Lost > Q.OverrunPackets)
            Q.OverrunPackets = Lost;
        }
      }

      uint32_t Budget = Q.Info.SlotCount;
      while (Q.Consumed < Claimed && Budget > 0) {
        Budget--;
        size_t Slot = static_cast<size_t>(Q.Consumed % Q.Info.SlotCount);
        volatile unsigned char *Src = Q.AppRing + Slot * AqlPacketBytes;
        uint16_t Header = loadHeader(Src + AqlHeaderOffset);
        if (packetType(Header) == PacketTypeInvalid)
          break; // claimed, but not finished being written

        forwardOnePacket(Q, Q.Consumed);

        // Re-arm the marker so a reused slot reads "empty" again; otherwise
        // the next lap finds the previous packet's still-valid header and we
        // cannot tell "reused" from "freshly written". ROCr does the same to
        // its own proxy ring (intercept_queue.cpp:384).
        //
        // Assumed safe because the application cannot reuse this slot until
        // the GPU has consumed our copy. Moving this above the publish -- so
        // the window where the application could rewrite the slot cannot open
        // at all -- was tried as a fix for the S10b hang and changed nothing,
        // so whatever that hang is, it is not this. Left where it was rather
        // than carrying an unvalidated reordering of a concurrency path.
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
}

void PacketMonitor::stopPoller() {
  pthread_mutex_lock(&QueueLock);
  const bool Started = PollerStarted;
  pthread_mutex_unlock(&QueueLock);
  if (!Started)
    return;
  __atomic_store_n(&PollerStopRequested, true, __ATOMIC_RELEASE);
  pthread_join(PollerThread, nullptr);
  pthread_mutex_lock(&QueueLock);
  PollerStarted = false;
  pthread_mutex_unlock(&QueueLock);
}

void PacketMonitor::drainPacketCallbacks() {
  static constexpr uint64_t GracePasses = 2;
  static constexpr unsigned MaxAttempts = 100;

  // Nothing has ever polled, so nothing can be in flight.
  if (!__atomic_load_n(&PollerStarted, __ATOMIC_ACQUIRE))
    return;

  const uint64_t Start = __atomic_load_n(&PollPass, __ATOMIC_ACQUIRE);
  for (unsigned Attempt = 0; Attempt < MaxAttempts; Attempt++) {
    if (__atomic_load_n(&PollPass, __ATOMIC_ACQUIRE) >= Start + GracePasses)
      return;
    struct timespec Ts = {0, 50 * 1000}; // 50 us
    nanosleep(&Ts, nullptr);
  }
}

int PacketMonitor::reclaimQueueSlot() {
  static constexpr uint64_t GracePasses = 2;
  /// 100 tries at 50 us is 5 ms -- two orders of magnitude above the 20 us
  /// poll period, so giving up means the poller has genuinely stopped, not
  /// that it was merely busy.
  static constexpr unsigned MaxAttempts = 100;

  for (unsigned Attempt = 0; Attempt < MaxAttempts; Attempt++) {
    uint64_t Pass = __atomic_load_n(&PollPass, __ATOMIC_ACQUIRE);

    // The lock is what makes the state and its pass number one decision: it
    // is also held across both stores in deactivateQueue, so a half-updated
    // dead entry is never visible here.
    pthread_mutex_lock(&QueueLock);
    for (int I = 0; I < QueueCount; I++) {
      const int St = __atomic_load_n(&Queues[I].State, __ATOMIC_ACQUIRE);
      const bool Reusable =
          St == SlotFree || // never polled, so no grace period is needed
          (St == SlotDead && Pass >= Queues[I].DeadAtPass + GracePasses);
      if (!Reusable)
        continue;
      __atomic_store_n(&Queues[I].State, SlotReserved, __ATOMIC_RELEASE);

      // Take the dead queue's ring with us and clear it from the entry, so
      // the caller cannot mistake it for the new queue's.
      const int OldFd = Queues[I].Fd;
      const uint64_t OldHandle = Queues[I].RingHandle;
      const uint32_t OldGpu = Queues[I].Info.GpuId;
      void *OldPages = Queues[I].RingPages;
      const size_t OldBytes = Queues[I].RingPagesBytes;
      Queues[I].RingPages = nullptr;
      pthread_mutex_unlock(&QueueLock);

      // Outside the lock: this makes two driver calls, and the poller must
      // not be kept waiting behind them. The entry is already reserved, so
      // nothing else can touch it meanwhile.
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

int PacketMonitor::reserveQueueSlot() {
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

void PacketMonitor::releaseQueueSlot(int Idx) {
  __atomic_store_n(&Queues[Idx].State, SlotFree, __ATOMIC_RELEASE);
}

void PacketMonitor::commitQueueSlot(int Idx, volatile unsigned char *AppRing,
                                    volatile unsigned char *ShimRing,
                                    volatile uint64_t *AppWritePointer,
                                    const QueueInfo &Info, int Fd,
                                    uint64_t RingHandle,
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
  Q.MaxLag = 0;
  Q.OverrunPackets = 0;
  Q.Summarized = 0;
  Q.DeadAtPass = 0;

  __atomic_add_fetch(&WrappedQueueTotal, 1, __ATOMIC_RELEASE);

  // Publish last. The poller reads these fields without the lock; its acquire
  // load of State pairs with this store. Until then the entry reads
  // SlotReserved and is skipped, so a reused entry never exposes a mixture of
  // the old queue's pointers and the new one's.
  __atomic_store_n(&Q.State, SlotLive, __ATOMIC_RELEASE);

  pthread_mutex_lock(&QueueLock);
  if (!PollerStarted)
    PollerStarted =
        (pthread_create(&PollerThread, nullptr, pollerTrampoline, this) == 0);
  pthread_mutex_unlock(&QueueLock);
}

void PacketMonitor::summarizeQueue(ForwardedQueue &Q) {
  if (__atomic_exchange_n(&Q.Summarized, 1, __ATOMIC_ACQ_REL))
    return;
  fprintf(stderr,
          "%ssummary gpu=%u q=%u: forwarded=%llu packets, dispatches=%llu, "
          "max_lag=%llu/%u\n",
          LogPrefix, Q.Info.GpuId, Q.Info.QueueId,
          static_cast<unsigned long long>(Q.Consumed),
          static_cast<unsigned long long>(Q.DispatchCount),
          static_cast<unsigned long long>(Q.MaxLag), Q.Info.SlotCount);
  // Separate line, and worded as a result rather than a diagnostic: a tool's
  // totals are short by at least this much, and a plausible-looking total is
  // exactly how this went unnoticed.
  if (Q.OverrunPackets != 0)
    fprintf(stderr,
            "%sWARNING gpu=%u q=%u: the application lapped us by at least "
            "%llu packet(s); those dispatches were never seen and any total "
            "reported for this queue is an undercount\n",
            LogPrefix, Q.Info.GpuId, Q.Info.QueueId,
            static_cast<unsigned long long>(Q.OverrunPackets));
}

void PacketMonitor::summarizeAll() {
  int N = __atomic_load_n(&QueueCount, __ATOMIC_ACQUIRE);
  for (int I = 0; I < N; I++)
    if (__atomic_load_n(&Queues[I].State, __ATOMIC_ACQUIRE) == SlotLive)
      summarizeQueue(Queues[I]);
}

void PacketMonitor::deactivateQueue(uint32_t QueueId) {
  pthread_mutex_lock(&QueueLock);
  for (int I = 0; I < QueueCount; I++) {
    if (__atomic_load_n(&Queues[I].State, __ATOMIC_ACQUIRE) != SlotLive)
      continue;
    if (Queues[I].Info.QueueId != QueueId)
      continue;
    summarizeQueue(Queues[I]);

    // Mark dead first, then take the timestamp -- that order is what bounds
    // the grace period, and reclaimQueueSlot explains why. Both stores happen
    // under the lock, which the reclaimer also holds, so it never sees one
    // without the other.
    __atomic_store_n(&Queues[I].State, SlotDead, __ATOMIC_RELEASE);
    Queues[I].DeadAtPass = __atomic_load_n(&PollPass, __ATOMIC_ACQUIRE);
  }
  pthread_mutex_unlock(&QueueLock);
}

int PacketMonitor::handleCreateQueue(int Fd, unsigned long Request, void *Arg) {
  auto *Q = static_cast<struct kfd_ioctl_create_queue_args *>(Arg);

  // Only AQL compute queues are wrapped; everything else passes through.
  if (Q->queue_type != KFD_IOC_QUEUE_TYPE_COMPUTE_AQL)
    return realIoctl(Fd, Request, Arg);

  // The tool's own queues are not the application's. Instrumenting them would
  // feed our dispatches to our own callback.
  if (insideToolRegion()) {
    __atomic_add_fetch(&ExcludedQueueTotal, 1, __ATOMIC_RELEASE);
    LLVM_DEBUG(fprintf(
        stderr,
        "%sleaving AQL queue on gpu=%u unwrapped: the tool created it\n",
        LogPrefix, Q->gpu_id));
    return realIoctl(Fd, Request, Arg);
  }

  int SlotIdx = reserveQueueSlot();
  if (SlotIdx < 0) {
    // Assumption A3: the application still gets a working queue and a success
    // return -- it simply is not instrumented. stderr is easy to miss, so
    // this is a silent, permanent loss of interception for that queue rather
    // than a visible failure. If that ever needs to be detectable rather than
    // merely logged, expose the count and let the harness assert zero.
    fprintf(stderr,
            "%sWARNING: %d queues are alive at once, which is the tracking "
            "limit; the queue on gpu=%u was created UNWRAPPED and its packets "
            "will not reach the callback\n",
            LogPrefix, MaxTrackedQueues, Q->gpu_id);
    return realIoctl(Fd, Request, Arg);
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
    return realIoctl(Fd, Request, Arg);
  }

  // The substitution itself: from here the GPU reads our buffer.
  Q->ring_base_address =
      static_cast<__u64>(reinterpret_cast<uintptr_t>(ShimRing));

  int Ret = realIoctl(Fd, Request, Arg);
  int SavedErrno = errno;
  if (Ret != 0) {
    fprintf(stderr, "%sCREATE_QUEUE failed ret=%d errno=%d (%s)\n", LogPrefix,
            Ret, SavedErrno, strerror(SavedErrno));
    // Hand the entry back. The poller never saw it, so this needs no grace
    // period -- and skipping it would burn one entry per failed creation
    // until the table was full of reservations nothing could ever release.
    releaseRing(Fd, RingHandle, Q->gpu_id, ShimRing, ShimRingBytes);
    releaseQueueSlot(SlotIdx);
    errno = SavedErrno; // do not let our logging clobber the caller's errno
    return Ret;
  }

  // Establish the "empty" marker in the application's ring. Safe here: the
  // queue has only just been created, so nothing can have been submitted.
  for (uint32_t Off = 0; Off + AqlPacketBytes <= RingBytes;
       Off += AqlPacketBytes)
    __atomic_store_n(
        reinterpret_cast<uint16_t *>(AppRingVa + Off + AqlHeaderOffset),
        static_cast<uint16_t>(PacketTypeInvalid), __ATOMIC_RELEASE);

  QueueInfo Info{};
  Info.GpuId = Q->gpu_id;
  Info.QueueId = Q->queue_id;
  Info.RingByteSize = RingBytes;
  Info.SlotCount = RingBytes / AqlPacketBytes;

  fprintf(stderr,
          "%swrapped AQL queue gpu=%u queue_id=%u app_ring=0x%llx "
          "shim_ring=%p slots=%u\n",
          LogPrefix, Info.GpuId, Info.QueueId,
          static_cast<unsigned long long>(AppRingVa), ShimRing, Info.SlotCount);

  commitQueueSlot(SlotIdx,
                  reinterpret_cast<volatile unsigned char *>(AppRingVa),
                  static_cast<volatile unsigned char *>(ShimRing),
                  reinterpret_cast<volatile uint64_t *>(AppWptrVa), Info, Fd,
                  RingHandle, ShimRingBytes);
  return Ret;
}

} // namespace luthier

#undef DEBUG_TYPE
