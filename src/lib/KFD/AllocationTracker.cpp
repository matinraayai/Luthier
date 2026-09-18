//===-- AllocationTracker.cpp - KFD-level GPU allocation tracking ---------===//
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
/// See \c luthier/KFD/AllocationTracker.h for what this is for and why the
/// design follows \c libhsakmt/src/fmm.c.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/AllocationTracker.h"

#include "luthier/Common/GenericLuthierError.h"

#include <llvm/Support/FormatVariadic.h>

#include <linux/kfd_ioctl.h>
#include <unistd.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <shared_mutex>

namespace luthier::kfd {

namespace detail {

void AllocationMap::record(const Allocation &A) {
  // Drop allocations with no virtual address. hsakmt asks for these on purpose
  // (va_addr == 0, "if allocate vram-only, use an invalid VA", fmm.c:1161-1162).
  // Keeping one would seat a range at 0 and make every low address resolve to it,
  // and a wrong hit reads exactly like a right one. RecordedTotal is deliberately
  // not bumped: nothing was recorded, so counting it would break the
  // recorded-versus-live distinction the counters exist to draw.
  if (A.Base == 0)
    return;

  // A base address can legitimately reappear after the previous allocation there
  // was freed, so overwrite rather than reject. If it reappears *without* an
  // intervening free the driver has handed out overlapping ranges, which we
  // cannot fix from here -- the newer record is the better answer either way.
  ByBase[A.Base] = A;
  ByHandle[A.Handle] = A.Base;
  ++RecordedTotal;
}

bool AllocationMap::forget(uint64_t Handle) {
  auto H = ByHandle.find(Handle);
  if (H == ByHandle.end())
    return false;

  const uint64_t Base = H->second;
  ByHandle.erase(H);

  // Only drop the base entry if it still belongs to this handle. A base whose
  // allocation was freed and then reallocated is now owned by a newer handle, and
  // erasing it here would discard a live record because of a stale free.
  auto B = ByBase.find(Base);
  if (B != ByBase.end() && B->second.Handle == Handle) {
    ByBase.erase(B);
    return true;
  }
  return false;
}

std::optional<Allocation> AllocationMap::find(uint64_t Addr) const {
  // Nearest record at or below Addr, then test containment -- the same search
  // hsakmt performs in vm_find_object_by_address_range (fmm.c:574), and the same
  // idiom ROCr uses over its own map (runtime.cpp:425).
  //
  // upper_bound gives the first base strictly greater than Addr, so the candidate
  // is the element before it. Guarding against begin() is what makes an address
  // below every known allocation return nothing instead of stepping off the
  // front.
  auto It = ByBase.upper_bound(Addr);
  if (It == ByBase.begin())
    return std::nullopt;
  --It;

  if (It->second.contains(Addr))
    return It->second;
  return std::nullopt;
}

void AllocationMap::clear() {
  ByBase.clear();
  ByHandle.clear();
  RecordedTotal = 0;
}

} // namespace detail

namespace detail {

unsigned orderAllocationChain(const AllocationCallbackEntry *Entries,
                              unsigned Count, unsigned *Out) {
  unsigned N = 0;
  for (unsigned I = 0; I < Count; I++)
    if (Entries[I].CB != nullptr)
      Out[N++] = I;

  // Insertion sort: N is at most MaxAllocationCallbacks, and a stable, obviously
  // correct comparison matters more here than the sort's shape, since this ordering
  // is the whole justification for the mechanism.
  for (unsigned I = 1; I < N; I++) {
    const unsigned Idx = Out[I];
    unsigned J = I;
    while (J > 0) {
      const AllocationCallbackEntry &Prev = Entries[Out[J - 1]];
      const AllocationCallbackEntry &Cur = Entries[Idx];
      // Higher priority first; on a tie, the later registration first.
      const bool CurGoesEarlier = Cur.Priority > Prev.Priority ||
                                  (Cur.Priority == Prev.Priority &&
                                   Cur.Seq > Prev.Seq);
      if (!CurGoesEarlier)
        break;
      Out[J] = Out[J - 1];
      J--;
    }
    Out[J] = Idx;
  }
  return N;
}

} // namespace detail

namespace {

constexpr int FreeHandleBias = static_cast<int>(MaxAllocationCallbacks);

} // namespace

/// Add to a chain under the caller's lock.
///
/// Handles encode which chain they belong to, so removing a free observer with
/// the allocation remover cannot silently unhook the wrong entry: allocation
/// handles are non-negative and free handles are offset past the array.
int AllocationTracker::addToChain(detail::AllocationCallbackEntry *Chain,
                                  void *CB, void *UserData, int Priority,
                                  int HandleBias) {
  if (CB == nullptr)
    return InvalidAllocationCallbackHandle;
  for (unsigned I = 0; I < MaxAllocationCallbacks; I++) {
    if (Chain[I].CB != nullptr)
      continue;
    Chain[I].CB = CB;
    Chain[I].UserData = UserData;
    Chain[I].Priority = Priority;
    Chain[I].Seq = NextSeq++;
    return static_cast<int>(I) + HandleBias;
  }
  return InvalidAllocationCallbackHandle;
}


//===----------------------------------------------------------------------===//
// Installation
//===----------------------------------------------------------------------===//

AllocationTracker::AllocationTracker(llvm::Error &Err) {
  llvm::ErrorAsOutParameter EAO(Err);
  Active = this;
  const int R = audit_hooks::register_wrap<&AllocationTracker::ioctlHook,
                                           &AllocationTracker::RealIoctl>(
      IoctlToolName, "ioctl");
  if (R != 0) {
    Active = nullptr;
    Err = LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "ah_register_hook(\"{0}\", \"ioctl\") returned {1}, so no GPU "
        "allocation will be observed and every address lookup will miss. Check "
        "that the application was started under LD_AUDIT=libLuthierAudit.so "
        "with this tool named in AH_PLUGINS.",
        IoctlToolName, R));
    return;
  }
  Err = llvm::Error::success();
}

AllocationTracker::~AllocationTracker() {
  // The hook stays installed -- audit_hook has no unregister call -- so clearing
  // the slot is what makes a later call forward untouched rather than reach
  // freed state.
  if (Active == this)
    Active = nullptr;
  reset();
}

int AllocationTracker::ioctlHook(int Fd, unsigned long Request, void *Arg) {
  AllocationTracker *T = Active;
  if (T == nullptr)
    return RealIoctl(Fd, Request, Arg);
  return T->handleIoctl(Fd, Request, Arg);
}

//===----------------------------------------------------------------------===//
// Decoding the boundary
//
// Request decoding is hand-rolled because <sys/ioctl.h> cannot be included
// alongside the driver's own request definitions.
//===----------------------------------------------------------------------===//

namespace {

constexpr unsigned ioctlNr(unsigned long Request) {
  return static_cast<unsigned>(Request & 0xFFu);
}
constexpr unsigned AcquireVmNr = AMDKFD_IOC_ACQUIRE_VM & 0xFFu;
constexpr unsigned AllocMemoryNr = AMDKFD_IOC_ALLOC_MEMORY_OF_GPU & 0xFFu;
constexpr unsigned FreeMemoryNr = AMDKFD_IOC_FREE_MEMORY_OF_GPU & 0xFFu;

} // namespace

int AllocationTracker::handleIoctl(int Fd, unsigned long Request, void *Arg) {
  const unsigned Nr = ioctlNr(Request);

  // No /dev/kfd descriptor check here, unlike the packet monitor's hook. These
  // three request numbers are read from the KFD ioctl space, and Arg is only
  // dereferenced after the number matches -- so a same-numbered request on some
  // other device would be misread. That is the one thing this shares with the
  // monitor's Assumption A2, and the reason the monitor keeps its fstat: it acts
  // on the arguments, whereas this only records them.
  if (Arg == nullptr)
    return RealIoctl(Fd, Request, Arg);

  // Remember which DRM descriptor each GPU's memory is bound to. Observed rather
  // than opened, because an mmap offset only resolves on the descriptor that
  // created the allocation -- see recordGpuDrmFd.
  if (Nr == AcquireVmNr) {
    const auto *V = static_cast<const struct kfd_ioctl_acquire_vm_args *>(Arg);
    const int Ret = RealIoctl(Fd, Request, Arg);
    if (Ret == 0)
      recordGpuDrmFd(V->gpu_id, static_cast<int>(V->drm_fd));
    return Ret;
  }

  if (Nr == AllocMemoryNr)
    return handleAllocMemory(Fd, Request, Arg);
  if (Nr == FreeMemoryNr)
    return handleFreeMemory(Fd, Request, Arg);

  return RealIoctl(Fd, Request, Arg);
}

int AllocationTracker::handleAllocMemory(int Fd, unsigned long Request,
                                         void *Arg) {
  const int Ret = RealIoctl(Fd, Request, Arg);
  if (Ret != 0)
    return Ret;

  const auto *A =
      static_cast<const struct kfd_ioctl_alloc_memory_of_gpu_args *>(Arg);
  const Allocation Recorded{A->va_addr, A->size,   A->flags,
                            A->gpu_id,  A->handle, A->mmap_offset};
  recordAllocation(Recorded);
  // After recording, so an observer that immediately looks the address up finds
  // it.
  runAllocationCallbacks(Recorded);
  return Ret;
}

int AllocationTracker::handleFreeMemory(int Fd, unsigned long Request,
                                        void *Arg) {
  const auto *F =
      static_cast<const struct kfd_ioctl_free_memory_of_gpu_args *>(Arg);
  const uint64_t Handle = F->handle;

  const int Ret = RealIoctl(Fd, Request, Arg);
  if (Ret != 0)
    return Ret;

  forgetAllocation(Handle);
  // Notified whether or not we had a record: an observer may be tracking
  // allocations we never saw, for instance because it attached earlier than we
  // did.
  runAllocationFreeCallbacks(Handle);
  return Ret;
}


void AllocationTracker::recordAllocation(const Allocation &A) {
  std::unique_lock Lock(Mutex);
  Map.record(A);
}

bool AllocationTracker::forgetAllocation(uint64_t Handle) {
  std::unique_lock Lock(Mutex);
  return Map.forget(Handle);
}

std::optional<Allocation> AllocationTracker::findAllocation(uint64_t Addr) const {
  std::shared_lock Lock(Mutex);
  return Map.find(Addr);
}

uint64_t AllocationTracker::liveAllocationCount() const {
  std::shared_lock Lock(Mutex);
  return Map.liveCount();
}

uint64_t AllocationTracker::recordedAllocationTotal() const {
  std::shared_lock Lock(Mutex);
  return Map.recordedTotal();
}

AllocationCallbackHandle AllocationTracker::addAllocationCallback(AllocationCallback CB,
                                               void *UserData, int Priority) {
  std::unique_lock Lock(Mutex);
  return addToChain(AllocCallbacks, reinterpret_cast<void *>(CB), UserData,
                    Priority, 0);
}

AllocationCallbackHandle AllocationTracker::addAllocationFreeCallback(AllocationFreeCallback CB,
                                                   void *UserData,
                                                   int Priority) {
  std::unique_lock Lock(Mutex);
  return addToChain(FreeCallbacks, reinterpret_cast<void *>(CB), UserData,
                    Priority, FreeHandleBias);
}

UnhookedCallback AllocationTracker::removeAllocationCallback(AllocationCallbackHandle H) {
  if (H < 0 || H >= FreeHandleBias)
    return {};
  std::unique_lock Lock(Mutex);
  const UnhookedCallback Was{AllocCallbacks[H].CB, AllocCallbacks[H].UserData};
  AllocCallbacks[H].CB = nullptr;
  return Was;
}

UnhookedCallback AllocationTracker::removeAllocationFreeCallback(AllocationCallbackHandle H) {
  const int Idx = H - FreeHandleBias;
  if (Idx < 0 || Idx >= static_cast<int>(MaxAllocationCallbacks))
    return {};
  std::unique_lock Lock(Mutex);
  const UnhookedCallback Was{FreeCallbacks[Idx].CB,
                             FreeCallbacks[Idx].UserData};
  FreeCallbacks[Idx].CB = nullptr;
  return Was;
}

/// Copy the chain out, then call outside the lock.
///
/// An observer is arbitrary code: it may allocate GPU memory, and so re-enter this
/// module. Holding the lock across the call would deadlock on the first observer
/// that does -- and \c std::shared_mutex is not recursive.
void AllocationTracker::runAllocationCallbacks(const Allocation &A) {
  detail::AllocationCallbackEntry Snapshot[MaxAllocationCallbacks];
  unsigned Order[MaxAllocationCallbacks];
  unsigned N;
  {
    std::shared_lock Lock(Mutex);
    for (unsigned I = 0; I < MaxAllocationCallbacks; I++)
      Snapshot[I] = AllocCallbacks[I];
    N = detail::orderAllocationChain(Snapshot, MaxAllocationCallbacks, Order);
  }
  for (unsigned I = 0; I < N; I++) {
    const auto &E = Snapshot[Order[I]];
    reinterpret_cast<AllocationCallback>(E.CB)(A, E.UserData);
  }
}

void AllocationTracker::runAllocationFreeCallbacks(uint64_t Handle) {
  detail::AllocationCallbackEntry Snapshot[MaxAllocationCallbacks];
  unsigned Order[MaxAllocationCallbacks];
  unsigned N;
  {
    std::shared_lock Lock(Mutex);
    for (unsigned I = 0; I < MaxAllocationCallbacks; I++)
      Snapshot[I] = FreeCallbacks[I];
    N = detail::orderAllocationChain(Snapshot, MaxAllocationCallbacks, Order);
  }
  for (unsigned I = 0; I < N; I++) {
    const auto &E = Snapshot[Order[I]];
    reinterpret_cast<AllocationFreeCallback>(E.CB)(Handle, E.UserData);
  }
}

void AllocationTracker::recordGpuDrmFd(uint32_t GpuId, int DrmFd) {
  if (DrmFd < 0)
    return;
  std::unique_lock Lock(Mutex);
  auto &Fds = DrmFds;
  if (Fds.count(GpuId) != 0)
    return; // first one wins; re-acquiring the same VM changes nothing for us

  // dup rather than storing the application's number: it may close its copy, and
  // dup keeps the same open file description alive -- which is what carries the
  // namespace an mmap_offset resolves in. Reopening the node would not.
  const int Copy = dup(DrmFd);
  if (Copy >= 0)
    Fds[GpuId] = Copy;
}

int AllocationTracker::gpuDrmFd(uint32_t GpuId) const {
  std::shared_lock Lock(Mutex);
  auto It = DrmFds.find(GpuId);
  return It == DrmFds.end() ? -1 : It->second;
}

void AllocationTracker::reset() {
  std::unique_lock Lock(Mutex);
  Map.clear();
  for (auto &Entry : DrmFds)
    if (Entry.second >= 0)
      close(Entry.second);
  DrmFds.clear();
  for (unsigned I = 0; I < MaxAllocationCallbacks; I++) {
    AllocCallbacks[I] = detail::AllocationCallbackEntry{};
    FreeCallbacks[I] = detail::AllocationCallbackEntry{};
  }
  NextSeq = 1;
}


} // namespace luthier::kfd
