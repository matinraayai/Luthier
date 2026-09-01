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

#include <cstdint>
#include <unistd.h>

#include <map>

#include <mutex>
#include <shared_mutex>

namespace {
/// Clears the C-linkage binding side-tables. Declared here and defined next to
/// those tables, which sit below the function that needs it.
void clearCBindings();
} // namespace

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

//===----------------------------------------------------------------------===//
// The process-wide instance
//===----------------------------------------------------------------------===//

namespace {

/// Function-local statics, so construction happens on first use rather than in
/// static-init order. This library is preloaded into arbitrary processes and its
/// first ioctl can arrive before any of our own initialisation would otherwise
/// have run.
detail::AllocationMap &theMap() {
  static detail::AllocationMap M;
  return M;
}

/// Lookups are far more frequent than mutations, and ioctls genuinely arrive on
/// several threads, so a shared mutex rather than a plain one.
///
/// \c std::shared_mutex rather than \c llvm::sys::RWMutex, which is Luthier's
/// convention elsewhere: this module is deliberately LLVM-free so it can be
/// preloaded into any process, and its build target does not carry LLVM's include
/// directories.
std::shared_mutex &theMutex() {
  static std::shared_mutex M;
  return M;
}

} // namespace

void recordAllocation(const Allocation &A) {
  std::unique_lock Lock(theMutex());
  theMap().record(A);
}

bool forgetAllocation(uint64_t Handle) {
  std::unique_lock Lock(theMutex());
  return theMap().forget(Handle);
}

std::optional<Allocation> findAllocation(uint64_t Addr) {
  std::shared_lock Lock(theMutex());
  return theMap().find(Addr);
}

uint64_t liveAllocationCount() {
  std::shared_lock Lock(theMutex());
  return theMap().liveCount();
}

uint64_t recordedAllocationTotal() {
  std::shared_lock Lock(theMutex());
  return theMap().recordedTotal();
}

namespace {

/// gpu_id -> duplicated DRM render-node descriptor. A plain map because a process
/// has a handful of GPUs, and it is written once per GPU during initialisation.
std::map<uint32_t, int> &theDrmFds() {
  static std::map<uint32_t, int> M;
  return M;
}

} // namespace

//===----------------------------------------------------------------------===//
// The observer chains
//===----------------------------------------------------------------------===//

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

detail::AllocationCallbackEntry AllocCallbacks[MaxAllocationCallbacks];
detail::AllocationCallbackEntry FreeCallbacks[MaxAllocationCallbacks];
unsigned long long NextSeq = 1;

/// Add to a chain under the caller's lock.
///
/// Handles encode which chain they belong to, so removing a free observer with the
/// allocation remover cannot silently unhook the wrong entry: allocation handles
/// are non-negative and free handles are offset past the array.
int addToChain(detail::AllocationCallbackEntry *Chain, void *CB, void *UserData,
               int Priority, int HandleBias) {
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

constexpr int FreeHandleBias = static_cast<int>(MaxAllocationCallbacks);

} // namespace

AllocationCallbackHandle addAllocationCallback(AllocationCallback CB,
                                               void *UserData, int Priority) {
  std::unique_lock Lock(theMutex());
  return addToChain(AllocCallbacks, reinterpret_cast<void *>(CB), UserData,
                    Priority, 0);
}

AllocationCallbackHandle addAllocationFreeCallback(AllocationFreeCallback CB,
                                                   void *UserData,
                                                   int Priority) {
  std::unique_lock Lock(theMutex());
  return addToChain(FreeCallbacks, reinterpret_cast<void *>(CB), UserData,
                    Priority, FreeHandleBias);
}

/// \return what was unhooked, so a C-linkage trampoline can release the binding
/// slot it owns -- and \e only that one. See luthierKfdRemoveAllocationCallback.
UnhookedCallback removeAllocationCallback(AllocationCallbackHandle H) {
  if (H < 0 || H >= FreeHandleBias)
    return {};
  std::unique_lock Lock(theMutex());
  const UnhookedCallback Was{AllocCallbacks[H].CB, AllocCallbacks[H].UserData};
  AllocCallbacks[H].CB = nullptr;
  return Was;
}

UnhookedCallback removeAllocationFreeCallback(AllocationCallbackHandle H) {
  const int Idx = H - FreeHandleBias;
  if (Idx < 0 || Idx >= static_cast<int>(MaxAllocationCallbacks))
    return {};
  std::unique_lock Lock(theMutex());
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
void runAllocationCallbacks(const Allocation &A) {
  detail::AllocationCallbackEntry Snapshot[MaxAllocationCallbacks];
  unsigned Order[MaxAllocationCallbacks];
  unsigned N;
  {
    std::shared_lock Lock(theMutex());
    for (unsigned I = 0; I < MaxAllocationCallbacks; I++)
      Snapshot[I] = AllocCallbacks[I];
    N = detail::orderAllocationChain(Snapshot, MaxAllocationCallbacks, Order);
  }
  for (unsigned I = 0; I < N; I++) {
    const auto &E = Snapshot[Order[I]];
    reinterpret_cast<AllocationCallback>(E.CB)(A, E.UserData);
  }
}

void runAllocationFreeCallbacks(uint64_t Handle) {
  detail::AllocationCallbackEntry Snapshot[MaxAllocationCallbacks];
  unsigned Order[MaxAllocationCallbacks];
  unsigned N;
  {
    std::shared_lock Lock(theMutex());
    for (unsigned I = 0; I < MaxAllocationCallbacks; I++)
      Snapshot[I] = FreeCallbacks[I];
    N = detail::orderAllocationChain(Snapshot, MaxAllocationCallbacks, Order);
  }
  for (unsigned I = 0; I < N; I++) {
    const auto &E = Snapshot[Order[I]];
    reinterpret_cast<AllocationFreeCallback>(E.CB)(Handle, E.UserData);
  }
}

void recordGpuDrmFd(uint32_t GpuId, int DrmFd) {
  if (DrmFd < 0)
    return;
  std::unique_lock Lock(theMutex());
  auto &Fds = theDrmFds();
  if (Fds.count(GpuId) != 0)
    return; // first one wins; re-acquiring the same VM changes nothing for us

  // dup rather than storing the application's number: it may close its copy, and
  // dup keeps the same open file description alive -- which is what carries the
  // namespace an mmap_offset resolves in. Reopening the node would not.
  const int Copy = dup(DrmFd);
  if (Copy >= 0)
    Fds[GpuId] = Copy;
}

int gpuDrmFd(uint32_t GpuId) {
  std::shared_lock Lock(theMutex());
  auto It = theDrmFds().find(GpuId);
  return It == theDrmFds().end() ? -1 : It->second;
}

void resetAllocationTracker() {
  std::unique_lock Lock(theMutex());
  theMap().clear();
  for (auto &Entry : theDrmFds())
    if (Entry.second >= 0)
      close(Entry.second);
  theDrmFds().clear();
  for (unsigned I = 0; I < MaxAllocationCallbacks; I++) {
    AllocCallbacks[I] = detail::AllocationCallbackEntry{};
    FreeCallbacks[I] = detail::AllocationCallbackEntry{};
  }
  NextSeq = 1;
  // The chains and the C-linkage binding arrays are two separate pools, and an
  // earlier version cleared only the first. A test using the C API twice then
  // exhausted the bindings while the chain still reported free slots.
  clearCBindings();
}

} // namespace luthier::kfd

extern "C" int luthierKfdFindAllocation(unsigned long long Addr,
                                        unsigned long long *Base,
                                        unsigned long long *Size,
                                        unsigned *Flags, unsigned *GpuId,
                                        unsigned long long *MmapOffset) {
  auto A = luthier::kfd::findAllocation(Addr);
  if (!A)
    return 0;
  if (Base != nullptr)
    *Base = A->Base;
  if (Size != nullptr)
    *Size = A->Size;
  if (Flags != nullptr)
    *Flags = A->Flags;
  if (GpuId != nullptr)
    *GpuId = A->GpuId;
  if (MmapOffset != nullptr)
    *MmapOffset = A->MmapOffset;
  return 1;
}

namespace {

/// Trampolines for the C-linkage registrations. The scalar signature keeps a
/// component from depending on struct Allocation's layout.
using CAllocFn = void (*)(unsigned long long, unsigned long long, unsigned,
                          unsigned, unsigned long long, unsigned long long,
                          void *);

struct CAllocBinding {
  CAllocFn CB;
  void *UserData;
};

/// One binding per slot, since the C callback needs somewhere to keep both its own
/// function and its user pointer while the chain stores only one void*.
CAllocBinding CAllocBindings[luthier::kfd::MaxAllocationCallbacks];

void cAllocTrampoline(const luthier::kfd::Allocation &A, void *Slot) {
  const auto *B = static_cast<const CAllocBinding *>(Slot);
  B->CB(A.Base, A.Size, A.Flags, A.GpuId, A.Handle, A.MmapOffset, B->UserData);
}

using CFreeFn = void (*)(unsigned long long, void *);

struct CFreeBinding {
  CFreeFn CB;
  void *UserData;
};

CFreeBinding CFreeBindings[luthier::kfd::MaxAllocationCallbacks];

void cFreeTrampoline(uint64_t Handle, void *Slot) {
  const auto *B = static_cast<const CFreeBinding *>(Slot);
  B->CB(Handle, B->UserData);
}

void clearCBindings() {
  for (unsigned I = 0; I < luthier::kfd::MaxAllocationCallbacks; I++) {
    CAllocBindings[I] = CAllocBinding{};
    CFreeBindings[I] = CFreeBinding{};
  }
}

} // namespace

extern "C" int luthierKfdAddAllocationCallback(CAllocFn CB, void *UserData,
                                               int Priority) {
  if (CB == nullptr)
    return luthier::kfd::InvalidAllocationCallbackHandle;
  // Find a free binding slot first: registering succeeds or it does not, and a
  // half-registered observer would be worse than a refusal.
  for (unsigned I = 0; I < luthier::kfd::MaxAllocationCallbacks; I++) {
    if (CAllocBindings[I].CB != nullptr)
      continue;
    CAllocBindings[I] = CAllocBinding{CB, UserData};
    const int H = luthier::kfd::addAllocationCallback(
        cAllocTrampoline, &CAllocBindings[I], Priority);
    if (H == luthier::kfd::InvalidAllocationCallbackHandle)
      CAllocBindings[I] = CAllocBinding{};
    return H;
  }
  return luthier::kfd::InvalidAllocationCallbackHandle;
}

extern "C" int luthierKfdAddAllocationFreeCallback(CFreeFn CB, void *UserData,
                                                   int Priority) {
  if (CB == nullptr)
    return luthier::kfd::InvalidAllocationCallbackHandle;
  for (unsigned I = 0; I < luthier::kfd::MaxAllocationCallbacks; I++) {
    if (CFreeBindings[I].CB != nullptr)
      continue;
    CFreeBindings[I] = CFreeBinding{CB, UserData};
    const int H = luthier::kfd::addAllocationFreeCallback(
        cFreeTrampoline, &CFreeBindings[I], Priority);
    if (H == luthier::kfd::InvalidAllocationCallbackHandle)
      CFreeBindings[I] = CFreeBinding{};
    return H;
  }
  return luthier::kfd::InvalidAllocationCallbackHandle;
}

extern "C" void luthierKfdRemoveAllocationCallback(int Handle) {
  const luthier::kfd::UnhookedCallback Was =
      luthier::kfd::removeAllocationCallback(Handle);
  // Release the binding slot -- but only if what came off the chain was this
  // trampoline. A C++ registration's UserData belongs to its own caller, and
  // treating it as a binding pointer would corrupt unrelated state.
  if (Was.CB == reinterpret_cast<void *>(&cAllocTrampoline) &&
      Was.UserData != nullptr)
    *static_cast<CAllocBinding *>(Was.UserData) = CAllocBinding{};
}

extern "C" void luthierKfdRemoveAllocationFreeCallback(int Handle) {
  const luthier::kfd::UnhookedCallback Was =
      luthier::kfd::removeAllocationFreeCallback(Handle);
  if (Was.CB == reinterpret_cast<void *>(&cFreeTrampoline) &&
      Was.UserData != nullptr)
    *static_cast<CFreeBinding *>(Was.UserData) = CFreeBinding{};
}

extern "C" int luthierKfdGpuDrmFd(unsigned GpuId) {
  return luthier::kfd::gpuDrmFd(static_cast<uint32_t>(GpuId));
}

