//===-- AllocationTracker.h - KFD-level GPU allocation tracking -*- C++ -*-===//
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
/// Records GPU memory allocations made directly through KFD \c ioctl calls, so
/// an address can be resolved back to the allocation that contains it.
///
/// \par Why this exists
/// A dispatch packet identifies its kernel by a bare address (\c kernel_object).
/// To instrument that kernel Luthier must read its instructions, which first
/// requires knowing which allocation the address falls in, how large it is and
/// which GPU owns it. On the HSA path \c hsa_amd_pointer_info answers that (see
/// \c HsaMemoryAllocationAccessor). Below HSA nothing does, and the driver offers
/// no "what is allocated?" query -- which is why both userspace layers keep their
/// own books, and why we must too.
///
/// \par Reference implementation
/// \c libhsakmt/src/fmm.c, which tracks the same ioctls one layer up. Its
/// \c vm_object (\c fmm.c:91) is the model for \c Allocation, and
/// \c vm_find_object_by_address_range (\c fmm.c:574) is the model for
/// \c findAllocation: locate the nearest record at or below the address, then
/// test containment.
///
/// \par What the ioctl view looks like, and why it is simpler than hsakmt's
/// hsakmt can split one request into several \c ALLOC_MEMORY_OF_GPU calls,
/// gathering the handles into a \c handles[] array (\c fmm.c:1195-1210), and then
/// frees by walking that array one \c FREE per handle (\c fmm.c:1220-1225). So a
/// single \c vm_object above can correspond to several records here.
///
/// In practice this is rare rather than routine, which an earlier version of this
/// comment got wrong: the split threshold is
/// \c BIGGEST_SINGLE_BUF_SIZE == (1ULL << 39) - GPU_HUGE_PAGE_SIZE (\c fmm.c:347),
/// roughly 512 GB, and the loop only runs for \c KFD_IOC_ALLOC_MEM_FLAGS_USERPTR
/// (\c fmm.c:1166-1173). Each chunk does get its own base (\c args.va_addr += size,
/// \c fmm.c:1199), so the handle-to-base index stays one-to-one and a containment
/// lookup finds the right chunk without our reassembling anything.
///
/// \par Managed memory is covered, contrary to what its addresses suggest
/// Measured on gfx908 with SCALE. \c cudaMallocManaged returns addresses from a
/// different aperture than \c cudaMalloc (\c 0x5090… versus \c 0x520d…) and the
/// SVM ioctl (0x20) does fire, which together look like a separate allocation
/// path. They are not: managed memory still arrives through
/// \c ALLOC_MEMORY_OF_GPU and is tracked here. Only the flags differ ---
/// \c GTT|WRITABLE|PUBLIC|AQL_QUEUE_MEM|UNCACHED for managed against
/// \c VRAM|WRITABLE|PUBLIC for device memory.
///
/// Recorded because the address ranges are misleading: reasoning from them alone
/// gives the wrong answer, and only the ioctl stream settles it.
///
/// \par Not covered
/// \li SVM attribute calls (\c AMDKFD_IOC_SVM, 0x20). What they do on top of an
///     ordinary allocation is not yet established, so a range whose properties
///     were changed that way will still be described here by its original
///     allocation flags.
/// \li DMABUF import (0x1D), which brings in memory allocated by another process
///     entirely -- no allocation ioctl of ours ever sees it.
/// \li \c UNMAP_MEMORY_FROM_GPU (0x19). Unmapping changes reachability, not
///     existence, and v1 answers "which allocation is this" rather than "can this
///     GPU reach it".
/// \li The \c userptr second index hsakmt keeps (\c fmm.c:101), needed only to
///     look an allocation up by the host pointer it was registered with.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_ALLOCATION_TRACKER_H
#define LUTHIER_KFD_ALLOCATION_TRACKER_H

#include <cstdint>
#include <map>
#include <optional>

namespace luthier::kfd {

/// \brief One GPU allocation, as seen at the driver boundary.
///
/// A deliberate subset of hsakmt's \c vm_object (\c fmm.c:91): the fields that
/// come straight out of \c kfd_ioctl_alloc_memory_of_gpu_args, and nothing that
/// would have to be derived or maintained.
struct Allocation {
  /// Start of the allocation in the process's virtual address space
  /// (\c va_addr).
  uint64_t Base;
  /// Size in bytes, exactly as passed to the driver  (assumption A7).
  ///
  /// The driver backs whole pages, so an address just past this size may still sit
  /// in memory the driver allocated -- and it will \b not resolve here. That is
  /// deliberate. We considered rounding up to a page, on the grounds that hsakmt's
  /// \c vm_object.size is described as "size allocated on GPU. When the user
  /// requests a random size, Thunk aligns it to page size", but that comment
  /// describes what hsakmt's \e callers hand it, not something hsakmt does at that
  /// point -- \c vm_align_area_size (\c fmm.c:620) only adds guard pages. So there
  /// is no reference behaviour to match, and reporting a range wider than the one
  /// requested would be our inference rather than the driver's answer.
  ///
  /// In practice this costs nothing for the case this exists to serve: a
  /// \c kernel_object lands well inside its allocation, not in the page slack.
  uint64_t Size;
  /// Raw \c KFD_IOC_ALLOC_MEM_FLAGS_* bits -- the driver's flags, not hsakmt's
  /// \c HsaMemFlags, because this layer observes the driver interface.
  ///
  /// Worth keeping even though v1 does not interpret them: whether the memory is
  /// host-readable is decided by these bits, and that determines whether reading
  /// a kernel's instructions is a plain load or needs a device-to-host copy.
  uint32_t Flags;
  /// KFD's identifier for the owning GPU (\c gpu_id).
  uint32_t GpuId;
  /// The handle the driver returned. Retained because \c FREE_MEMORY_OF_GPU
  /// carries \b only a handle, with no address.
  uint64_t Handle;

  /// The offset the driver returned in \c mmap_offset, which is how a host
  /// mapping of this allocation is obtained.
  ///
  /// hsakmt \c mmap()s exactly this offset on the GPU's \b DRM \b render \b node
  /// -- not \c /dev/kfd -- to place a CPU mapping over the allocation's device
  /// address, with \c PROT_READ|PROT_WRITE when
  /// \c KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC is set and \c PROT_NONE when it is not
  /// (\c fmm.c:1569-1573, called from \c fmm.c:1799-1808). Recorded because it is
  /// the only handle on host-readability we get from below, and it is free to
  /// capture at allocation time but unrecoverable afterwards.
  ///
  /// \note Measured on gfx908 (MI100, BAR0 = 32 GB against 32 GB of VRAM, i.e.
  /// large BAR): mapping this offset on the render node with \c PROT_READ succeeds
  /// and reads correctly even for an allocation \b without \c PUBLIC, and two such
  /// mappings alias each other. So hsakmt's \c PROT_NONE is a policy that follows
  /// the flag, not a limit the hardware imposes. Do not generalise it: on a
  /// small-BAR GPU only part of VRAM has a CPU aperture, so a caller must treat a
  /// failed mapping as an ordinary outcome rather than an error in its own logic.
  uint64_t MmapOffset;

  /// \return whether \p Addr falls inside this allocation.
  ///
  /// Written as a subtraction rather than as \c Addr < Base + Size because the
  /// latter overflows for a size close to \c UINT64_MAX and then wraps, turning a
  /// huge range into one that contains nothing. \c Size and \c Base both come
  /// straight out of ioctl arguments, so neither is ours to trust.
  [[nodiscard]] bool contains(uint64_t Addr) const {
    return Size > 0 && Addr >= Base && (Addr - Base) < Size;
  }
};

/// \brief Record a successful allocation.
///
/// Call only after the underlying \c ioctl has succeeded: before that there is no
/// handle, and on failure there is no allocation.
///
/// An allocation whose \c Base is 0 is \b ignored, and that is not defensive
/// programming against a case that cannot happen -- it is a case hsakmt creates
/// deliberately. Allocating from its \c mem_handle_aperture passes
/// \c va_addr == 0, commented "if allocate vram-only, use an invalid VA"
/// (\c fmm.c:1161-1162). Such an allocation has no virtual address, so it can
/// never contain a queried address; recording it would put a range starting at 0
/// into the map, after which \e every address below its size would resolve to it.
/// That failure mode is worse than a miss, because a wrong hit is
/// indistinguishable from a right one.
void recordAllocation(const Allocation &A);

/// \brief Forget the allocation with this handle.
///
/// Keyed on the handle because that is all \c kfd_ioctl_free_memory_of_gpu_args
/// contains. Note hsakmt never needs this direction -- its own API frees by
/// address (\c fmm.c:1220) -- so the reverse index is a consequence of sitting
/// below it rather than beside it.
///
/// \return whether a record was actually removed
bool forgetAllocation(uint64_t Handle);

/// \brief Find the allocation containing \p Addr.
///
/// \return the containing allocation, or \c std::nullopt if no tracked
/// allocation covers \p Addr -- which happens legitimately for SVM and imported
/// memory, so callers must handle it rather than treat it as an error.
std::optional<Allocation> findAllocation(uint64_t Addr);

/// \brief How many allocations are currently tracked.
uint64_t liveAllocationCount();

/// \brief How many allocations have ever been recorded. Cumulative.
///
/// Paired with \c liveAllocationCount for the same reason \c wrappedQueueCount is
/// paired with \c excludedQueueCount: a test that sees an empty map cannot
/// otherwise tell "recorded and correctly freed" from "never recorded at all".
uint64_t recordedAllocationTotal();

//===----------------------------------------------------------------------===//
// Several components watching the same allocations
//===----------------------------------------------------------------------===//

/// \brief Identifies a registered observer, for removing it again.
using AllocationCallbackHandle = int;

/// Returned when there is no room left in the chain.
static constexpr AllocationCallbackHandle InvalidAllocationCallbackHandle = -1;

/// \brief Called after an allocation has been recorded.
using AllocationCallback = void (*)(const Allocation &A, void *UserData);

/// \brief Called after an allocation has been released, with the only thing
/// \c FREE_MEMORY_OF_GPU carries.
using AllocationFreeCallback = void (*)(uint64_t Handle, void *UserData);

/// \brief Most observers of each kind that can be registered at once.
static constexpr unsigned MaxAllocationCallbacks = 8;

/// \brief Watch every allocation the application makes.
///
/// \par Why components register here instead of each intercepting ioctl
/// Several tools can wrap \c ioctl independently -- that is what GOTCHA is for, and
/// it works. But only \e one of them can substitute a queue's ring buffer, because
/// \c handleCreateQueue overwrites \c ring_base_address and the innermost
/// substitution is the one the driver sees; an outer wrapper's poller would then
/// read a ring the GPU never touches and report clean counts while observing
/// nothing. So this boundary has a single owner by necessity. Allocations are
/// decoded from the same ioctl stream at the same point, and giving them a
/// different registration model would be arbitrary.
///
/// It also means a consumer does not re-derive what
/// \c kfd_ioctl_alloc_memory_of_gpu_args means, that a record is only valid after
/// the ioctl succeeded, or that a free carries a handle and no address.
///
/// \par Order
/// **Higher priority runs first**, and ties are broken last-registered-first.
/// "Higher first" is chosen to match GOTCHA's own rule -- its documentation says
/// lower values are called innermost -- so a component's mental model is the same
/// whether it registers here or wraps \c ioctl itself.
///
/// \warning Do \b not unify this with \c addPacketCallback, whose order is
/// last-registered-first with no priority at all. That is a deliberate match to
/// ROCr's \c intercept_queue.cpp:375 and is asserted on hardware by the
/// \c S14c-two-callbacks scenario. If priority is ever wanted there it has to
/// arrive as an optional parameter defaulting to the present behaviour.
///
/// \param Priority higher runs earlier; 0 is a reasonable default
/// \return a handle, or \c InvalidAllocationCallbackHandle if the chain is full
AllocationCallbackHandle addAllocationCallback(AllocationCallback CB,
                                               void *UserData, int Priority);

/// \brief Watch every release. Ordering rules as for \c addAllocationCallback.
AllocationCallbackHandle
addAllocationFreeCallback(AllocationFreeCallback CB, void *UserData,
                          int Priority);

/// \brief What a removal took off a chain.
///
/// Returned so a C-linkage wrapper can release the side-table slot it owns while
/// leaving a C++ registration's \c UserData alone -- the two are not
/// distinguishable from a handle.
struct UnhookedCallback {
  void *CB{nullptr};
  void *UserData{nullptr};
};

/// \brief Remove an observer, leaving the order of the rest unchanged.
///
/// \return what was unhooked, or a zeroed \c UnhookedCallback if the handle named
/// no live entry. Existing callers may ignore the result.
UnhookedCallback removeAllocationCallback(AllocationCallbackHandle H);

/// \brief Remove a free observer, leaving the order of the rest unchanged.
UnhookedCallback removeAllocationFreeCallback(AllocationCallbackHandle H);

/// \brief Notify the registered observers. Called by the ioctl handler.
void runAllocationCallbacks(const Allocation &A);

/// \brief Notify the registered free observers. Called by the ioctl handler.
void runAllocationFreeCallbacks(uint64_t Handle);

namespace detail {

/// \brief One registered observer.
///
/// \c Seq records registration order, so a tie on \c Priority can be broken
/// without depending on where in the array an entry happens to sit -- slots are
/// reused after a removal, so position says nothing about age.
struct AllocationCallbackEntry {
  void *CB{nullptr};
  void *UserData{nullptr};
  int Priority{0};
  unsigned long long Seq{0};
};

/// \brief Order a chain: higher priority first, then last registered first.
///
/// A free function over a plain array so the ordering guarantee can be tested
/// without a GPU or a driver. That matters more here than for the packet chain: the
/// whole reason for adopting GOTCHA over the existing callback array was
/// deterministic ordering, so an ordering rule that is merely asserted in a comment
/// would leave the justification unchecked.
///
/// \param Out receives indices into \p Entries, in the order they should run.
/// \return how many indices were written.
unsigned orderAllocationChain(const AllocationCallbackEntry *Entries,
                              unsigned Count, unsigned *Out);

} // namespace detail

/// \brief Remember the DRM render-node descriptor a GPU's memory was bound to.
///
/// Taken from \c AMDKFD_IOC_ACQUIRE_VM, whose arguments are exactly
/// \c {drm_fd, gpu_id} -- the application hands KFD the descriptor it opened, and
/// from then on that GPU's allocations belong to \e that DRM file.
///
/// \par Why this is not optional
/// The \c mmap_offset an allocation reports is only meaningful on the DRM file
/// that created it. Measured: mapping a valid offset through a second,
/// independently opened descriptor for the same render node fails with
/// \c EACCES, and that is exactly how a tool which opens its own descriptor
/// fails against an application's allocation. So a host mapping is impossible
/// without the application's own descriptor, and this ioctl is the only place it
/// passes by.
///
/// The descriptor is duplicated, so the record survives the application closing
/// its copy. \c dup shares the underlying open file description, which is what
/// carries the namespace the offset resolves in -- a freshly opened descriptor
/// would not do.
///
/// \par Assumption A6
/// The first \c ACQUIRE_VM per GPU wins and is kept for the life of the process.
/// If an application closed its render node and opened a new one, allocations made
/// afterwards would carry offsets valid on the \e new description while we still
/// hold the old -- and mapping them would fail with \c EACCES, the same way it
/// does for a descriptor we opened ourselves. Not observed, not tested; recorded
/// because the failure would look like a permissions problem rather than a stale
/// handle.
void recordGpuDrmFd(uint32_t GpuId, int DrmFd);

/// \brief The remembered DRM descriptor for \p GpuId, or -1 if none was seen.
///
/// -1 means \c ACQUIRE_VM was never observed for this GPU, which for a process
/// that allocated GPU memory means interception started too late.
[[nodiscard]] int gpuDrmFd(uint32_t GpuId);

/// \brief Drop all state. For tests only.
void resetAllocationTracker();

namespace detail {

/// \brief The tracker's logic, with no global state.
///
/// Separated from the free functions above so the boundary behaviour can be
/// tested without a GPU, a driver, or the process-wide instance -- the same
/// reason \c runCallbackChain is a free function over a plain array. The
/// interesting cases here are all boundaries: an address exactly at the base, at
/// the final byte, one past the end, and a base that is freed and then reused.
class AllocationMap {
public:
  void record(const Allocation &A);
  bool forget(uint64_t Handle);
  [[nodiscard]] std::optional<Allocation> find(uint64_t Addr) const;
  [[nodiscard]] uint64_t liveCount() const { return ByBase.size(); }
  [[nodiscard]] uint64_t recordedTotal() const { return RecordedTotal; }
  void clear();

private:
  /// Keyed by base address, and ordered because lookup needs the nearest record
  /// at or below an address. The standard library's red-black tree stands in for
  /// hsakmt's hand-rolled one (\c libhsakmt/src/rbtree.c), which exists to serve
  /// its allocator rather than its lookups.
  ///
  /// One flat map, not a tree per aperture as hsakmt keeps: addresses are unique
  /// within the process, and per-aperture arenas serve an allocator we are not
  /// writing.
  std::map<uint64_t, Allocation> ByBase;

  /// Handle to base, because \c FREE_MEMORY_OF_GPU identifies its target by
  /// handle alone.
  std::map<uint64_t, uint64_t> ByHandle;

  uint64_t RecordedTotal = 0;
};

} // namespace detail

} // namespace luthier::kfd

extern "C" {

/// \brief C-linkage lookup, so a test binary can resolve an address without
/// linking the tracker.
///
/// Mirrors \c luthierKfdSetPacketCallback's rationale: the hardware suite runs the
/// same binary with and without the wrapper preloaded, and a link-time dependency
/// would force it to load either way.
///
/// \return 1 and fills the out-parameters when found, 0 otherwise. Any
/// out-parameter may be null.
int luthierKfdFindAllocation(unsigned long long Addr, unsigned long long *Base,
                             unsigned long long *Size, unsigned *Flags,
                             unsigned *GpuId, unsigned long long *MmapOffset);

/// \brief C-linkage form of \c luthier::kfd::addAllocationCallback.
///
/// The allocation is passed as separate scalars rather than as a struct, so a
/// component built against a different Luthier revision cannot silently disagree
/// about the layout of \c Allocation.
int luthierKfdAddAllocationCallback(void (*CB)(unsigned long long Base,
                                               unsigned long long Size,
                                               unsigned Flags, unsigned GpuId,
                                               unsigned long long Handle,
                                               unsigned long long MmapOffset,
                                               void *UserData),
                                    void *UserData, int Priority);

/// \brief C-linkage form of \c luthier::kfd::addAllocationFreeCallback.
int luthierKfdAddAllocationFreeCallback(void (*CB)(unsigned long long Handle,
                                                   void *UserData),
                                        void *UserData, int Priority);

/// \brief C-linkage form of \c luthier::kfd::removeAllocationCallback.
void luthierKfdRemoveAllocationCallback(int Handle);

/// \brief C-linkage form of \c luthier::kfd::removeAllocationFreeCallback.
void luthierKfdRemoveAllocationFreeCallback(int Handle);

/// \brief C-linkage form of \c luthier::kfd::gpuDrmFd.
///
/// Exported for the same reason as the lookup above: a component that wants to map
/// an allocation for host access needs the application's DRM descriptor, and it
/// cannot obtain one for itself -- see \c recordGpuDrmFd.
int luthierKfdGpuDrmFd(unsigned GpuId);

}

#endif // LUTHIER_KFD_ALLOCATION_TRACKER_H
