//===-- KfdAllocationResolver.h ---------------------------------*- C++ -*-===//
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
/// Resolves addresses against allocations made directly through KFD \c ioctl
/// calls, i.e. without the HSA runtime.
///
/// \par What this is for
/// Luthier locates a kernel's code by asking a \c MemoryAllocationAccessor which
/// allocation contains an address. \c HsaMemoryAllocationAccessor answers that
/// from the HSA loader, then from \c hsa_amd_pointer_info. An application that
/// issues KFD ioctls itself never populates HSA's books, so neither finds
/// anything, and in such a process HSA cannot even be initialized (see below).
/// This resolver answers the same question from \c luthier::kfd::AllocationTracker's
/// record of the ioctl stream, and the HSA accessor consults it as its last
/// source.
///
/// \par Why the tracker is reached through \c dlsym rather than linked
/// The records live in whichever module intercepted the ioctls -- in practice
/// \c libluthier-kfd-queue-wrapper.so, preloaded into the application. The
/// tracker's storage is a function-local \c static
/// (\c AllocationTracker.cpp:116), and a \c static in a shared library is per
/// \e library, not per process. So linking the tracker into this module as well
/// would give it a \b second, permanently empty map -- which is worse than an
/// error, because "no allocations" is a legal answer that reads exactly like an
/// application which allocated nothing. The process-wide instance is therefore
/// located at run time, the same way \c HsaOracleMain finds
/// \c luthierKfdSetPacketCallback.
///
/// \par How the host-readable view is obtained
/// A caller needs a host-readable pointer, because the code lifter dereferences
/// it. Below HSA there is no \c hsa_memory_copy to fall back on, so we use
/// hsakmt's own mechanism: \c mmap the GPU's DRM render node at the offset
/// \c ALLOC_MEMORY_OF_GPU returned (\c fmm.c:1569-1573). hsakmt maps with
/// \c PROT_NONE unless the allocation carries \c KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC,
/// but that is a policy following the flag rather than a hardware limit: measured
/// on gfx908 with a 32 GB BAR, a \c PROT_READ mapping of a non-\c PUBLIC VRAM
/// allocation succeeds and reads correctly; re-confirmed on gfx942. On a
/// small-BAR GPU only part of VRAM has a CPU aperture, so a failed mapping is an
/// ordinary outcome and is reported as an \c llvm::Error naming the reason.
///
/// \par The mapping must use the application's own DRM descriptor
/// This is the part that is easy to get wrong, and it fails in a way that looks
/// like a permissions problem rather than a design error. An \c mmap_offset names
/// a GEM object in the namespace of the DRM \e file that created the allocation.
/// Measured: opening a second descriptor for the same render node and mapping a
/// perfectly valid offset through it fails with \c EACCES. So this resolver cannot
/// open its own node; it uses the descriptor the application passed to
/// \c AMDKFD_IOC_ACQUIRE_VM, which the wrapper records for exactly this purpose
/// (\c luthier::kfd::recordGpuDrmFd).
///
/// \par Why we map even though the application usually already has
/// An application may well have mapped the allocation over its own device VA --
/// tinygrad does, with a \c MAP_FIXED mmap of the render node right after every
/// non-\c USERPTR allocation (\c ops_amd.py:766-768), exactly as hsakmt does
/// (\c fmm.c:1799-1808). Relying on that would be relying on one application's
/// policy rather than on anything the driver guarantees, so we map for ourselves.
/// The two mappings alias the same GEM object, which is why both read correctly.
///
/// \par What this cannot do, by construction
/// \li It cannot report a parsed code object, and its interface cannot express
///     one. There is no loader below HSA to have parsed an ELF. Callers that need
///     one -- \c CodeDiscoveryPass does -- fall back to a synthetic
///     \c kernel-<addr> name (\c CodeDiscoveryPass.cpp:761).
/// \li The answer is the \b driver-level allocation, which is coarser than HSA's.
///     Measured: a \c kernel_object at \c 0x5202400003c0 resolves to a 2 MB
///     allocation based at \c 0x520240000000, i.e. a runtime's suballocation arena
///     containing many kernels, where HSA would have named the code object's own
///     loaded range. This is why the HSA accessor stops as soon as HSA gives a
///     non-empty answer rather than preferring this one.
/// \li The host view is a live mapping of device memory, not a snapshot
///     (assumption A5). The HSA accessor's fallback path copies; this does not.
///     Equivalent for reading kernel code, which is written once before dispatch,
///     but it is a different guarantee.
/// \li An \c Allocation has no field for the owning GPU (assumption A8). Our
///     record knows it; the interface cannot express it. Harmless with one device,
///     wrong the moment two devices hand out the same virtual address.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_KFD_KFD_ALLOCATION_RESOLVER_H
#define LUTHIER_KFD_KFD_ALLOCATION_RESOLVER_H
#include "luthier/KFD/Topology.h"
#include "luthier/ToolCodeGen/DriverAllocationResolver.h"

#include <llvm/ADT/DenseMap.h>

#include <cstdint>

namespace luthier {

namespace kfd {

/// \brief Signature of \c luthierKfdFindAllocation, resolved at run time.
using FindAllocationFn = int (*)(unsigned long long Addr,
                                 unsigned long long *Base,
                                 unsigned long long *Size, unsigned *Flags,
                                 unsigned *GpuId,
                                 unsigned long long *MmapOffset);

/// \brief Signature of \c luthierKfdGpuDrmFd, resolved at run time.
using GpuDrmFdFn = int (*)(unsigned GpuId);

} // namespace kfd

/// \brief \c DriverAllocationResolver over allocations observed at the KFD ioctl
/// boundary. See the file comment for the mechanism and its limits.
class KfdAllocationResolver final : public DriverAllocationResolver {
public:
  /// \param Find the allocation lookup to use. Defaults to resolving
  /// \c luthierKfdFindAllocation from the process at construction time; passing
  /// one explicitly is how tests drive the resolver without a preloaded wrapper.
  /// \param GetDrmFd where the application's DRM descriptor comes from; defaults
  /// to resolving \c luthierKfdGpuDrmFd the same way.
  explicit KfdAllocationResolver(kfd::FindAllocationFn Find = nullptr,
                                 kfd::GpuDrmFdFn GetDrmFd = nullptr);

  ~KfdAllocationResolver() override;

  KfdAllocationResolver(const KfdAllocationResolver &) = delete;
  KfdAllocationResolver &operator=(const KfdAllocationResolver &) = delete;

  [[nodiscard]] llvm::Expected<Allocation>
  resolve(uint64_t DeviceAddr) const override;

  /// \brief Whether the tracker was found in this process. False means no module
  /// is recording KFD allocations -- normally because the wrapper was not
  /// preloaded -- and a caller should not read anything into an empty result.
  [[nodiscard]] bool isAvailable() const override { return Find != nullptr; }

private:
  /// One host mapping of one allocation, unmapped when the resolver dies.
  struct HostMapping {
    void *Address{nullptr};
    size_t Length{0};
  };

  kfd::FindAllocationFn Find{nullptr};

  kfd::GpuDrmFdFn GetDrmFd{nullptr};

  /// Keyed on the allocation's base. Mappings are cached because each one costs an
  /// \c mmap and a file descriptor, and the lifter queries the same allocation
  /// once per instruction it disassembles.
  ///
  /// \par What this cache is valid for, and what it cannot see  (assumption A9)
  /// One pipeline run. \c MemoryAllocationAnalysis owns the accessor that owns
  /// this by \c unique_ptr, and the analysis manager holding it is a local in
  /// \c runInstrumentationPipelineForDispatch, so this resolver is built and
  /// destroyed within the lifting of a single kernel dispatch -- milliseconds.
  ///
  /// Within that window a mapping is \b not invalidated. If the application frees
  /// an allocation and a later one lands at the same base, a lookup of that base
  /// returns the older mapping and therefore the wrong bytes, silently. Two facts
  /// make precise invalidation impossible without changing an exported C
  /// signature: \c forgetAllocation runs \e before the free observers, and
  /// \c luthierKfdFindAllocation does not return the handle -- so an observer
  /// cannot resolve handle to base, and this resolver never learns the handle of
  /// anything it mapped.
  ///
  /// Left as a documented limit rather than fixed, deliberately: for it to bite,
  /// the application would have to free \e and reuse an address this resolver had
  /// already queried, during one kernel's lifting.
  mutable llvm::DenseMap<uint64_t, HostMapping> Mappings;

  /// \return a host-readable pointer to the start of the allocation.
  [[nodiscard]] llvm::Expected<void *>
  mapAllocation(uint64_t Base, uint64_t Size, uint32_t Flags, uint32_t GpuId,
                uint64_t MmapOffset) const;
};

} // namespace luthier

#endif // LUTHIER_KFD_KFD_ALLOCATION_RESOLVER_H
