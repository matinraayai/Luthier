//===-- KfdAllocationResolver.cpp ------------------------------------------===//
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
/// Implements \c luthier::KfdAllocationResolver. See its header for why the
/// tracker is reached through \c dlsym and how the host mapping is obtained.
///
/// \note Kept in its own translation unit, separate from \c AllocationTracker.cpp,
/// because that file is deliberately LLVM-free so it can be preloaded into any
/// process. This one uses LLVM and must therefore never end up in the preloadable
/// shared library -- see the target split in this directory's CMakeLists.txt.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/KfdAllocationResolver.h"

#include "luthier/Common/GenericLuthierError.h"
#include "luthier/KFD/Topology.h"

#include <llvm/Support/FormatVariadic.h>

#include <cstdio>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/kfd_ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace luthier {

KfdAllocationResolver::KfdAllocationResolver(kfd::FindAllocationFn Find,
                                             kfd::GpuDrmFdFn GetDrmFd)
    : Find(Find), GetDrmFd(GetDrmFd) {
  // RTLD_DEFAULT rather than a link-time dependency: the records live in whatever
  // module intercepted the ioctls, which is loaded by LD_PRELOAD and is not
  // something we can link against. A null result is not fatal here -- it is
  // reported through isAvailable(), so a caller can skip this source rather than
  // mistaking an unwatched process for one that allocated nothing.
  if (this->Find == nullptr)
    this->Find = reinterpret_cast<kfd::FindAllocationFn>(
        dlsym(RTLD_DEFAULT, "luthierKfdFindAllocation"));
  if (this->GetDrmFd == nullptr)
    this->GetDrmFd = reinterpret_cast<kfd::GpuDrmFdFn>(
        dlsym(RTLD_DEFAULT, "luthierKfdGpuDrmFd"));
}

KfdAllocationResolver::~KfdAllocationResolver() {
  for (const auto &Entry : Mappings)
    if (Entry.second.Address != nullptr)
      munmap(Entry.second.Address, Entry.second.Length);
  // The DRM descriptors are not ours -- they belong to the tracker, which
  // duplicated the application's -- so they are not closed here.
}

llvm::Expected<void *> KfdAllocationResolver::mapAllocation(
    uint64_t Base, uint64_t Size, uint32_t Flags, uint32_t GpuId,
    uint64_t MmapOffset) const {
  if (auto It = Mappings.find(Base); It != Mappings.end())
    return It->second.Address;

  // The application's descriptor, not one of ours. An mmap offset only resolves
  // on the DRM file that created the allocation: opening this GPU's render node
  // ourselves and mapping a valid offset through it fails with EACCES, measured.
  if (GetDrmFd == nullptr)
    return LUTHIER_MAKE_GENERIC_ERROR(
        "luthierKfdGpuDrmFd was not found in this process, so the DRM "
        "descriptor the application bound its GPU memory to is unknown. A host "
        "mapping is impossible without it: an mmap offset only resolves on the "
        "descriptor that created the allocation.");

  const int Fd = GetDrmFd(GpuId);
  if (Fd < 0) {
    std::optional<std::string> Path = kfd::renderNodeForGpuId(GpuId);
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "No DRM descriptor was recorded for KFD gpu_id {0} ({1}), so the "
        "allocation at {2:x} cannot be mapped for host access. That descriptor "
        "is observed from AMDKFD_IOC_ACQUIRE_VM, so this means interception "
        "started after the application had already bound this GPU -- opening "
        "the render node here instead would fail with EACCES, because an mmap "
        "offset only resolves on the descriptor that created the allocation.",
        GpuId, Path ? *Path : std::string("no render node in the topology"),
        Base));
  }

  // The same call hsakmt makes (fmm.c:1569-1573), except that we ask for
  // PROT_READ regardless of the PUBLIC flag. PROT_READ only: this resolver never
  // writes device memory, and asking for less makes a refusal more likely to be
  // about readability than about permissions.
  void *Host = mmap(nullptr, static_cast<size_t>(Size), PROT_READ, MAP_SHARED,
                    Fd, static_cast<off_t>(MmapOffset));
  if (Host == MAP_FAILED) {
    // Expected on a small-BAR GPU, where only part of VRAM has a CPU aperture.
    // Naming the flag matters: a caller seeing this needs to know whether to
    // blame its own logic or the hardware.
    const bool IsPublic = (Flags & KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC) != 0;
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "Failed to map the allocation at {0:x} (size {1:x}, flags {2:x}, "
        "gpu_id {3}) for host access at mmap offset {4:x}: {5}. The allocation "
        "is {6}marked KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC. Device memory without "
        "that flag is only host-mappable where the whole of VRAM has a CPU "
        "aperture, i.e. on a large-BAR GPU.",
        Base, Size, Flags, GpuId, MmapOffset, strerror(errno),
        IsPublic ? "" : "not "));
  }

  Mappings[Base] = HostMapping{Host, static_cast<size_t>(Size)};
  return Host;
}

llvm::Expected<DriverAllocationResolver::Allocation>
KfdAllocationResolver::resolve(uint64_t DeviceAddr) const {
  // Empty rather than an error, even though this resolver can answer nothing at
  // all: isAvailable() is how a caller learns the difference, and it can check it
  // once instead of on every lookup along a disassembly walk.
  if (Find == nullptr)
    return Allocation();

  unsigned long long Base = 0, Size = 0, MmapOffset = 0;
  unsigned Flags = 0, GpuId = 0;
  if (Find(DeviceAddr, &Base, &Size, &Flags, &GpuId, &MmapOffset) != 1) {
    // Not an error. Legitimately happens for SVM and imported memory, which no
    // allocation ioctl of ours ever sees.
    return Allocation();
  }

  // A registered host allocation is already host memory: the application
  // allocated it itself and handed the driver the pointer, so the "device"
  // address IS the host address and there is nothing on the render node to map.
  // Mapping it anyway fails with EPERM, because the offset names no GEM object.
  //
  // This is not an edge case invented here -- it is the same branch every
  // userspace above the driver takes. hsakmt maps only non-USERPTR allocations
  // (fmm.c:1799-1808), and so does tinygrad (ops_amd.py:767). The KFD test
  // harness dispatches a kernel out of exactly such an allocation, which is how
  // the omission surfaced.
  if ((Flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) != 0) {
    const auto *HostBase =
        reinterpret_cast<const std::byte *>(static_cast<uintptr_t>(Base));
    return Allocation{HostBase, HostBase, static_cast<size_t>(Size)};
  }

  llvm::Expected<void *> HostOrErr =
      mapAllocation(Base, Size, Flags, GpuId, MmapOffset);
  LUTHIER_RETURN_ON_ERROR(HostOrErr.takeError());

  return Allocation{reinterpret_cast<const std::byte *>(
                        static_cast<uintptr_t>(Base)),
                    static_cast<const std::byte *>(*HostOrErr),
                    static_cast<size_t>(Size)};
}

} // namespace luthier
