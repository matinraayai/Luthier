//===-- IsaInfo.cpp --------------------------------------------------------===//
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
/// Implements \c luthier/KFD/IsaInfo.h. See that header for why none of this may
/// go through HSA.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/IsaInfo.h"

#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/KFD/Topology.h"
#include "luthier/LLVM/streams.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TargetParser/AMDGPUTargetParser.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <hsakmt/hsakmttypes.h>
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define DEBUG_TYPE "luthier-kfd-isa-info"

namespace luthier::kfd {

namespace {

/// A mask selecting one field of the node's \c capability word, derived from
/// hsakmt's own \c HSA_CAPABILITY bitfield rather than by counting bits.
///
/// Counting was the first implementation and it is a bad idea here: an off-by-one
/// still yields a plausible answer, and on our gfx908 the neighbouring
/// \c SVMAPISupported bit happens to hold the same value as \c SRAM_EDCSupport,
/// so even comparing against what HSA reports does not catch the mistake on this
/// hardware. Letting the compiler lay the field out removes the possibility.
template <typename SetField> uint32_t capabilityMask(SetField Set) {
  HSA_CAPABILITY C{};
  C.Value = 0;
  Set(C);
  return C.Value;
}

/// Whether GFX internal SRAM ECC is active on this device.
const uint32_t SrameccMask =
    capabilityMask([](HSA_CAPABILITY &C) { C.ui32.SRAM_EDCSupport = 1; });

/// \c DEPRECATED_SRAM_EDCSupport, whose comment in \c hsakmttypes.h says "Old
/// buggy user mode depends on this being 0". Now that the layout comes from the
/// header rather than from counting, this checks something else and still worth
/// checking: that the header Luthier compiled against describes the same word the
/// running kernel publishes. A mismatch there shifts every field silently.
const uint32_t DeprecatedSrameccMask = capabilityMask(
    [](HSA_CAPABILITY &C) { C.ui32.DEPRECATED_SRAM_EDCSupport = 1; });

/// Query the driver's xnack mode for this process.
///
/// Not a node property -- xnack is process state, which is why this needs an
/// ioctl where everything else here is a file read. Passing -1 queries rather
/// than sets, which is what hsakmt's \c hsaKmtGetXNACKMode does.
///
/// A failure is not an error: \c EPERM means the chip has no xnack at all, and
/// ROCR treats every other failure as "assume disabled" too
/// (\c amd_kfd_driver.cpp, \c BindXnackMode). Reporting an error would make a
/// perfectly instrumentable chip unusable over a feature bit.
bool queryXnackEnabled() {
  const int Kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (Kfd < 0) {
    LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                   "[KfdTargetMachine] /dev/kfd could not be opened to query "
                   "the xnack mode ({0}); assuming disabled.\n",
                   strerror(errno)));
    return false;
  }

  struct kfd_ioctl_set_xnack_mode_args Args {};
  Args.xnack_enabled = -1; // query, do not set
  const bool Ok = ioctl(Kfd, AMDKFD_IOC_SET_XNACK_MODE, &Args) == 0;
  const int SavedErrno = errno;
  close(Kfd);

  if (!Ok) {
    LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                   "[KfdTargetMachine] the xnack mode query failed ({0}); "
                   "assuming disabled, as ROCR does.\n",
                   strerror(SavedErrno)));
    return false;
  }
  return Args.xnack_enabled != 0;
}

} // namespace

llvm::Expected<IsaInfo> queryIsaInfo(uint32_t GpuId) {
  std::optional<uint32_t> Node = topologyNodeForGpuId(GpuId);
  if (!Node)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "No KFD topology node reports gpu_id {0}, so nothing is known about "
        "the device a kernel was dispatched on. Note that a gpu_id is not a "
        "topology node index; see luthier/KFD/Topology.h.",
        GpuId));

  std::optional<uint64_t> Version =
      readNodeProperty(*Node, "gfx_target_version");
  if (!Version || *Version == 0)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "KFD topology node {0} (gpu_id {1}) reports no gfx_target_version, "
        "which is what a CPU node reports. Only a GPU can be a dispatch "
        "target.",
        *Node, GpuId));

  // 90008 -> 9, 0, 8. The same decode hsakmt performs (topology.c:1217).
  IsaInfo Info;
  Info.Major = static_cast<unsigned>(*Version / 10000);
  Info.Minor = static_cast<unsigned>((*Version / 100) % 100);
  Info.Stepping = static_cast<unsigned>(*Version % 100);

  std::optional<uint64_t> Capability = readNodeProperty(*Node, "capability");
  if (!Capability)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "KFD topology node {0} (gpu_id {1}) publishes no capability word, so "
        "whether sramecc is active cannot be determined.",
        *Node, GpuId));

  if ((*Capability & DeprecatedSrameccMask) != 0)
    return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
        "gpu_id {0}'s capability word ({1:x}) has DEPRECATED_SRAM_EDCSupport "
        "(mask {2:x}) set, which the driver is documented always to leave "
        "clear. The hsakmt header this was built against therefore describes a "
        "different layout than the running kernel publishes, and every field "
        "read out of this word -- sramecc included -- would be silently "
        "shifted.",
        GpuId, *Capability, DeprecatedSrameccMask));

  Info.SrameccEnabled = (*Capability & SrameccMask) != 0;
  Info.XnackEnabled = queryXnackEnabled();

  LLVM_DEBUG(luthier::dbgs() << llvm::formatv(
                 "[KfdTargetMachine] gpu_id {0} is node {1}: gfx{2}.{3}.{4}, "
                 "sramecc {5}, xnack {6}.\n",
                 GpuId, *Node, Info.Major, Info.Minor, Info.Stepping,
                 Info.SrameccEnabled ? "on" : "off",
                 Info.XnackEnabled ? "on" : "off"));
  return Info;
}

llvm::Expected<std::string> archNameForIsaVersion(unsigned Major,
                                                  unsigned Minor,
                                                  unsigned Stepping) {
  llvm::SmallVector<llvm::StringRef, 64> Arches;
  llvm::AMDGPU::fillValidArchListAMDGCN(Arches);

  for (const llvm::StringRef Arch : Arches) {
    const llvm::AMDGPU::IsaVersion V = llvm::AMDGPU::getIsaVersion(Arch);
    if (V.Major == Major && V.Minor == Minor && V.Stepping == Stepping)
      return Arch.str();
  }

  return LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
      "The driver reports ISA version {0}.{1}.{2}, which the LLVM this tool "
      "was built against does not know any AMDGPU architecture for. A newer "
      "LLVM is needed to lift code for this device.",
      Major, Minor, Stepping));
}

} // namespace luthier::kfd
