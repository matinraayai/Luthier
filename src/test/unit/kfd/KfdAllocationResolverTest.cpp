//===-- KfdAllocationResolverTest.cpp -------------------------------------===//
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
/// GPU-free tests for \c luthier::KfdAllocationResolver.
///
/// The resolver takes its allocation lookup as a constructor argument precisely so
/// these can run: everything except the successful \c mmap can be driven from a
/// stub, and the \c mmap needs a real driver allocation, so it is covered by the
/// hardware suite instead.
///
/// The distinction these tests care about most is between the three states the
/// resolver can be in, which are easy to collapse into one another:
/// \li \b unavailable -- nothing in this process is tracking allocations at all,
///     reported once through \c isAvailable rather than as an error on every
///     lookup. Confusing this with "absent" is what would make an untracked
///     process look like one that allocated nothing, so the two are separate
///     questions rather than two shades of one answer;
/// \li \b absent -- an empty result from an available resolver, which is not an
///     error and happens for SVM and imported memory;
/// \li \b found, or a failure to map something found -- the latter is an
///     \c llvm::Error, because the address genuinely is inside an allocation.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/KfdAllocationResolver.h"

#include <gtest/gtest.h>

#include <llvm/Support/Error.h>

using luthier::KfdAllocationResolver;

namespace {

/// A lookup that never finds anything.
int findNothing(unsigned long long, unsigned long long *, unsigned long long *,
                unsigned *, unsigned *, unsigned long long *) {
  return 0;
}

/// A DRM-descriptor source that never has one.
int noDrmFd(unsigned) { return -1; }

/// A lookup that reports a plausible allocation, taken from a measured run.
int findOnImpossibleGpu(unsigned long long, unsigned long long *Base,
                        unsigned long long *Size, unsigned *Flags,
                        unsigned *GpuId, unsigned long long *MmapOffset) {
  if (Base != nullptr)
    *Base = 0x520240000000ULL;
  if (Size != nullptr)
    *Size = 0x200000ULL;
  if (Flags != nullptr)
    *Flags = 0x90000001U; // VRAM|WRITABLE|NO_SUBSTITUTE, as measured
  if (GpuId != nullptr)
    *GpuId = 0xFFFFFFFFU;
  if (MmapOffset != nullptr)
    *MmapOffset = 0x1000ULL;
  return 1;
}

/// Consume an Expected's error and return its message, so a test can assert on
/// what a failure actually says rather than only that it failed.
std::string errorMessage(llvm::Error E) {
  return llvm::toString(std::move(E));
}

/// "Nothing is tracking allocations" is answered once, by \c isAvailable, and not
/// re-reported as an error on every lookup.
///
/// This is the opposite of what this resolver used to do, and the change is
/// deliberate. It now sits behind \c HsaMemoryAllocationAccessor, which walks a
/// disassembly one instruction at a time; an error there aborts the whole
/// analysis, whereas an empty result is how a walk is supposed to end. So the
/// unanswerable case has to be a property a caller checks once, not an answer it
/// receives repeatedly.
TEST(KfdAllocationResolver, NoTrackerIsReportedByAvailabilityNotPerLookup) {
  // Explicitly null rather than relying on dlsym failing: the test binary links
  // the tracker, so dlsym would succeed here and the test would measure nothing.
  KfdAllocationResolver A{nullptr, noDrmFd};
  EXPECT_FALSE(A.isAvailable())
      << "an untracked process must be distinguishable from one that allocated "
         "nothing, and isAvailable is where that distinction lives";

  auto R = A.resolve(0x520240000000ULL);
  ASSERT_TRUE(static_cast<bool>(R))
      << "an unavailable resolver must not abort a caller's disassembly walk";
  EXPECT_TRUE(R->empty());
}

TEST(KfdAllocationResolver, UntrackedAddressIsEmptyNotAnError) {
  KfdAllocationResolver A{findNothing, noDrmFd};
  EXPECT_TRUE(A.isAvailable());

  auto R = A.resolve(0x1234ULL);
  ASSERT_TRUE(static_cast<bool>(R))
      << "an address with no tracked allocation is an ordinary outcome -- it "
         "happens for SVM and imported memory -- and must not be an error";
  EXPECT_TRUE(R->empty());
  EXPECT_EQ(0u, R->Size);
}

/// The failure that took a full hardware round-trip to understand: an mmap offset
/// only resolves on the DRM descriptor that created the allocation, so without the
/// application's descriptor there is nothing to map through. Measured: opening the
/// render node here instead fails with EACCES.
TEST(KfdAllocationResolver, WithoutTheApplicationsDrmFdMappingIsImpossible) {
  KfdAllocationResolver A{findOnImpossibleGpu, noDrmFd};

  auto R = A.resolve(0x5202400003c0ULL);
  ASSERT_FALSE(static_cast<bool>(R))
      << "a tracked allocation that cannot be mapped is an error, not an empty "
         "descriptor -- the address IS in an allocation";
  const std::string Msg = errorMessage(R.takeError());
  // A caller seeing this needs to know it is about the descriptor, not about its
  // own address arithmetic.
  EXPECT_NE(std::string::npos, Msg.find("ACQUIRE_VM"));
  EXPECT_NE(std::string::npos, Msg.find("EACCES"));
  EXPECT_NE(std::string::npos, Msg.find("4294967295")); // the bogus gpu_id
}

/// A missing DRM descriptor and a missing lookup are different situations with
/// different fixes, and after the merge they are also different \e kinds of
/// answer -- one is an error, the other is an availability flag.
///
/// \note This passes \c noDrmFd explicitly rather than \c nullptr. The constructor
/// resolves a null argument with \c dlsym(RTLD_DEFAULT, ...), and inside a test
/// \e executable that lookup fails even though the tracker is linked in, because
/// an executable does not export its symbols dynamically unless asked to. In
/// production the tracker lives in a preloaded shared library, where the lookup
/// works -- so relying on it here would test the linker, not the accessor.
TEST(KfdAllocationResolver, UnavailableAndUnmappableAreDifferentOutcomes) {
  KfdAllocationResolver NoLookup{nullptr, noDrmFd};
  auto NoLookupResult = NoLookup.resolve(0x5202400003c0ULL);
  ASSERT_TRUE(static_cast<bool>(NoLookupResult));
  EXPECT_FALSE(NoLookup.isAvailable());
  EXPECT_TRUE(NoLookupResult->empty());

  KfdAllocationResolver NoDrmFd{findOnImpossibleGpu, noDrmFd};
  auto NoDrmFdResult = NoDrmFd.resolve(0x5202400003c0ULL);
  EXPECT_TRUE(NoDrmFd.isAvailable())
      << "a resolver with a working lookup is available even when a particular "
         "allocation cannot be mapped";
  ASSERT_FALSE(static_cast<bool>(NoDrmFdResult))
      << "the address IS inside a tracked allocation, so failing to map it is a "
         "failure rather than a miss";
  EXPECT_NE(std::string::npos,
            errorMessage(NoDrmFdResult.takeError()).find("ACQUIRE_VM"));
}

//===----------------------------------------------------------------------===//
// gpu_id -> DRM render node resolution
//===----------------------------------------------------------------------===//

TEST(KfdRenderNode, ImpossibleGpuIdResolvesToNothing) {
  EXPECT_FALSE(luthier::kfd::renderNodeForGpuId(0xFFFFFFFFU).has_value());
}

/// gpu_id 0 marks a CPU node in KFD's topology, and a CPU node owns no device
/// memory. Worth pinning: the scan skips those explicitly, and without that it
/// would return the first node's render minor for any caller passing 0.
TEST(KfdRenderNode, GpuIdZeroIsNotAGpu) {
  EXPECT_FALSE(luthier::kfd::renderNodeForGpuId(0U).has_value());
}

/// On a machine with an AMD GPU, whatever the topology reports must be a path
/// that exists. Skipped rather than failed elsewhere, so the suite stays runnable
/// without hardware.
TEST(KfdRenderNode, RealGpuIdResolvesToAnExistingNode) {
  std::optional<uint32_t> SomeGpuId;
  for (unsigned Node = 0; Node < 64 && !SomeGpuId; Node++) {
    std::string Path = "/sys/class/kfd/kfd/topology/nodes/" +
                       std::to_string(Node) + "/gpu_id";
    FILE *F = fopen(Path.c_str(), "r");
    if (F == nullptr)
      continue;
    unsigned Id = 0;
    if (fscanf(F, "%u", &Id) == 1 && Id != 0)
      SomeGpuId = Id;
    fclose(F);
  }
  if (!SomeGpuId)
    GTEST_SKIP() << "no KFD GPU node on this machine";

  auto Node = luthier::kfd::renderNodeForGpuId(*SomeGpuId);
  ASSERT_TRUE(Node.has_value()) << "gpu_id " << *SomeGpuId
                                << " is in the topology but resolved to nothing";
  EXPECT_EQ(0, access(Node->c_str(), F_OK))
      << *Node << " does not exist, so the topology and /dev disagree";
}

//===----------------------------------------------------------------------===//
// gpu_id -> topology node index
//===----------------------------------------------------------------------===//

/// The inverse of the render-node walk, factored out because naming a GPU's ISA
/// needs the node's sysfs properties and every ioctl only ever hands us a
/// \c gpu_id. The two are different namespaces -- a \c gpu_id is a large opaque
/// number, a node index is a small dense counter that also covers CPU nodes -- so
/// a test that pins one is not pinning the other.
TEST(KfdTopologyNode, ImpossibleGpuIdResolvesToNothing) {
  EXPECT_FALSE(luthier::kfd::topologyNodeForGpuId(0xFFFFFFFFU).has_value());
}

TEST(KfdTopologyNode, GpuIdZeroIsNotAGpu) {
  EXPECT_FALSE(luthier::kfd::topologyNodeForGpuId(0U).has_value());
}

/// Round-trip: whatever node the topology reports for a real gpu_id must be the
/// node whose gpu_id file holds that value. Guards against the two namespaces
/// being silently swapped, which reads correctly and resolves the wrong device.
TEST(KfdTopologyNode, RealGpuIdRoundTripsThroughTheNodeIndex) {
  std::optional<uint32_t> SomeGpuId;
  for (unsigned Node = 0; Node < 64 && !SomeGpuId; Node++) {
    std::string Path = "/sys/class/kfd/kfd/topology/nodes/" +
                       std::to_string(Node) + "/gpu_id";
    FILE *F = fopen(Path.c_str(), "r");
    if (F == nullptr)
      continue;
    unsigned Id = 0;
    if (fscanf(F, "%u", &Id) == 1 && Id != 0)
      SomeGpuId = Id;
    fclose(F);
  }
  if (!SomeGpuId)
    GTEST_SKIP() << "no KFD GPU node on this machine";

  auto Node = luthier::kfd::topologyNodeForGpuId(*SomeGpuId);
  ASSERT_TRUE(Node.has_value());

  std::string Path = "/sys/class/kfd/kfd/topology/nodes/" +
                     std::to_string(*Node) + "/gpu_id";
  FILE *F = fopen(Path.c_str(), "r");
  ASSERT_NE(nullptr, F) << Path << " does not exist";
  unsigned Id = 0;
  const int Scanned = fscanf(F, "%u", &Id);
  fclose(F);
  ASSERT_EQ(1, Scanned);
  EXPECT_EQ(*SomeGpuId, Id)
      << "node index " << *Node << " does not report gpu_id " << *SomeGpuId;
}

} // namespace
