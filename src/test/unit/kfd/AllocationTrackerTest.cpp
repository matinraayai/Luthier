//===-- AllocationTrackerTest.cpp - GPU-free tests for the tracker --------===//
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
/// The tracker's lookup is a containment search over an ordered map, so all of
/// its interesting behaviour is at boundaries -- and boundaries are exactly what
/// a hardware run will not exercise, because a real \c kernel_object lands
/// comfortably inside a real allocation.
///
/// \par Verified against a mutation, and what that showed
/// Removing the containment test from \c find -- returning the nearest record at
/// or below the address regardless of whether it actually covers it -- fails
/// exactly four of these: \c Boundaries,
/// \c AddressAboveEverythingDoesNotResolve, \c GapBetweenAllocations and
/// \c ZeroSizedAllocationContainsNothing. The other ten still pass, because they
/// cover different properties (handle bookkeeping, base reuse, chunking, the
/// counters).
///
/// That distinction is worth writing down rather than claiming every test guards
/// everything: if the four above are ever deleted as redundant, the containment
/// check becomes untested while thirteen green tests suggest otherwise.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/AllocationTracker.h"

#include <gtest/gtest.h>

#include <unistd.h>

using luthier::kfd::Allocation;
using luthier::kfd::detail::AllocationMap;

namespace {

/// One allocation with distinguishable field values, so a test that resolves the
/// wrong record fails on the contents rather than merely on the address.
Allocation makeAlloc(uint64_t Base, uint64_t Size, uint64_t Handle,
                     uint32_t Flags = 0x1u, uint32_t GpuId = 38979u,
                     uint64_t MmapOffset = 0u) {
  return Allocation{Base, Size, Flags, GpuId, Handle, MmapOffset};
}


TEST(AllocationTracker, EmptyMapResolvesNothing) {
  AllocationMap M;
  EXPECT_FALSE(M.find(0x1000).has_value());
  EXPECT_EQ(0u, M.liveCount());
}

TEST(AllocationTracker, ResolvesAddressInsideAllocation) {
  AllocationMap M;
  M.record(makeAlloc(0x520d00440000, 0x400000, /*Handle=*/1));

  auto A = M.find(0x520d00440000 + 0x1234);
  ASSERT_TRUE(A.has_value());
  EXPECT_EQ(0x520d00440000u, A->Base);
  EXPECT_EQ(0x400000u, A->Size);
  EXPECT_EQ(1u, A->Handle);
  EXPECT_EQ(38979u, A->GpuId);
}

/// The three boundaries. The last-byte and one-past-the-end cases are what
/// distinguish a containment test from a nearest-record lookup, and getting them
/// wrong would silently attribute an address to a neighbouring allocation.
TEST(AllocationTracker, Boundaries) {
  AllocationMap M;
  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/1));

  EXPECT_TRUE(M.find(0x1000).has_value()) << "first byte is inside";
  EXPECT_TRUE(M.find(0x10FF).has_value()) << "last byte is inside";
  EXPECT_FALSE(M.find(0x1100).has_value()) << "one past the end is outside";
  EXPECT_FALSE(M.find(0x0FFF).has_value()) << "one before the base is outside";
}

/// An address above every record must not resolve to the highest one. This is the
/// case a nearest-record lookup gets wrong in the most confusing way, since it
/// returns a plausible-looking allocation.
TEST(AllocationTracker, AddressAboveEverythingDoesNotResolve) {
  AllocationMap M;
  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/1));
  M.record(makeAlloc(0x2000, 0x100, /*Handle=*/2));

  EXPECT_FALSE(M.find(0x9000).has_value());
}

/// A gap between two allocations belongs to neither.
TEST(AllocationTracker, GapBetweenAllocations) {
  AllocationMap M;
  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/1));
  M.record(makeAlloc(0x3000, 0x100, /*Handle=*/2));

  EXPECT_FALSE(M.find(0x2000).has_value());

  auto Low = M.find(0x1050);
  ASSERT_TRUE(Low.has_value());
  EXPECT_EQ(1u, Low->Handle);

  auto High = M.find(0x3050);
  ASSERT_TRUE(High.has_value());
  EXPECT_EQ(2u, High->Handle);
}

TEST(AllocationTracker, AdjacentAllocationsResolveToTheirOwn) {
  AllocationMap M;
  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/1));
  M.record(makeAlloc(0x1100, 0x100, /*Handle=*/2));

  EXPECT_EQ(1u, M.find(0x10FF)->Handle);
  EXPECT_EQ(2u, M.find(0x1100)->Handle);
}

TEST(AllocationTracker, ForgetRemovesTheRecord) {
  AllocationMap M;
  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/7));
  ASSERT_TRUE(M.find(0x1050).has_value());

  EXPECT_TRUE(M.forget(7));
  EXPECT_FALSE(M.find(0x1050).has_value());
  EXPECT_EQ(0u, M.liveCount());
}

TEST(AllocationTracker, ForgetUnknownHandleIsHarmless) {
  AllocationMap M;
  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/7));

  EXPECT_FALSE(M.forget(99));
  EXPECT_TRUE(M.find(0x1050).has_value()) << "unrelated record survives";
  EXPECT_EQ(1u, M.liveCount());
}

/// The case that motivated handling deallocation at all. A virtual address freed
/// and then handed out again must resolve to the *new* allocation; returning the
/// old one is the stale-state failure this whole design guards against, and it
/// would present as a wrong size or GPU rather than as an obvious error.
TEST(AllocationTracker, BaseReusedAfterFreeResolvesToTheNewAllocation) {
  AllocationMap M;
  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/1, /*Flags=*/0x1u));
  ASSERT_TRUE(M.forget(1));

  M.record(makeAlloc(0x1000, 0x800, /*Handle=*/2, /*Flags=*/0x2u));

  auto A = M.find(0x1400);
  ASSERT_TRUE(A.has_value()) << "address is inside the new, larger allocation";
  EXPECT_EQ(2u, A->Handle);
  EXPECT_EQ(0x800u, A->Size);
  EXPECT_EQ(0x2u, A->Flags);
}

/// A free arriving for a handle whose base has already been reallocated must not
/// take the new record with it. Out-of-order frees are plausible with several
/// threads, and this is the one way a late free can destroy live state.
TEST(AllocationTracker, StaleFreeDoesNotEvictTheNewOwnerOfABase) {
  AllocationMap M;
  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/1));
  M.record(makeAlloc(0x1000, 0x200, /*Handle=*/2)); // same base, new handle

  EXPECT_FALSE(M.forget(1)) << "handle 1 no longer owns this base";

  auto A = M.find(0x1050);
  ASSERT_TRUE(A.has_value());
  EXPECT_EQ(2u, A->Handle);
}

/// Chunking means one application-level allocation becomes several records
/// (fmm.c:1195). Each must resolve on its own, since we deliberately do not
/// reassemble them.
TEST(AllocationTracker, ChunkedAllocationsResolveIndividually) {
  AllocationMap M;
  const uint64_t Chunk = 0x1000;
  for (uint64_t I = 0; I < 4; ++I)
    M.record(makeAlloc(0x8000 + I * Chunk, Chunk, /*Handle=*/I + 1));

  EXPECT_EQ(4u, M.liveCount());
  for (uint64_t I = 0; I < 4; ++I) {
    auto A = M.find(0x8000 + I * Chunk + 0x10);
    ASSERT_TRUE(A.has_value()) << "chunk " << I;
    EXPECT_EQ(I + 1, A->Handle) << "chunk " << I;
  }
}

/// Live and cumulative counts answer different questions: a test that sees an
/// empty map cannot otherwise tell "recorded then correctly freed" from "never
/// recorded".
TEST(AllocationTracker, CountsDistinguishFreedFromNeverRecorded) {
  AllocationMap M;
  EXPECT_EQ(0u, M.recordedTotal());

  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/1));
  M.record(makeAlloc(0x2000, 0x100, /*Handle=*/2));
  ASSERT_TRUE(M.forget(1));

  EXPECT_EQ(1u, M.liveCount()) << "one still alive";
  EXPECT_EQ(2u, M.recordedTotal()) << "two were seen";
}

TEST(AllocationTracker, ZeroSizedAllocationContainsNothing) {
  AllocationMap M;
  M.record(makeAlloc(0x1000, 0, /*Handle=*/1));
  EXPECT_FALSE(M.find(0x1000).has_value())
      << "an empty range cannot contain its own base";
}

TEST(AllocationTracker, ClearResetsEverything) {
  AllocationMap M;
  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/1));
  M.clear();

  EXPECT_EQ(0u, M.liveCount());
  EXPECT_EQ(0u, M.recordedTotal());
  EXPECT_FALSE(M.find(0x1050).has_value());
}


//===----------------------------------------------------------------------===//
// C1: an allocation with no virtual address must not enter the map
//===----------------------------------------------------------------------===//

/// hsakmt asks for these deliberately -- allocating from its mem_handle_aperture
/// passes va_addr == 0, commented "if allocate vram-only, use an invalid VA"
/// (fmm.c:1161-1162). Mutation: delete the `if (A.Base == 0) return;` guard in
/// AllocationMap::record and this test fails.
TEST(AllocationTracker, ZeroBaseAllocationIsNotRecorded) {
  AllocationMap M;
  M.record(makeAlloc(0, 0x1000, /*Handle=*/1));

  EXPECT_EQ(0u, M.liveCount());
  // Nothing was recorded, so the cumulative counter must not move either --
  // otherwise "recorded but freed" and "rejected" become indistinguishable.
  EXPECT_EQ(0u, M.recordedTotal());
  EXPECT_FALSE(M.find(0).has_value());
}

/// The reason the guard matters, stated as behaviour rather than as an internal
/// detail: a zero-based record would sit below every real allocation, so the
/// nearest-at-or-below search would land on it for any low address.
TEST(AllocationTracker, ZeroBaseAllocationNeverShadowsARealOne) {
  AllocationMap M;
  M.record(makeAlloc(0, 0x40000000, /*Handle=*/1)); // 1 GB starting at 0
  M.record(makeAlloc(0x1000, 0x100, /*Handle=*/2, /*Flags=*/0x2u));

  // An address inside the real allocation resolves to the real allocation.
  auto Inside = M.find(0x1080);
  ASSERT_TRUE(Inside.has_value());
  EXPECT_EQ(0x1000u, Inside->Base);
  EXPECT_EQ(2u, Inside->Handle);

  // And an address that only the bogus record could have covered resolves to
  // nothing at all.
  EXPECT_FALSE(M.find(0x10).has_value());
}

//===----------------------------------------------------------------------===//
// Overflow safety in the containment test
//===----------------------------------------------------------------------===//

/// Base and Size both come from ioctl arguments, so a size near UINT64_MAX is
/// reachable. Mutation: write contains() as `Addr < Base + Size` and this fails,
/// because the sum wraps and the range appears to contain nothing.
TEST(AllocationTracker, HugeSizeDoesNotWrapContainment) {
  AllocationMap M;
  M.record(makeAlloc(0x30000, UINT64_MAX - 8, /*Handle=*/1));

  auto A = M.find(0x30000);
  ASSERT_TRUE(A.has_value());
  EXPECT_EQ(UINT64_MAX - 8, A->Size);
  // Far inside the range, at an address that would be excluded by a wrapped sum.
  EXPECT_TRUE(M.find(0x8000000000000000ULL).has_value());
}

/// The mmap offset survives a round trip, since the accessor will need it to
/// place a host mapping over the allocation.
TEST(AllocationTracker, MmapOffsetIsRetained) {
  AllocationMap M;
  M.record(makeAlloc(0x40000, 0x1000, /*Handle=*/1, /*Flags=*/0x1u,
                     /*GpuId=*/38979u, /*MmapOffset=*/0xdeadb000u));

  auto A = M.find(0x40000);
  ASSERT_TRUE(A.has_value());
  EXPECT_EQ(0xdeadb000u, A->MmapOffset);
}

} // namespace
