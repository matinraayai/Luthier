//===-- AllocationChainTest.cpp --------------------------------------------===//
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
/// Tests the allocation observer chain: that several components each see every
/// event, and in an order they chose rather than one that fell out of load order.
///
/// \par Why this file carries more weight than it looks
/// Deterministic ordering is the \e only thing GOTCHA offers over the callback array
/// Luthier already had. If the ordering rule is merely asserted in a comment, the
/// justification for the whole mechanism is unchecked. So the central test here is
/// the one that would fail if order followed registration instead of priority --
/// registering the same observers both ways round and requiring the same result.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/AllocationTracker.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using luthier::kfd::Allocation;
using luthier::kfd::detail::AllocationCallbackEntry;
using luthier::kfd::detail::orderAllocationChain;

namespace {

/// Build an entry with a recognisable identity. \c CB only has to be non-null for
/// the entry to count as live, so a distinct fake pointer per entry doubles as a
/// label.
AllocationCallbackEntry entry(int Priority, unsigned long long Seq,
                              uintptr_t Id) {
  AllocationCallbackEntry E;
  E.CB = reinterpret_cast<void *>(Id);
  E.UserData = nullptr;
  E.Priority = Priority;
  E.Seq = Seq;
  return E;
}

/// The identities of the entries the chain would call, in order.
std::vector<uintptr_t> order(const std::vector<AllocationCallbackEntry> &Entries) {
  std::vector<unsigned> Out(Entries.size() + 1);
  const unsigned N = orderAllocationChain(
      Entries.data(), static_cast<unsigned>(Entries.size()), Out.data());
  std::vector<uintptr_t> Ids;
  for (unsigned I = 0; I < N; I++)
    Ids.push_back(reinterpret_cast<uintptr_t>(Entries[Out[I]].CB));
  return Ids;
}

TEST(AllocationChain, EmptyChainRunsNothing) {
  EXPECT_TRUE(order({}).empty());
}

TEST(AllocationChain, RemovedEntriesAreSkipped) {
  std::vector<AllocationCallbackEntry> E{entry(0, 1, 0xA), AllocationCallbackEntry{},
                                         entry(0, 3, 0xC)};
  // The hole in the middle is how removal works, without shuffling the others.
  EXPECT_EQ(std::vector<uintptr_t>({0xC, 0xA}), order(E));
}

TEST(AllocationChain, HigherPriorityRunsFirst) {
  std::vector<AllocationCallbackEntry> E{entry(1, 1, 0x100), entry(100, 2, 0x200)};
  EXPECT_EQ(std::vector<uintptr_t>({0x200, 0x100}), order(E));
}

/// THE test. Registration order must not decide anything when priorities differ.
///
/// Mutation: make orderAllocationChain compare Seq before Priority, and this fails
/// while every other test in this file still passes -- which is exactly the
/// situation in which GOTCHA would have bought nothing.
TEST(AllocationChain, PriorityBeatsRegistrationOrderBothWaysRound) {
  // Same two observers, registered in opposite orders.
  const std::vector<AllocationCallbackEntry> HighFirst{entry(100, 1, 0x200),
                                                       entry(1, 2, 0x100)};
  const std::vector<AllocationCallbackEntry> LowFirst{entry(1, 1, 0x100),
                                                      entry(100, 2, 0x200)};

  const auto A = order(HighFirst);
  const auto B = order(LowFirst);
  EXPECT_EQ(A, B) << "the observation order changed when the two components "
                     "registered in the opposite order, which is precisely what "
                     "priority exists to prevent";
  EXPECT_EQ(std::vector<uintptr_t>({0x200, 0x100}), A);
}

/// Negative priorities are how a component asks to run last, so they must order
/// like any other number rather than being treated as unset.
TEST(AllocationChain, NegativePrioritiesOrderNormally) {
  std::vector<AllocationCallbackEntry> E{entry(0, 1, 0x300), entry(-5, 2, 0x400),
                                         entry(5, 3, 0x500)};
  EXPECT_EQ(std::vector<uintptr_t>({0x500, 0x300, 0x400}), order(E));
}

/// A tie falls back to last-registered-first, matching the packet chain, so a
/// component that expresses no preference behaves as it would there.
TEST(AllocationChain, TiesBreakLastRegisteredFirst) {
  std::vector<AllocationCallbackEntry> E{entry(7, 1, 0x600), entry(7, 2, 0x700),
                                         entry(7, 3, 0x800)};
  EXPECT_EQ(std::vector<uintptr_t>({0x800, 0x700, 0x600}), order(E));
}

/// Sequence numbers, not array positions, decide a tie. Slots are reused after a
/// removal, so a rebuilt entry sits in an old position while being the newest --
/// ordering on position would get this backwards.
TEST(AllocationChain, TiesUseAgeNotArrayPosition) {
  std::vector<AllocationCallbackEntry> E{entry(0, 9, 0x900),
                                         entry(0, 2, 0xA00)};
  EXPECT_EQ(std::vector<uintptr_t>({0x900, 0xA00}), order(E));
}

//===----------------------------------------------------------------------===//
// Registration through the public API
//===----------------------------------------------------------------------===//

std::vector<std::string> *Log = nullptr;

void observeA(const Allocation &, void *) { Log->push_back("A"); }
void observeB(const Allocation &, void *) { Log->push_back("B"); }
void observeFree(uint64_t H, void *) {
  Log->push_back("free:" + std::to_string(H));
}

class AllocationChainApi : public ::testing::Test {
protected:
  std::vector<std::string> Entries;
  void SetUp() override {
    luthier::kfd::resetAllocationTracker();
    Entries.clear();
    Log = &Entries;
  }
  void TearDown() override {
    luthier::kfd::resetAllocationTracker();
    Log = nullptr;
  }
};

/// Both components see the event, which is the cascading requirement itself.
TEST_F(AllocationChainApi, EveryObserverSeesEveryAllocation) {
  ASSERT_NE(luthier::kfd::InvalidAllocationCallbackHandle,
            luthier::kfd::addAllocationCallback(observeA, nullptr, 10));
  ASSERT_NE(luthier::kfd::InvalidAllocationCallbackHandle,
            luthier::kfd::addAllocationCallback(observeB, nullptr, 20));

  luthier::kfd::runAllocationCallbacks(
      Allocation{0x1000, 0x100, 0x1u, 38979u, 7, 0});
  // B has the higher priority, so it runs first regardless of registering second.
  EXPECT_EQ(std::vector<std::string>({"B", "A"}), Entries);
}

TEST_F(AllocationChainApi, RemovingOneLeavesTheOther) {
  const auto HA = luthier::kfd::addAllocationCallback(observeA, nullptr, 10);
  luthier::kfd::addAllocationCallback(observeB, nullptr, 20);
  luthier::kfd::removeAllocationCallback(HA);

  luthier::kfd::runAllocationCallbacks(
      Allocation{0x1000, 0x100, 0x1u, 38979u, 7, 0});
  EXPECT_EQ(std::vector<std::string>({"B"}), Entries);
}

/// Allocation and free handles must not be interchangeable, or removing one kind
/// would silently unhook an unrelated observer of the other.
TEST_F(AllocationChainApi, AllocationAndFreeHandlesAreNotInterchangeable) {
  const auto HAlloc = luthier::kfd::addAllocationCallback(observeA, nullptr, 0);
  const auto HFree = luthier::kfd::addAllocationFreeCallback(observeFree, nullptr, 0);
  ASSERT_NE(HAlloc, HFree);

  // Remove with the wrong remover: neither observer may be affected.
  luthier::kfd::removeAllocationFreeCallback(HAlloc);
  luthier::kfd::removeAllocationCallback(HFree);

  luthier::kfd::runAllocationCallbacks(
      Allocation{0x1000, 0x100, 0x1u, 38979u, 7, 0});
  luthier::kfd::runAllocationFreeCallbacks(7);
  EXPECT_EQ(std::vector<std::string>({"A", "free:7"}), Entries);
}

TEST_F(AllocationChainApi, ChainFullIsReportedNotSilentlyDropped) {
  for (unsigned I = 0; I < luthier::kfd::MaxAllocationCallbacks; I++)
    ASSERT_NE(luthier::kfd::InvalidAllocationCallbackHandle,
              luthier::kfd::addAllocationCallback(observeA, nullptr, 0));
  EXPECT_EQ(luthier::kfd::InvalidAllocationCallbackHandle,
            luthier::kfd::addAllocationCallback(observeB, nullptr, 0));
}

/// An observer may allocate GPU memory itself, which re-enters the tracker. The
/// chain is copied before being walked precisely so that does not deadlock on a
/// non-recursive mutex.
TEST_F(AllocationChainApi, AnObserverMayReenterTheTracker) {
  static bool Reentered = false;
  Reentered = false;
  luthier::kfd::addAllocationCallback(
      [](const Allocation &, void *) {
        // Any tracker call that takes the lock is enough to expose a deadlock.
        (void)luthier::kfd::findAllocation(0x1000);
        (void)luthier::kfd::liveAllocationCount();
        Reentered = true;
      },
      nullptr, 0);

  luthier::kfd::runAllocationCallbacks(
      Allocation{0x1000, 0x100, 0x1u, 38979u, 7, 0});
  EXPECT_TRUE(Reentered);
}


//===----------------------------------------------------------------------===//
// The C-linkage registration side-tables
//===----------------------------------------------------------------------===//
// The C API keeps its own array of {callback, user-pointer} bindings alongside the
// chain, because the chain stores only one void* per slot. Two pools, two chances
// to leak one. Nothing exercised them until these tests.

void cObserve(unsigned long long, unsigned long long, unsigned, unsigned,
              unsigned long long, unsigned long long, void *) {}
void cObserveFree(unsigned long long, void *) {}

/// Mutation: drop the clearCBindings() call from resetAllocationTracker, and the
/// second round of registrations runs out of bindings while the chain reports
/// free slots.
TEST_F(AllocationChainApi, ResetReleasesTheCLinkageBindingsToo) {
  for (unsigned Round = 0; Round < 2; Round++) {
    for (unsigned I = 0; I < luthier::kfd::MaxAllocationCallbacks; I++)
      ASSERT_NE(luthier::kfd::InvalidAllocationCallbackHandle,
                luthierKfdAddAllocationCallback(cObserve, nullptr, 0))
          << "round " << Round << ", slot " << I;
    for (unsigned I = 0; I < luthier::kfd::MaxAllocationCallbacks; I++)
      ASSERT_NE(luthier::kfd::InvalidAllocationCallbackHandle,
                luthierKfdAddAllocationFreeCallback(cObserveFree, nullptr, 0));
    luthier::kfd::resetAllocationTracker();
  }
}

/// Mutation: stop releasing the binding slot in the remove trampoline, and this
/// exhausts the side-table after MaxAllocationCallbacks add/remove pairs even
/// though never more than one is registered at a time.
TEST_F(AllocationChainApi, RemovingViaTheCApiReleasesItsBinding) {
  for (unsigned I = 0; I < luthier::kfd::MaxAllocationCallbacks * 3; I++) {
    const int H = luthierKfdAddAllocationCallback(cObserve, nullptr, 0);
    ASSERT_NE(luthier::kfd::InvalidAllocationCallbackHandle, H)
        << "add/remove cycle " << I;
    luthierKfdRemoveAllocationCallback(H);
  }
  for (unsigned I = 0; I < luthier::kfd::MaxAllocationCallbacks * 3; I++) {
    const int H = luthierKfdAddAllocationFreeCallback(cObserveFree, nullptr, 0);
    ASSERT_NE(luthier::kfd::InvalidAllocationCallbackHandle, H)
        << "free add/remove cycle " << I;
    luthierKfdRemoveAllocationFreeCallback(H);
  }
}

/// The reason the remove trampoline must check *what* it unhooked.
///
/// Both APIs hand back a plain int from the same numbering space, so passing a
/// handle from the C++ registration to the C remover is an easy mistake. When that
/// happens, the entry coming off the chain carries the C++ caller's own user
/// pointer -- and a trampoline that assumed every entry was one of its own would
/// zero two words of whatever that pointer names.
///
/// Mutation: drop the `Was.CB == &cAllocTrampoline` test and clear the binding
/// from `Was.UserData` unconditionally. This test then scribbles on Guarded.
TEST_F(AllocationChainApi, TheCRemoverIgnoresAnEntryThatIsNotItsOwn) {
  // Shaped like the trampoline's own side-table entry -- two pointers -- because
  // that is what would be written through if the ownership check were dropped.
  struct TwoPointers {
    void *A;
    void *B;
  };
  TwoPointers Guarded{reinterpret_cast<void *>(0xAAAA),
                      reinterpret_cast<void *>(0xBBBB)};

  const auto HCpp = luthier::kfd::addAllocationCallback(
      [](const Allocation &, void *) {}, &Guarded, 0);
  ASSERT_NE(luthier::kfd::InvalidAllocationCallbackHandle, HCpp);

  // The mistake: a C++ handle handed to the C remover.
  luthierKfdRemoveAllocationCallback(HCpp);

  EXPECT_EQ(reinterpret_cast<void *>(0xAAAA), Guarded.A)
      << "the C remover wrote through a user pointer that was not a binding";
  EXPECT_EQ(reinterpret_cast<void *>(0xBBBB), Guarded.B);
}

/// And the same handle really does come off the chain, so the test above is not
/// passing merely because nothing was unhooked.
TEST_F(AllocationChainApi, ACppHandleDoesUnhookViaTheCRemover) {
  static int Fired = 0;
  Fired = 0;
  const auto HCpp = luthier::kfd::addAllocationCallback(
      [](const Allocation &, void *) { Fired++; }, nullptr, 0);
  ASSERT_NE(luthier::kfd::InvalidAllocationCallbackHandle, HCpp);

  luthierKfdRemoveAllocationCallback(HCpp);
  luthier::kfd::runAllocationCallbacks(
      Allocation{0x1000, 0x100, 0x1u, 38979u, 7, 0});
  EXPECT_EQ(0, Fired) << "the entry was not actually unhooked, so the ownership "
                         "test above proves nothing";
}

} // namespace
