//===-- CallbackChainTest.cpp - order and composition of packet callbacks -===//
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
/// "Last registered runs first" is exactly the kind of claim that gets written
/// in a comment and never checked, so it is checked here -- with no GPU, no
/// queue and no driver, because the ordering does not depend on any of them.
///
/// The chain walk is a free function over a plain array precisely so this file
/// can exist. Each test below was confirmed to fail against a deliberately
/// reversed walk before being kept.
//===----------------------------------------------------------------------===//
#include "luthier/KFD/QueueWrapper.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using luthier::kfd::QueueInfo;
using luthier::kfd::detail::CallbackEntry;
using luthier::kfd::detail::runCallbackChain;

namespace {

/// What the callbacks did, in the order they did it.
struct Trace {
  std::vector<std::string> Calls;
  std::vector<uint64_t> Indices;
};

/// A callback that records its own name. \c UserData carries a Recorder.
struct Recorder {
  Trace *T;
  std::string Name;
};

void record(const QueueInfo &, uint64_t Index, luthier::hsa::AqlPacket &,
            void *UserData) {
  auto *R = static_cast<Recorder *>(UserData);
  R->T->Calls.push_back(R->Name);
  R->T->Indices.push_back(Index);
}

/// A callback that appends to the packet's header, so a later assertion can
/// tell whether it saw an earlier callback's edit.
void addOneToHeader(const QueueInfo &, uint64_t, luthier::hsa::AqlPacket &P,
                    void *) {
  P.Packet.Header = static_cast<uint16_t>(P.Packet.Header + 1);
}

void doubleHeader(const QueueInfo &, uint64_t, luthier::hsa::AqlPacket &P,
                  void *) {
  P.Packet.Header = static_cast<uint16_t>(P.Packet.Header * 2);
}

QueueInfo someQueue() {
  QueueInfo Q{};
  Q.GpuId = 1;
  Q.QueueId = 2;
  Q.RingByteSize = 4096;
  Q.SlotCount = 64;
  return Q;
}

luthier::hsa::AqlPacket packetWithHeader(uint16_t H) {
  luthier::hsa::AqlPacket P{};
  P.Packet.Header = H;
  return P;
}

} // namespace

TEST(CallbackChain, RunsTheLastRegisteredFirst) {
  Trace T;
  Recorder First{&T, "first"};
  Recorder Second{&T, "second"};
  Recorder Third{&T, "third"};

  // Registration order: first, second, third.
  CallbackEntry Entries[] = {{record, &First}, {record, &Second},
                             {record, &Third}};

  auto P = packetWithHeader(2);
  runCallbackChain(Entries, 3, someQueue(), 7, P);

  // ROCr's order: the most recently attached tool sees the packet as the
  // application wrote it, and the earliest-attached tool sees it last, just
  // before the GPU does.
  const std::vector<std::string> Expected{"third", "second", "first"};
  EXPECT_EQ(T.Calls, Expected);
}

TEST(CallbackChain, EachCallbackSeesThePreviousOnesEdits) {
  // The order matters arithmetically here, which is the point: +1 then double
  // gives a different answer from double then +1, so this cannot pass under a
  // reversed walk by coincidence.
  //
  // Registered as {addOne, double} means double runs first: (3 * 2) + 1 == 7.
  CallbackEntry Entries[] = {{addOneToHeader, nullptr},
                             {doubleHeader, nullptr}};

  auto P = packetWithHeader(3);
  runCallbackChain(Entries, 2, someQueue(), 0, P);
  EXPECT_EQ(P.Packet.Header, 7);
}

TEST(CallbackChain, TheOppositeRegistrationOrderGivesTheOppositeAnswer) {
  // The companion to the test above. Together they pin the direction down: a
  // reversed walk would swap both answers, so neither can be satisfied by
  // accident.
  //
  // Registered as {double, addOne} means addOne runs first: (3 + 1) * 2 == 8.
  CallbackEntry Entries[] = {{doubleHeader, nullptr},
                             {addOneToHeader, nullptr}};

  auto P = packetWithHeader(3);
  runCallbackChain(Entries, 2, someQueue(), 0, P);
  EXPECT_EQ(P.Packet.Header, 8);
}

TEST(CallbackChain, SkipsRemovedCallbacksWithoutDisturbingTheRest) {
  Trace T;
  Recorder First{&T, "first"};
  Recorder Third{&T, "third"};

  // A null entry is how removal is expressed: the hole stays, so the callbacks
  // around it keep both their order and their handles.
  CallbackEntry Entries[] = {
      {record, &First}, {nullptr, nullptr}, {record, &Third}};

  auto P = packetWithHeader(2);
  runCallbackChain(Entries, 3, someQueue(), 0, P);

  const std::vector<std::string> Expected{"third", "first"};
  EXPECT_EQ(T.Calls, Expected);
}

TEST(CallbackChain, AnEmptyChainLeavesThePacketAlone) {
  // Not a trivial case: this is what every queue does before a tool attaches,
  // and the packet must reach the GPU byte-for-byte as the application wrote
  // it.
  CallbackEntry Entries[1] = {{nullptr, nullptr}};
  auto P = packetWithHeader(0x1234);
  runCallbackChain(Entries, 0, someQueue(), 0, P);
  EXPECT_EQ(P.Packet.Header, 0x1234);
}

TEST(CallbackChain, EveryCallbackSeesTheSamePacketIndex) {
  // A chain must not renumber packets between tools: two tools counting
  // dispatches have to agree on which dispatch they are looking at.
  Trace T;
  Recorder A{&T, "a"};
  Recorder B{&T, "b"};
  CallbackEntry Entries[] = {{record, &A}, {record, &B}};

  auto P = packetWithHeader(2);
  runCallbackChain(Entries, 2, someQueue(), 41, P);

  ASSERT_EQ(T.Indices.size(), 2u);
  EXPECT_EQ(T.Indices[0], 41u);
  EXPECT_EQ(T.Indices[1], 41u);
}

//===----------------------------------------------------------------------===//
// Registration bookkeeping
//
// The tests above exercise the walk over a hand-built array. These exercise the
// real registry -- handles, the published count, holes left by removal, and what
// setPacketCallback does to a chain that already exists. That bookkeeping is
// where this kind of code goes wrong, and none of it is reachable from the walk
// alone.
//===----------------------------------------------------------------------===//

namespace {

/// Each test starts from a known-empty registry. The registry is process-wide
/// static state, so leaving it dirty would make these tests order-dependent --
/// the exact property this suite has already been bitten by twice.
class CallbackRegistry : public ::testing::Test {
protected:
  void SetUp() override { luthier::kfd::setPacketCallback(nullptr, nullptr); }
  void TearDown() override { luthier::kfd::setPacketCallback(nullptr, nullptr); }

  /// Run the real registered chain and return who ran, in order.
  std::vector<std::string> fire() {
    T.Calls.clear();
    auto P = packetWithHeader(2);
    luthier::kfd::runRegisteredCallbacks(someQueue(), 0, P);
    return T.Calls;
  }

  Trace T;
};

} // namespace

TEST_F(CallbackRegistry, AddedCallbacksRunLastRegisteredFirst) {
  Recorder A{&T, "a"}, B{&T, "b"};
  ASSERT_GE(luthier::kfd::addPacketCallback(record, &A), 0);
  ASSERT_GE(luthier::kfd::addPacketCallback(record, &B), 0);

  const std::vector<std::string> Expected{"b", "a"};
  EXPECT_EQ(fire(), Expected);
}

TEST_F(CallbackRegistry, RemovingOneLeavesTheOthersAloneAndKeepsTheirHandles) {
  Recorder A{&T, "a"}, B{&T, "b"}, C{&T, "c"};
  const auto HA = luthier::kfd::addPacketCallback(record, &A);
  const auto HB = luthier::kfd::addPacketCallback(record, &B);
  const auto HC = luthier::kfd::addPacketCallback(record, &C);
  ASSERT_GE(HA, 0);
  ASSERT_GE(HB, 0);
  ASSERT_GE(HC, 0);

  luthier::kfd::removePacketCallback(HB);

  const std::vector<std::string> AfterRemoval{"c", "a"};
  EXPECT_EQ(fire(), AfterRemoval);

  // The survivors' handles must still be theirs: removing B and then removing
  // C must not take out A because indices shifted.
  luthier::kfd::removePacketCallback(HC);
  const std::vector<std::string> OnlyA{"a"};
  EXPECT_EQ(fire(), OnlyA);
}

TEST_F(CallbackRegistry, AnAddReusesTheHoleLeftByARemoval) {
  // Otherwise a tool that repeatedly attaches and detaches exhausts the array,
  // and the failure is silent: the callback is simply never invoked again.
  Recorder A{&T, "a"}, B{&T, "b"}, D{&T, "d"};
  const auto HA = luthier::kfd::addPacketCallback(record, &A);
  const auto HB = luthier::kfd::addPacketCallback(record, &B);
  ASSERT_GE(HA, 0);
  ASSERT_GE(HB, 0);

  luthier::kfd::removePacketCallback(HA);
  const auto HD = luthier::kfd::addPacketCallback(record, &D);
  EXPECT_EQ(HD, HA) << "the freed slot should have been reused";

  // D now sits where A did -- earlier in the array than B -- so B still runs
  // first. Position in the chain follows the slot, not the time of the call.
  const std::vector<std::string> Expected{"b", "d"};
  EXPECT_EQ(fire(), Expected);
}

TEST_F(CallbackRegistry, RegistrationSurvivesRepeatedAttachAndDetach) {
  // Many more cycles than the array has slots. If removal did not free the
  // slot this stops registering partway through and says nothing about it.
  Recorder A{&T, "a"};
  for (unsigned I = 0; I < luthier::kfd::MaxPacketCallbacks * 4; I++) {
    const auto H = luthier::kfd::addPacketCallback(record, &A);
    ASSERT_GE(H, 0) << "ran out of callback slots on cycle " << I;
    luthier::kfd::removePacketCallback(H);
  }
  EXPECT_TRUE(fire().empty());
}

TEST_F(CallbackRegistry, SetReplacesTheWholeChain) {
  Recorder A{&T, "a"}, B{&T, "b"}, Only{&T, "only"};
  ASSERT_GE(luthier::kfd::addPacketCallback(record, &A), 0);
  ASSERT_GE(luthier::kfd::addPacketCallback(record, &B), 0);

  luthier::kfd::setPacketCallback(record, &Only);

  const std::vector<std::string> Expected{"only"};
  EXPECT_EQ(fire(), Expected);
}

TEST_F(CallbackRegistry, AddingAfterSetDoesNotResurrectTheReplacedCallbacks) {
  // set() shrinks the published count. An add that then appends must land on a
  // slot that is genuinely free, not on one still holding a replaced entry.
  Recorder A{&T, "a"}, B{&T, "b"}, C{&T, "c"}, Only{&T, "only"}, Extra{&T, "extra"};
  ASSERT_GE(luthier::kfd::addPacketCallback(record, &A), 0);
  ASSERT_GE(luthier::kfd::addPacketCallback(record, &B), 0);
  ASSERT_GE(luthier::kfd::addPacketCallback(record, &C), 0);

  luthier::kfd::setPacketCallback(record, &Only);
  ASSERT_GE(luthier::kfd::addPacketCallback(record, &Extra), 0);

  const std::vector<std::string> Expected{"extra", "only"};
  EXPECT_EQ(fire(), Expected);
}

TEST_F(CallbackRegistry, RefusesMoreThanTheArrayHolds) {
  // And says so, rather than returning a handle that never fires.
  Recorder R{&T, "r"};
  for (unsigned I = 0; I < luthier::kfd::MaxPacketCallbacks; I++)
    ASSERT_GE(luthier::kfd::addPacketCallback(record, &R), 0);
  EXPECT_EQ(luthier::kfd::addPacketCallback(record, &R),
            luthier::kfd::InvalidCallbackHandle);
}

TEST_F(CallbackRegistry, ANullCallbackIsRejectedRatherThanStored) {
  EXPECT_EQ(luthier::kfd::addPacketCallback(nullptr, nullptr),
            luthier::kfd::InvalidCallbackHandle);
}

TEST_F(CallbackRegistry, RemovingAnOutOfRangeHandleIsHarmless) {
  Recorder A{&T, "a"};
  ASSERT_GE(luthier::kfd::addPacketCallback(record, &A), 0);
  luthier::kfd::removePacketCallback(-1);
  luthier::kfd::removePacketCallback(9999);
  const std::vector<std::string> Expected{"a"};
  EXPECT_EQ(fire(), Expected);
}
