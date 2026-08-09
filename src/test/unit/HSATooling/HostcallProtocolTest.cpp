//===-- HostcallProtocolTest.cpp ------------------------------------------===//
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
/// Exercises the hostcall protocol against a buffer in ordinary host memory,
/// with this test standing in for the device: it pops packets off the free
/// stack, fills them the way a wave would, pushes them onto the ready stack,
/// and then checks what \c HostcallBuffer::processPackets wrote back.
///
/// None of this needs a GPU, which is the point — the packet loop, the tagged
/// pointers, the message reassembly and the services are all host logic, and
/// this is the only place they can be tested deterministically. The one thing
/// it cannot cover is whether real device code agrees about the layout, so the
/// first test asserts that layout explicitly against an independent
/// declaration of what the device library expects to see.
//===----------------------------------------------------------------------===//
#include "luthier/HSATooling/HostcallHandler.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <string>
#include <vector>

using namespace luthier;

namespace {

//===----------------------------------------------------------------------===//
// The device's view of a hostcall buffer
//===----------------------------------------------------------------------===//

/// Independent transcription of \c buffer_t, the struct the ROCm device
/// libraries overlay on a hostcall buffer (\c ockl/src/hostcall_impl.cl).
/// Declared here rather than reused from \c HostcallBuffer on purpose: if the
/// two ever disagree, the layout assertions below are what catches it.
struct DeviceBufferView {
  HostcallPacketHeader *Headers;
  HostcallPayload *Payloads;
  uint64_t Doorbell;
  uint64_t FreeStack;
  std::atomic<uint64_t> ReadyStack;
  uint64_t IndexMask;
};

constexpr uint32_t ControlReadyFlag = 1U << 0;

/// A hostcall buffer plus the storage behind it, with the helpers this test
/// needs to act like a device wave.
class TestBuffer {
public:
  explicit TestBuffer(uint32_t NumPackets) : NumPackets(NumPackets) {
    const size_t Size = HostcallBuffer::getRequiredSize(NumPackets);
    Storage.assign(Size, 0);
    Buffer = reinterpret_cast<HostcallBuffer *>(Storage.data());
    // The services this test drives (printf, function call) never touch the
    // AMD extension table, so a zeroed one is enough to build the state they
    // hang off. A devmem request would need a real table and a real pool, and
    // is covered by the end-to-end loader test instead.
    Services = std::make_unique<HostcallServiceState>(
        hsa::ApiTableContainer<::AmdExtTable>(DummyExtTable), hsa_agent_t{0},
        hsa_amd_memory_pool_t{0}, /*DeviceMemoryPoolAlignment=*/4096);
    llvm::Error Err = Buffer->initialize(NumPackets, *Services);
    EXPECT_FALSE(static_cast<bool>(Err));
    llvm::consumeError(std::move(Err));
  }

  HostcallBuffer &get() const { return *Buffer; }

  DeviceBufferView &deviceView() const {
    return *reinterpret_cast<DeviceBufferView *>(Storage.data());
  }

  /// Pops a packet off the free stack the way a wave does, and returns its
  /// tagged pointer. Single-threaded, so no CAS loop is needed.
  uint64_t popFreePacket() const {
    DeviceBufferView &View = deviceView();
    const uint64_t Popped = View.FreeStack;
    EXPECT_NE(Popped, 0u) << "the free stack ran dry";
    View.FreeStack = View.Headers[Popped & View.IndexMask].Next;
    return Popped;
  }

  /// Fills a packet in the way a wave submitting \p Service would, and pushes
  /// it onto the ready stack. \p LanePayloads supplies the eight-word payload
  /// of every lane named in \p ActiveMask, in lane order.
  uint64_t submit(uint32_t Service, uint64_t ActiveMask,
                  llvm::ArrayRef<std::array<uint64_t, 8>> LanePayloads) const {
    DeviceBufferView &View = deviceView();
    const uint64_t Ptr = popFreePacket();
    const uint64_t Index = Ptr & View.IndexMask;

    HostcallPacketHeader &Header = View.Headers[Index];
    Header.Service = Service;
    Header.ActiveMask = ActiveMask;

    HostcallPayload &Payload = View.Payloads[Index];
    size_t Which = 0;
    for (unsigned Lane = 0; Lane < 64; ++Lane) {
      if ((ActiveMask & (uint64_t{1} << Lane)) == 0)
        continue;
      EXPECT_LT(Which, LanePayloads.size());
      std::memcpy(Payload.Slots[Lane], LanePayloads[Which].data(),
                  sizeof(Payload.Slots[Lane]));
      ++Which;
    }

    Header.Control.store(ControlReadyFlag, std::memory_order_relaxed);
    Header.Next = View.ReadyStack.load(std::memory_order_relaxed);
    View.ReadyStack.store(Ptr, std::memory_order_release);
    return Ptr;
  }

  /// Pushes a packet back onto the free stack the way a wave does once it has
  /// seen the host clear the packet's READY flag. The host never recycles
  /// packets itself, so a test that sends more messages than the buffer has
  /// packets has to do this or the free stack runs dry.
  void releasePacket(uint64_t Ptr) const {
    DeviceBufferView &View = deviceView();
    // The tag is bumped on every push so a recycled pointer is never mistaken
    // for the null one that terminates the stack, matching the non-zero tag
    // initialize() gives the deepest entry.
    const uint64_t TagIncrement = View.IndexMask + 1;
    uint64_t Tagged = Ptr + TagIncrement;
    if (Tagged == 0)
      Tagged = TagIncrement;
    View.Headers[Tagged & View.IndexMask].Next = View.FreeStack;
    View.FreeStack = Tagged;
  }

  bool isReady(uint64_t Ptr) const {
    DeviceBufferView &View = deviceView();
    return (View.Headers[Ptr & View.IndexMask].Control.load(
                std::memory_order_acquire) &
            ControlReadyFlag) != 0;
  }

  const uint64_t *lanePayload(uint64_t Ptr, unsigned Lane) const {
    DeviceBufferView &View = deviceView();
    return View.Payloads[Ptr & View.IndexMask].Slots[Lane];
  }

  uint32_t packetCount() const { return NumPackets; }

private:
  uint32_t NumPackets;
  mutable std::vector<uint8_t> Storage;
  HostcallBuffer *Buffer{nullptr};
  ::AmdExtTable DummyExtTable{};
  std::unique_ptr<HostcallServiceState> Services;
};

//===----------------------------------------------------------------------===//
// Message descriptor construction
//===----------------------------------------------------------------------===//

/// Builds the word-0 descriptor a wave puts at the head of a printf payload.
uint64_t makeDescriptor(bool Begin, bool End, uint64_t Len, uint64_t ID = 0) {
  return (Begin ? uint64_t{1} << 0 : 0) | (End ? uint64_t{1} << 1 : 0) |
         (Len << 5) | (ID << 8);
}

uint64_t descriptorID(uint64_t Descriptor) { return Descriptor >> 8; }

/// Packs a NUL-terminated format string plus already-encoded arguments into
/// the word stream a printf message body is made of.
std::vector<uint64_t> makePrintfBody(llvm::StringRef Format,
                                     llvm::ArrayRef<uint64_t> Args) {
  const size_t FormatWords = (Format.size() + 1 + 7) / 8;
  std::vector<uint64_t> Body(FormatWords, 0);
  std::memcpy(Body.data(), Format.data(), Format.size());
  Body.insert(Body.end(), Args.begin(), Args.end());
  return Body;
}

/// Runs \p Body through the buffer as one or more printf hostcalls, seven
/// content words at a time, and returns what the services printed.
std::string
servicePrintfMessage(TestBuffer &Buffer, llvm::ArrayRef<uint64_t> Body,
                     uint64_t ControlWord = 0) {
  // A printf message leads with a control word selecting the stream.
  std::vector<uint64_t> Stream;
  Stream.push_back(ControlWord);
  Stream.insert(Stream.end(), Body.begin(), Body.end());

  testing::internal::CaptureStdout();
  uint64_t MessageID = 0;
  for (size_t Sent = 0; Sent < Stream.size();) {
    const size_t Len = std::min<size_t>(7, Stream.size() - Sent);
    const bool Begin = Sent == 0;
    const bool End = Sent + Len == Stream.size();

    std::array<uint64_t, 8> Slot{};
    Slot[0] = makeDescriptor(Begin, End, Len, Begin ? 0 : MessageID);
    for (size_t I = 0; I < Len; ++I)
      Slot[1 + I] = Stream[Sent + I];

    const uint64_t Ptr =
        Buffer.submit(HOSTCALL_SERVICE_PRINTF, /*ActiveMask=*/1, {Slot});
    Buffer.get().processPackets();
    EXPECT_FALSE(Buffer.isReady(Ptr));

    if (Begin)
      MessageID = descriptorID(Buffer.lanePayload(Ptr, 0)[0]);
    // Hand the packet back now that its response has been read, so a long run
    // of messages keeps reusing the same few packets.
    Buffer.releasePacket(Ptr);
    Sent += Len;
  }
  std::fflush(stdout);
  return testing::internal::GetCapturedStdout();
}

//===----------------------------------------------------------------------===//
// Buffer layout
//===----------------------------------------------------------------------===//

// The device library indexes the buffer's leading fields at fixed offsets, so
// nothing may be reordered ahead of or between them.
TEST(HostcallBufferLayout, MatchesTheDeviceLibraryStruct) {
  EXPECT_EQ(offsetof(DeviceBufferView, Headers), 0u);
  EXPECT_EQ(offsetof(DeviceBufferView, Payloads), 8u);
  EXPECT_EQ(offsetof(DeviceBufferView, Doorbell), 16u);
  EXPECT_EQ(offsetof(DeviceBufferView, FreeStack), 24u);
  EXPECT_EQ(offsetof(DeviceBufferView, ReadyStack), 32u);
  EXPECT_EQ(offsetof(DeviceBufferView, IndexMask), 40u);

  // A payload is one eight-word slot per lane of a wave64, and a header is
  // the two link words plus the service and control fields.
  EXPECT_EQ(sizeof(HostcallPayload), 64u * 8u * sizeof(uint64_t));
  EXPECT_EQ(sizeof(HostcallPacketHeader), 24u);
}

TEST(HostcallBufferLayout, PacketArraysLieInsideTheAllocation) {
  constexpr uint32_t NumPackets = 4;
  TestBuffer Buffer(NumPackets);
  const auto *Base = reinterpret_cast<const uint8_t *>(&Buffer.get());
  const size_t Size = HostcallBuffer::getRequiredSize(NumPackets);

  DeviceBufferView &View = Buffer.deviceView();
  const auto *Headers = reinterpret_cast<const uint8_t *>(View.Headers);
  const auto *Payloads = reinterpret_cast<const uint8_t *>(View.Payloads);

  EXPECT_GE(Headers, Base + sizeof(HostcallBuffer));
  EXPECT_GE(Payloads, Headers + NumPackets * sizeof(HostcallPacketHeader));
  EXPECT_LE(Payloads + NumPackets * sizeof(HostcallPayload), Base + Size);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(Payloads) % alignof(HostcallPayload),
            0u);
}

TEST(HostcallBufferInit, IndexMaskAndStacksStartCorrect) {
  TestBuffer Buffer(8);
  DeviceBufferView &View = Buffer.deviceView();
  EXPECT_EQ(View.IndexMask, 7u);
  EXPECT_EQ(View.ReadyStack.load(), 0u) << "nothing is ready before a submit";
  EXPECT_NE(View.FreeStack, 0u) << "every packet starts free";
}

// The free stack must reach every packet exactly once, and no tagged pointer
// on it may be zero — zero is how a stack says it is empty, so a packet whose
// index and tag were both zero would be lost.
TEST(HostcallBufferInit, FreeStackThreadsEveryPacketExactlyOnce) {
  constexpr uint32_t NumPackets = 8;
  TestBuffer Buffer(NumPackets);
  DeviceBufferView &View = Buffer.deviceView();

  std::vector<bool> Seen(NumPackets, false);
  uint64_t Ptr = View.FreeStack;
  unsigned Count = 0;
  while (Ptr != 0 && Count <= NumPackets) {
    const uint64_t Index = Ptr & View.IndexMask;
    ASSERT_LT(Index, NumPackets);
    EXPECT_FALSE(Seen[Index]) << "packet " << Index << " is on the stack twice";
    Seen[Index] = true;
    ++Count;
    Ptr = View.Headers[Index].Next;
  }
  EXPECT_EQ(Count, NumPackets);
  for (unsigned I = 0; I < NumPackets; ++I)
    EXPECT_TRUE(Seen[I]) << "packet " << I << " is unreachable";
}

TEST(HostcallBufferInit, RejectsPacketCountsItCannotIndex) {
  std::vector<uint8_t> Storage(HostcallBuffer::getRequiredSize(4), 0);
  auto *Buffer = reinterpret_cast<HostcallBuffer *>(Storage.data());
  ::AmdExtTable DummyExtTable{};
  HostcallServiceState Services(
      hsa::ApiTableContainer<::AmdExtTable>(DummyExtTable), hsa_agent_t{0},
      hsa_amd_memory_pool_t{0}, 4096);

  for (uint32_t Bad : {0u, 1u, 3u, 6u}) {
    llvm::Error Err = Buffer->initialize(Bad, Services);
    EXPECT_TRUE(static_cast<bool>(Err)) << "accepted " << Bad << " packets";
    llvm::consumeError(std::move(Err));
  }
}

//===----------------------------------------------------------------------===//
// Packet servicing
//===----------------------------------------------------------------------===//

// Clearing the ready flag is what unblocks the submitting wave, so it has to
// happen for every packet the host takes off the ready stack — including ones
// it could not service, or the wave hangs forever.
TEST(HostcallServicing, UnknownServiceStillReleasesThePacket) {
  TestBuffer Buffer(4);
  std::array<uint64_t, 8> Slot{};
  constexpr uint32_t UnknownService = 48879;
  const uint64_t Ptr = Buffer.submit(UnknownService, /*ActiveMask=*/1, {Slot});

  testing::internal::CaptureStderr();
  Buffer.get().processPackets();
  const std::string Diagnostics = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(Buffer.isReady(Ptr)) << "an unserviceable packet must not hang "
                                       "the wave that submitted it";
  EXPECT_NE(Diagnostics.find(std::to_string(UnknownService)),
            std::string::npos)
      << "the unhandled service should be reported; got: " << Diagnostics;
}

TEST(HostcallServicing, AnEmptyReadyStackIsANoOp) {
  TestBuffer Buffer(4);
  Buffer.get().processPackets();
  EXPECT_EQ(Buffer.deviceView().ReadyStack.load(), 0u);
}

TEST(HostcallServicing, DrainsEveryPacketOnTheStack) {
  TestBuffer Buffer(8);
  std::array<uint64_t, 8> Slot{};
  llvm::SmallVector<uint64_t, 4> Submitted;
  for (unsigned I = 0; I < 4; ++I)
    Submitted.push_back(Buffer.submit(/*Service=*/48879, /*ActiveMask=*/1,
                                      {Slot}));

  testing::internal::CaptureStderr();
  Buffer.get().processPackets();
  (void)testing::internal::GetCapturedStderr();

  for (uint64_t Ptr : Submitted)
    EXPECT_FALSE(Buffer.isReady(Ptr));
  EXPECT_EQ(Buffer.deviceView().ReadyStack.load(), 0u);
}

//===----------------------------------------------------------------------===//
// Function-call service
//===----------------------------------------------------------------------===//

// The function-call service hands the payload's trailing seven words to a host
// function pointer and copies its two results back over the head of the slot.
uint64_t FunctionCallInputs[7];
void testHostFunction(uint64_t *Output, const uint64_t *Input) {
  std::memcpy(FunctionCallInputs, Input, sizeof(FunctionCallInputs));
  Output[0] = Input[0] + Input[1];
  Output[1] = 0xA5A5A5A5A5A5A5A5ULL;
}

TEST(HostcallFunctionCall, InvokesTheHostFunctionAndReturnsItsResults) {
  TestBuffer Buffer(4);
  std::memset(FunctionCallInputs, 0, sizeof(FunctionCallInputs));

  std::array<uint64_t, 8> Slot{};
  Slot[0] = reinterpret_cast<uintptr_t>(&testHostFunction);
  for (unsigned I = 1; I < 8; ++I)
    Slot[I] = 100 + I;

  const uint64_t Ptr =
      Buffer.submit(HOSTCALL_SERVICE_FUNCTION_CALL, /*ActiveMask=*/1, {Slot});
  Buffer.get().processPackets();

  EXPECT_FALSE(Buffer.isReady(Ptr));
  for (unsigned I = 0; I < 7; ++I)
    EXPECT_EQ(FunctionCallInputs[I], 100u + I + 1);
  EXPECT_EQ(Buffer.lanePayload(Ptr, 0)[0], 101u + 102u);
  EXPECT_EQ(Buffer.lanePayload(Ptr, 0)[1], 0xA5A5A5A5A5A5A5A5ULL);
}

TEST(HostcallFunctionCall, ANullFunctionPointerIsReportedNotCalled) {
  TestBuffer Buffer(4);
  std::array<uint64_t, 8> Slot{};
  Slot[0] = 0;

  testing::internal::CaptureStderr();
  const uint64_t Ptr =
      Buffer.submit(HOSTCALL_SERVICE_FUNCTION_CALL, /*ActiveMask=*/1, {Slot});
  Buffer.get().processPackets();
  const std::string Diagnostics = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(Buffer.isReady(Ptr));
  EXPECT_NE(Diagnostics.find("null function pointer"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Printf service
//===----------------------------------------------------------------------===//

TEST(HostcallPrintf, RendersAMessageThatFitsOnePacket) {
  TestBuffer Buffer(4);
  // "hi\n" is three bytes, so format plus one argument fits in the seven
  // content words a single hostcall carries.
  const std::vector<uint64_t> Body = makePrintfBody("v=%d\n", {uint64_t{42}});
  EXPECT_EQ(servicePrintfMessage(Buffer, Body), "v=42\n");
}

TEST(HostcallPrintf, ReportsTheCharacterCountToTheDevice) {
  TestBuffer Buffer(4);
  const std::vector<uint64_t> Body = makePrintfBody("abcd", {});

  std::vector<uint64_t> Stream{0};
  Stream.insert(Stream.end(), Body.begin(), Body.end());
  ASSERT_LE(Stream.size(), 7u);

  std::array<uint64_t, 8> Slot{};
  Slot[0] = makeDescriptor(/*Begin=*/true, /*End=*/true, Stream.size());
  for (size_t I = 0; I < Stream.size(); ++I)
    Slot[1 + I] = Stream[I];

  testing::internal::CaptureStdout();
  const uint64_t Ptr =
      Buffer.submit(HOSTCALL_SERVICE_PRINTF, /*ActiveMask=*/1, {Slot});
  Buffer.get().processPackets();
  std::fflush(stdout);
  const std::string Out = testing::internal::GetCapturedStdout();

  EXPECT_EQ(Out, "abcd");
  EXPECT_EQ(Buffer.lanePayload(Ptr, 0)[0], 4u) << "printf returns its count";
}

// A message longer than seven words arrives as a run of hostcalls: the host
// allocates an id on the BEGIN packet, hands it back through the payload, and
// the device quotes it on every packet that follows.
TEST(HostcallPrintf, ReassemblesAMessageSplitAcrossPackets) {
  TestBuffer Buffer(4);
  const std::string LongFormat =
      "a very long format string that will not fit in one hostcall payload "
      "no matter how it is sliced: %d %d %d\n";
  const std::vector<uint64_t> Body =
      makePrintfBody(LongFormat, {uint64_t{1}, uint64_t{2}, uint64_t{3}});
  ASSERT_GT(Body.size(), 7u) << "this test needs a multi-packet message";

  const std::string Expected =
      "a very long format string that will not fit in one hostcall payload "
      "no matter how it is sliced: 1 2 3\n";
  EXPECT_EQ(servicePrintfMessage(Buffer, Body), Expected);
}

// The host answers a BEGIN packet by clearing the flag and writing the id it
// allocated back into the descriptor, which is how the device learns what to
// quote on the packets that follow.
TEST(HostcallPrintf, AnswersBeginByClearingTheFlagAndReturningTheDescriptor) {
  TestBuffer Buffer(4);
  std::array<uint64_t, 8> Slot{};
  Slot[0] = makeDescriptor(/*Begin=*/true, /*End=*/false, /*Len=*/1);
  Slot[1] = 0; // control word

  const uint64_t Ptr =
      Buffer.submit(HOSTCALL_SERVICE_PRINTF, /*ActiveMask=*/1, {Slot});
  Buffer.get().processPackets();

  const uint64_t Returned = Buffer.lanePayload(Ptr, 0)[0];
  EXPECT_FALSE(Buffer.isReady(Ptr));
  EXPECT_EQ(Returned & 1u, 0u) << "the BEGIN flag must be cleared";
  EXPECT_EQ((Returned >> 1) & 1u, 0u) << "END was not set on this packet";
  // The length field is left as the device wrote it; only the flag and the id
  // are the host's to change.
  EXPECT_EQ((Returned >> 5) & 0x7u, 1u);
}

// Continuing a message the host is not assembling must be reported rather than
// silently appended to some unrelated message.
TEST(HostcallPrintf, RejectsAContinuationOfAnUnknownMessage) {
  TestBuffer Buffer(4);
  std::array<uint64_t, 8> Slot{};
  Slot[0] = makeDescriptor(/*Begin=*/false, /*End=*/true, /*Len=*/1,
                           /*ID=*/9999);

  testing::internal::CaptureStderr();
  const uint64_t Ptr =
      Buffer.submit(HOSTCALL_SERVICE_PRINTF, /*ActiveMask=*/1, {Slot});
  Buffer.get().processPackets();
  const std::string Diagnostics = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(Buffer.isReady(Ptr));
  EXPECT_NE(Diagnostics.find("9999"), std::string::npos)
      << "got: " << Diagnostics;
}

// Every lane named in the active mask carries its own independent message, so
// one packet from a wave where several lanes printed must render all of them.
TEST(HostcallPrintf, ServicesEveryActiveLaneOfAPacket) {
  TestBuffer Buffer(4);
  constexpr unsigned Lanes[] = {0, 3, 63};

  llvm::SmallVector<std::array<uint64_t, 8>, 3> Payloads;
  for (unsigned Lane : Lanes) {
    const std::vector<uint64_t> Body =
        makePrintfBody("L%d\n", {static_cast<uint64_t>(Lane)});
    std::vector<uint64_t> Stream{0};
    Stream.insert(Stream.end(), Body.begin(), Body.end());
    EXPECT_LE(Stream.size(), 7u);

    std::array<uint64_t, 8> Slot{};
    Slot[0] = makeDescriptor(true, true, Stream.size());
    for (size_t I = 0; I < Stream.size(); ++I)
      Slot[1 + I] = Stream[I];
    Payloads.push_back(Slot);
  }

  const uint64_t ActiveMask =
      (uint64_t{1} << 0) | (uint64_t{1} << 3) | (uint64_t{1} << 63);

  testing::internal::CaptureStdout();
  const uint64_t Ptr =
      Buffer.submit(HOSTCALL_SERVICE_PRINTF, ActiveMask, Payloads);
  Buffer.get().processPackets();
  std::fflush(stdout);
  const std::string Out = testing::internal::GetCapturedStdout();

  EXPECT_FALSE(Buffer.isReady(Ptr));
  EXPECT_NE(Out.find("L0\n"), std::string::npos) << "got: " << Out;
  EXPECT_NE(Out.find("L3\n"), std::string::npos) << "got: " << Out;
  EXPECT_NE(Out.find("L63\n"), std::string::npos) << "got: " << Out;
}

// Message slots are recycled, so a long run of printfs must not grow the
// assembler's table without bound or start losing ids.
TEST(HostcallPrintf, RecyclesMessageSlotsAcrossManyMessages) {
  TestBuffer Buffer(4);
  for (unsigned I = 0; I < 64; ++I) {
    const std::vector<uint64_t> Body =
        makePrintfBody("%d\n", {static_cast<uint64_t>(I)});
    EXPECT_EQ(servicePrintfMessage(Buffer, Body), std::to_string(I) + "\n");
  }
}

} // namespace
