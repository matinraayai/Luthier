//===-- HiddenArgumentsTest.cpp -------------------------------------------===//
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
/// Tests the hidden kernel argument fill against a hand-built kernarg segment.
/// The values a dispatch owes its kernel are derived from the dispatch packet
/// and the queue, both of which a test can supply directly, so none of this
/// needs a GPU.
///
/// The layouts asserted here are ABI: \c DeviceGridSyncInfo and
/// \c DeviceAqlWrap are read by the ROCm device libraries and the
/// device-enqueue scheduler respectively, so their sizes and offsets are
/// pinned rather than merely exercised.
//===----------------------------------------------------------------------===//
#include "luthier/HSATooling/HiddenArgBuffers.h"
#include "luthier/HSATooling/InstrumentedKernelLoaderAndLauncher.h"

#include <gtest/gtest.h>

#include <cstring>
#include <hsa/amd_hsa_queue.h>
#include <llvm/Support/Endian.h>
#include <llvm/Support/Error.h>
#include <vector>

using namespace luthier;

namespace {

//===----------------------------------------------------------------------===//
// Access to the loader's protected statics
//===----------------------------------------------------------------------===//

/// The hidden-argument fill is a protected static of the loader, which is
/// where it belongs — it is an implementation detail of a dispatch. Deriving
/// re-exports it for the test without widening the public surface.
struct LauncherAccess : InstrumentedKernelLoaderAndLauncher {
  using HiddenArgInfo = InstrumentedKernelLoaderAndLauncher::HiddenArgInfo;
  using HiddenArgBufferAddresses =
      InstrumentedKernelLoaderAndLauncher::HiddenArgBufferAddresses;
  using LoadedKernelInfo =
      InstrumentedKernelLoaderAndLauncher::LoadedKernelInfo;
  using InstrumentedKernelLoaderAndLauncher::declaresHiddenArg;
  using InstrumentedKernelLoaderAndLauncher::fillExtendedKernargBuffer;
  using InstrumentedKernelLoaderAndLauncher::writeHiddenKernelArguments;
};

using ValueKind = amdgpu::hsamd::ValueKind;
using HiddenArgInfo = LauncherAccess::HiddenArgInfo;

constexpr uint32_t PrivateApertureHi = 0x7FFF0000u;
constexpr uint32_t GroupApertureHi = 0x7FFE0000u;

/// A queue whose AMD extension fields carry recognizable apertures, so the
/// test can tell the two base arguments apart.
class TestQueue {
public:
  TestQueue() {
    std::memset(&AmdQueue, 0, sizeof(AmdQueue));
    AmdQueue.private_segment_aperture_base_hi = PrivateApertureHi;
    AmdQueue.group_segment_aperture_base_hi = GroupApertureHi;
  }
  const hsa_queue_t &get() const { return AmdQueue.hsa_queue; }

private:
  amd_queue_t AmdQueue;
};

/// A one-work-item dispatch packet, the only shape the loader launches.
hsa_kernel_dispatch_packet_t makeSingleWorkItemPacket() {
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.setup = 1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Packet.workgroup_size_x = 1;
  Packet.workgroup_size_y = 1;
  Packet.workgroup_size_z = 1;
  Packet.grid_size_x = 1;
  Packet.grid_size_y = 1;
  Packet.grid_size_z = 1;
  return Packet;
}

/// Runs the fill over a kernarg segment big enough for \p Args, and hands back
/// the resulting bytes.
std::vector<uint8_t> fill(llvm::ArrayRef<HiddenArgInfo> Args,
                          const LauncherAccess::HiddenArgBufferAddresses
                              &Buffers = {},
                          size_t KernargSize = 256) {
  std::vector<uint8_t> Kernarg(KernargSize, 0);
  const hsa_kernel_dispatch_packet_t Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  llvm::Error Err = LauncherAccess::writeHiddenKernelArguments(
      Kernarg, Args, Packet, Queue.get(), Buffers);
  EXPECT_FALSE(static_cast<bool>(Err))
      << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));
  return Kernarg;
}

/// Runs the fill expecting it to reject the argument list.
llvm::Error fillExpectingFailure(llvm::ArrayRef<HiddenArgInfo> Args,
                                 size_t KernargSize = 256) {
  std::vector<uint8_t> Kernarg(KernargSize, 0);
  const hsa_kernel_dispatch_packet_t Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  return LauncherAccess::writeHiddenKernelArguments(Kernarg, Args, Packet,
                                                    Queue.get(), {});
}

uint16_t read16(llvm::ArrayRef<uint8_t> Buf, size_t Offset) {
  return llvm::support::endian::read16le(Buf.data() + Offset);
}
uint32_t read32(llvm::ArrayRef<uint8_t> Buf, size_t Offset) {
  return llvm::support::endian::read32le(Buf.data() + Offset);
}
uint64_t read64(llvm::ArrayRef<uint8_t> Buf, size_t Offset) {
  return llvm::support::endian::read64le(Buf.data() + Offset);
}

//===----------------------------------------------------------------------===//
// Dispatch geometry
//===----------------------------------------------------------------------===//

TEST(HiddenArguments, DerivesBlockCountsFromTheDispatchGeometry) {
  const std::vector<uint8_t> Kernarg =
      fill({{ValueKind::HiddenBlockCountX, 0, 4},
            {ValueKind::HiddenBlockCountY, 4, 4},
            {ValueKind::HiddenBlockCountZ, 8, 4}});
  EXPECT_EQ(read32(Kernarg, 0), 1u);
  EXPECT_EQ(read32(Kernarg, 4), 1u);
  EXPECT_EQ(read32(Kernarg, 8), 1u);
}

TEST(HiddenArguments, DerivesGroupSizesAndRemainders) {
  const std::vector<uint8_t> Kernarg =
      fill({{ValueKind::HiddenGroupSizeX, 0, 2},
            {ValueKind::HiddenGroupSizeY, 2, 2},
            {ValueKind::HiddenGroupSizeZ, 4, 2},
            {ValueKind::HiddenRemainderX, 6, 2},
            {ValueKind::HiddenRemainderY, 8, 2},
            {ValueKind::HiddenRemainderZ, 10, 2}});
  EXPECT_EQ(read16(Kernarg, 0), 1u);
  EXPECT_EQ(read16(Kernarg, 2), 1u);
  EXPECT_EQ(read16(Kernarg, 4), 1u);
  // A 1x1x1 grid of 1x1x1 workgroups divides evenly.
  EXPECT_EQ(read16(Kernarg, 6), 0u);
  EXPECT_EQ(read16(Kernarg, 8), 0u);
  EXPECT_EQ(read16(Kernarg, 10), 0u);
}

TEST(HiddenArguments, ReadsGridDimensionsOutOfThePacketSetupField) {
  const std::vector<uint8_t> Kernarg =
      fill({{ValueKind::HiddenGridDims, 0, 2}});
  EXPECT_EQ(read16(Kernarg, 0), 1u);
}

TEST(HiddenArguments, StartsTheGridAtTheOrigin) {
  const std::vector<uint8_t> Kernarg =
      fill({{ValueKind::HiddenGlobalOffsetX, 0, 8},
            {ValueKind::HiddenGlobalOffsetY, 8, 8},
            {ValueKind::HiddenGlobalOffsetZ, 16, 8}});
  EXPECT_EQ(read64(Kernarg, 0), 0u);
  EXPECT_EQ(read64(Kernarg, 8), 0u);
  EXPECT_EQ(read64(Kernarg, 16), 0u);
}

TEST(HiddenArguments, RequestsNoDynamicGroupSegment) {
  const std::vector<uint8_t> Kernarg =
      fill({{ValueKind::HiddenDynamicLDSSize, 0, 4}});
  EXPECT_EQ(read32(Kernarg, 0), 0u);
}

//===----------------------------------------------------------------------===//
// Queue-derived arguments
//===----------------------------------------------------------------------===//

// The two aperture arguments come out of the AMD extension of the queue
// struct, and must not be swapped for one another.
TEST(HiddenArguments, TakesTheAperturesFromTheQueueWithoutSwappingThem) {
  const std::vector<uint8_t> Kernarg =
      fill({{ValueKind::HiddenPrivateBase, 0, 4},
            {ValueKind::HiddenSharedBase, 4, 4}});
  EXPECT_EQ(read32(Kernarg, 0), PrivateApertureHi);
  EXPECT_EQ(read32(Kernarg, 4), GroupApertureHi);
}

TEST(HiddenArguments, PassesTheQueueItself) {
  std::vector<uint8_t> Kernarg(256, 0);
  const hsa_kernel_dispatch_packet_t Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  const HiddenArgInfo Args[] = {{ValueKind::HiddenQueuePtr, 0, 8}};

  llvm::Error Err = LauncherAccess::writeHiddenKernelArguments(
      Kernarg, Args, Packet, Queue.get(), {});
  ASSERT_FALSE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(read64(Kernarg, 0), reinterpret_cast<uint64_t>(&Queue.get()));
}

//===----------------------------------------------------------------------===//
// Buffer-backed arguments
//===----------------------------------------------------------------------===//

TEST(HiddenArguments, PassesEveryBufferTheDispatchStoodUp) {
  LauncherAccess::HiddenArgBufferAddresses Buffers;
  Buffers.HostcallBuffer = reinterpret_cast<void *>(uintptr_t{0x1000});
  Buffers.PrintfBuffer = reinterpret_cast<void *>(uintptr_t{0x2000});
  Buffers.Heap = reinterpret_cast<void *>(uintptr_t{0x3000});
  Buffers.GridSyncInfo = reinterpret_cast<void *>(uintptr_t{0x4000});
  Buffers.CompletionAction = reinterpret_cast<void *>(uintptr_t{0x5000});

  const std::vector<uint8_t> Kernarg =
      fill({{ValueKind::HiddenHostcallBuffer, 0, 8},
            {ValueKind::HiddenPrintfBuffer, 8, 8},
            {ValueKind::HiddenHeapV1, 16, 8},
            {ValueKind::HiddenMultiGridSyncArg, 24, 8},
            {ValueKind::HiddenCompletionAction, 32, 8}},
           Buffers);

  EXPECT_EQ(read64(Kernarg, 0), 0x1000u);
  EXPECT_EQ(read64(Kernarg, 8), 0x2000u);
  EXPECT_EQ(read64(Kernarg, 16), 0x3000u);
  EXPECT_EQ(read64(Kernarg, 24), 0x4000u);
  EXPECT_EQ(read64(Kernarg, 32), 0x5000u);
}

// A dispatch that stood nothing up leaves the slot null, which is what the
// device-side null checks expect.
TEST(HiddenArguments, LeavesUnbackedBufferArgumentsNull) {
  const std::vector<uint8_t> Kernarg =
      fill({{ValueKind::HiddenHostcallBuffer, 0, 8},
            {ValueKind::HiddenPrintfBuffer, 8, 8},
            {ValueKind::HiddenHeapV1, 16, 8}});
  EXPECT_EQ(read64(Kernarg, 0), 0u);
  EXPECT_EQ(read64(Kernarg, 8), 0u);
  EXPECT_EQ(read64(Kernarg, 16), 0u);
}

// Luthier runs no device-enqueue scheduler, so the default queue is
// deliberately null: a queue nobody drains would turn a clean enqueue_kernel
// failure into children that silently never run.
TEST(HiddenArguments, LeavesTheDeviceEnqueueQueueNull) {
  std::vector<uint8_t> Kernarg(256, 0xAB);
  const hsa_kernel_dispatch_packet_t Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  const HiddenArgInfo Args[] = {{ValueKind::HiddenDefaultQueue, 0, 8}};

  llvm::Error Err = LauncherAccess::writeHiddenKernelArguments(
      Kernarg, Args, Packet, Queue.get(), {});
  ASSERT_FALSE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
  // The slot is skipped, so whatever the caller zeroed stays put — the loader
  // always hands in a zeroed segment.
  EXPECT_EQ(Kernarg[0], 0xABu) << "the fill must not touch this slot";
}

TEST(HiddenArguments, SkipsPaddingSlots) {
  std::vector<uint8_t> Kernarg(256, 0xCD);
  const hsa_kernel_dispatch_packet_t Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  const HiddenArgInfo Args[] = {{ValueKind::HiddenNone, 0, 8}};

  llvm::Error Err = LauncherAccess::writeHiddenKernelArguments(
      Kernarg, Args, Packet, Queue.get(), {});
  ASSERT_FALSE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
  EXPECT_EQ(Kernarg[0], 0xCDu);
}

//===----------------------------------------------------------------------===//
// Bounds and width checking
//===----------------------------------------------------------------------===//

TEST(HiddenArguments, RejectsASlotThatRunsPastTheKernargSegment) {
  llvm::Error Err = fillExpectingFailure(
      {{ValueKind::HiddenBlockCountX, /*Offset=*/250, /*Size=*/8}},
      /*KernargSize=*/256);
  EXPECT_TRUE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

TEST(HiddenArguments, RejectsASlotOffsetPastTheKernargSegment) {
  llvm::Error Err = fillExpectingFailure(
      {{ValueKind::HiddenBlockCountX, /*Offset=*/4096, /*Size=*/4}});
  EXPECT_TRUE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

// Silently truncating a pointer into a slot the metadata declared too narrow
// would hand the device a wild address, so it is rejected instead.
TEST(HiddenArguments, RejectsAValueTooWideForItsDeclaredSlot) {
  LauncherAccess::HiddenArgBufferAddresses Buffers;
  Buffers.Heap = reinterpret_cast<void *>(uintptr_t{0x1'0000'0000ULL});

  std::vector<uint8_t> Kernarg(256, 0);
  const hsa_kernel_dispatch_packet_t Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  const HiddenArgInfo Args[] = {{ValueKind::HiddenHeapV1, 0, /*Size=*/4}};

  llvm::Error Err = LauncherAccess::writeHiddenKernelArguments(
      Kernarg, Args, Packet, Queue.get(), Buffers);
  EXPECT_TRUE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

TEST(HiddenArguments, AnEmptyArgumentListIsFine) {
  std::vector<uint8_t> Kernarg(0);
  const hsa_kernel_dispatch_packet_t Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  llvm::Error Err = LauncherAccess::writeHiddenKernelArguments(
      Kernarg, {}, Packet, Queue.get(), {});
  EXPECT_FALSE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

TEST(HiddenArguments, RejectsAZeroSizedWorkgroupDimension) {
  std::vector<uint8_t> Kernarg(256, 0);
  hsa_kernel_dispatch_packet_t Packet = makeSingleWorkItemPacket();
  Packet.workgroup_size_y = 0;
  TestQueue Queue;

  llvm::Error Err = LauncherAccess::writeHiddenKernelArguments(
      Kernarg, {}, Packet, Queue.get(), {});
  EXPECT_TRUE(static_cast<bool>(Err))
      << "a zero workgroup dimension would divide by zero";
  llvm::consumeError(std::move(Err));
}

//===----------------------------------------------------------------------===//
// declaresHiddenArg
//===----------------------------------------------------------------------===//

TEST(HiddenArguments, DetectsWhichArgumentsAKernelDeclares) {
  LauncherAccess::LoadedKernelInfo Kernel;
  Kernel.HiddenArgs.push_back({ValueKind::HiddenHostcallBuffer, 0, 8});
  Kernel.HiddenArgs.push_back({ValueKind::HiddenGridDims, 8, 2});

  EXPECT_TRUE(LauncherAccess::declaresHiddenArg(
      Kernel, ValueKind::HiddenHostcallBuffer));
  EXPECT_TRUE(
      LauncherAccess::declaresHiddenArg(Kernel, ValueKind::HiddenGridDims));
  EXPECT_FALSE(
      LauncherAccess::declaresHiddenArg(Kernel, ValueKind::HiddenHeapV1));
  EXPECT_FALSE(LauncherAccess::declaresHiddenArg(
      Kernel, ValueKind::HiddenPrintfBuffer));
}

//===----------------------------------------------------------------------===//
// Extended kernarg buffer composition
//===----------------------------------------------------------------------===//
//
// The extended kernarg buffer is what the launcher stands up in front of an
// instrumented dispatch when the patcher's kernarg expansion applies. Its
// shape is a per-kernel flag away from the shape writeHiddenKernelArguments
// already handles: either an 8-byte app-kernarg pointer prefixes the hidden
// block, or the hidden block starts at offset 0. These tests exercise both
// shapes against the composition helper.

TEST(ExtendedKernargBuffer, CopiesTheAppKernargPointerIntoTheFirstEightBytes) {
  std::vector<uint8_t> Kernarg(256, 0);
  const auto Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  const auto *AppKernarg = reinterpret_cast<const void *>(uintptr_t{0xCAFED00Du});

  llvm::Error Err = LauncherAccess::fillExtendedKernargBuffer(
      Kernarg, /*HasAppKernargPrefix=*/true, AppKernarg, /*HiddenArgs=*/{},
      Packet, Queue.get(), {});
  ASSERT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(read64(Kernarg, 0), reinterpret_cast<uint64_t>(AppKernarg));
}

TEST(ExtendedKernargBuffer, LeavesTheFirstEightBytesUntouchedWithoutAPrefix) {
  std::vector<uint8_t> Kernarg(256, 0);
  const auto Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  const auto *AppKernarg = reinterpret_cast<const void *>(uintptr_t{0xCAFED00Du});

  // Even with a non-null app kernarg address handed in, HasAppKernargPrefix
  // == false means the extended buffer has no prefix slot; nothing must
  // land in bytes [0, 8).
  llvm::Error Err = LauncherAccess::fillExtendedKernargBuffer(
      Kernarg, /*HasAppKernargPrefix=*/false, AppKernarg, /*HiddenArgs=*/{},
      Packet, Queue.get(), {});
  ASSERT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(read64(Kernarg, 0), 0u);
}

TEST(ExtendedKernargBuffer, WritesHiddenArgsAfterThePrefix) {
  std::vector<uint8_t> Kernarg(256, 0);
  const auto Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  const auto *AppKernarg = reinterpret_cast<const void *>(uintptr_t{0xDEADBEEFu});
  const HiddenArgInfo Hidden[] = {
      {ValueKind::HiddenBlockCountX, /*Offset=*/8, /*Size=*/4}};

  llvm::Error Err = LauncherAccess::fillExtendedKernargBuffer(
      Kernarg, /*HasAppKernargPrefix=*/true, AppKernarg, Hidden, Packet,
      Queue.get(), {});
  ASSERT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(read64(Kernarg, 0), reinterpret_cast<uint64_t>(AppKernarg))
      << "prefix must still be there after the hidden fill";
  EXPECT_EQ(read32(Kernarg, 8), 1u)
      << "single-workgroup dispatch means BlockCountX == 1";
}

TEST(ExtendedKernargBuffer, WritesHiddenArgsAtOffsetZeroWithoutAPrefix) {
  std::vector<uint8_t> Kernarg(256, 0);
  const auto Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  const HiddenArgInfo Hidden[] = {
      {ValueKind::HiddenBlockCountX, /*Offset=*/0, /*Size=*/4}};

  llvm::Error Err = LauncherAccess::fillExtendedKernargBuffer(
      Kernarg, /*HasAppKernargPrefix=*/false, /*AppKernargPtr=*/nullptr,
      Hidden, Packet, Queue.get(), {});
  ASSERT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(read32(Kernarg, 0), 1u);
}

TEST(ExtendedKernargBuffer, WritesTheAppKernargBufferPointerCorrectlyWhenNull) {
  // A null app kernarg pointer is legal — the app may dispatch a kernel that
  // takes only implicit args. The prefix slot has to hold that null exactly,
  // not silently pick up whatever the buffer was previously initialized to.
  std::vector<uint8_t> Kernarg(256, 0xAB);
  const auto Packet = makeSingleWorkItemPacket();
  TestQueue Queue;

  llvm::Error Err = LauncherAccess::fillExtendedKernargBuffer(
      Kernarg, /*HasAppKernargPrefix=*/true, /*AppKernargPtr=*/nullptr,
      /*HiddenArgs=*/{}, Packet, Queue.get(), {});
  ASSERT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(read64(Kernarg, 0), 0u);
}

TEST(ExtendedKernargBuffer, RejectsAPrefixThatWouldNotFitTheBuffer) {
  std::vector<uint8_t> Kernarg(4, 0);
  const auto Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  const auto *AppKernarg = reinterpret_cast<const void *>(uintptr_t{0x1000u});

  llvm::Error Err = LauncherAccess::fillExtendedKernargBuffer(
      Kernarg, /*HasAppKernargPrefix=*/true, AppKernarg, /*HiddenArgs=*/{},
      Packet, Queue.get(), {});
  EXPECT_TRUE(static_cast<bool>(Err))
      << "a 4-byte buffer cannot hold the 8-byte prefix";
  llvm::consumeError(std::move(Err));
}

TEST(ExtendedKernargBuffer, SurfacesErrorsFromTheHiddenArgFill) {
  // If the composition wrote the prefix and then swallowed a downstream
  // hidden-arg error, the launcher would install a buffer whose implicit
  // args are partially filled. Verify the error propagates instead.
  std::vector<uint8_t> Kernarg(32, 0);
  const auto Packet = makeSingleWorkItemPacket();
  TestQueue Queue;
  const HiddenArgInfo Hidden[] = {
      {ValueKind::HiddenBlockCountX, /*Offset=*/64, /*Size=*/4}};

  llvm::Error Err = LauncherAccess::fillExtendedKernargBuffer(
      Kernarg, /*HasAppKernargPrefix=*/false, /*AppKernargPtr=*/nullptr,
      Hidden, Packet, Queue.get(), {});
  EXPECT_TRUE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

//===----------------------------------------------------------------------===//
// Backing structures
//===----------------------------------------------------------------------===//

// These layouts are read by the ROCm device libraries, so they are pinned
// rather than merely exercised.
TEST(HiddenArgBuffers, GridSyncInfoMatchesTheDeviceLibraryLayout) {
  EXPECT_EQ(sizeof(DeviceGridSyncData), 8u);
  EXPECT_EQ(offsetof(DeviceGridSyncInfo, MultiGridSync), 0u);
  EXPECT_EQ(offsetof(DeviceGridSyncInfo, GridID), 8u);
  EXPECT_EQ(offsetof(DeviceGridSyncInfo, NumGrids), 12u);
  EXPECT_EQ(offsetof(DeviceGridSyncInfo, PrevGridSum), 16u);
  EXPECT_EQ(offsetof(DeviceGridSyncInfo, AllGridSum), 24u);
  EXPECT_EQ(offsetof(DeviceGridSyncInfo, SingleGridSync), 32u);
  EXPECT_EQ(offsetof(DeviceGridSyncInfo, NumWorkgroups), 40u);
}

TEST(HiddenArgBuffers, InitializesASingleGridBarrier) {
  DeviceGridSyncInfo Info;
  std::memset(&Info, 0xFF, sizeof(Info));
  initializeSingleGridSyncInfo(Info, /*NumWorkgroups=*/1);

  EXPECT_EQ(Info.MultiGridSync, nullptr)
      << "a single-grid launch has no multi-grid barrier to join";
  EXPECT_EQ(Info.GridID, 0u);
  EXPECT_EQ(Info.NumGrids, 1u);
  EXPECT_EQ(Info.PrevGridSum, 0u);
  EXPECT_EQ(Info.AllGridSum, 1u);
  EXPECT_EQ(Info.SingleGridSync.W0, 0u) << "the barrier starts unclaimed";
  EXPECT_EQ(Info.SingleGridSync.W1, 0u);
  EXPECT_EQ(Info.NumWorkgroups, 1u);
}

TEST(HiddenArgBuffers, GridSyncBarrierCountsTheWorkgroupsGiven) {
  DeviceGridSyncInfo Info;
  initializeSingleGridSyncInfo(Info, /*NumWorkgroups=*/16);
  EXPECT_EQ(Info.NumWorkgroups, 16u);
  EXPECT_EQ(Info.AllGridSum, 16u)
      << "one grid means the whole launch is this grid";
}

TEST(HiddenArgBuffers, AqlWrapMatchesTheSchedulerLayout) {
  EXPECT_EQ(offsetof(DeviceAqlWrap, State), 0u);
  EXPECT_EQ(offsetof(DeviceAqlWrap, EnqueueFlags), 4u);
  EXPECT_EQ(offsetof(DeviceAqlWrap, CommandID), 8u);
  EXPECT_EQ(offsetof(DeviceAqlWrap, ChildCounter), 12u);
  EXPECT_EQ(offsetof(DeviceAqlWrap, Completion), 16u);
  EXPECT_EQ(offsetof(DeviceAqlWrap, ParentWrap), 24u);
  EXPECT_EQ(offsetof(DeviceAqlWrap, WaitList), 32u);
  EXPECT_EQ(offsetof(DeviceAqlWrap, WaitNum), 40u);
  EXPECT_EQ(offsetof(DeviceAqlWrap, Aql), 64u);
  EXPECT_EQ(sizeof(DeviceAqlWrap), 128u);
}

TEST(HiddenArgBuffers, InitializesAnAlreadyCompletedCompletionAction) {
  DeviceAqlWrap Wrap;
  std::memset(&Wrap, 0xFF, sizeof(Wrap));
  initializeCompletionAction(Wrap);

  EXPECT_EQ(Wrap.State, static_cast<uint32_t>(DEVICE_AQL_WRAP_DONE));
  EXPECT_EQ(Wrap.ChildCounter, 0u) << "nothing is outstanding against it";
  EXPECT_EQ(Wrap.ParentWrap, 0u) << "the host launched this kernel directly";
  EXPECT_EQ(Wrap.Completion, 0u);
  EXPECT_EQ(Wrap.WaitList, 0u);
  EXPECT_EQ(Wrap.WaitNum, 0u);
}

// The device library clears exactly this much of the heap, which is its own
// statement of how large its management structure can be.
TEST(HiddenArgBuffers, DeviceHeapIsSizedForTheManagementStructure) {
  EXPECT_EQ(DeviceHeapSize, 131072u);
}

} // namespace
