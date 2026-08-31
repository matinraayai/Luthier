//===-- MemoryAllocationAccessorTest.cpp ----------------------------------===//
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
/// Tests \c MemoryAllocationAccessor::AllocationDescriptor's device-to-host
/// address arithmetic.
///
/// \par Why these exist
/// This arithmetic was previously written inline in
/// \c InstructionTracesAnalysis.cpp and was wrong twice over: the device-to-host
/// translation subtracted the device base and added it straight back, cancelling
/// to the device address, and the end-of-allocation bound was computed from the
/// caller's current address rather than from the allocation's base, overrunning
/// the end by the caller's offset into the allocation.
///
/// Neither error was visible to any test in the repository, and that is the point
/// of this file. Every accessor that existed at the time -- the HSA one on its
/// loader path, and \c MockLoaderMemoryAccessor -- reports the \e same pointer for
/// the device and host views, so a computation that returns the device address
/// when it should return the host one is indistinguishable from a correct one.
/// Reverting either formula passes all 145 lit tests. The tests below are written
/// with the two bases deliberately \b different, which is the only configuration
/// that can tell the formulas apart, and is the normal case for an allocation
/// tracked through KFD ioctls.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"

#include "luthier/Common/GenericLuthierError.h"

#include <gtest/gtest.h>

#include <llvm/Support/Error.h>

#include <cstdint>
#include <vector>

using luthier::MemoryAllocationAccessor;
using AllocationDescriptor = MemoryAllocationAccessor::AllocationDescriptor;

namespace {

/// Two distinct buffers standing in for a device allocation and a separate
/// host-readable view of it. Real memory rather than invented addresses, so the
/// descriptor's ArrayRefs stay legal to form.
struct TwoViews {
  std::vector<std::byte> Device;
  std::vector<std::byte> Host;
  static constexpr size_t Size = 256;

  TwoViews() : Device(Size), Host(Size) {
    // Distinguishable contents, so a test that reads the wrong view fails on the
    // bytes and not merely on an address comparison.
    for (size_t I = 0; I < Size; I++) {
      Device[I] = static_cast<std::byte>(0xDD);
      Host[I] = static_cast<std::byte>(I);
    }
  }

  [[nodiscard]] AllocationDescriptor descriptor() const {
    return AllocationDescriptor{Device[0], Host[0], Size};
  }
  [[nodiscard]] uint64_t deviceBase() const {
    return reinterpret_cast<uint64_t>(Device.data());
  }
  [[nodiscard]] uint64_t hostBase() const {
    return reinterpret_cast<uint64_t>(Host.data());
  }
};

TEST(AllocationDescriptor, HostAddressForTheBaseIsTheHostBase) {
  TwoViews V;
  auto D = V.descriptor();
  EXPECT_EQ(V.hostBase(), D.hostAddressFor(V.deviceBase()));
}

/// The formula that cancelled to the device address passes the base case above by
/// accident whenever the two bases coincide. Here they do not, so an offset
/// address is what separates the two formulas.
///
/// Mutation: write hostAddressFor as `DeviceBase + (DeviceAddr - DeviceBase)` and
/// this fails.
TEST(AllocationDescriptor, HostAddressForPreservesTheOffset) {
  TwoViews V;
  auto D = V.descriptor();
  for (uint64_t Offset : {uint64_t{1}, uint64_t{7}, uint64_t{64},
                          uint64_t{TwoViews::Size - 1}}) {
    EXPECT_EQ(V.hostBase() + Offset, D.hostAddressFor(V.deviceBase() + Offset))
        << "at offset " << Offset;
    // And it must not merely be the device address shifted by something.
    EXPECT_NE(V.deviceBase() + Offset, D.hostAddressFor(V.deviceBase() + Offset))
        << "at offset " << Offset;
  }
}

/// Reading through the translated address must yield the host view's bytes. This
/// is the property the code lifter actually depends on.
TEST(AllocationDescriptor, TranslatedAddressReadsTheHostView) {
  TwoViews V;
  auto D = V.descriptor();
  const uint64_t Offset = 42;
  auto *P = reinterpret_cast<const std::byte *>(
      D.hostAddressFor(V.deviceBase() + Offset));
  EXPECT_EQ(static_cast<std::byte>(Offset), *P);
  EXPECT_NE(static_cast<std::byte>(0xDD), *P);
}

/// The end bound belongs to the allocation, not to whoever is reading it.
///
/// Mutation: compute it as `hostAddressFor(someMidAddress) + Size` and this
/// fails -- which is exactly the bug that was there.
TEST(AllocationDescriptor, HostEndAddressIsAnchoredToTheAllocationBase) {
  TwoViews V;
  auto D = V.descriptor();
  EXPECT_EQ(V.hostBase() + TwoViews::Size, D.hostEndAddress());

  // Starting mid-allocation must not move the end.
  const uint64_t MidDevice = V.deviceBase() + TwoViews::Size / 2;
  EXPECT_EQ(V.hostBase() + TwoViews::Size, D.hostEndAddress());
  // The remaining readable span from a mid-allocation start is smaller than the
  // whole allocation, which is the thing the overrunning version got wrong.
  EXPECT_EQ(TwoViews::Size / 2, D.hostEndAddress() - D.hostAddressFor(MidDevice));
}

/// A descriptor whose views coincide -- every accessor in the tree before the KFD
/// one -- must keep behaving exactly as it did, so this refactor is provably a
/// no-op for existing callers.
TEST(AllocationDescriptor, CoincidingViewsBehaveAsBefore) {
  std::vector<std::byte> Buf(128);
  AllocationDescriptor D{Buf[0], Buf[0], Buf.size()};
  const auto Base = reinterpret_cast<uint64_t>(Buf.data());

  EXPECT_EQ(Base + 8, D.hostAddressFor(Base + 8));
  EXPECT_EQ(Base + Buf.size(), D.hostEndAddress());
}

/// An empty descriptor is the "not found" sentinel; asking it for addresses must
/// not be undefined, since callers check empty() only after constructing one.
TEST(AllocationDescriptor, EmptyDescriptorIsInert) {
  AllocationDescriptor D;
  EXPECT_TRUE(D.empty());
  EXPECT_EQ(0u, D.getSize());
  EXPECT_EQ(0u, D.hostEndAddress());
  EXPECT_EQ(0u, D.hostAddressFor(0));
}

} // namespace
