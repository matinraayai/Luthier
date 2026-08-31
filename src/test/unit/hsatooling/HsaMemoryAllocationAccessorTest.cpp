//===-- HsaMemoryAllocationAccessorTest.cpp -------------------------------===//
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
/// GPU-free tests for how \c HsaMemoryAllocationAccessor chooses between its
/// sources.
///
/// \par What makes these runnable without a GPU
/// The accessor reaches HSA through rocprofiler-captured API table snapshots. A
/// snapshot whose registration callback has never fired reports so, and that is
/// exactly the state the accessor is in inside an application that drives the
/// KFD driver itself -- such an application holds the DRM virtual address space
/// for its GPUs, the kernel permits only one per GPU per process, and \c hsa_init
/// therefore fails. Constructing snapshots and leaving them unregistered
/// reproduces that state faithfully rather than approximating it.
///
/// \par What these replace
/// The behaviour asserted here used to belong to \c
/// CompositeMemoryAllocationAccessor, which held a list of accessors and walked
/// it. That class is gone: the driver-level source is now a component of this
/// accessor rather than a sibling of it. The rules it enforced are unchanged and
/// are what these tests pin -- an empty answer means "ask the next source", an
/// error means "this source owns the address and something broke", and an
/// allocation is used whole rather than having its fields mixed with another
/// source's.
//===----------------------------------------------------------------------===//
#include "luthier/HSATooling/HsaMemoryAllocationAccessor.h"

#include "luthier/Common/GenericLuthierError.h"

#include "common/HsaApiTable.h"
#include "common/ProviderTestAccess.h"

#include <gtest/gtest.h>

#include <llvm/Support/Error.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

using luthier::DriverAllocationResolver;
using luthier::HsaMemoryAllocationAccessor;
using luthier::LoadedCodeObjectCache;
using luthier::rocprofiler::HsaApiTableSnapshot;
using luthier::rocprofiler::HsaExtensionTableSnapshot;
using luthier::test::buildCoreApiTable;
using luthier::test::buildHsaApiTable;

namespace {

//===----------------------------------------------------------------------===//
// Stand-in resolvers
//===----------------------------------------------------------------------===//

/// A resolver with no records at all -- the state in a process where the KFD
/// wrapper was never preloaded.
class UnavailableResolver final : public DriverAllocationResolver {
public:
  llvm::Expected<Allocation> resolve(uint64_t) const override {
    return Allocation();
  }
  bool isAvailable() const override { return false; }
};

/// A resolver that is watching, but has not seen this address.
class EmptyResolver final : public DriverAllocationResolver {
public:
  mutable unsigned Calls{0};
  llvm::Expected<Allocation> resolve(uint64_t) const override {
    Calls++;
    return Allocation();
  }
  bool isAvailable() const override { return true; }
};

/// A resolver that answers with two distinct buffers, so a test can tell the
/// device view from the host view. Real memory rather than invented addresses,
/// so the descriptor's ArrayRefs stay legal to form.
class AnsweringResolver final : public DriverAllocationResolver {
public:
  std::vector<std::byte> Device{64, std::byte{0xAA}};
  std::vector<std::byte> Host{64, std::byte{0xBB}};

  llvm::Expected<Allocation> resolve(uint64_t) const override {
    return Allocation{Device.data(), Host.data(), Device.size()};
  }
  bool isAvailable() const override { return true; }
};

/// A resolver for which the address is real but unreadable -- a mapping the
/// hardware refused, say.
class FailingResolver final : public DriverAllocationResolver {
public:
  llvm::Expected<Allocation> resolve(uint64_t) const override {
    return LUTHIER_MAKE_GENERIC_ERROR("the mapping was refused");
  }
  bool isAvailable() const override { return true; }
};

//===----------------------------------------------------------------------===//
// An accessor whose HSA half was never initialized
//===----------------------------------------------------------------------===//

/// Owns the snapshots an accessor needs and hands out an accessor over them.
///
/// The snapshots are deliberately never delivered to, which leaves
/// \c wasRegistrationCallbackInvoked false -- the KFD-only process this suite is
/// about. They are leaked rather than destroyed, following
/// ../rocprofiler/RocprofilerTest.cpp: a snapshot's teardown expects
/// rocprofiler-sdk's lifecycle, which no unit test has.
struct UninitializedHsa {
  HsaApiTableSnapshot<::CoreApiTable> *Core;
  HsaApiTableSnapshot<::AmdExtTable> *AmdExt;
  HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER> *Loader;
  std::unique_ptr<LoadedCodeObjectCache> COC;

  UninitializedHsa() {
    llvm::Error Err = llvm::Error::success();
    Core = new HsaApiTableSnapshot<::CoreApiTable>(Err);
    AmdExt = new HsaApiTableSnapshot<::AmdExtTable>(Err);
    Loader = new HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>(Err);
    llvm::consumeError(std::move(Err));
    COC = std::make_unique<LoadedCodeObjectCache>(*Core, *Loader);
  }

  std::unique_ptr<HsaMemoryAllocationAccessor>
  accessorWith(std::unique_ptr<DriverAllocationResolver> R) const {
    return std::make_unique<HsaMemoryAllocationAccessor>(
        *COC, *Core, *AmdExt, *Loader, std::move(R));
  }
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

/// The load-bearing one. Reading an uncaptured API table snapshot is a fatal
/// error by design (HsaApiTableSnapshot.h:125-128), so an accessor that reached
/// for HSA before checking would take the whole process down here rather than
/// returning anything at all. That this test returns is the assertion.
TEST(HsaMemoryAllocationAccessor, UninitializedHsaDoesNotAbortAndDefersToDriver) {
  const UninitializedHsa Hsa;
  auto Resolver = std::make_unique<AnsweringResolver>();
  const auto *Raw = Resolver.get();
  auto A = Hsa.accessorWith(std::move(Resolver));

  auto D = A->getAllocationDescriptor(0x520240000000ULL);
  ASSERT_TRUE(static_cast<bool>(D)) << llvm::toString(D.takeError());
  ASSERT_FALSE(D->empty());
  EXPECT_EQ(Raw->Device.size(), D->getSize());
}

/// The device and host views must survive the trip into the descriptor
/// separately. They are equal for every source that reads memory the host
/// already owns, so a descriptor built with one base in both slots looks correct
/// until a source appears whose host view lives elsewhere -- which is precisely
/// what the KFD resolver is.
TEST(HsaMemoryAllocationAccessor, DriverAnswerKeepsDeviceAndHostViewsApart) {
  const UninitializedHsa Hsa;
  auto Resolver = std::make_unique<AnsweringResolver>();
  const auto *Raw = Resolver.get();
  auto A = Hsa.accessorWith(std::move(Resolver));

  auto D = A->getAllocationDescriptor(0x520240000000ULL);
  ASSERT_TRUE(static_cast<bool>(D)) << llvm::toString(D.takeError());

  EXPECT_EQ(reinterpret_cast<const uint8_t *>(Raw->Device.data()),
            D->getDeviceAllocation().data());
  EXPECT_EQ(reinterpret_cast<const uint8_t *>(Raw->Host.data()),
            D->getHostAllocation().data());
  // And the arithmetic built on them agrees.
  EXPECT_EQ(reinterpret_cast<uint64_t>(Raw->Host.data()),
            D->hostAddressFor(reinterpret_cast<uint64_t>(Raw->Device.data())));
}

/// A driver-level source cannot produce a parsed code object -- there is no
/// loader below HSA to have parsed one -- and CodeDiscoveryPass depends on the
/// distinction: a null code object makes it name the kernel after its address,
/// whereas a code object that lacks the expected symbol is a hard error.
TEST(HsaMemoryAllocationAccessor, DriverAnswerCarriesNoCodeObject) {
  const UninitializedHsa Hsa;
  auto A = Hsa.accessorWith(std::make_unique<AnsweringResolver>());

  auto D = A->getAllocationDescriptor(0x520240000000ULL);
  ASSERT_TRUE(static_cast<bool>(D)) << llvm::toString(D.takeError());
  EXPECT_EQ(nullptr, D->getAllocationCodeObject());
}

/// An error from the resolver means "the address is mine and something broke",
/// so it propagates rather than being flattened into "no allocation here".
TEST(HsaMemoryAllocationAccessor, DriverErrorPropagates) {
  const UninitializedHsa Hsa;
  auto A = Hsa.accessorWith(std::make_unique<FailingResolver>());

  auto D = A->getAllocationDescriptor(0x520240000000ULL);
  ASSERT_FALSE(static_cast<bool>(D));
  EXPECT_NE(std::string::npos,
            llvm::toString(D.takeError()).find("the mapping was refused"));
}

/// Nobody knows the address. Empty, not an error: InstructionTraces treats an
/// empty descriptor as the normal way a disassembly walk ends, and an error as a
/// reason to abort the whole analysis.
TEST(HsaMemoryAllocationAccessor, NobodyKnowsTheAddressIsEmptyNotAnError) {
  const UninitializedHsa Hsa;
  auto Resolver = std::make_unique<EmptyResolver>();
  const auto *Raw = Resolver.get();
  auto A = Hsa.accessorWith(std::move(Resolver));

  auto D = A->getAllocationDescriptor(0x1234ULL);
  ASSERT_TRUE(static_cast<bool>(D)) << llvm::toString(D.takeError());
  EXPECT_TRUE(D->empty());
  EXPECT_EQ(1u, Raw->Calls) << "the resolver should have been asked exactly once";
}

/// An unavailable resolver is not asked to guess. It reports emptiness the same
/// way a resolver with no matching record does, but the accessor must not treat
/// the two as interchangeable when reporting -- hence a separate case.
TEST(HsaMemoryAllocationAccessor, UnavailableResolverYieldsEmpty) {
  const UninitializedHsa Hsa;
  auto A = Hsa.accessorWith(std::make_unique<UnavailableResolver>());

  auto D = A->getAllocationDescriptor(0x520240000000ULL);
  ASSERT_TRUE(static_cast<bool>(D)) << llvm::toString(D.takeError());
  EXPECT_TRUE(D->empty());
}

/// No resolver at all is the configuration of a plain HSA tool. It must behave
/// exactly as an HSA-only accessor did before the merge, which for an
/// uninitialized HSA means an empty answer rather than a crash.
TEST(HsaMemoryAllocationAccessor, NoResolverIsAnHsaOnlyAccessor) {
  const UninitializedHsa Hsa;
  auto A = Hsa.accessorWith(nullptr);

  auto D = A->getAllocationDescriptor(0x520240000000ULL);
  ASSERT_TRUE(static_cast<bool>(D)) << llvm::toString(D.takeError());
  EXPECT_TRUE(D->empty());
}


//===----------------------------------------------------------------------===//
// An accessor whose HSA half *is* initialized
//===----------------------------------------------------------------------===//
//
// Needed to test the one rule that cannot be reached with HSA absent: when HSA
// claims an address, the driver-level source must not be asked. Everything above
// exercises the HSA-unavailable path, where the resolver is always consulted, so
// a regression that consulted it unconditionally would pass every test so far.
//
// The API tables here are fabricated and delivered to the snapshots directly, the
// way ../rocprofiler/RocprofilerTest.cpp does. No HSA runtime is initialized --
// which is just as well, since in the process this whole feature exists for, it
// cannot be.

/// Stub state. The HSA tables take bare function pointers with no user-data slot,
/// so the stubs and the test communicate through these.
hsa_amd_pointer_type_t StubPointerType = HSA_EXT_POINTER_TYPE_UNKNOWN;
std::byte StubAgentMemory[64] = {};
std::byte StubHostMemory[64] = {};

/// Reports the loader knowing nothing, which is what sends the accessor to
/// hsa_amd_pointer_info -- the branch where the decision under test lives.
hsa_status_t stubQueryExecutable(const void *, hsa_executable_t *) {
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t stubPointerInfo(const void *, hsa_amd_pointer_info_t *Info,
                             void *(*)(size_t), uint32_t *, hsa_agent_t **) {
  Info->type = StubPointerType;
  Info->agentBaseAddress = StubAgentMemory;
  Info->hostBaseAddress = StubHostMemory;
  Info->sizeInBytes = sizeof(StubAgentMemory);
  return HSA_STATUS_SUCCESS;
}

/// Hands out the fabricated loader extension table. This is the hook the
/// extension snapshot uses to fetch its table, so overriding it is what lets a
/// loader snapshot be delivered without a live runtime.
hsa_status_t stubGetMajorExtensionTable(uint16_t, uint16_t, size_t TableLength,
                                        void *Table) {
  hsa_ven_amd_loader_1_03_pfn_t Loader{};
  Loader.hsa_ven_amd_loader_query_executable = &stubQueryExecutable;
  std::memcpy(Table, &Loader, std::min(TableLength, sizeof(Loader)));
  return HSA_STATUS_SUCCESS;
}

::AmdExtTable buildAmdExtTable() {
  ::AmdExtTable T{};
  T.version.major_id = HSA_AMD_EXT_API_TABLE_MAJOR_VERSION;
  T.version.minor_id = sizeof(::AmdExtTable);
  T.version.step_id = HSA_AMD_EXT_API_TABLE_STEP_VERSION;
  T.hsa_amd_pointer_info_fn = &stubPointerInfo;
  return T;
}

/// Same shape as UninitializedHsa, but the snapshots are delivered to, so
/// wasRegistrationCallbackInvoked reports true and the accessor takes its HSA
/// path.
struct RegisteredHsa {
  ::CoreApiTable Core = buildCoreApiTable();
  ::AmdExtTable AmdExt = buildAmdExtTable();
  ::HsaApiTable Root{};

  HsaApiTableSnapshot<::CoreApiTable> *CoreSnap;
  HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnap;
  HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER> *LoaderSnap;
  std::unique_ptr<LoadedCodeObjectCache> COC;

  RegisteredHsa() {
    Core.hsa_system_get_major_extension_table_fn = &stubGetMajorExtensionTable;
    Root = buildHsaApiTable(&Core, &AmdExt);

    llvm::Error Err = llvm::Error::success();
    CoreSnap = new HsaApiTableSnapshot<::CoreApiTable>(Err);
    AmdExtSnap = new HsaApiTableSnapshot<::AmdExtTable>(Err);
    LoaderSnap = new HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>(Err);
    llvm::consumeError(std::move(Err));

    ::HsaApiTable *Tables[1] = {&Root};
    using Access = luthier::test::ProviderTestAccess<ROCPROFILER_HSA_TABLE>;
    Access::deliver(CoreSnap, ROCPROFILER_HSA_TABLE, 0, 0,
                    reinterpret_cast<void **>(Tables), 1);
    Access::deliver(AmdExtSnap, ROCPROFILER_HSA_TABLE, 0, 0,
                    reinterpret_cast<void **>(Tables), 1);
    Access::deliver(LoaderSnap, ROCPROFILER_HSA_TABLE, 0, 0,
                    reinterpret_cast<void **>(Tables), 1);

    COC = std::make_unique<LoadedCodeObjectCache>(*CoreSnap, *LoaderSnap);
  }

  std::unique_ptr<HsaMemoryAllocationAccessor>
  accessorWith(std::unique_ptr<DriverAllocationResolver> R) const {
    return std::make_unique<HsaMemoryAllocationAccessor>(
        *COC, *CoreSnap, *AmdExtSnap, *LoaderSnap, std::move(R));
  }
};

/// The decision this whole arrangement turns on: a non-empty HSA answer is final,
/// even though hsa_amd_pointer_info never carries a parsed code object and it is
/// tempting to treat that as a half-answer worth improving on.
///
/// Improving on it would make things worse in a way no local test would catch.
/// The driver-level source describes the *coarser* enclosing allocation -- a
/// measured kernel_object at 0x5202400003c0 sits inside a 2 MB suballocation
/// arena -- and InstructionTraces disassembles forward to the end of whatever
/// allocation it is handed, so the coarser answer sends it megabytes past the end
/// of the kernel, through other kernels and data.
TEST(HsaMemoryAllocationAccessor, HsaAnswerIsFinalAndTheResolverIsNotAsked) {
  StubPointerType = HSA_EXT_POINTER_TYPE_HSA;
  const RegisteredHsa Hsa;
  auto Resolver = std::make_unique<EmptyResolver>();
  const auto *Raw = Resolver.get();
  auto A = Hsa.accessorWith(std::move(Resolver));

  auto D = A->getAllocationDescriptor(0x520240000000ULL);
  ASSERT_TRUE(static_cast<bool>(D)) << llvm::toString(D.takeError());
  ASSERT_FALSE(D->empty()) << "HSA claimed the address, so it must be described";
  EXPECT_EQ(reinterpret_cast<const uint8_t *>(StubAgentMemory),
            D->getDeviceAllocation().data());
  EXPECT_EQ(0u, Raw->Calls)
      << "HSA answered, so the driver-level source must not be consulted -- a "
         "coarser answer would replace a precise one";
}

/// The other half of the same branch. HSA is present and working, but does not
/// manage this address, so the resolver is exactly what should answer.
TEST(HsaMemoryAllocationAccessor, HsaNotManagingAnAddressReachesTheResolver) {
  StubPointerType = HSA_EXT_POINTER_TYPE_UNKNOWN;
  const RegisteredHsa Hsa;
  auto Resolver = std::make_unique<EmptyResolver>();
  const auto *Raw = Resolver.get();
  auto A = Hsa.accessorWith(std::move(Resolver));

  auto D = A->getAllocationDescriptor(0x520240000000ULL);
  ASSERT_TRUE(static_cast<bool>(D)) << llvm::toString(D.takeError());
  EXPECT_TRUE(D->empty());
  EXPECT_EQ(1u, Raw->Calls)
      << "an address HSA does not manage is precisely what the driver-level "
         "source exists to answer";
}

} // namespace
