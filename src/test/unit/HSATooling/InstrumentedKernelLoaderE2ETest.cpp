//===-- InstrumentedKernelLoaderE2ETest.cpp -------------------------------===//
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
/// End-to-end tests for \c InstrumentedKernelLoaderAndLauncher on real
/// hardware. This binary registers itself as an in-process rocprofiler-sdk
/// tool so it can obtain the HSA API table snapshots the loader is built on,
/// then loads the relocatable built from \c InitFiniKernels.hip and checks
/// that the loader:
///
/// \li runs \c amdgcn.device.init before \c loadInstrumented returns;
/// \li services the \c printf that constructor makes, which is a hostcall —
///     the dispatch blocks until the loader's listener answers it, so a
///     regression here shows up as a hang rather than a wrong value;
/// \li runs \c amdgcn.device.fini when the record is unloaded, while the
///     executable is still alive.
///
/// Everything skips when no GPU is present, when the device relocatable was
/// not built, or when it was built for a different ISA than the machine
/// running the test.
//===----------------------------------------------------------------------===//
#include "common/GpuAvailability.h"

#include "luthier/HSA/Agent.h"
#include "luthier/HSA/ExecutableSymbol.h"
#include "luthier/HSA/ISA.h"
#include "luthier/HSA/Memory.h"
#include "luthier/HSA/MemoryPool.h"
#include "luthier/HSA/Queue.h"
#include "luthier/HSATooling/InstrumentedKernelLoaderAndLauncher.h"
#include "luthier/Rocprofiler/ApiTableSnapshot.h"

#include <gtest/gtest.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <memory>
#include <string>

using namespace luthier;

namespace {

//===----------------------------------------------------------------------===//
// In-process rocprofiler tool
//===----------------------------------------------------------------------===//

rocprofiler::HsaApiTableSnapshot<::CoreApiTable> *CoreSnapshot = nullptr;
rocprofiler::HsaApiTableSnapshot<::AmdExtTable> *AmdExtSnapshot = nullptr;
rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
    *LoaderSnapshot = nullptr;

void toolInit() {
  llvm::Error Err = llvm::Error::success();
  CoreSnapshot = new rocprofiler::HsaApiTableSnapshot<::CoreApiTable>(Err);
  LUTHIER_ABORT_ON_FATAL_ERROR(Err);
  AmdExtSnapshot = new rocprofiler::HsaApiTableSnapshot<::AmdExtTable>(Err);
  LUTHIER_ABORT_ON_FATAL_ERROR(Err);
  LoaderSnapshot =
      new rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>(Err);
  LUTHIER_ABORT_ON_FATAL_ERROR(Err);
}

void toolFini(void *) {
  delete LoaderSnapshot;
  delete AmdExtSnapshot;
  delete CoreSnapshot;
}

rocprofiler_tool_configure_result_t *
toolConfigure(uint32_t, const char *, uint32_t,
              rocprofiler_client_id_t *ClientID) {
  ClientID->name = "Luthier instrumented-kernel-loader e2e unit-test tool";
  toolInit();
  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr, &toolFini, nullptr};
  return &Cfg;
}

//===----------------------------------------------------------------------===//
// Fixture
//===----------------------------------------------------------------------===//

class InstrumentedKernelLoaderE2E : public ::testing::Test {
protected:
  inline static bool HsaUp = false;

  static void SetUpTestSuite() {
    ASSERT_EQ(rocprofiler_force_configure(&toolConfigure),
              ROCPROFILER_STATUS_SUCCESS);
    HsaUp = (hsa_init() == HSA_STATUS_SUCCESS);
  }

  static void TearDownTestSuite() {
    if (HsaUp)
      (void)hsa_shut_down();
  }

  /// Skips unless a GPU is present, the snapshots were captured, and the
  /// device relocatable exists and matches this machine's ISA.
  void SetUp() override {
    LUTHIER_SKIP_IF_NO_HSA_GPU();
    if (CoreSnapshot == nullptr ||
        !CoreSnapshot->wasRegistrationCallbackInvoked())
      GTEST_SKIP() << "rocprofiler did not deliver the HSA table snapshots";

    llvm::SmallVector<hsa_agent_t, 1> GpuAgents;
    llvm::Error Err = hsa::getAllAgentsWithDeviceType<HSA_DEVICE_TYPE_GPU>(
        CoreSnapshot->getTable(), GpuAgents);
    ASSERT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
    llvm::consumeError(std::move(Err));
    if (GpuAgents.empty())
      GTEST_SKIP() << "no GPU agent";
    Agent = GpuAgents.front();

    // The relocatable was compiled for one architecture at build time; a
    // machine with a different GPU cannot load it.
    llvm::SmallVector<hsa_isa_t, 1> ISAs;
    Err = hsa::agentGetSupportedISAs(CoreSnapshot->getTable(), Agent, ISAs);
    ASSERT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
    llvm::consumeError(std::move(Err));
    ASSERT_FALSE(ISAs.empty());
    auto GpuNameOrErr =
        hsa::isaGetGPUName(CoreSnapshot->getTable(), ISAs.front());
    ASSERT_TRUE(static_cast<bool>(GpuNameOrErr))
        << llvm::toString(GpuNameOrErr.takeError());
    if (*GpuNameOrErr != LUTHIER_TEST_INIT_FINI_ARCH)
      GTEST_SKIP() << "device relocatable was built for "
                   << LUTHIER_TEST_INIT_FINI_ARCH << ", this agent is "
                   << *GpuNameOrErr;

    auto BufferOrErr =
        llvm::MemoryBuffer::getFile(LUTHIER_TEST_INIT_FINI_OBJECT,
                                    /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
    if (!BufferOrErr)
      GTEST_SKIP() << "device relocatable "
                   << LUTHIER_TEST_INIT_FINI_OBJECT
                   << " was not built: " << BufferOrErr.getError().message();
    Relocatable = std::move(*BufferOrErr);

    // The two code objects meant to be added to an entry that already holds
    // the one above. Tests that need them skip if they were not built.
    if (auto AddendumOrErr = llvm::MemoryBuffer::getFile(
            LUTHIER_TEST_ADDENDUM_OBJECT, /*IsText=*/false,
            /*RequiresNullTerminator=*/false))
      Addendum = std::move(*AddendumOrErr);
    if (auto AddendumOrErr = llvm::MemoryBuffer::getFile(
            LUTHIER_TEST_ADDENDUM_NO_KERNEL_OBJECT, /*IsText=*/false,
            /*RequiresNullTerminator=*/false))
      AddendumNoKernel = std::move(*AddendumOrErr);

    Loader = std::make_unique<InstrumentedKernelLoaderAndLauncher>(
        *CoreSnapshot, *AmdExtSnapshot, *LoaderSnapshot);

    // loadInstrumented resolves the owning agent of the "original" kernel
    // descriptor through hsa_amd_pointer_info, so the key has to be a real
    // device allocation rather than an arbitrary pointer.
    auto PoolOrErr =
        hsa::agentFindCoarseGrainedPool(AmdExtSnapshot->getTable(), Agent);
    ASSERT_TRUE(static_cast<bool>(PoolOrErr))
        << llvm::toString(PoolOrErr.takeError());
    ASSERT_TRUE(PoolOrErr->has_value());
    auto KeyOrErr = hsa::memoryPoolAllocate(AmdExtSnapshot->getTable(),
                                            **PoolOrErr, /*Size=*/64);
    ASSERT_TRUE(static_cast<bool>(KeyOrErr))
        << llvm::toString(KeyOrErr.takeError());
    OriginalKDStorage = *KeyOrErr;
  }

  void TearDown() override {
    if (Loader)
      llvm::consumeError(Loader->unloadAll());
    Loader.reset();
    if (OriginalKDStorage != nullptr)
      llvm::consumeError(hsa::memoryPoolFree(AmdExtSnapshot->getTable(),
                                             OriginalKDStorage));
    OriginalKDStorage = nullptr;
  }

  const llvm::amdhsa::kernel_descriptor_t *originalKD() const {
    return static_cast<const llvm::amdhsa::kernel_descriptor_t *>(
        OriginalKDStorage);
  }

  /// A fresh copy of the relocatable; \c loadInstrumented takes ownership of
  /// the buffer it is handed.
  std::unique_ptr<llvm::MemoryBuffer> relocatableCopy() const {
    return llvm::MemoryBuffer::getMemBufferCopy(Relocatable->getBuffer(),
                                                "luthier-test-relocatable");
  }

  /// A fresh copy of the additional code object that carries a kernel.
  std::unique_ptr<llvm::MemoryBuffer> addendumCopy() const {
    return llvm::MemoryBuffer::getMemBufferCopy(Addendum->getBuffer(),
                                                "luthier-test-addendum");
  }

  /// A fresh copy of the additional code object that carries no kernel.
  std::unique_ptr<llvm::MemoryBuffer> addendumNoKernelCopy() const {
    return llvm::MemoryBuffer::getMemBufferCopy(
        AddendumNoKernel->getBuffer(), "luthier-test-addendum-no-kernel");
  }

  /// Owns an \c hsa_queue_t for the duration of a test scope. Every
  /// dispatch-packet-override case needs a queue to hand to the loader —
  /// the loader reads the aperture bases out of its AMD extension struct
  /// while filling the hidden args — but never actually pushes the packet
  /// onto it, so a single-slot queue is enough.
  class OwnedQueue {
  public:
    OwnedQueue(const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> &Core,
               hsa_agent_t Agent) {
      auto MinSizeOrErr = hsa::agentGetQueueMinSize(Core.getTable(), Agent);
      if (!MinSizeOrErr) {
        Err = MinSizeOrErr.takeError();
        return;
      }
      auto QueueOrErr = hsa::queueCreate(Core.getTable(), Agent, *MinSizeOrErr);
      if (!QueueOrErr) {
        Err = QueueOrErr.takeError();
        return;
      }
      CoreRef = &Core;
      Queue = *QueueOrErr;
    }
    ~OwnedQueue() {
      if (Queue != nullptr && CoreRef != nullptr)
        llvm::consumeError(hsa::queueDestroy(CoreRef->getTable(), Queue));
      llvm::consumeError(std::move(Err));
    }
    OwnedQueue(const OwnedQueue &) = delete;
    OwnedQueue &operator=(const OwnedQueue &) = delete;

    llvm::Error takeError() { return std::move(Err); }
    hsa_queue_t &operator*() { return *Queue; }
    const hsa_queue_t &operator*() const { return *Queue; }

  private:
    const rocprofiler::HsaApiTableSnapshot<::CoreApiTable> *CoreRef{nullptr};
    hsa_queue_t *Queue{nullptr};
    llvm::Error Err = llvm::Error::success();
  };

  /// A dispatch packet valid enough for the hidden-argument fill to succeed:
  /// non-zero workgroup and grid dimensions, and a 1-D setup. Tests point
  /// \c kernel_object at whatever the current case demands.
  static hsa_kernel_dispatch_packet_t makeDispatchPacket() {
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

  /// Reads an \c int device global out of the loaded instrumented copy.
  llvm::Expected<int> readDeviceInt(llvm::StringRef Name) const {
    auto SymOrErr = Loader->lookupGlobalVariable(Name, originalKD());
    if (!SymOrErr)
      return SymOrErr.takeError();
    auto AddrOrErr =
        hsa::executableSymbolGetAddress(CoreSnapshot->getTable(), *SymOrErr);
    if (!AddrOrErr)
      return AddrOrErr.takeError();
    int Value = 0;
    if (llvm::Error Err = hsa::memoryCopy(
            CoreSnapshot->getTable(), &Value,
            reinterpret_cast<const void *>(*AddrOrErr), sizeof(Value)))
      return std::move(Err);
    return Value;
  }

  hsa_agent_t Agent{};
  std::unique_ptr<llvm::MemoryBuffer> Relocatable;
  std::unique_ptr<llvm::MemoryBuffer> Addendum;
  std::unique_ptr<llvm::MemoryBuffer> AddendumNoKernel;
  std::unique_ptr<InstrumentedKernelLoaderAndLauncher> Loader;
  void *OriginalKDStorage{nullptr};
};

//===----------------------------------------------------------------------===//
// Loading
//===----------------------------------------------------------------------===//

TEST_F(InstrumentedKernelLoaderE2E, LoadsARelocatableAndExposesItsKernel) {
  auto SymOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(SymOrErr))
      << llvm::toString(SymOrErr.takeError());

  auto NameOrErr =
      hsa::executableSymbolGetName(CoreSnapshot->getTable(), *SymOrErr);
  ASSERT_TRUE(static_cast<bool>(NameOrErr))
      << llvm::toString(NameOrErr.takeError());
  EXPECT_EQ(*NameOrErr, "luthierTestKernel.kd");
}

// Adding a code object to a key binds it against what is already there, so a
// byte-for-byte copy of the code object already loaded cannot be added: every
// global it defines is a global the entry already defines at another address.
TEST_F(InstrumentedKernelLoaderE2E, RejectsReloadingAnIdenticalCodeObject) {
  auto FirstOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());

  auto SecondOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  EXPECT_FALSE(static_cast<bool>(SecondOrErr))
      << "a code object that redefines the entry's globals must not load";
  llvm::consumeError(SecondOrErr.takeError());
}

TEST_F(InstrumentedKernelLoaderE2E, DistinctPresetsCoexist) {
  auto FirstOrErr =
      Loader->loadInstrumented(relocatableCopy(), originalKD(), /*Preset=*/0);
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());
  auto SecondOrErr =
      Loader->loadInstrumented(relocatableCopy(), originalKD(), /*Preset=*/1);
  ASSERT_TRUE(static_cast<bool>(SecondOrErr))
      << llvm::toString(SecondOrErr.takeError());
  EXPECT_NE(FirstOrErr->handle, SecondOrErr->handle)
      << "each preset gets its own instrumented copy";
}

TEST_F(InstrumentedKernelLoaderE2E, RejectsANullRelocatable) {
  auto SymOrErr = Loader->loadInstrumented(nullptr, originalKD());
  EXPECT_FALSE(static_cast<bool>(SymOrErr));
  llvm::consumeError(SymOrErr.takeError());
}

TEST_F(InstrumentedKernelLoaderE2E, RejectsANullKernelDescriptor) {
  auto SymOrErr = Loader->loadInstrumented(relocatableCopy(), nullptr);
  EXPECT_FALSE(static_cast<bool>(SymOrErr));
  llvm::consumeError(SymOrErr.takeError());
}

//===----------------------------------------------------------------------===//
// Global constructor / destructor kernels
//===----------------------------------------------------------------------===//

// loadInstrumented must dispatch amdgcn.device.init before it returns, so a
// caller that immediately launches the kernel sees initialized globals.
TEST_F(InstrumentedKernelLoaderE2E, RunsTheConstructorKernelDuringLoad) {
  auto SymOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(SymOrErr))
      << llvm::toString(SymOrErr.takeError());

  auto CtorRanOrErr = readDeviceInt("LuthierTestCtorRan");
  ASSERT_TRUE(static_cast<bool>(CtorRanOrErr))
      << llvm::toString(CtorRanOrErr.takeError());
  EXPECT_EQ(*CtorRanOrErr, 1) << "amdgcn.device.init did not run";

  // The destructor must not have run yet.
  auto DtorRanOrErr = readDeviceInt("LuthierTestDtorRan");
  ASSERT_TRUE(static_cast<bool>(DtorRanOrErr))
      << llvm::toString(DtorRanOrErr.takeError());
  EXPECT_EQ(*DtorRanOrErr, 0) << "amdgcn.device.fini ran too early";
}

// Both variable constructors must have run: 10*2+1 plus 20*2+1.
TEST_F(InstrumentedKernelLoaderE2E, RunsEveryVariableConstructor) {
  auto SymOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(SymOrErr))
      << llvm::toString(SymOrErr.takeError());

  auto SumOrErr = readDeviceInt("LuthierTestCtorSum");
  ASSERT_TRUE(static_cast<bool>(SumOrErr))
      << llvm::toString(SumOrErr.takeError());
  EXPECT_EQ(*SumOrErr, 21 + 41);
}

// The destructor kernel has to run while the executable is still alive, which
// is the whole reason eraseRecordLocked dispatches it before tearing down.
TEST_F(InstrumentedKernelLoaderE2E, RunsTheDestructorKernelOnUnload) {
  auto LoadedOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(LoadedOrErr))
      << llvm::toString(LoadedOrErr.takeError());

  llvm::Error Err = Loader->unloadInstrumentedIfExists(originalKD());
  EXPECT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));

  // The record is gone, so the global is no longer reachable through it. That
  // the unload succeeded is what says the fini dispatch completed; had it
  // faulted or hung, this would have failed or timed out.
  auto AfterOrErr = readDeviceInt("LuthierTestDtorRan");
  EXPECT_FALSE(static_cast<bool>(AfterOrErr))
      << "the record should no longer be cached";
  llvm::consumeError(AfterOrErr.takeError());
}

TEST_F(InstrumentedKernelLoaderE2E, UnloadingAnUnknownKeyIsASuccess) {
  llvm::Error Err = Loader->unloadInstrumentedIfExists(originalKD());
  EXPECT_FALSE(static_cast<bool>(Err)) << "unload must be idempotent";
  llvm::consumeError(std::move(Err));
}

TEST_F(InstrumentedKernelLoaderE2E, UnloadAllRunsEveryDestructor) {
  auto FirstOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD(), 0);
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());
  auto SecondOrErr =
      Loader->loadInstrumented(relocatableCopy(), originalKD(), 1);
  ASSERT_TRUE(static_cast<bool>(SecondOrErr))
      << llvm::toString(SecondOrErr.takeError());

  llvm::Error Err = Loader->unloadAll();
  EXPECT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));

  Err = Loader->unloadAll();
  EXPECT_FALSE(static_cast<bool>(Err)) << "unloadAll must be idempotent";
  llvm::consumeError(std::move(Err));
}

//===----------------------------------------------------------------------===//
// Hostcall servicing
//===----------------------------------------------------------------------===//

// The constructor calls printf, which is a SERVICE_PRINTF hostcall: the wave
// spins until the host answers it. If the loader failed to stand up a hostcall
// buffer, register it with a listener, or answer the packet, this dispatch
// never completes and the test hangs — so reaching the assertion at all is
// most of what is being checked here.
TEST_F(InstrumentedKernelLoaderE2E, ServicesTheConstructorsPrintfHostcall) {
  testing::internal::CaptureStdout();
  auto SymOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  std::fflush(stdout);
  const std::string Out = testing::internal::GetCapturedStdout();

  ASSERT_TRUE(static_cast<bool>(SymOrErr))
      << llvm::toString(SymOrErr.takeError());
  EXPECT_NE(Out.find("luthier-test: ctor ran"), std::string::npos)
      << "the constructor's printf hostcall was not serviced; got: " << Out;
  // The format string's argument has to have been rendered from the message
  // the device streamed, not dropped. It reads a statically initialized global
  // rather than one a variable constructor writes, because llvm.global_ctors
  // does not order this constructor against the variable initializers.
  EXPECT_NE(Out.find("arg=62"), std::string::npos) << "got: " << Out;
}

TEST_F(InstrumentedKernelLoaderE2E, ServicesTheDestructorsPrintfHostcall) {
  auto LoadedOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(LoadedOrErr))
      << llvm::toString(LoadedOrErr.takeError());

  testing::internal::CaptureStdout();
  llvm::Error Err = Loader->unloadInstrumentedIfExists(originalKD());
  std::fflush(stdout);
  const std::string Out = testing::internal::GetCapturedStdout();

  EXPECT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));
  EXPECT_NE(Out.find("luthier-test: dtor ran"), std::string::npos)
      << "the destructor's printf hostcall was not serviced; got: " << Out;
}

// One listener serves every record, and a buffer must be deregistered before
// its record's storage goes away. Cycling several records through the loader
// exercises that registration/deregistration handshake against a live
// listener thread.
TEST_F(InstrumentedKernelLoaderE2E, SurvivesRepeatedLoadUnloadCycles) {
  for (unsigned I = 0; I < 4; ++I) {
    testing::internal::CaptureStdout();
    auto SymOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
    ASSERT_TRUE(static_cast<bool>(SymOrErr))
        << "cycle " << I << ": " << llvm::toString(SymOrErr.takeError());
    llvm::Error Err = Loader->unloadInstrumentedIfExists(originalKD());
    std::fflush(stdout);
    const std::string Out = testing::internal::GetCapturedStdout();

    ASSERT_FALSE(static_cast<bool>(Err))
        << "cycle " << I << ": " << llvm::toString(std::move(Err));
    llvm::consumeError(std::move(Err));
    EXPECT_NE(Out.find("ctor ran"), std::string::npos) << "cycle " << I;
    EXPECT_NE(Out.find("dtor ran"), std::string::npos) << "cycle " << I;
  }
}

//===----------------------------------------------------------------------===//
// Additional code objects
//
// A second loadInstrumented for a key that already has code objects adds
// another one, bound against the globals the earlier ones already loaded.
// AdditionalCodeObject.hip reads LuthierTestCtorSum — a global that lives in
// the *first* code object — inside its own amdgcn.device.init, and stashes
// what it saw in LuthierTestAddendumSawSum, so the binding is checked by
// reading that back rather than by dispatching anything here.
//===----------------------------------------------------------------------===//

/// Skips unless the additional code objects were built.
#define LUTHIER_SKIP_IF_NO_ADDENDUM()                                          \
  do {                                                                         \
    if (Addendum == nullptr || AddendumNoKernel == nullptr)                    \
      GTEST_SKIP() << "the additional code objects were not built";            \
  } while (0)

TEST_F(InstrumentedKernelLoaderE2E, AddsASecondCodeObjectToAnExistingKey) {
  LUTHIER_SKIP_IF_NO_ADDENDUM();
  auto FirstOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());

  auto SecondOrErr = Loader->loadInstrumented(addendumCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(SecondOrErr))
      << llvm::toString(SecondOrErr.takeError());

  auto NameOrErr =
      hsa::executableSymbolGetName(CoreSnapshot->getTable(), *SecondOrErr);
  ASSERT_TRUE(static_cast<bool>(NameOrErr))
      << llvm::toString(NameOrErr.takeError());
  EXPECT_EQ(*NameOrErr, "luthierTestAddendumKernel.kd")
      << "an addition carrying a kernel hands that kernel back";
}

// The whole point of the feature: the addition's reference to a global defined
// by the code object already loaded under the key has to resolve to the
// address that code object was loaded at.
TEST_F(InstrumentedKernelLoaderE2E, AdditionResolvesAgainstTheEarlierObject) {
  LUTHIER_SKIP_IF_NO_ADDENDUM();
  auto FirstOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());
  // The first code object's own constructors have run by now, so the global
  // the addition is about to read holds 21+41.
  auto SumOrErr = readDeviceInt("LuthierTestCtorSum");
  ASSERT_TRUE(static_cast<bool>(SumOrErr))
      << llvm::toString(SumOrErr.takeError());
  ASSERT_EQ(*SumOrErr, 21 + 41);

  auto SecondOrErr = Loader->loadInstrumented(addendumCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(SecondOrErr))
      << llvm::toString(SecondOrErr.takeError());

  // Written by the addition's constructor, out of the other code object's
  // global. Its initial value is -1, which is what it keeps if the reference
  // was left dangling.
  auto SawOrErr = readDeviceInt("LuthierTestAddendumSawSum");
  ASSERT_TRUE(static_cast<bool>(SawOrErr))
      << llvm::toString(SawOrErr.takeError());
  EXPECT_EQ(*SawOrErr, 21 + 41)
      << "the addition did not read the earlier code object's global";
}

// Only the first code object under a key has to carry a kernel; an addition
// may be nothing but a constructor and globals.
TEST_F(InstrumentedKernelLoaderE2E, AddsACodeObjectCarryingNoKernel) {
  LUTHIER_SKIP_IF_NO_ADDENDUM();
  auto FirstOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());

  auto SecondOrErr =
      Loader->loadInstrumented(addendumNoKernelCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(SecondOrErr))
      << llvm::toString(SecondOrErr.takeError());
  EXPECT_EQ(SecondOrErr->handle, 0u)
      << "an addition carrying no kernel hands back a zero-handle symbol";

  // It still loaded and still ran its constructor against the earlier object.
  auto SawOrErr = readDeviceInt("LuthierTestAddendumSawSum");
  ASSERT_TRUE(static_cast<bool>(SawOrErr))
      << llvm::toString(SawOrErr.takeError());
  EXPECT_EQ(*SawOrErr, 21 + 41);
}

// Globals of every code object under the key are reachable through the entry.
TEST_F(InstrumentedKernelLoaderE2E, LooksUpGlobalsAcrossEveryCodeObject) {
  LUTHIER_SKIP_IF_NO_ADDENDUM();
  auto FirstOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());
  auto SecondOrErr = Loader->loadInstrumented(addendumCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(SecondOrErr))
      << llvm::toString(SecondOrErr.takeError());

  // One from the first code object, one from the addition.
  auto FromFirstOrErr = readDeviceInt("LuthierTestCtorRan");
  EXPECT_TRUE(static_cast<bool>(FromFirstOrErr))
      << llvm::toString(FromFirstOrErr.takeError());
  auto FromAdditionOrErr = readDeviceInt("LuthierTestAddendumSawSum");
  EXPECT_TRUE(static_cast<bool>(FromAdditionOrErr))
      << llvm::toString(FromAdditionOrErr.takeError());

  // The addition was handed the earlier object's global as an external agent
  // variable, so both name the one address it actually lives at.
  auto FirstSymOrErr =
      Loader->lookupGlobalVariable("LuthierTestCtorSum", originalKD());
  ASSERT_TRUE(static_cast<bool>(FirstSymOrErr))
      << llvm::toString(FirstSymOrErr.takeError());
  auto AddrOrErr =
      hsa::executableSymbolGetAddress(CoreSnapshot->getTable(), *FirstSymOrErr);
  ASSERT_TRUE(static_cast<bool>(AddrOrErr))
      << llvm::toString(AddrOrErr.takeError());
  EXPECT_NE(*AddrOrErr, 0u);
}

// The instrumented kernel a dispatch runs is the first code object's; adding
// more must not move it.
TEST_F(InstrumentedKernelLoaderE2E, AdditionsDoNotChangeTheDispatchedKernel) {
  LUTHIER_SKIP_IF_NO_ADDENDUM();
  auto FirstOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());
  auto FirstKOOrErr =
      hsa::executableSymbolGetAddress(CoreSnapshot->getTable(), *FirstOrErr);
  ASSERT_TRUE(static_cast<bool>(FirstKOOrErr))
      << llvm::toString(FirstKOOrErr.takeError());

  auto SecondOrErr = Loader->loadInstrumented(addendumCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(SecondOrErr))
      << llvm::toString(SecondOrErr.takeError());

  OwnedQueue Queue(*CoreSnapshot, Agent);
  ASSERT_FALSE(static_cast<bool>(Queue.takeError()));
  hsa_kernel_dispatch_packet_t Packet = makeDispatchPacket();
  Packet.kernel_object = reinterpret_cast<uint64_t>(originalKD());
  auto BufOrErr = Loader->overrideWithInstrumented(Packet, *Queue);
  ASSERT_TRUE(static_cast<bool>(BufOrErr))
      << llvm::toString(BufOrErr.takeError());
  EXPECT_EQ(Packet.kernel_object, *FirstKOOrErr)
      << "the first code object's kernel stays the instrumented one";
}

// Unloading the entry has to take every code object with it, newest first.
TEST_F(InstrumentedKernelLoaderE2E, UnloadTearsDownEveryCodeObject) {
  LUTHIER_SKIP_IF_NO_ADDENDUM();
  auto FirstOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());
  auto SecondOrErr = Loader->loadInstrumented(addendumCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(SecondOrErr))
      << llvm::toString(SecondOrErr.takeError());

  llvm::Error Err = Loader->unloadInstrumentedIfExists(originalKD());
  EXPECT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));

  // Neither code object's globals are reachable any more.
  auto FirstGoneOrErr = readDeviceInt("LuthierTestCtorRan");
  EXPECT_FALSE(static_cast<bool>(FirstGoneOrErr));
  llvm::consumeError(FirstGoneOrErr.takeError());
  auto SecondGoneOrErr = readDeviceInt("LuthierTestAddendumSawSum");
  EXPECT_FALSE(static_cast<bool>(SecondGoneOrErr));
  llvm::consumeError(SecondGoneOrErr.takeError());

  // And the key is free again, so a fresh load succeeds.
  auto ReloadOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  EXPECT_TRUE(static_cast<bool>(ReloadOrErr))
      << llvm::toString(ReloadOrErr.takeError());
  llvm::consumeError(ReloadOrErr.takeError());
}

// Presets stay independent: an addition joins the entry it was loaded for.
TEST_F(InstrumentedKernelLoaderE2E, AdditionsAreScopedToTheirPreset) {
  LUTHIER_SKIP_IF_NO_ADDENDUM();
  auto FirstOrErr =
      Loader->loadInstrumented(relocatableCopy(), originalKD(), /*Preset=*/0);
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());
  auto OtherOrErr =
      Loader->loadInstrumented(relocatableCopy(), originalKD(), /*Preset=*/1);
  ASSERT_TRUE(static_cast<bool>(OtherOrErr))
      << llvm::toString(OtherOrErr.takeError());

  auto AddedOrErr =
      Loader->loadInstrumented(addendumCopy(), originalKD(), /*Preset=*/0);
  ASSERT_TRUE(static_cast<bool>(AddedOrErr))
      << llvm::toString(AddedOrErr.takeError());

  auto PresentOrErr = Loader->lookupGlobalVariable("LuthierTestAddendumSawSum",
                                                   originalKD(), /*Preset=*/0);
  EXPECT_TRUE(static_cast<bool>(PresentOrErr))
      << llvm::toString(PresentOrErr.takeError());
  auto MissingOrErr = Loader->lookupGlobalVariable("LuthierTestAddendumSawSum",
                                                   originalKD(), /*Preset=*/1);
  EXPECT_FALSE(static_cast<bool>(MissingOrErr))
      << "preset 1 never had the addition loaded into it";
  llvm::consumeError(MissingOrErr.takeError());
}

//===----------------------------------------------------------------------===//
// Global variable lookup
//===----------------------------------------------------------------------===//

TEST_F(InstrumentedKernelLoaderE2E, RejectsAnUnknownGlobalVariable) {
  auto LoadedOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(LoadedOrErr))
      << llvm::toString(LoadedOrErr.takeError());
  auto SymOrErr =
      Loader->lookupGlobalVariable("NoSuchGlobal", originalKD());
  EXPECT_FALSE(static_cast<bool>(SymOrErr));
  llvm::consumeError(SymOrErr.takeError());
}

TEST_F(InstrumentedKernelLoaderE2E, RejectsALookupAgainstAnUnloadedKey) {
  auto SymOrErr =
      Loader->lookupGlobalVariable("LuthierTestCtorRan", originalKD());
  EXPECT_FALSE(static_cast<bool>(SymOrErr));
  llvm::consumeError(SymOrErr.takeError());
}

//===----------------------------------------------------------------------===//
// Dispatch packet override
//===----------------------------------------------------------------------===//

TEST_F(InstrumentedKernelLoaderE2E, OverridesTheDispatchPacketsKernelObject) {
  auto SymOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(SymOrErr))
      << llvm::toString(SymOrErr.takeError());
  auto InstrumentedKOOrErr =
      hsa::executableSymbolGetAddress(CoreSnapshot->getTable(), *SymOrErr);
  ASSERT_TRUE(static_cast<bool>(InstrumentedKOOrErr))
      << llvm::toString(InstrumentedKOOrErr.takeError());

  OwnedQueue Queue(*CoreSnapshot, Agent);
  ASSERT_FALSE(static_cast<bool>(Queue.takeError()));
  hsa_kernel_dispatch_packet_t Packet = makeDispatchPacket();
  Packet.kernel_object = reinterpret_cast<uint64_t>(originalKD());
  Packet.private_segment_size = 0;

  auto BufOrErr = Loader->overrideWithInstrumented(Packet, *Queue);
  ASSERT_TRUE(static_cast<bool>(BufOrErr))
      << llvm::toString(BufOrErr.takeError());

  EXPECT_EQ(Packet.kernel_object, *InstrumentedKOOrErr);
}

TEST_F(InstrumentedKernelLoaderE2E, OverrideNeverLowersTheScratchRequest) {
  auto LoadedOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(LoadedOrErr))
      << llvm::toString(LoadedOrErr.takeError());

  constexpr uint32_t CallerRequest = 1u << 20;
  OwnedQueue Queue(*CoreSnapshot, Agent);
  ASSERT_FALSE(static_cast<bool>(Queue.takeError()));
  hsa_kernel_dispatch_packet_t Packet = makeDispatchPacket();
  Packet.kernel_object = reinterpret_cast<uint64_t>(originalKD());
  Packet.private_segment_size = CallerRequest;

  auto BufOrErr = Loader->overrideWithInstrumented(Packet, *Queue);
  ASSERT_TRUE(static_cast<bool>(BufOrErr))
      << llvm::toString(BufOrErr.takeError());

  EXPECT_GE(Packet.private_segment_size, CallerRequest)
      << "the override must not shrink what the caller already reserved";
}

// The instrumented kernel expects an extended kernarg buffer: an 8-byte app
// kernarg prefix followed by the COV5 hidden block. Verify the loader stands
// one up, points the packet at it, and copies the original kernarg pointer
// into the prefix.
TEST_F(InstrumentedKernelLoaderE2E, OverrideBuildsAnExtendedKernargBuffer) {
  auto LoadedOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(LoadedOrErr))
      << llvm::toString(LoadedOrErr.takeError());

  // The address the app dispatch would have carried. Only its value is
  // asserted; the loader does not dereference it.
  const auto AppKernargAddress =
      reinterpret_cast<void *>(uintptr_t{0xF00DFACEu});

  OwnedQueue Queue(*CoreSnapshot, Agent);
  ASSERT_FALSE(static_cast<bool>(Queue.takeError()));
  hsa_kernel_dispatch_packet_t Packet = makeDispatchPacket();
  Packet.kernel_object = reinterpret_cast<uint64_t>(originalKD());
  Packet.kernarg_address = AppKernargAddress;

  auto BufOrErr = Loader->overrideWithInstrumented(Packet, *Queue);
  ASSERT_TRUE(static_cast<bool>(BufOrErr))
      << llvm::toString(BufOrErr.takeError());
  ExtendedKernargBuffer Buf = std::move(*BufOrErr);

  ASSERT_FALSE(Buf.empty());
  ASSERT_NE(Buf.getKernargAddress(), nullptr);
  EXPECT_EQ(Packet.kernarg_address, Buf.getKernargAddress())
      << "the packet must point at the extended buffer, not the app's kernarg";

  const void *Prefix = nullptr;
  ASSERT_FALSE(static_cast<bool>(
      hsa::memoryCopy(CoreSnapshot->getTable(), &Prefix,
                      Buf.getKernargAddress(), sizeof(Prefix))));
  EXPECT_EQ(Prefix, AppKernargAddress)
      << "the extended buffer's prefix must hold the original kernarg address";

  // Explicit release is what the caller uses in a completion callback; verify
  // it succeeds and empties the handle.
  llvm::Error Rel = Buf.release();
  EXPECT_FALSE(static_cast<bool>(Rel)) << llvm::toString(std::move(Rel));
  llvm::consumeError(std::move(Rel));
  EXPECT_TRUE(Buf.empty());
}

// A second release from the same handle is a no-op — the RAII destructor
// relies on this to not double-free when a caller has already released
// explicitly in a completion callback.
TEST_F(InstrumentedKernelLoaderE2E, ExtendedKernargBufferReleaseIsIdempotent) {
  auto LoadedOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(LoadedOrErr))
      << llvm::toString(LoadedOrErr.takeError());

  OwnedQueue Queue(*CoreSnapshot, Agent);
  ASSERT_FALSE(static_cast<bool>(Queue.takeError()));
  hsa_kernel_dispatch_packet_t Packet = makeDispatchPacket();
  Packet.kernel_object = reinterpret_cast<uint64_t>(originalKD());

  auto BufOrErr = Loader->overrideWithInstrumented(Packet, *Queue);
  ASSERT_TRUE(static_cast<bool>(BufOrErr))
      << llvm::toString(BufOrErr.takeError());
  ExtendedKernargBuffer Buf = std::move(*BufOrErr);

  llvm::Error First = Buf.release();
  ASSERT_FALSE(static_cast<bool>(First)) << llvm::toString(std::move(First));
  llvm::consumeError(std::move(First));

  llvm::Error Second = Buf.release();
  EXPECT_FALSE(static_cast<bool>(Second))
      << "a second release must be a no-op";
  llvm::consumeError(std::move(Second));
}

TEST_F(InstrumentedKernelLoaderE2E, OverrideRejectsAnUnloadedKernelObject) {
  OwnedQueue Queue(*CoreSnapshot, Agent);
  ASSERT_FALSE(static_cast<bool>(Queue.takeError()));
  hsa_kernel_dispatch_packet_t Packet = makeDispatchPacket();
  Packet.kernel_object = reinterpret_cast<uint64_t>(originalKD());

  auto BufOrErr = Loader->overrideWithInstrumented(Packet, *Queue);
  EXPECT_FALSE(static_cast<bool>(BufOrErr));
  llvm::consumeError(BufOrErr.takeError());
}

} // namespace
