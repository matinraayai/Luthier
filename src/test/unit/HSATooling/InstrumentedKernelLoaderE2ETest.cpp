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

TEST_F(InstrumentedKernelLoaderE2E, RejectsASecondLoadOfTheSameKey) {
  auto FirstOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(FirstOrErr))
      << llvm::toString(FirstOrErr.takeError());

  auto SecondOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  EXPECT_FALSE(static_cast<bool>(SecondOrErr))
      << "the same (kernel descriptor, preset) must not load twice";
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

  hsa_kernel_dispatch_packet_t Packet{};
  Packet.kernel_object = reinterpret_cast<uint64_t>(originalKD());
  Packet.private_segment_size = 0;

  llvm::Error Err = Loader->overrideWithInstrumented(Packet);
  ASSERT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));

  EXPECT_EQ(Packet.kernel_object, *InstrumentedKOOrErr);
}

TEST_F(InstrumentedKernelLoaderE2E, OverrideNeverLowersTheScratchRequest) {
  auto LoadedOrErr = Loader->loadInstrumented(relocatableCopy(), originalKD());
  ASSERT_TRUE(static_cast<bool>(LoadedOrErr))
      << llvm::toString(LoadedOrErr.takeError());

  constexpr uint32_t CallerRequest = 1u << 20;
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.kernel_object = reinterpret_cast<uint64_t>(originalKD());
  Packet.private_segment_size = CallerRequest;

  llvm::Error Err = Loader->overrideWithInstrumented(Packet);
  ASSERT_FALSE(static_cast<bool>(Err)) << llvm::toString(std::move(Err));
  llvm::consumeError(std::move(Err));

  EXPECT_GE(Packet.private_segment_size, CallerRequest)
      << "the override must not shrink what the caller already reserved";
}

TEST_F(InstrumentedKernelLoaderE2E, OverrideRejectsAnUnloadedKernelObject) {
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.kernel_object = reinterpret_cast<uint64_t>(originalKD());

  llvm::Error Err = Loader->overrideWithInstrumented(Packet);
  EXPECT_TRUE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

} // namespace
