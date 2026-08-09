//===-- RocprofilerToolE2ETest.cpp ------------------------------*- C++ -*-===//
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
/// End-to-end unit tests that run this test binary as an in-process
/// rocprofiler-sdk tool (via \c rocprofiler_force_configure) and exercise the
/// real registration flow on actual hardware. GPU-dependent tests skip when no
/// AMD GPU is present / the HSA runtime fails to initialize.
//===----------------------------------------------------------------------===//
#include "common/GpuAvailability.h"

#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/Rocprofiler/ApiTableWrapperInstaller.h"

#include <gtest/gtest.h>
#include <llvm/Support/Error.h>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <mutex>
#include <tuple>
#include <vector>

using namespace luthier::rocprofiler;

namespace {

//===----------------------------------------------------------------------===//
// In-process tool state (created during rocprofiler configuration)
//===----------------------------------------------------------------------===//

HsaApiTableSnapshot<::CoreApiTable> *CoreSnapshot = nullptr;
HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER> *LoaderSnapshot = nullptr;
// HSA_EXTENSION_FINALIZER is deliberately not snapshotted here. It is the HSAIL
// finalizer, which the runtime no longer implements: it still answers
// hsa_system_extension_supported, but hsa_system_get_major_extension_table
// fails for it, and the snapshot reports that as a fatal error from inside
// rocprofiler's registration callback, taking the whole binary down with it.
HsaExtensionTableSnapshot<HSA_EXTENSION_IMAGES> *ImageSnapshot = nullptr;
HsaApiTableWrapperInstaller<::CoreApiTable> *Wrapper = nullptr;
HipApiTableSnapshot<ROCPROFILER_HIP_RUNTIME_TABLE> *HipSnapshot = nullptr;

using HsaInitFn = decltype(::CoreApiTable::hsa_init_fn);
// Pointer-to-member type for CoreApiTable::hsa_init_fn. Spelled via decltype of
// the member to avoid the ambiguous `HsaInitFn ::CoreApiTable::*` form (which
// some parsers misread as a qualified name rooted at the typedef HsaInitFn).
using HsaInitFnMemberPtr = decltype(&::CoreApiTable::hsa_init_fn);
HsaInitFn UnderlyingInit = nullptr;
std::atomic<int> WrapperInitCalls{0};
hsa_status_t hsaInitWrapper() {
  WrapperInitCalls.fetch_add(1);
  return UnderlyingInit ? UnderlyingInit() : HSA_STATUS_SUCCESS;
}

// Independent observer to witness the lib_instance values rocprofiler delivers.
std::mutex ObservedMutex;
std::vector<uint64_t> ObservedHsaInstances;
void hsaInstanceObserver(rocprofiler_intercept_table_t, uint64_t, uint64_t Inst,
                         void **, uint64_t, void *) noexcept {
  std::lock_guard<std::mutex> Lock(ObservedMutex);
  ObservedHsaInstances.push_back(Inst);
}

uint64_t maxObservedInstance() {
  std::lock_guard<std::mutex> Lock(ObservedMutex);
  uint64_t Max = 0;
  for (uint64_t I : ObservedHsaInstances)
    Max = std::max(Max, I);
  return Max;
}

void toolInit() {
  llvm::Error Err = llvm::Error::success();
  CoreSnapshot = new HsaApiTableSnapshot<::CoreApiTable>(Err);
  LUTHIER_ABORT_ON_FATAL_ERROR(Err);
  LoaderSnapshot = new HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>(Err);
  LUTHIER_ABORT_ON_FATAL_ERROR(Err);
  ImageSnapshot = new HsaExtensionTableSnapshot<HSA_EXTENSION_IMAGES>(Err);
  LUTHIER_ABORT_ON_FATAL_ERROR(Err);

  std::tuple<HsaInitFnMemberPtr, HsaInitFn &, HsaInitFn> Spec(
      &::CoreApiTable::hsa_init_fn, UnderlyingInit, &hsaInitWrapper);
  Wrapper = new HsaApiTableWrapperInstaller<::CoreApiTable>(Err, Spec);
  LUTHIER_ABORT_ON_FATAL_ERROR(Err);

  HipSnapshot = new HipApiTableSnapshot<ROCPROFILER_HIP_RUNTIME_TABLE>(Err);
  LUTHIER_ABORT_ON_FATAL_ERROR(Err);

  // Independent witness of lib_instance progression.
  (void)rocprofiler_at_intercept_table_registration(
      &hsaInstanceObserver, ROCPROFILER_HSA_TABLE, nullptr);
}

void toolFini(void *) {
  // Runs during rocprofiler finalization (fini_status != 0), so the providers
  // destruct on the safe path.
  delete Wrapper;
  delete ImageSnapshot;
  delete LoaderSnapshot;
  delete CoreSnapshot;
  delete HipSnapshot;
}

rocprofiler_tool_configure_result_t *
toolConfigure(uint32_t, const char *, uint32_t,
              rocprofiler_client_id_t *ClientID) {
  ClientID->name = "Luthier rocprofiler e2e unit-test tool";
  toolInit();
  static auto Cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), nullptr, &toolFini, nullptr};
  return &Cfg;
}

//===----------------------------------------------------------------------===//
// Fixture: force-configure once, bring up HSA, detect GPU
//===----------------------------------------------------------------------===//

class RocprofilerToolE2E : public ::testing::Test {
protected:
  inline static bool HsaUp = false;

  static void SetUpTestSuite() {
    // Register this binary as an in-process rocprofiler tool. This succeeds and
    // initializes rocprofiler regardless of GPU availability.
    ASSERT_EQ(rocprofiler_force_configure(&toolConfigure),
              ROCPROFILER_STATUS_SUCCESS);
    // Bring up HSA: this is what triggers rocprofiler's HSA table registration
    // (and therefore our snapshot/wrapper callbacks). Fails with no GPU/kfd.
    HsaUp = (hsa_init() == HSA_STATUS_SUCCESS);
  }

  static void TearDownTestSuite() {
    if (HsaUp)
      (void)hsa_shut_down();
  }
};

//===----------------------------------------------------------------------===//
// Always-run: registration failure is safe to destroy
//===----------------------------------------------------------------------===//

// After SetUpTestSuite force-configured rocprofiler (status > 0), a newly
// constructed provider's registration is rejected, so it must be safe to
// destroy (no dangling registration, no abort). Needs no GPU.
TEST_F(RocprofilerToolE2E, RegistrationFailureIsSafeToDestroy) {
  llvm::Error Err = llvm::Error::success();
  auto *Snap = new HsaApiTableSnapshot<::CoreApiTable>(Err);
  EXPECT_TRUE(static_cast<bool>(Err)) << "expected registration to be locked";
  llvm::consumeError(std::move(Err));
  delete Snap; // must not abort
  SUCCEED();
}

//===----------------------------------------------------------------------===//
// GPU-gated: real registration flow
//===----------------------------------------------------------------------===//

TEST_F(RocprofilerToolE2E, HsaCoreSnapshotCaptured) {
  LUTHIER_SKIP_IF_NO_HSA_GPU();
  ASSERT_TRUE(CoreSnapshot->wasRegistrationCallbackInvoked());
  auto Table = CoreSnapshot->getTable();
  // A real, callable core function pointer was captured.
  EXPECT_NE(&Table.getFunction<&::CoreApiTable::hsa_system_get_info_fn>(),
            nullptr);
}

TEST_F(RocprofilerToolE2E, AmdLoaderExtensionFilled) {
  LUTHIER_SKIP_IF_NO_HSA_GPU();
  ASSERT_TRUE(LoaderSnapshot->wasRegistrationCallbackInvoked());
  EXPECT_NE(LoaderSnapshot->getTable().hsa_ven_amd_loader_query_host_address,
            nullptr);
}

// Validates the hsa_system_get_major_extension_table fix: images previously
// failed (version-major mismatch) via hsa_system_get_extension_table.
TEST_F(RocprofilerToolE2E, ImageExtensionFilled) {
  LUTHIER_SKIP_IF_NO_HSA_GPU();
  ASSERT_TRUE(ImageSnapshot->wasRegistrationCallbackInvoked());
  EXPECT_NE(ImageSnapshot->getTable().hsa_ext_image_create, nullptr);
}

TEST_F(RocprofilerToolE2E, WrapperInstalledInLiveTable) {
  LUTHIER_SKIP_IF_NO_HSA_GPU();
  // installWrapperEntry saved the runtime's original hsa_init into
  // UnderlyingInit and replaced the live entry with hsaInitWrapper.
  EXPECT_NE(UnderlyingInit, nullptr);
}

// Full HSA finalize -> re-init cycle: lib_instance increments, the snapshot is
// unchanged (capture-once), and wrappers are re-installed.
TEST_F(RocprofilerToolE2E, ReinitIncrementsInstanceAndPreservesSnapshot) {
  LUTHIER_SKIP_IF_NO_HSA_GPU();
  ASSERT_TRUE(CoreSnapshot->wasRegistrationCallbackInvoked());
  auto *CapturedInit =
      &CoreSnapshot->getTable().getFunction<&::CoreApiTable::hsa_init_fn>();
  UnderlyingInit = nullptr; // observe re-install on the next cycle

  // Drop HSA's refcount to zero and bring it back: triggers re-registration.
  (void)hsa_shut_down();
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  EXPECT_GE(maxObservedInstance(), 1u) << "re-init should bump lib_instance";
  EXPECT_EQ(
      &CoreSnapshot->getTable().getFunction<&::CoreApiTable::hsa_init_fn>(),
      CapturedInit)
      << "snapshot must remain the first capture";
  EXPECT_NE(UnderlyingInit, nullptr) << "wrappers must be re-installed";
}

TEST_F(RocprofilerToolE2E, HipRuntimeSnapshotCaptured) {
  LUTHIER_SKIP_IF_NO_HIP_GPU();
  // A HIP runtime call publishes the dispatch table and registers it with
  // rocprofiler, firing our HIP snapshot callback.
  int Count = 0;
  (void)hipGetDeviceCount(&Count);
  ASSERT_TRUE(HipSnapshot->wasRegistrationCallbackInvoked());
}

} // namespace
