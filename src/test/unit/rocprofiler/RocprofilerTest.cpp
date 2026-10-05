//===-- RocprofilerTest.cpp -------------------------------------*- C++ -*-===//
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
/// GPU-free unit tests for the Luthier rocprofiler library. These exercise the
/// snapshot/wrapper callback logic by delivering fabricated API tables directly
/// (no GPU, no rocprofiler-sdk runtime), plus compile-time traits and the fatal
/// paths via death tests.
///
/// \note This binary is NOT a rocprofiler tool, so provider construction
/// registers with rocprofiler-sdk while it is uninitialized (status 0) and
/// succeeds. Such a provider's destructor would (correctly) abort because
/// rocprofiler is "up and not finalizing", so the non-death logic tests
/// intentionally LEAK their providers (heap, never freed) to avoid that — the
/// destructor behavior itself is covered by a dedicated death test.
//===----------------------------------------------------------------------===//
#include "HipApiTable.h"
#include "common/GpuAvailability.h"
#include "common/HsaApiTable.h"
#include "common/ProviderTestAccess.h"

#include "luthier/Rocprofiler/ApiTableSnapshot.h"
#include "luthier/Rocprofiler/ApiTableWrapperInstaller.h"
#include "luthier/Rocprofiler/RocprofilerSdkLibrary.h"

#include <gtest/gtest.h>
#include <llvm/Support/Error.h>

#include <tuple>
#include <type_traits>

using namespace luthier::rocprofiler;
using luthier::test::buildCoreApiTable;
using luthier::test::buildHipDispatchTable;
using luthier::test::buildHsaApiTable;
template <rocprofiler_intercept_table_t T>
using Access = luthier::test::ProviderTestAccess<T>;

namespace {

/// Consumes the (possibly-failure) Error an out-param ctor leaves behind.
void discard(llvm::Error E) { llvm::consumeError(std::move(E)); }

/// The rocprofiler-sdk library handle shared by all providers in this test;
/// Intentionally leaked since leaked providers reference it
const luthier::DynamicLibrary &rocprofilerLib() {
  static auto *Lib = [] {
    auto LibOrErr = openRocprofilerSdkLibrary();
    if (!LibOrErr)
      llvm::report_fatal_error(LibOrErr.takeError());
    return new luthier::DynamicLibrary(std::move(*LibOrErr));
  }();
  return *Lib;
}

//===----------------------------------------------------------------------===//
// Compile-time traits / SFINAE
//===----------------------------------------------------------------------===//

static_assert(
    std::is_same_v<ApiTableEnumInfo<ROCPROFILER_HSA_TABLE>::ApiTableType,
                   ::HsaApiTable>);
static_assert(std::is_same_v<
              ApiTableEnumInfo<ROCPROFILER_HIP_RUNTIME_TABLE>::ApiTableType,
              ::HipDispatchTable>);
static_assert(std::is_same_v<
              ApiTableEnumInfo<ROCPROFILER_HIP_COMPILER_TABLE>::ApiTableType,
              ::HipCompilerDispatchTable>);
static_assert(ApiTableEnumInfo<ROCPROFILER_HSA_TABLE>::NumApiTables == 1);

TEST(RocprofilerTraits, EnumInfoNames) {
  EXPECT_STREQ(ApiTableEnumInfo<ROCPROFILER_HSA_TABLE>::ApiTableName, "HSA");
  EXPECT_STREQ(ApiTableEnumInfo<ROCPROFILER_HIP_RUNTIME_TABLE>::ApiTableName,
               "HIP Runtime");
  EXPECT_STREQ(ApiTableEnumInfo<ROCPROFILER_HIP_COMPILER_TABLE>::ApiTableName,
               "HIP Compiler");
}

//===----------------------------------------------------------------------===//
// HSA snapshot — capture-once + re-fire ignored
//===----------------------------------------------------------------------===//

TEST(HsaApiTableSnapshotTest, CapturesFirstAndIgnoresReinit) {
  llvm::Error Err = llvm::Error::success();
  auto *Snap = new HsaApiTableSnapshot<::CoreApiTable>(
      rocprofilerLib(), Err); // leaked on purpose
  discard(std::move(Err));

  ::CoreApiTable CoreA = buildCoreApiTable(); // hsa_init_fn == &hsa_init
  ::HsaApiTable TblA = buildHsaApiTable(&CoreA);
  ::HsaApiTable *ArrA[1] = {&TblA};
  Access<ROCPROFILER_HSA_TABLE>::deliver(Snap, ROCPROFILER_HSA_TABLE, 0, 0,
                                         reinterpret_cast<void **>(ArrA), 1);

  ASSERT_TRUE(Snap->wasRegistrationCallbackInvoked());
  EXPECT_EQ(&Snap->getTable().getFunction<&::CoreApiTable::hsa_init_fn>(),
            &hsa_init);

  // Re-init (instance 1) with a table whose hsa_init_fn is a different (but
  // identically-typed) function. Capture-once must ignore it.
  ::CoreApiTable CoreB = buildCoreApiTable();
  CoreB.hsa_init_fn = &hsa_shut_down; // distinct sentinel
  ::HsaApiTable TblB = buildHsaApiTable(&CoreB);
  ::HsaApiTable *ArrB[1] = {&TblB};
  Access<ROCPROFILER_HSA_TABLE>::deliver(Snap, ROCPROFILER_HSA_TABLE, 0, 1,
                                         reinterpret_cast<void **>(ArrB), 1);

  EXPECT_EQ(&Snap->getTable().getFunction<&::CoreApiTable::hsa_init_fn>(),
            &hsa_init); // still the first capture, not &hsa_shut_down
}

//===----------------------------------------------------------------------===//
// HIP snapshot — capture-once + memcpy bounding
//===----------------------------------------------------------------------===//

TEST(HipApiTableSnapshotTest, CapturesFirstAndIgnoresReinit) {
  llvm::Error Err = llvm::Error::success();
  auto *Snap = new HipApiTableSnapshot<ROCPROFILER_HIP_RUNTIME_TABLE>(
      rocprofilerLib(), Err); // leaked
  discard(std::move(Err));

  ::HipDispatchTable A = buildHipDispatchTable();
  ::HipDispatchTable *ArrA[1] = {&A};
  Access<ROCPROFILER_HIP_RUNTIME_TABLE>::deliver(
      Snap, ROCPROFILER_HIP_RUNTIME_TABLE, 0, 0,
      reinterpret_cast<void **>(ArrA), 1);
  ASSERT_TRUE(Snap->wasRegistrationCallbackInvoked());
  EXPECT_EQ(&Snap->getTable().getFunction<&::HipDispatchTable::hipApiName_fn>(),
            &hipApiName);

  ::HipDispatchTable B = buildHipDispatchTable();
  B.hipApiName_fn = nullptr; // distinct
  ::HipDispatchTable *ArrB[1] = {&B};
  Access<ROCPROFILER_HIP_RUNTIME_TABLE>::deliver(
      Snap, ROCPROFILER_HIP_RUNTIME_TABLE, 0, 1,
      reinterpret_cast<void **>(ArrB), 1);
  EXPECT_EQ(&Snap->getTable().getFunction<&::HipDispatchTable::hipApiName_fn>(),
            &hipApiName); // unchanged
}

TEST(HipApiTableSnapshotTest, OversizedRuntimeTableDoesNotOverflow) {
  // Simulate a newer ROCm whose table is larger than what we compiled against:
  // a heap allocation of exactly sizeof(HipDispatchTable) whose `size` field
  // claims to be larger. The bounded memcpy must copy only sizeof(...) bytes
  // (caught by ASan if it over-reads the source allocation).
  llvm::Error Err = llvm::Error::success();
  auto *Snap = new HipApiTableSnapshot<ROCPROFILER_HIP_RUNTIME_TABLE>(
      rocprofilerLib(), Err); // leaked
  discard(std::move(Err));

  auto *Src = new ::HipDispatchTable(buildHipDispatchTable());
  Src->size = sizeof(::HipDispatchTable) + 4096; // lies: claims to be bigger
  ::HipDispatchTable *Arr[1] = {Src};
  Access<ROCPROFILER_HIP_RUNTIME_TABLE>::deliver(
      Snap, ROCPROFILER_HIP_RUNTIME_TABLE, 0, 0, reinterpret_cast<void **>(Arr),
      1);
  EXPECT_EQ(&Snap->getTable().getFunction<&::HipDispatchTable::hipApiName_fn>(),
            &hipApiName);
  delete Src;
}

//===----------------------------------------------------------------------===//
// HSA wrapper installer — install + re-install on re-registration
//===----------------------------------------------------------------------===//

// A distinct wrapper function with hsa_init's signature.
hsa_status_t initWrapper() { return HSA_STATUS_SUCCESS; }

TEST(HsaApiTableWrapperInstallerTest, InstallsAndReinstalls) {
  using InitFn = decltype(::CoreApiTable::hsa_init_fn);
  using InitFnMemberPtr = decltype(&::CoreApiTable::hsa_init_fn);
  static InitFn Underlying = nullptr;
  InitFn Wrapper = &initWrapper;

  std::tuple<InitFnMemberPtr, InitFn &, InitFn> Spec(
      &::CoreApiTable::hsa_init_fn, Underlying, Wrapper);

  llvm::Error Err = llvm::Error::success();
  auto *Inst = new HsaApiTableWrapperInstaller<::CoreApiTable>(
      rocprofilerLib(), Err, Spec); // leaked
  discard(std::move(Err));

  // First registration: a pristine table whose hsa_init_fn is &hsa_init.
  ::CoreApiTable Core = buildCoreApiTable();
  ::HsaApiTable Tbl = buildHsaApiTable(&Core);
  ::HsaApiTable *Arr[1] = {&Tbl};
  Access<ROCPROFILER_HSA_TABLE>::deliver(Inst, ROCPROFILER_HSA_TABLE, 0, 0,
                                         reinterpret_cast<void **>(Arr), 1);
  EXPECT_EQ(Core.hsa_init_fn, Wrapper); // wrapper installed
  EXPECT_EQ(Underlying, &hsa_init);     // original saved

  // Re-init (instance 1): a fresh table with pristine entries — wrappers must
  // be re-applied.
  ::CoreApiTable Core2 = buildCoreApiTable();
  ::HsaApiTable Tbl2 = buildHsaApiTable(&Core2);
  ::HsaApiTable *Arr2[1] = {&Tbl2};
  Access<ROCPROFILER_HSA_TABLE>::deliver(Inst, ROCPROFILER_HSA_TABLE, 0, 1,
                                         reinterpret_cast<void **>(Arr2), 1);
  EXPECT_EQ(Core2.hsa_init_fn, Wrapper); // re-installed in the fresh table
  EXPECT_EQ(Underlying, &hsa_init);      // re-saved the pristine underlying
}

//===----------------------------------------------------------------------===//
// Fatal paths — death tests (LUTHIER fatals exit(1)/abort; both match)
//===----------------------------------------------------------------------===//

TEST(RocprofilerDeathTest, WrongNumTablesFatals) {
  EXPECT_DEATH(
      {
        llvm::Error Err = llvm::Error::success();
        HsaApiTableSnapshot<::CoreApiTable> Snap(rocprofilerLib(), Err);
        discard(std::move(Err));
        ::CoreApiTable Core = buildCoreApiTable();
        ::HsaApiTable Tbl = buildHsaApiTable(&Core);
        ::HsaApiTable *Arr[1] = {&Tbl};
        Access<ROCPROFILER_HSA_TABLE>::deliver(
            &Snap, ROCPROFILER_HSA_TABLE, 0, 0, reinterpret_cast<void **>(Arr),
            /*NumTables=*/2);
      },
      "");
}

TEST(RocprofilerDeathTest, WrongTableTypeFatals) {
  EXPECT_DEATH(
      {
        llvm::Error Err = llvm::Error::success();
        HsaApiTableSnapshot<::CoreApiTable> Snap(rocprofilerLib(), Err);
        discard(std::move(Err));
        ::CoreApiTable Core = buildCoreApiTable();
        ::HsaApiTable Tbl = buildHsaApiTable(&Core);
        ::HsaApiTable *Arr[1] = {&Tbl};
        Access<ROCPROFILER_HSA_TABLE>::deliver(
            &Snap, ROCPROFILER_HIP_RUNTIME_TABLE, 0, 0,
            reinterpret_cast<void **>(Arr), 1);
      },
      "");
}

TEST(RocprofilerDeathTest, NullSubTablePointerFatals) {
  EXPECT_DEATH(
      {
        llvm::Error Err = llvm::Error::success();
        HsaApiTableSnapshot<::CoreApiTable> Snap(rocprofilerLib(), Err);
        discard(std::move(Err));
        ::HsaApiTable Tbl = buildHsaApiTable(/*Core=*/nullptr); // null core_
        ::HsaApiTable *Arr[1] = {&Tbl};
        Access<ROCPROFILER_HSA_TABLE>::deliver(
            &Snap, ROCPROFILER_HSA_TABLE, 0, 0, reinterpret_cast<void **>(Arr),
            1);
      },
      "");
}

TEST(RocprofilerDeathTest, WrongMajorVersionFatals) {
  EXPECT_DEATH(
      {
        llvm::Error Err = llvm::Error::success();
        HsaApiTableSnapshot<::CoreApiTable> Snap(rocprofilerLib(), Err);
        discard(std::move(Err));
        ::CoreApiTable Core = buildCoreApiTable();
        ::HsaApiTable Tbl = buildHsaApiTable(&Core);
        Tbl.version.major_id = HSA_API_TABLE_MAJOR_VERSION + 1; // bogus
        ::HsaApiTable *Arr[1] = {&Tbl};
        Access<ROCPROFILER_HSA_TABLE>::deliver(
            &Snap, ROCPROFILER_HSA_TABLE, 0, 0, reinterpret_cast<void **>(Arr),
            1);
      },
      "");
}

// Destroying a successfully-registered provider while rocprofiler is up and not
// finalizing must abort (dangling-registration guard). In this non-tool binary,
// the ctor's registration succeeds, so letting the stack object destruct here
// triggers the guard.
TEST(RocprofilerDeathTest, DestroyWhileNotFinalizingAborts) {
  EXPECT_DEATH(
      {
        llvm::Error Err = llvm::Error::success();
        HsaApiTableSnapshot<::CoreApiTable> Snap(rocprofilerLib(), Err);
        discard(std::move(Err));
        // Snap destructs at end of scope -> abort.
      },
      "");
}

} // namespace
