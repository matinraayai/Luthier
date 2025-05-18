//===-- ApiTableInterceptor.hpp ----------------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// This file defines the <tt>luthier::hsa::HsaApiTableInterceptor</tt>,
/// in charge of capturing the HSA API Table using rocprofiler-sdk and providing
/// the captured table to other components of Luthier so that they can install
/// wrapper functions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HSA_API_TABLE_INTERCEPTOR_HPP
#define LUTHIER_HSA_HSA_API_TABLE_INTERCEPTOR_HPP
#include "common/Singleton.hpp"
#include "luthier/hsa/HsaError.h"
#include "luthier/rocprofiler-sdk/RocprofilerSDKError.h"
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <rocprofiler-sdk/intercept_table.h>

namespace luthier::hsa {

class HsaApiTableInterceptor {
protected:
  /// Pointer to the HSA runtime's API table, provided by rocprofiler-sdk.
  /// The HSA runtime uses this table to look up and call HSA functions
  ::HsaApiTable *RuntimeApiTable{nullptr};

  /// Saved copy of the "real" API table
  ::HsaApiTableContainer SavedApiTable{};

  /// Holds function pointers to the AMD's loader API
  ::hsa_ven_amd_loader_1_03_pfn_s AmdTable{};

  /// Mutex to protect the \c RegistrationCallbacks field
  std::mutex CBMutex;

  /// A set of callbacks invoked when rocprofiler-sdk provides the HSA API table
  llvm::SmallVector<std::function<void(::HsaApiTable *)>> RegistrationCallbacks;

  HsaApiTableInterceptor() = default;

  virtual ~HsaApiTableInterceptor() = default;

  bool isTableCaptured() const { return RuntimeApiTable != nullptr; }

  [[nodiscard]] const hsa_ven_amd_loader_1_03_pfn_t &
  getHsaVenAmdLoaderTable() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        isTableCaptured(), "HSA API table is not captured"));
    return AmdTable;
  }

  [[nodiscard]] const ::HsaApiTable &getHsaApiTable() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        isTableCaptured(), "HSA API table is not captured"));
    return SavedApiTable.root;
  }

  llvm::Error
  addRegistrationCallback(const std::function<void(::HsaApiTable *)> &CB) {
    std::lock_guard Lock(CBMutex);
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        !isTableCaptured(), "HSA API table is not captured"));
    RegistrationCallbacks.emplace_back(CB);
    return llvm::Error::success();
  }

  llvm::Error checkApiTableCopySuccess() const;
};

template <size_t Idx>
class ROCPROFILER_HIDDEN_API UniqueApiTableInterceptor final
    : public HsaApiTableInterceptor,
      public Singleton<UniqueApiTableInterceptor<Idx>> {
private:
  /// API table registration callback for rocprofiler-sdk
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data) {
    LUTHIER_REPORT_FATAL_ON_ERROR(
        LUTHIER_ERROR_CHECK(UniqueApiTableInterceptor::isInitialized(),
                            "HSA API Table Interceptor is not initialized"));
    auto &I = UniqueApiTableInterceptor::instance();
    std::lock_guard Lock(I.CBMutex);
    /// Check for errors
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        NumTables == 1,
        "Expected HSA to register only a single API table, instead got {0}",
        NumTables));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        Type == ROCPROFILER_HSA_TABLE,
        "Expected to get HSA API table, but the API table type is {0}", Type));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        LibInstance == 0, "Multiple instances of HSA library."));
    auto *Table = static_cast<HsaApiTable *>(Tables[0]);
    LUTHIER_REPORT_FATAL_ON_ERROR(
        LUTHIER_ERROR_CHECK(Table != nullptr, "HSA API table is nullptr"));
    I.RuntimeApiTable = Table;
    copyTables(I.RuntimeApiTable, &I.SavedApiTable.root);
    /// Check if the copy was successful
    LUTHIER_REPORT_FATAL_ON_ERROR(I.checkApiTableCopySuccess());
    /// Copy the loader table
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        I.SavedApiTable.core.hsa_system_get_major_extension_table_fn(
            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_03_pfn_t),
            &I.AmdTable)));
    /// Invoke registered callbacks
    for (const auto &CB : I.RegistrationCallbacks) {
      CB(I.RuntimeApiTable);
    }
  }

public:
  static llvm::Expected<std::unique_ptr<UniqueApiTableInterceptor>>
  initialize() {
    auto Out = std::unique_ptr<UniqueApiTableInterceptor>(
        new (std::nothrow) UniqueApiTableInterceptor());
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        Out != nullptr, "Failed to initialize HSA API Table interceptor."));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ROCPROFILER_SUCCESS_CHECK(
        rocprofiler_at_intercept_table_registration(
            UniqueApiTableInterceptor::apiRegistrationCallback,
            ROCPROFILER_HSA_TABLE, nullptr)));
    return std::move(Out);
  }

  ~UniqueApiTableInterceptor() override = default;
};

template <size_t Idx>
ROCPROFILER_HIDDEN_API UniqueApiTableInterceptor<Idx>
    *Singleton<UniqueApiTableInterceptor<Idx>>::Instance{nullptr};

} // namespace luthier::hsa

#endif
