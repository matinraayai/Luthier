//===-- ApiTableRegistrationCallbackProvider.h -------------------*- C++-*-===//
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
/// \file
/// Defines the \c ApiTableRegistrationCallbackProvider class which provide
/// a callback to its user when an API table is registered with rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_API_TABLE_REGISTRATION_CALLBACK_PROVIDER_H
#define LUTHIER_ROCPROFILER_API_TABLE_REGISTRATION_CALLBACK_PROVIDER_H
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/GenericLuthierError.h"
#include "luthier/hsa/ApiTable.h"
#include "luthier/hsa/HsaError.h"
#include "luthier/rocprofiler-sdk/RocprofilerError.h"
#include <atomic>
#include <hip/amd_detail/hip_api_trace.hpp>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>

namespace luthier::rocprofiler {

/// \brief Struct providing static information regarding the
/// \c rocprofiler_intercept_table_t enum, including its table type struct
/// and the number of tables that are registered with rocprofiler-sdk
template <rocprofiler_intercept_table_t TableType>
struct ApiTableEnumInfo;

template <> struct ApiTableEnumInfo<ROCPROFILER_HSA_TABLE> {
  using ApiTableType = ::HsaApiTable;
  constexpr unsigned int NumApiTables = 1;
  constexpr static auto ApiTableName = "HSA";
};

template <> struct ApiTableEnumInfo<ROCPROFILER_HIP_COMPILER_TABLE> {
  using ApiTableType = ::HipCompilerDispatchTable;
  constexpr unsigned int NumApiTables = 1;
  constexpr static auto ApiTableName = "HIP Compiler";
};

template <> struct ApiTableEnumInfo<ROCPROFILER_HIP_RUNTIME_TABLE> {
  using ApiTableType = ::HipDispatchTable;
  constexpr unsigned int NumApiTables = 1;
  constexpr static auto ApiTableName = "HIP Runtime";
};

/// \brief a generic class used to request a callback to be invoked when
/// an Api table is registered with rocprofiler-sdk
template <rocprofiler_intercept_table_t TableType>
class ApiTableRegistrationCallbackProvider {
protected:
  /// Keeps track of whether the registration callback has been invoked by
  /// rocprofiler-sdk
  std::atomic<bool> WasRegistrationInvoked{false};

  using CallbackType = std::function<void(
      std::conditional<
          ApiTableEnumInfo<TableType>::NumApiTables == 1,
          typename ApiTableEnumInfo<TableType>::ApiTableType &,
          llvm::ArrayRef<
              typename ApiTableEnumInfo<TableType>::ApiTableType &>>,
      uint64_t, uint64_t)>;

  /// Callback invoked inside the registration callback
  const CallbackType Callback;

  /// API table registration callback for used by rocprofiler-sdk
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data) {
    /// Check for errors
    if (NumTables != ApiTableEnumInfo<TableType>::NumApiTables) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(
              llvm::formatv("Expected HIP to register only a single API table, "
                            "instead got {0}",
                            NumTables)));
    }
    if (Type != TableType) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(llvm::formatv(
              "Expected to get {0} API table, but the API table type is {0}",
              ApiTableEnumInfo<TableType>::ApiTableName, Type)));
    }
    if (Tables == nullptr) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(
              "API tables passed by rocprofiler is nullptr"));
    }

    auto &RegProvider =
        *static_cast<ApiTableRegistrationCallbackProvider *>(Data);

    if constexpr (ApiTableEnumInfo<TableType>::NumApiTables == 1) {
      const auto *Table = static_cast<
          typename ApiTableEnumInfo<TableType>::ApiTableType *>(
          Tables[0]);
      if (Table == nullptr) {
        LUTHIER_REPORT_FATAL_ON_ERROR(
            llvm::make_error<rocprofiler::RocprofilerError>(
                "API table passed by rocprofiler is nullptr"));
      }
      RegProvider.Callback(Table, LibVersion, LibInstance);
    } else {
      llvm::SmallVector<
          typename ApiTableEnumInfo<TableType>::ApiTableType &, 4>
          CallbackTables;

      for (const auto *Table : llvm::ArrayRef<
               typename ApiTableEnumInfo<TableType>::ApiTableType *>(
               Tables, NumTables)) {
        if (!Table) {
          LUTHIER_REPORT_FATAL_ON_ERROR(
              llvm::make_error<rocprofiler::RocprofilerError>(
                  "API table passed by rocprofiler is nullptr"));
        }
        CallbackTables.push_back(*Table);
      }

      RegProvider.Callback(CallbackTables, LibVersion, LibInstance);
    }
    RegProvider.WasRegistrationInvoked.store(true);
  }

  ApiTableRegistrationCallbackProvider(CallbackType CB, llvm::Error &Err)
      : Callback(std::move(CB)) {
    llvm::ErrorAsOutParameter EAO(Err);
    Err = std::move(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        rocprofiler_at_intercept_table_registration(
            ApiTableRegistrationCallbackProvider::apiRegistrationCallback,
            TableType, this),
        llvm::formatv("Failed to request a callback on {0} API table "
                      "initialization from "
                      "rocprofiler-sdk",
                      ApiTableEnumInfo<TableType>::ApiTableName)));
  };

  /// Utility used to extract the type of the member pointer and the class
  /// it belongs to
  template <typename T> struct RemoveMemberPointer {
    using type = T;
  };

  template <typename C, typename T> struct RemoveMemberPointer<T C::*> {
    using type = T;
    using outer = C;
  };

public:
  static llvm::Expected<std::unique_ptr<ApiTableRegistrationCallbackProvider>>
  requestCallback(CallbackType CB) {
    llvm::Error Err = llvm::Error::success();
    auto Out = std::make_unique<ApiTableRegistrationCallbackProvider>(
        std::move(CB), Err);
    if (Err)
      return std::move(Err);
    return std::move(Out);
  }

  virtual ~ApiTableRegistrationCallbackProvider() {
    int RocprofilerFiniStatus;
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        rocprofiler_is_finalized(&RocprofilerFiniStatus),
        "Failed to check rocprofiler's finalization status."));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        WasRegistrationInvoked || RocprofilerFiniStatus != 0,
        "HSA Api table snapshot has been destroyed before rocprofiler-sdk "
        "could perform the api table registration callback"));
  }

  /// Checks whether rocprofiler-sdk has invoked the registration callback
  [[nodiscard]] bool wasRegistrationCallbackInvoked() const {
    return WasRegistrationInvoked.load();
  }

  /// If the HSA table is not initialized, forces its initialization by
  /// directly calling a "harmless" library function directly
  /// \note Must only be used sparingly, when absolutely sure the library
  /// is not going to be initialized otherwise
  void forceTriggerApiTableCallback() {
    if (!WasRegistrationInvoked.load()) {
      (void)hsa_status_string(HSA_STATUS_SUCCESS, nullptr);
    }
  }
};

} // namespace luthier::rocprofiler

#endif