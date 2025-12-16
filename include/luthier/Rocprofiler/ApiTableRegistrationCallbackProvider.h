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
/// a callback to its user or sub-classes when an API table is registered with
/// rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_API_TABLE_REGISTRATION_CALLBACK_PROVIDER_H
#define LUTHIER_ROCPROFILER_API_TABLE_REGISTRATION_CALLBACK_PROVIDER_H
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/HSA/ApiTable.h"
#include "luthier/HSA/HsaError.h"
#include "luthier/Rocprofiler/RocprofilerError.h"
#include <atomic>
#include <hip/amd_detail/hip_api_trace.hpp>
#include <hip/hip_runtime.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>

namespace luthier::rocprofiler {

/// \brief Struct providing static information regarding the
/// \c rocprofiler_intercept_table_t enum, including its table type
/// and the number of tables that are registered with rocprofiler-sdk
template <rocprofiler_intercept_table_t TableType> struct ApiTableEnumInfo;

template <> struct ApiTableEnumInfo<ROCPROFILER_HSA_TABLE> {
  using ApiTableType = ::HsaApiTable;
  constexpr static auto NumApiTables = 1;
  constexpr static auto ApiTableName = "HSA";
};

template <> struct ApiTableEnumInfo<ROCPROFILER_HIP_COMPILER_TABLE> {
  using ApiTableType = ::HipCompilerDispatchTable;
  constexpr static auto NumApiTables = 1;
  constexpr static auto ApiTableName = "HIP Compiler";
};

template <> struct ApiTableEnumInfo<ROCPROFILER_HIP_RUNTIME_TABLE> {
  using ApiTableType = ::HipDispatchTable;
  constexpr static auto NumApiTables = 1;
  constexpr static auto ApiTableName = "HIP Runtime";
};

/// \brief a generic class used to safely request a callback to be invoked when
/// an Api table is registered with rocprofiler-sdk. Can be either used as is
/// or inherited by a sub-class
template <rocprofiler_intercept_table_t TableType>
class ApiTableRegistrationCallbackProvider {
protected:
  /// Keeps track of whether the registration callback has been invoked by
  /// rocprofiler-sdk
  std::atomic<bool> WasRegistrationInvoked{false};

  using CallbackType = std::function<void(
      llvm::ArrayRef<typename ApiTableEnumInfo<TableType>::ApiTableType *>
          Tables,
      uint64_t LibVersion, uint64_t LibInstance)>;

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
          llvm::make_error<rocprofiler::RocprofilerError>(llvm::formatv(
              "Expected rocprofiler to register {0} API table(s), "
              "instead got {0}",
              ApiTableEnumInfo<TableType>::NumApiTables, NumTables)));
    }
    if (Type != TableType) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(llvm::formatv(
              "Expected to get API table of type {0}, but instead got "
              "{0}",
              TableType, Type)));
    }
    if (Tables == nullptr) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(
              "API tables passed by rocprofiler is nullptr"));
    }

    auto &RegProvider =
        *static_cast<ApiTableRegistrationCallbackProvider *>(Data);

    llvm::ArrayRef TablesAsArrayRef(
        reinterpret_cast<typename ApiTableEnumInfo<TableType>::ApiTableType **>(
            Tables),
        NumTables);

    for (const auto *Table : TablesAsArrayRef) {
      if (!Table) {
        LUTHIER_REPORT_FATAL_ON_ERROR(
            llvm::make_error<rocprofiler::RocprofilerError>(
                "API table passed by rocprofiler is nullptr"));
      }
    }

    RegProvider.Callback(TablesAsArrayRef, LibVersion, LibInstance);
    RegProvider.WasRegistrationInvoked.store(true);
  }

public:
  /// Constructor
  /// \note Must only be invoked before rocprofiler-sdk has been
  /// fully configured (i.e, inside the \c rocprofiler_configure function)
  /// \param CB The callback to be invoked once rocprofiler-sdk has reported
  /// back with the requested API table. The callback provides the following
  /// arguments: a) A list of pointers of the API tables passed by
  /// rocprofiler-sdk; These pointers are checked to not be \c nullptr before
  /// they are passed to the client. b)\c uint64_t version of the library as
  /// passed by rocprofiler-sdk; and c) \c uint64_t indicating the number of
  /// times this library has registered itself with rocprofiler-sdk before. The
  /// class also checks if rocprofiler-sdk has passed the correct number of
  /// tables expected for the library of choice, as described in the \c
  /// ApiTableEnumInfo of the \c TableType before invoking the \p CB
  /// \param Err an externally initialized \c llvm::Error that will report
  /// back any errors encountered by this constructor
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

  /// Destructor
  /// \note The object must only be destroyed only if: a) rocprofiler is in the
  /// process of being finalized; or b) rocprofiler has performed the scheduled
  /// callback by this object, which can be checked via \c
  /// wasRegistrationCallbackInvoked
  virtual ~ApiTableRegistrationCallbackProvider() {
    int RocprofilerFiniStatus;
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        rocprofiler_is_finalized(&RocprofilerFiniStatus),
        "Failed to check rocprofiler's finalization status."));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        WasRegistrationInvoked || RocprofilerFiniStatus != 0,
        "Api table callback provider has been destroyed before rocprofiler-sdk "
        "could perform the api table registration callback"));
  }

  /// Checks whether rocprofiler-sdk has invoked the registration callback and
  /// the requested user callback has been invoked
  [[nodiscard]] bool wasRegistrationCallbackInvoked() const {
    return WasRegistrationInvoked.load();
  }

  /// If the API table is not registered by the application with
  /// rocprofiler-sdk, forces its initialization by directly calling a
  /// "harmless" library function directly
  /// \note Only use when absolutely sure the underlying library is not going to
  /// be initialized otherwise
  template <std::enable_if<TableType == ROCPROFILER_HSA_TABLE ||
                           TableType == ROCPROFILER_HIP_RUNTIME_TABLE>>
  void forceTriggerApiTableCallback() {
    if (!WasRegistrationInvoked.load()) {
      if constexpr (TableType == ROCPROFILER_HSA_TABLE)
        (void)hsa_status_string(HSA_STATUS_SUCCESS, nullptr);
      else
        (void)hipApiName(0);
    }
  }
};

} // namespace luthier::rocprofiler

#endif