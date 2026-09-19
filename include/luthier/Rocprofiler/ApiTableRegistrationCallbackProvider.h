//===-- ApiTableRegistrationCallbackProvider.h -------------------*- C++-*-===//
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
/// \file
/// Defines the \c ApiTableRegistrationCallbackProvider class which provide
/// the rocprofiler-sdk API table registration callback to its user or
/// sub-classes.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_API_TABLE_REGISTRATION_CALLBACK_PROVIDER_H
#define LUTHIER_ROCPROFILER_API_TABLE_REGISTRATION_CALLBACK_PROVIDER_H
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Rocprofiler/ApiTableEnumInfo.h"
#include "luthier/Rocprofiler/RocprofilerError.h"
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>

namespace luthier::rocprofiler {

/// \brief a generic class used to request a callback to be invoked when
/// an Api table is registered with rocprofiler-sdk.
/// \note After creation, the object must not be destroyed until it is
/// confirmed that rocprofiler-sdk in the application space has been finalized.
template <rocprofiler_intercept_table_t TableType>
class ApiTableRegistrationCallbackProvider {
public:
  using CallbackType = std::function<void(
    llvm::ArrayRef<typename ApiTableEnumInfo<TableType>::ApiTableType *>
        Tables,
    uint64_t LibVersion, uint64_t LibInstance)>;
private:

  /// Callback invoked inside the registration callback
  const CallbackType Callback;

  /// API table registration callback consumed by rocprofiler-sdk
  /// \note Declared \c noexcept to turns any unexpected exceptions into a
  /// \c std::terminate.
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data) noexcept {
    /// Check for errors
    if (NumTables != ApiTableEnumInfo<TableType>::NumApiTables) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
          llvm::formatv("Expected rocprofiler to register {0} API table(s), "
                        "instead got {1}",
                        ApiTableEnumInfo<TableType>::NumApiTables, NumTables)));
    }
    if (Type != TableType) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          LUTHIER_MAKE_ROCPROFILER_ERROR(llvm::formatv(
              "Expected to get API table of type {0}, but instead got "
              "{1}",
              TableType, Type)));
    }
    if (Tables == nullptr) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
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
        LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
            "API table passed by rocprofiler is nullptr"));
      }
    }

    RegProvider.Callback(TablesAsArrayRef, LibVersion, LibInstance);
  }

public:
  /// Constructor
  /// \note Must only be invoked inside the \c rocprofiler_configure function
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


  virtual ~ApiTableRegistrationCallbackProvider() = default;
};

} // namespace luthier::rocprofiler

#endif