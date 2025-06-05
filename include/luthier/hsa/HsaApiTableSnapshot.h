//===-- HsaApiTableSnapshot.h ------------------------------------*- C++-*-===//
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
/// Defines the <tt>HsaApiTableSnapshot</tt>, in charge of capturing a snapshot
/// of the HSA API Table using rocprofiler-sdk and providing it for other
/// components so that they can make HSA calls without them being wrapped
/// by other components.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HSA_API_TABLE_SNAPSHOT_H
#define LUTHIER_HSA_HSA_API_TABLE_SNAPSHOT_H
#include <hsa/hsa_api_trace.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/common/GenericLuthierError.h>
#include <luthier/hsa/HsaError.h>
#include <luthier/rocprofiler/RocprofilerError.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>

namespace luthier::hsa {

class HsaApiTableSnapshot {
private:
  /// The snapshot of the HSA API Table
  ::HsaApiTableContainer ApiTable{};

  /// Keeps track of whether the registration callback has been invoked by
  /// rocprofiler-sdk and the snapshot has been initialized as a result
  bool IsSnapshotInitialized{false};

  /// API table registration callback for used by rocprofiler-sdk
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data) {
    /// Check for errors
    if (NumTables != 1) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(
              llvm::formatv("Expected HSA to register only a single API table, "
                            "instead got {0}",
                            NumTables)));
    }
    if (Type != ROCPROFILER_HSA_TABLE) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(llvm::formatv(
              "Expected to get HSA API table, but the API table type is {0}",
              Type)));
    }
    if (LibInstance != 0) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(llvm::formatv(
              LibInstance == 0, "Multiple instances of HSA library.")));
    }
    const auto *Table = static_cast<HsaApiTable *>(Tables[0]);
    if (Table == nullptr) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(
              "HSA API table is nullptr"));
    }

    auto &TableSnapshot = *static_cast<HsaApiTableSnapshot *>(Data);
    ::copyTables(Table, &TableSnapshot.ApiTable.root);

    /// Check if the API table copy has been performed by the copy constructor
    const ApiTableVersion &DestApiTableVersion =
        TableSnapshot.ApiTable.root.version;
    if (DestApiTableVersion.major_id == 0 ||
        DestApiTableVersion.minor_id == 0 || DestApiTableVersion.step_id) {
      LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<HsaError>(
          "Failed to correctly copy the HSA API tables"));
    }
    TableSnapshot.IsSnapshotInitialized = true;
  }

  explicit HsaApiTableSnapshot() = default;

public:
  static llvm::Expected<std::unique_ptr<HsaApiTableSnapshot>>
  requestSnapshot() {
    auto Out = std::make_unique<HsaApiTableSnapshot>();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        rocprofiler_at_intercept_table_registration(
            HsaApiTableSnapshot::apiRegistrationCallback, ROCPROFILER_HSA_TABLE,
            Out.get()),
        "Failed to request HSA API tables from rocprofiler-sdk"));
    return std::move(Out);
  }

  ~HsaApiTableSnapshot() {
    int RocprofilerFiniStatus;
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        rocprofiler_is_finalized(&RocprofilerFiniStatus),
        "Failed to check if rocprofiler is finalized."));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        IsSnapshotInitialized || RocprofilerFiniStatus != 0,
        "HSA Api table snapshot has been destroyed before rocprofiler-sdk "
        "could perform the api table registration callback"));
  }

  [[nodiscard]] const ::HsaApiTable &getSnapshot() const {
    return ApiTable.root;
  }

  /// \return \c true if the API table snapshot has been initialized by the
  /// rocprofiler-sdk, \c false otherwise
  [[nodiscard]] bool isSnapShotInitialized() const {
    return IsSnapshotInitialized;
  }
};

} // namespace luthier::hsa

#endif