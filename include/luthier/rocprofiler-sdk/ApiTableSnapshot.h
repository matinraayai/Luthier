//===-- ApiTableSnapshot.h ---------------------------------------*- C++-*-===//
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
/// Defines the a set of classes for HSA and HIP that provide a way to capture a
/// snapshot of their API tables when they are registered with rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_API_TABLE_SNAPSHOT_H
#define LUTHIER_ROCPROFILER_API_TABLE_SNAPSHOT_H
#include "luthier/hip/ApiTable.h"
#include "luthier/rocprofiler-sdk/ApiTableRegistrationCallbackProvider.h"

namespace luthier::rocprofiler {

/// \brief Provides a snapshot of the HSA API table to other components
/// using rocprofiler-sdk
/// \details To invoke HSA methods inside a Luthier tool without them being
/// recursively intercepted by the tool's wrapper functions, components must
/// request a shared snapshot of the HSA API table from rocprofiler-sdk before
/// proceeding to install wrappers during their initialization stage.
/// To obtain the API table snapshot from rocprofiler-sdk a tool must invoke
/// \c HsaApiTableSnapshot::requestSnapshot before its other components are
/// initialized. \n
/// \c requestSnapshot returns an instance of <tt>HsaApiTableSnapshot</tt>,
/// which can be passed by reference to other components of the tool.
/// \note It goes without saying that any components that depend on the
/// snapshot must be destroyed before the snapshot instance.
/// \note The instance of \c HsaApiTableSnapshot must be preserved by the tool
/// until the HSA API tables have been provided by rocprofiler-sdk, or after
/// rocprofiler-sdk has started finalizing the tool. Failure to do so will
/// result in a fatal error.
template <typename HsaApiTableType>
class HsaApiTableSnapshot final
    : public ApiTableRegistrationCallbackProvider<ROCPROFILER_HSA_TABLE> {
private:
  /// Where the snapshot of the HSA API Table is stored
  HsaApiTableType ApiTable{};

public:
  /// On initialization, requests a snapshot of the HSA API table to be provided
  /// by rocprofiler-sdk
  /// \note Must only be invoked during rocprofiler-sdk's configuration stage
  /// \param Err an external \c llvm::Error that will hold any errors
  /// encountered by the constructor
  explicit HsaApiTableSnapshot(llvm::Error &Err)
      : ApiTableRegistrationCallbackProvider(
            [&](llvm::ArrayRef<::HsaApiTable *> Tables, uint64_t, uint64_t) {
              auto &Table = *Tables[0];
              if (Table.version.major_id != HSA_API_TABLE_MAJOR_VERSION) {
                LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<
                                              hsa::HsaError>(llvm::formatv(
                    "Expected HSA API table major version to be {0}, got {1} "
                    "instead.",
                    HSA_API_TABLE_MAJOR_VERSION, Table.version.major_id)));
              }

              constexpr auto RootAccessor = hsa::ApiTableInfo<
                  HsaApiTableType>::PointerToMemberRootAccessor;
              if (!hsa::apiTableHasEntry<RootAccessor>(Table)) {
                LUTHIER_REPORT_FATAL_ON_ERROR(
                    llvm::make_error<hsa::HsaError>(llvm::formatv(
                        "Captured HSA table doesn't support extension {0}",
                        hsa::ApiTableInfo<HsaApiTableType>::Name)));
              }
              ::copyElement(&ApiTable.version,
                            &((Table.*RootAccessor)->version));

              /// Check if the API table copy has been correctly performed by
              /// the copy constructor
              const ApiTableVersion &DestApiTableVersion = Table.version;
              if (DestApiTableVersion.major_id == 0 ||
                  DestApiTableVersion.minor_id == 0 ||
                  DestApiTableVersion.step_id) {
                LUTHIER_REPORT_FATAL_ON_ERROR(
                    llvm::make_error<hsa::HsaError>(llvm::formatv(
                        "Failed to correctly copy the HSA {0} extension",
                        hsa::ApiTableInfo<HsaApiTableType>::Name)));
              }
            },
            Err) {};

  /// \returns the API table snapshot; Will report a fatal error if the snapshot
  /// has not been initialized by rocprofiler-sdk
  hsa::ApiTableContainer<HsaApiTableType> getTable() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        wasRegistrationCallbackInvoked(), "Snapshot is not initialized"));
    return hsa::ApiTableContainer(ApiTable);
  }

  ~HsaApiTableSnapshot() override = default;
};

/// \brief Similar to \c HsaApiTableSnapshot but instead requests a snapshot
/// of an \c hsa_extension_t API table from HSA once the HSA API table has
/// been registered
/// \tparam ExtensionType enum for the type of extension to be requested from
/// HSA
/// \tparam ExtensionApiTableType type of the table corresponding to that
/// extension; By default, \c hsa::ExtensionApiTableInfo is used to query the
/// latest version of the extension available from the HSA library the tool
/// was compiled against
template <hsa_extension_t ExtensionType,
          typename ExtensionApiTableType =
              typename hsa::ExtensionApiTableInfo<ExtensionType>::TableType>
class HsaExtensionTableSnapshot final
    : public ApiTableRegistrationCallbackProvider<ROCPROFILER_HSA_TABLE> {

  ExtensionApiTableType ExtensionTable;

public:
  /// On initialization, requests a snapshot of the \c ExtensionType from the
  /// HSA library to be provided once the HSA API table has been captured
  /// by rocprofiler-sdk
  /// \note Must only be invoked during rocprofiler-sdk's configuration stage
  /// \param Err an external \c llvm::Error that will hold any errors
  /// encountered by the constructor
  explicit HsaExtensionTableSnapshot(llvm::Error &Err)
      : ApiTableRegistrationCallbackProvider(
            [&](llvm::ArrayRef<::HsaApiTable *> Tables, uint64_t, uint64_t) {
              auto &Table = *Tables[0];
              if (Table.version.major_id != HSA_API_TABLE_MAJOR_VERSION) {
                LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<
                                              hsa::HsaError>(llvm::formatv(
                    "Expected HSA API table major version to be {0}, got {1} "
                    "instead.",
                    HSA_API_TABLE_MAJOR_VERSION, Table.version.major_id)));
              }
              if (!hsa::apiTableHasEntry<&::HsaApiTable::core_>(Table)) {
                LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<hsa::HsaError>(
                    "Captured HSA table doesn't support the core extension"));
              }
              if (!hsa::apiTableHasEntry<
                      &::CoreApiTable::hsa_system_get_extension_table_fn>(
                      *Table.core_)) {
                LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<hsa::HsaError>(
                    "Captured HSA API table doesn't have "
                    "hsa_system_get_extension_table function"));
              }
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
                  Table.core_->hsa_system_get_extension_table_fn(
                      ExtensionType, 1, sizeof(ExtensionApiTableType),
                      &ExtensionTable),
                  "Failed to get the extension table"));
            },
            Err) {};

  ~HsaExtensionTableSnapshot() override = default;

  /// \returns the extension table snapshot; Will report a fatal error if the
  /// snapshot has not been initialized by rocprofiler-sdk
  const ExtensionApiTableType &getTable() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        wasRegistrationCallbackInvoked(), "Snapshot is not initialized"));
    return ExtensionTable;
  }
};

/// \brief Provides a snapshot of the HIP API table to other components
/// using rocprofiler-sdk
template <rocprofiler_intercept_table_t TableType>
class HipApiTableSnapshot final
    : public ApiTableRegistrationCallbackProvider<TableType> {
private:
  /// Where the snapshot of the HSA API Table is stored
  typename hip::ApiTableEnumInfo<TableType>::ApiTableType ApiTable{};

public:
  explicit HipApiTableSnapshot(llvm::Error &Err)
      : ApiTableRegistrationCallbackProvider<TableType>(
            [&](typename hip::ApiTableEnumInfo<TableType>::ApiTableType
                    &Table) { std::memcpy(&Table, &ApiTable, ApiTable.size); },
            Err) {};

  ~HipApiTableSnapshot() override = default;

  hip::ApiTableContainer<TableType> getTable() const {
    return hip::ApiTableContainer<TableType>(ApiTable);
  }
};

} // namespace luthier::rocprofiler

#endif