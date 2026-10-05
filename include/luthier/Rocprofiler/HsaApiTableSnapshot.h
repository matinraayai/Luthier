//===-- HsaApiTableSnapshot.h -----------------------------------*- C++ -*-===//
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
/// Defines the HSA API table snapshot classes that capture a snapshot of the
/// HSA API table (or its extension tables) when registered with
/// rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_HSA_API_TABLE_SNAPSHOT_H
#define LUTHIER_ROCPROFILER_HSA_API_TABLE_SNAPSHOT_H
#include "luthier/HSA/ApiTable.h"
#include "luthier/Rocprofiler/ApiTableRegistrationCallbackProvider.h"
#include "luthier/Rocprofiler/HsaApiTableEnumInfo.h"

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
/// until either the HSA API tables have been provided by rocprofiler-sdk,
/// or after rocprofiler-sdk has started finalizing the tool. Failure to do so
/// will result in a fatal error. The snapshot will not raise a fatal error if
/// a crash has happened in the program before rocprofiler-sdk gets a chance to
/// invoke the provided HSA API table callback
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
  /// \param RocprofilerLib the rocprofiler-sdk library of the target
  /// application; Must outlive this object
  /// \param Err an external \c llvm::Error that will hold any errors
  /// encountered by the constructor
  explicit HsaApiTableSnapshot(const DynamicLibrary &RocprofilerLib,
                               llvm::Error &Err)
      : ApiTableRegistrationCallbackProvider(
            [this](llvm::ArrayRef<::HsaApiTable *> Tables, uint64_t, uint64_t) {
              /// Capture only the first registration. When the application
              /// finalizes and re-initializes HSA, rocprofiler-sdk invokes this
              /// callback again with a non-zero library instance, but the
              /// runtime's function-pointer addresses are stable across the
              /// cycle, so the first snapshot remains valid; ignore re-fires.
              if (this->wasRegistrationCallbackInvoked())
                return;
              if (Tables.empty())
                LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
                    "No tables were passed to the callback"));
              auto &Table = *Tables[0];
              if (Table.version.major_id != HSA_API_TABLE_MAJOR_VERSION) {
                LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_HSA_ERROR(
                    llvm::formatv("Expected HSA API table major version to be "
                                  "{0}, got {1} "
                                  "instead.",
                                  HSA_API_TABLE_MAJOR_VERSION,
                                  Table.version.major_id)));
              }

              constexpr auto RootAccessor = hsa::ApiTableInfo<
                  HsaApiTableType>::PointerToMemberRootAccessor;
              if (!hsa::apiTableHasEntry<HsaApiTableType>(Table)) {
                LUTHIER_REPORT_FATAL_ON_ERROR(
                    LUTHIER_MAKE_HSA_ERROR(llvm::formatv(
                        "Captured HSA table doesn't support extension {0}",
                        hsa::ApiTableInfo<HsaApiTableType>::Name)));
              }
              /// The entry being within the table's size does not guarantee
              /// the runtime populated the sub-table pointer; an optional
              /// extension the runtime doesn't provide leaves it null.
              if ((Table.*RootAccessor) == nullptr) {
                LUTHIER_REPORT_FATAL_ON_ERROR(
                    LUTHIER_MAKE_HSA_ERROR(llvm::formatv(
                        "Captured HSA table's {0} extension pointer is null",
                        hsa::ApiTableInfo<HsaApiTableType>::Name)));
              }
              ::copyElement(&ApiTable.version,
                            &((Table.*RootAccessor)->version));

              /// Check if the API table copy has been correctly performed by
              /// the copy constructor
              const ApiTableVersion &DestApiTableVersion = ApiTable.version;
              if (DestApiTableVersion.major_id == 0 ||
                  DestApiTableVersion.minor_id == 0) {
                LUTHIER_REPORT_FATAL_ON_ERROR(
                    LUTHIER_MAKE_HSA_ERROR(llvm::formatv(
                        "Failed to correctly copy the HSA {0} extension",
                        hsa::ApiTableInfo<HsaApiTableType>::Name)));
              }
            },
            RocprofilerLib, Err) {
    /// Need to poulate the correct API table version here, otherwise the HSA
    /// copy table routine will fail
    ApiTable.version.major_id = hsa::ApiTableInfo<HsaApiTableType>::MajorVer;
    ApiTable.version.step_id = hsa::ApiTableInfo<HsaApiTableType>::StepVer;
    ApiTable.version.minor_id = hsa::ApiTableInfo<HsaApiTableType>::MinorVer;
  };

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

  ExtensionApiTableType ExtensionTable{};

public:
  /// On initialization, requests a snapshot of the \c ExtensionType from the
  /// HSA library to be provided once the HSA API table has been captured
  /// by rocprofiler-sdk
  /// \note Must only be invoked during rocprofiler-sdk's configuration stage
  /// \param RocprofilerLib the rocprofiler-sdk library of the target
  /// application; Must outlive this object
  /// \param Err an external \c llvm::Error that will hold any errors
  /// encountered by the constructor
  explicit HsaExtensionTableSnapshot(const DynamicLibrary &RocprofilerLib,
                                     llvm::Error &Err)
      : ApiTableRegistrationCallbackProvider(
            [this](llvm::ArrayRef<::HsaApiTable *> Tables, uint64_t,
                   uint64_t Instance) {
              /// Query the extension only on the first registration. When the
              /// application finalizes and re-initializes HSA, rocprofiler-sdk
              /// invokes this callback again with \c Instance > 0, but the
              /// runtime's function-pointer addresses are stable across the
              /// cycle, so the first snapshot remains valid; ignore re-fires.
              if (this->wasRegistrationCallbackInvoked())
                return;
              if (Tables.empty())
                LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
                    "No tables were passed in the callback"));
              auto &Table = *Tables[0];
              if (Table.version.major_id != HSA_API_TABLE_MAJOR_VERSION) {
                LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_HSA_ERROR(
                    llvm::formatv("Expected HSA API table major version to be "
                                  "{0}, got {1} "
                                  "instead.",
                                  HSA_API_TABLE_MAJOR_VERSION,
                                  Table.version.major_id)));
              }
              if (!hsa::apiTableHasEntry<::CoreApiTable>(Table)) {
                LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_HSA_ERROR(
                    "Captured HSA table doesn't support the core extension"));
              }
              /// The entry being within the table's size does not guarantee
              /// the runtime populated the core pointer.
              if (Table.core_ == nullptr) {
                LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_HSA_ERROR(
                    "Captured HSA table's core extension pointer is null"));
              }
              if (!hsa::apiTableHasEntry<
                      &::CoreApiTable::hsa_system_get_major_extension_table_fn>(
                      *Table.core_)) {
                LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_HSA_ERROR(
                    "Captured HSA API table doesn't have "
                    "hsa_system_get_major_extension_table function"));
              }
              /// Use \c hsa_system_get_major_extension_table (explicit length)
              /// rather than \c hsa_system_get_extension_table. The latter
              /// derives the table size from the (major, minor) version by
              /// name-matching a pfn struct (e.g. "hsa_ext_images_1_00_pfn_t"),
              /// whose major (1) disagrees with the major rocprofiler checks
              /// the request against (the API-table major, e.g. 2 for images
              /// and the finalizer) — so that path is unusable for those
              /// extensions. Passing \c sizeof(ExtensionTable) as the length
              /// also bounds the runtime's copy to our struct, so a newer
              /// runtime table can never overflow \c ExtensionTable.
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
                  Table.core_->hsa_system_get_major_extension_table_fn(
                      ExtensionType,
                      hsa::ExtensionApiTableInfo<ExtensionType>::MajorVer,
                      sizeof(ExtensionTable), &ExtensionTable),
                  "Failed to get the extension table"));
            },
            RocprofilerLib, Err) {};

  ~HsaExtensionTableSnapshot() override = default;

  /// \returns the extension table snapshot; Will report a fatal error if the
  /// snapshot has not been initialized by rocprofiler-sdk
  const ExtensionApiTableType &getTable() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        wasRegistrationCallbackInvoked(), "Snapshot is not initialized"));
    return ExtensionTable;
  }
};

} // namespace luthier::rocprofiler

#endif
