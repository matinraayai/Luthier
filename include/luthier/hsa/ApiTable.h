//===-- ApiTable.h -----------------------------------------------*- C++-*-===//
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
/// Defines a set of classes and utility functions regarding the HSA API table,
/// including obtaining a snapshot of the HSA API table, installing Wrappers
/// in the API table entries, and querying extension and function support based
/// on the table's version.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_API_TABLE_H
#define LUTHIER_HSA_API_TABLE_H
#include <hsa/hsa_api_trace.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/common/GenericLuthierError.h>
#include <luthier/hsa/HsaError.h>
#include <luthier/rocprofiler/RocprofilerError.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>

namespace luthier::hsa {

/// \brief Queries whether the HSA API \p Table includes the extension table
/// \p Ext inside it
/// \details the implementation of the HSA standard by AMD (i.e. ROCr) provides
/// a set of core functionality from the base HSA standard, along with a set of
/// vendor-specific extensions added by AMD. For each extension the
/// \c ::HsaApiTable has a corresponding table (e.g.,
/// <tt>::HsaApiTable::core_</tt>). Each extension table is a \c struct and each
/// API inside the extension has a corresponding function pointer field in the
/// extension table (e.g., <tt>::CoreApiTable::hsa_init_fn</tt>).\n
/// \c ::HsaApiTable and its tables are forward-compatible i.e.
/// newly added tables and functions are added below the existing ones.
/// This means that each entry remains at a fixed offset in a table,
/// ensuring a tool compiled against an older version of HSA can also run
/// against a newer version of HSA without any interference from the newly
/// added functionality. \n
/// To check if an entry is present in the HSA API table during runtime,
/// a fixed-size \c version field is provided at the very
/// beginning of \c ::HsaApiTable and its extension tables. Tools can use the
/// \c ::ApiTableVersion::minor_id field to obtain the total size of each
/// table in the active HSA runtime. Since table entries remain at a
/// fixed offset, a tool can confirm the instrumented HSA runtime supports
/// its required extension by making sure
/// the corresponding offset of the entry is smaller than the
/// size of the table. \n
/// For example, to check if the HSA API \p Table has the core extension,
/// one can use the following:
/// \code{.cpp}
/// tableHasEntry(Table, &::HsaApiTable::core_);
/// \endcode
/// To check if the \c ::CoreApiTable has the \c hsa_iterate_agents_fn field,
/// one can use the following:
/// \code{.cpp}
/// tableHasEntry(CoreApiTable, &::CoreApiTable::hsa_iterate_agents_fn);
/// \endcode
/// \param Table an HSA API Table or one of its sub-tables, likely obtained from
/// rocprofiler-sdk
/// \param Entry the member being queried to be present or not
/// \return \c true if \p Table contains the \p Entry and \c false otherwise
/// \sa <hsa/hsa_api_trace.h> in the ROCr runtime
/// \sa extensionSupportsFunction
template <typename TableType, typename EntryType>
bool tableHasEntry(const TableType &Table, EntryType TableType::*Entry) {
  return static_cast<size_t>(&(Table.*Entry)) < Table.version.minor_id;
}

/// \brief Provides a snapshot of the HSA API table to other components
/// using rocprofiler-sdk
/// \details To invoke HSA methods inside a Luthier tool without them being
/// recursively intercepted by the tool's components, components must require
/// a snapshot of the HSA API table from rocprofiler-sdk before
/// proceeding to install wrappers during their initialization stage.
/// To obtain the API table snapshot from rocprofiler-sdk a tool must invoke
/// \c hsa::ApiTableSnapshot::requestSnapshot before its components are
/// initialized. \n
/// \c requestSnapshot returns an instance of <tt>ApiTableSnapshot</tt>, which
/// can be passed by reference to other components of the tool. The components
/// can then request a callback from the \c ApiTableSnapshot instance during
/// their initialization stage, before they request installing API table
/// wrapper functions. \n
/// Inside the \c ApiSnapshot callback a component must ensure that the
/// HSA extensions and functions that it intends to use are inside the
/// captured API table snapshot. This can be done via the \c
/// LUTHIER_IS_EXTENSION_PRESENT_IN_HSA_API_TABLE and
/// \c LUTHIER_IS_FUNC_PRESENT_IN_HSA_EXT_TABLE macros. This check is done
/// to ensure that a tool compiled for one version of HSA can be safely
/// used with another version of HSA without requiring recompilation. After
/// this check, the components can safely store a reference to the captured
/// table snapshot and use it. \n
/// It goes without saying that any components that depend on the snapshot
/// must be destroyed before the snapshot instance itself.
/// \note The instance of \c ApiTableSnapshot must be preserved by the tool
/// until the HSA API tables have been provided by rocprofiler-sdk, or after
/// rocprofiler-sdk has started finalizing the tool. Failure to do so will
/// result in a fatal error.
/// \note \c ApiTableSnapshot is not thread-safe and is meant to be used
/// inside a single thread.
class ApiTableSnapshot {
private:
  /// Where the snapshot of the HSA API Table is stored
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

    auto &TableSnapshot = *static_cast<ApiTableSnapshot *>(Data);
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

  explicit ApiTableSnapshot() = default;

public:
  /// Requests a snapshot of the HSA API table to be provided by
  /// rocprofiler-sdk; Must only be invoked during rocprofiler-sdk's
  /// configuration stage
  /// \return Expects a new instance of \c ApiTableSnapshot
  static llvm::Expected<std::unique_ptr<ApiTableSnapshot>> requestSnapshot() {
    auto Out = std::make_unique<ApiTableSnapshot>();
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        rocprofiler_at_intercept_table_registration(
            ApiTableSnapshot::apiRegistrationCallback, ROCPROFILER_HSA_TABLE,
            Out.get()),
        "Failed to request the HSA API table from rocprofiler-sdk"));
    return std::move(Out);
  }

  ~ApiTableSnapshot() {
    int RocprofilerFiniStatus;
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        rocprofiler_is_finalized(&RocprofilerFiniStatus),
        "Failed to check if rocprofiler is finalized."));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        IsSnapshotInitialized || RocprofilerFiniStatus != 0,
        "HSA Api table snapshot has been destroyed before rocprofiler-sdk "
        "could perform the api table registration callback"));
  }

  /// Checks whether the snapshot has been installed or not (i.e. if
  /// rocprofiler-sdk has provided the API tables to this snapshot)
  [[nodiscard]] bool isSnapshotInitialized() const {
    return IsSnapshotInitialized;
  }

  /// \brief Checks if the API table snapshot contains the \p Extension
  /// \param Extension Pointer to member of the extension being queried
  /// \return \c true if the snapshot supports the <tt>Extension</tt>,
  /// \c false otherwise. Reports a fatal error if the snapshot
  /// has not been initialized by rocprofiler-sdk
  template <typename ExtType>
  bool tableSupportsExtension(ExtType HsaApiTable::*Extension) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        isSnapshotInitialized(), "Snapshot is not initialized"));
    return tableHasEntry(ApiTable.root, Extension);
  }

  /// \brief Checks if the API extension table \p Ext is inside the
  /// snapshot, and whether it contains \p Func
  /// \param Ext pointer-to-member of the extension table being queried
  /// \param Func pointer-to-member of the function entry inside the
  /// extension table being queried
  /// \return \c true if the function is available inside the
  /// API table, \c false otherwise. Reports a fatal error
  /// if the snapshot has not been initialized by rocprofiler-sdk
  template <typename ExtType, typename FuncType>
  bool extensionTableSupportsFunction(ExtType HsaApiTable::*Ext,
                                      FuncType ExtType::*Func) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        isSnapshotInitialized(), "Snapshot is not initialized"));
    return tableHasEntry(ApiTable.root, Ext) &&
           tableHasEntry(ApiTable.root.*Ext, Func);
  }

  /// \brief Provides the function pointer associated with the
  /// entry \p Func inside the HSA API table extension \p Ext inside
  /// the HSA Api table snapshot
  /// \param Ext pointer-to-member of the extension of \p Func
  /// \param Func pointer-to-member of the function inside the \p Ext
  /// \return \c const reference to the function entry inside the API table;
  /// Reports a fatal error if \p Func or \p Ext are not found inside the
  /// snapshot, or if the snapshot hasn't been initialized in the first place
  template <typename ExtType, typename FuncType>
  const auto &getFunction(ExtType HsaApiTable::*Ext, FuncType ExtType::*Func) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        extensionTableSupportsFunction(Ext, Func),
        "The passed function is not inside the table."));
    return *(ApiTable.*Ext.*Func);
  }
};

class ApiTableWrapperInstaller {
private:
  /// Keeps track of whether the registration callback has been invoked by
  /// rocprofiler-sdk
  bool WasRegCallbackInvoked{false};

  typedef std::function<void(HsaApiTable &)> CallbackType;
  /// Callback invoked when the API table has been passed to us
  /// by rocprofiler-sdk
  CallbackType Callback;

  /// API table registration callback for used by rocprofiler-sdk
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data);

  explicit ApiTableWrapperInstaller(CallbackType Callback)
      : Callback(std::move(Callback)) {};

  /// Installs a wrapper for an entry inside an extension table of \p Table
  /// \p WrapperSpec is a 4-entry tuple, containing the pointer to member
  /// accessor for the extension table inside <tt>::HsaApiTable</tt>, pointer
  /// to member accessor function for the entry inside the extension, reference
  /// to where the underlying function entry will be saved to, and a function
  /// pointer to the wrapper being installed. Reports a fatal error if
  /// the entry is not present in the table
  template <typename ExtType, typename ApiEntryAccessorType,
            typename ApiFuncType>
  void installWrapperEntry(
      ::HsaApiTable &Table,
      const std::tuple<ExtType HsaApiTable::*, ApiEntryAccessorType ExtType::*,
                       ApiFuncType *&, ApiFuncType &> &WrapperSpec) {
    auto &[ExtTable, ExtEntry, UnderlyingStoreLocation, WrapperFunc] =
        WrapperSpec;
    if (!tableHasEntry(Table, ExtTable)) {
      LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<HsaError>(llvm::formatv(
          "Failed to find entry inside the HSA API table at offset {0:x}.",
          static_cast<size_t>(&(Table.*ExtTable)))));
    }
    if (!tableHasEntry(Table.*ExtTable, ExtEntry)) {
      LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<HsaError>(
          llvm::formatv("Failed to find entry inside the HSA "
                        "extension table at offset {0:x}",
                        static_cast<size_t>(&(Table.*ExtTable.*ExtEntry)))));
    }
    UnderlyingStoreLocation = Table.*ExtTable->*ExtEntry;
    Table.*ExtTable->*ExtEntry = WrapperFunc;
  }

public:
  // Variadic template function to accept a variable-length list of tuples
  template <typename... Tuples>
  llvm::Expected<std::unique_ptr<ApiTableWrapperInstaller>>
  requestWrapperInstallation(const Tuples &...WrapperSpecs) {
    auto Out =
        std::make_unique<ApiTableWrapperInstaller>([&](::HsaApiTable &Table) {
          (installWrapperEntry(Table, WrapperSpecs), ...);
        });
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        rocprofiler_at_intercept_table_registration(
            ApiTableWrapperInstaller::apiRegistrationCallback,
            ROCPROFILER_HSA_TABLE, Out.get()),
        "Failed to request HSA API tables from rocprofiler-sdk"));
    return Out;
  }

  ~ApiTableWrapperInstaller();

  /// Checks whether the wrappers have been installed or not (i.e. if
  /// rocprofiler-sdk has provided the API tables to the wrapper installer)
  [[nodiscard]] bool areWrappersInstalled() const {
    return WasRegCallbackInvoked;
  }
};

} // namespace luthier::hsa

#endif
