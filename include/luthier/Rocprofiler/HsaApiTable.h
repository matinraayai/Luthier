//===-- HsaApiTable.h --------------------------------------------*- C++-*-===//
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
#ifndef LUTHIER_ROCPROFILER_HSA_API_TABLE_H
#define LUTHIER_ROCPROFILER_HSA_API_TABLE_H
#include <atomic>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/Rocprofiler/RocprofilerError.h>
#include <rocprofiler-sdk/intercept_table.h>

namespace luthier::rocprofiler {

/// \brief Queries whether the \p Table from the \c ::HsaApiTable includes the
/// \p Entry inside it
/// \details the implementation of the HSA standard by AMD (i.e. ROCr) provides
/// a set of core functionality from the base HSA standard, along with a set of
/// vendor-specific extensions added by AMD. For each extension the
/// \c ::HsaApiTable has a corresponding table (e.g.,
/// <tt>::HsaApiTable::core_</tt>). Each extension table is a \c struct and each
/// API inside the extension has a corresponding function pointer field in the
/// extension table (e.g., <tt>::CoreApiTable::hsa_init_fn</tt>).\n
/// The \c ::HsaApiTable and its tables are forward-compatible i.e.
/// newly added tables and functions are added below the existing ones.
/// This means that each entry remains at a fixed offset in a table,
/// ensuring a tool compiled against an older version of HSA can also run
/// against a newer version of HSA without any interference from the newly
/// added functionality. \n
/// To check if an entry is present in the HSA API table during runtime,
/// a fixed-size \c version field is provided at the very
/// beginning of \c ::HsaApiTable and its extension tables. Tools can use the
/// \c ::ApiTableVersion::minor_id field to obtain the size of each
/// table in the active HSA runtime. Since table entries remain at a
/// fixed offset, a tool can confirm the currently running HSA runtime supports
/// its required extension by calling this function, which makes sure the
/// corresponding offset of the entry is smaller than the size of the table. \n
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
/// \return \c true if \p Table contains <tt>Entry</tt>, \c false otherwise
/// \sa <hsa/hsa_api_trace.h> in ROCr
template <auto Entry> bool hsaApiTableHasEntry(const auto &Table) {
  return static_cast<size_t>(&(Table.*Entry)) < Table.version.minor_id;
}

/// \brief Struct containing \c constexpr compile-time info regarding individual
/// tables inside <tt>::HsaApiTable</tt>. Used by \c ApiTableSnapshot and
/// \c ApiTableWrapperInstaller to provide convenience accessors using the
/// table's type. If a table is not present here, simply define a specialization
/// of this template.
template <typename ApiTableType> struct HsaApiTableInfo;

template <> struct HsaApiTableInfo<::CoreApiTable> {
  static constexpr auto Name = "core";
  static constexpr auto PointerToMemberRootAccessor = &::HsaApiTable::core_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::core;
};

template <> struct HsaApiTableInfo<::AmdExtTable> {
  static constexpr auto Name = "amd";
  static constexpr auto PointerToMemberRootAccessor = &::HsaApiTable::amd_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::amd_ext;
};

template <> struct HsaApiTableInfo<::FinalizerExtTable> {
  static constexpr auto Name = "finalizer";
  static constexpr auto PointerToMemberRootAccessor =
      &::HsaApiTable::finalizer_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::finalizer_ext;
};

template <> struct HsaApiTableInfo<::ImageExtTable> {
  static constexpr auto Name = "image";
  static constexpr auto PointerToMemberRootAccessor =
      &::HsaApiTable::image_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::image_ext;
};

template <> struct HsaApiTableInfo<::ToolsApiTable> {
  static constexpr auto Name = "tools";
  static constexpr auto PointerToMemberRootAccessor = &::HsaApiTable::tools_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::tools;
};

template <> struct HsaApiTableInfo<::PcSamplingExtTable> {
  static constexpr auto Name = "pc sampling";
  static constexpr auto PointerToMemberRootAccessor =
      &::HsaApiTable::pc_sampling_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::pc_sampling_ext;
};

template <hsa_extension_t ExtType> struct HsaExtensionApiTableInfo;

template <> struct HsaExtensionApiTableInfo<HSA_EXTENSION_FINALIZER> {
  using TableType = hsa_ext_finalizer_1_00_pfn_t;
  static constexpr auto MajorVer = HSA_FINALIZER_API_TABLE_MAJOR_VERSION;
  static constexpr auto StepVer = HSA_FINALIZER_API_TABLE_STEP_VERSION;
};

template <> struct HsaExtensionApiTableInfo<HSA_EXTENSION_IMAGES> {
  using TableType = hsa_ext_images_1_pfn_t;
  static constexpr auto MajorVer = HSA_IMAGE_API_TABLE_MAJOR_VERSION;
  static constexpr auto StepVer = HSA_IMAGE_API_TABLE_STEP_VERSION;
};

template <> struct HsaExtensionApiTableInfo<HSA_EXTENSION_AMD_LOADER> {
  using TableType = hsa_ven_amd_loader_1_03_pfn_t;
  static constexpr auto MajorVer = 1;
  static constexpr auto StepVer = 3;
};

template <> struct HsaExtensionApiTableInfo<HSA_EXTENSION_AMD_PC_SAMPLING> {
  using TableType = hsa_ven_amd_pc_sampling_1_00_pfn_t;
  static constexpr auto MajorVer = HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION;
  static constexpr auto StepVer = HSA_PC_SAMPLING_API_TABLE_STEP_VERSION;
};

/// \brief a generic class used to request a callback to be invoked when the
/// \c ::HsaApiTable is registered with rocprofiler-sdk
class HsaApiTableRegistrationCallbackProvider {
protected:
  /// Keeps track of whether the registration callback has been invoked by
  /// rocprofiler-sdk
  std::atomic<bool> WasRegistrationInvoked{false};

  using CallbackType = std::function<void(::HsaApiTable &)>;

  /// Callback invoked inside the registration callback
  const CallbackType Callback;

  /// API table registration callback for used by rocprofiler-sdk
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data);

  explicit HsaApiTableRegistrationCallbackProvider(CallbackType CB,
                                                   llvm::Error &Err)
      : Callback(std::move(CB)) {
    llvm::ErrorAsOutParameter EAO(Err);
    auto ErrInner = LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        rocprofiler_at_intercept_table_registration(
            HsaApiTableRegistrationCallbackProvider::apiRegistrationCallback,
            ROCPROFILER_HSA_TABLE, this),
        "Failed to request a callback on HSA API table initialization from "
        "rocprofiler-sdk");
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
  static llvm::Expected<
      std::unique_ptr<HsaApiTableRegistrationCallbackProvider>>
  requestCallback(CallbackType CB);

  virtual ~HsaApiTableRegistrationCallbackProvider();

  /// Checks whether rocprofiler-sdk has invoked the registration callback
  [[nodiscard]] bool wasRegistrationCallbackInvoked() const {
    return WasRegistrationInvoked.load();
  }

  /// If the HSA table is not initialized, forces its initialization by directly
  /// calling a "harmless" library function directly
  /// \note Must only be used sparingly, when absolutely sure the library
  /// is not going to be initialized otherwise
  void forceTriggerApiTableCallback() {
    if (!WasRegistrationInvoked.load()) {
      (void)hsa_status_string(HSA_STATUS_SUCCESS, nullptr);
    }
  }
};

/// \brief Provides a snapshot of the HSA API table to other components
/// using rocprofiler-sdk
/// \details To invoke HSA methods inside a Luthier tool without them being
/// recursively intercepted by the tool's wrapper functions, components must
/// request a shared snapshot of the HSA API table from rocprofiler-sdk before
/// proceeding to install wrappers during their initialization stage.
/// To obtain the API table snapshot from rocprofiler-sdk a tool must invoke
/// \c hsa::ApiTableSnapshot::requestSnapshot before its components are
/// initialized. \n
/// \c requestSnapshot returns an instance of <tt>ApiTableSnapshot</tt>, which
/// can be passed by reference to other components of the tool.
/// \note It goes without saying that any components that depend on the snapshot
/// must be destroyed before the snapshot instance itself.
/// \note The instance of \c ApiTableSnapshot must be preserved by the tool
/// until the HSA API tables have been provided by rocprofiler-sdk, or after
/// rocprofiler-sdk has started finalizing the tool. Failure to do so will
/// result in a fatal error.
/// \note \c ApiTableSnapshot is not thread-safe and is meant to be used
/// inside a single thread
template <typename ApiTableType>
class HsaApiTableSnapshot final
    : public HsaApiTableRegistrationCallbackProvider {
private:
  /// Where the snapshot of the HSA API Table is stored
  ApiTableType ApiTable{};

  explicit HsaApiTableSnapshot(llvm::Error &Err)
      : HsaApiTableRegistrationCallbackProvider(
            [&](const ::HsaApiTable &Table) {
              if (Table.version.major_id != HSA_API_TABLE_MAJOR_VERSION) {
                LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<
                                              hsa::HsaError>(llvm::formatv(
                    "Expected HSA API table major version to be {0}, got {1} "
                    "instead.",
                    HSA_API_TABLE_MAJOR_VERSION, Table.version.major_id)));
              }

              constexpr auto RootAccessor =
                  HsaApiTableInfo<ApiTableType>::PointerToMemberRootAccessor;
              if (!hsaApiTableHasEntry<RootAccessor>(Table)) {
                LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<hsa::HsaError>(
                    "Captured HSA table doesn't support extension {0}",
                    HsaApiTableInfo<ApiTableType>::Name));
              }
              ::copyElement(&Table, &ApiTable.root.*RootAccessor);

              /// Check if the API table copy has been performed by the copy
              /// constructor
              const ApiTableVersion &DestApiTableVersion = Table.version;
              if (DestApiTableVersion.major_id == 0 ||
                  DestApiTableVersion.minor_id == 0 ||
                  DestApiTableVersion.step_id) {
                LUTHIER_REPORT_FATAL_ON_ERROR(
                    llvm::make_error<hsa::HsaError>(llvm::formatv(
                        "Failed to correctly copy the HSA {0} extension",
                        HsaApiTableInfo<ApiTableType>::Name)));
              }
            },
            Err) {};

public:
  /// Requests a snapshot of the HSA API table to be provided by
  /// rocprofiler-sdk; Must only be invoked during rocprofiler-sdk's
  /// configuration stage
  /// \return Expects a new instance of \c ApiTableSnapshot
  static llvm::Expected<std::unique_ptr<HsaApiTableSnapshot>>
  requestSnapshot() {
    llvm::Error Err = llvm::Error::success();
    auto Out = std::make_unique<HsaApiTableSnapshot>(Err);
    if (Err)
      return std::move(Err);
    return Out;
  }

  ~HsaApiTableSnapshot() override = default;

  /// \brief Checks if the \c Func is present in the API table snapshot
  /// \tparam Func pointer-to-member of the function entry inside the
  /// extension table being queried
  /// \return \c true if the function is available inside the
  /// API table, \c false otherwise. Reports a fatal error
  /// if the snapshot has not been initialized by rocprofiler-sdk
  template <auto ApiTableType::*Func>
  [[nodiscard]] bool tableSupportsFunction() const {
    return hsaApiTableHasEntry<Func>(ApiTable);
  }

  /// \returns the function inside the snapshot associated with the
  /// pointer-to-member accessor \c Func
  template <auto ApiTableType::*Func> const auto &getFunction() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        (tableSupportsFunction<Func>()),
        "The passed function is not inside the table."));
    return *(ApiTable.*Func);
  }
};

template <hsa_extension_t ExtensionType,
          typename ExtensionApiTableType =
              typename HsaExtensionApiTableInfo<ExtensionType>::TableType>
class HsaExtensionTableSnapshot final
    : public HsaApiTableRegistrationCallbackProvider {

  ExtensionApiTableType ExtensionTable;

  explicit HsaExtensionTableSnapshot(llvm::Error &Err)
      : HsaApiTableRegistrationCallbackProvider(
            [&](const ::HsaApiTable &Table) {
              if (Table.version.major_id != HSA_API_TABLE_MAJOR_VERSION) {
                LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<
                                              hsa::HsaError>(llvm::formatv(
                    "Expected HSA API table major version to be {0}, got {1} "
                    "instead.",
                    HSA_API_TABLE_MAJOR_VERSION, Table.version.major_id)));
              }
              if (!hsaApiTableHasEntry<&::HsaApiTable::core_>(Table)) {
                LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<hsa::HsaError>(
                    "Captured HSA table doesn't support the core extension"));
              }
              if (!hsaApiTableHasEntry<
                      &::CoreApiTable::hsa_system_get_extension_table_fn>(
                      Table.core_)) {
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

public:
  /// Requests a snapshot of the HSA API table to be provided by
  /// rocprofiler-sdk; Must only be invoked during rocprofiler-sdk's
  /// configuration stage
  /// \return Expects a new instance of \c ApiTableSnapshot
  static llvm::Expected<std::unique_ptr<HsaExtensionTableSnapshot>>
  requestSnapshot() {
    llvm::Error Err = llvm::Error::success();
    auto Out = std::make_unique<HsaExtensionTableSnapshot>(Err);
    if (Err)
      return std::move(Err);
    return Out;
  }

  ~HsaExtensionTableSnapshot() override = default;

  /// \returns the function inside the snapshot associated with the
  /// pointer-to-member accessor \c Func
  template <auto HsaExtensionApiTableInfo<ExtensionType>::TableType::*Func>
  const auto &getFunction() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        wasRegistrationCallbackInvoked(), "Snapshot is not initialized"));
    return *(ExtensionTable.*Func);
  }
};

class HsaApiTableWrapperInstaller final
    : public HsaApiTableRegistrationCallbackProvider {
private:
  template <typename... Tuples>
  explicit HsaApiTableWrapperInstaller(llvm::Error &Err,
                                       const Tuples &...WrapperSpecs)
      : HsaApiTableRegistrationCallbackProvider(
            [&](::HsaApiTable &Table) {
              (installWrapperEntry(Table, WrapperSpecs), ...);
            },
            Err){};

  /// Installs a wrapper for an entry inside an extension table of
  /// \p Table
  /// \p WrapperSpec is a 4-entry tuple, containing the pointer to
  /// member accessor for the extension table inside
  /// <tt>::HsaApiTable</tt>, pointer to member accessor function
  /// for the entry inside the extension, reference to where the
  /// underlying function entry will be saved to, and a function
  /// pointer to the wrapper being installed. Reports a fatal error
  /// if the entry is not present in the table
  template <auto Func>
  void installWrapperEntry(
      ::HsaApiTable &Table,
      const std::tuple<decltype(Func), auto *&, auto &> &WrapperSpec) {
    auto &[FuncEntry, UnderlyingStoreLocation, WrapperFunc] = WrapperSpec;
    using ExtTableType = typename RemoveMemberPointer<decltype(Func)>::outer;
    auto constexpr ExtTableRootAccessor =
        typename HsaApiTableInfo<ExtTableType>::PointerToMemberRootAccessor;
    if (!hsaApiTableHasEntry<ExtTableRootAccessor>(Table)) {
      LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<hsa::HsaError>(
          llvm::formatv("Failed to find entry inside the HSA API "
                        "table at offset {0:x}.",
                        static_cast<size_t>(&(Table.*ExtTableRootAccessor)))));
    }
    if (!hsaApiTableHasEntry<FuncEntry>(Table.*ExtTableRootAccessor)) {
      LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<hsa::HsaError>(
          llvm::formatv("Failed to find entry inside the HSA "
                        "extension table at offset {0:x}",
                        static_cast<size_t>(
                            &(Table.*ExtTableRootAccessor->*FuncEntry)))));
    }
    UnderlyingStoreLocation = Table.*ExtTableRootAccessor->*FuncEntry;
    Table.*ExtTableRootAccessor->*FuncEntry = WrapperFunc;
  }

public:
  template <typename... Tuples>
  llvm::Expected<std::unique_ptr<HsaApiTableWrapperInstaller>>
  requestWrapperInstallation(const Tuples &...WrapperSpecs) {
    llvm::Error Err = llvm::Error::success();
    auto Out =
        std::make_unique<HsaApiTableWrapperInstaller>(Err, WrapperSpecs...);
    if (Err)
      return std::move(Err);
    return Out;
  }

  ~HsaApiTableWrapperInstaller() override = default;
};

} // namespace luthier::rocprofiler

#endif
