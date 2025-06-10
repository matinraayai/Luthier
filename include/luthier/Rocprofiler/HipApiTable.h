//===-- HipApiTable.h -------------------------------------------*- C++ -*-===//
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
/// Implements utilities for obtaining a snapshot of the HIP API tables from
/// rocprofiler-sdk as well as installing wrappers inside it.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_HIP_API_TABLE_H
#define LUTHIER_ROCPROFILER_HIP_API_TABLE_H
#include <hip/amd_detail/hip_api_trace.hpp>
#include <luthier/Common/Singleton.h>
#include <luthier/HIP/HipError.h>
#include <luthier/Rocprofiler/RocprofilerError.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>

namespace luthier::rocprofiler {

template <auto Entry> bool hipApiTableHasEntry(const auto &Table) {
  return static_cast<size_t>(&(Table.*Entry)) < Table.size;
}

template <rocprofiler_intercept_table_t TableType> struct HipApiTableEnumInfo;

template <> struct HipApiTableEnumInfo<ROCPROFILER_HIP_COMPILER_TABLE> {
  using ApiTableType = ::HipCompilerDispatchTable;
};

template <> struct HipApiTableEnumInfo<ROCPROFILER_HIP_RUNTIME_TABLE> {
  using ApiTableType = ::HipDispatchTable;
};

/// \brief a generic class used to request a callback to be invoked when the
/// \c ::HsaApiTable is registered with rocprofiler-sdk
template <rocprofiler_intercept_table_t TableType>
class HipApiTableRegistrationCallbackProvider {
protected:
  /// Keeps track of whether the registration callback has been invoked by
  /// rocprofiler-sdk
  std::atomic<bool> WasRegistrationInvoked{false};

  using CallbackType = std::function<void(
      typename HipApiTableEnumInfo<TableType>::ApiTableType)>;

  /// Callback invoked inside the registration callback
  const CallbackType Callback;

  /// API table registration callback for used by rocprofiler-sdk
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data) {
    /// Check for errors
    if (NumTables != 1) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(
              llvm::formatv("Expected HIP to register only a single API table, "
                            "instead got {0}",
                            NumTables)));
    }
    if (Type != TableType) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(
              llvm::formatv("Expected to get HIP API table with type {0}, but "
                            "the API table type is {0}",
                            TableType, Type)));
    }
    if (LibInstance != 0) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(
              "Multiple instances of HIP library."));
    }
    auto *Table =
        static_cast<typename HipApiTableEnumInfo<TableType>::ApiTableType *>(
            Tables[0]);
    if (Table == nullptr) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<rocprofiler::RocprofilerError>(
              "HIP API table is nullptr"));
    }

    auto &RegProvider =
        *static_cast<HipApiTableRegistrationCallbackProvider *>(Data);
    RegProvider.WasRegistrationInvoked.store(true);
    RegProvider.Callback(*Table);
  }

  explicit HipApiTableRegistrationCallbackProvider(CallbackType CB,
                                                   llvm::Error &Err)
      : Callback(std::move(CB)) {
    Err = LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
        rocprofiler_at_intercept_table_registration(
            HipApiTableRegistrationCallbackProvider::apiRegistrationCallback,
            TableType, this),
        "Failed to request a callback on HIP API table initialization from "
        "rocprofiler-sdk");
  };

public:
  static llvm::Expected<
      std::unique_ptr<HipApiTableRegistrationCallbackProvider>>
  requestCallback(CallbackType CB) {
    llvm::Error Err = llvm::Error::success();
    auto Out = std::make_unique<HipApiTableRegistrationCallbackProvider>(
        std::move(CB), Err);
    if (Err)
      return std::move(Err);
    return std::move(Out);
  }

  virtual ~HipApiTableRegistrationCallbackProvider() {
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

  /// If the table of the HIP runtime is not initialized, forces
  /// the initialization of the HIP runtime table by calling a "harmless"
  /// library function
  /// \note Must only be used sparingly, when absolutely sure the library
  /// is not going to be initialized otherwise
  template <std::enable_if<TableType == ROCPROFILER_HIP_RUNTIME_TABLE>>
  void forceTriggerApiTableCallback() {
    if (!WasRegistrationInvoked.load()) {
      (void)hipApiName(0);
    }
  }
};

using HipCompilerApiTableRegistrationCallbackProvider =
    HipApiTableRegistrationCallbackProvider<ROCPROFILER_HIP_COMPILER_TABLE>;
using HipRuntimeApiTableRegistrationCallbackProvider =
    HipApiTableRegistrationCallbackProvider<ROCPROFILER_HIP_RUNTIME_TABLE>;

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
template <rocprofiler_intercept_table_t TableType>
class HipApiTableSnapshot final
    : public HipApiTableRegistrationCallbackProvider<TableType> {
private:
  /// Where the snapshot of the HSA API Table is stored
  typename HipApiTableEnumInfo<TableType>::ApiTableType ApiTable{};

  explicit HipApiTableSnapshot(llvm::Error &Err)
      : HipApiTableRegistrationCallbackProvider<TableType>(
            [&](typename HipApiTableEnumInfo<TableType>::ApiTableType &Table) {
              std::memcpy(&Table, &ApiTable, ApiTable.size);
            },
            Err) {};

public:
  /// Requests a snapshot of the HSA API table to be provided by
  /// rocprofiler-sdk; Must only be invoked during rocprofiler-sdk's
  /// configuration stage
  /// \return Expects a new instance of \c ApiTableSnapshot
  static llvm::Expected<std::unique_ptr<HipApiTableSnapshot>> requestSnapshot() {
    llvm::Error Err = llvm::Error::success();
    auto Out = std::make_unique<HipApiTableSnapshot>(Err);
    if (Err)
      return std::move(Err);
    return Out;
  }

  ~HipApiTableSnapshot() override = default;

  /// \brief Checks if the \c Func is present in the API table snapshot
  /// \tparam Func pointer-to-member of the function entry inside the
  /// extension table being queried
  /// \return \c true if the function is available inside the
  /// API table, \c false otherwise. Reports a fatal error
  /// if the snapshot has not been initialized by rocprofiler-sdk
  template <auto Func> [[nodiscard]] bool tableSupportsFunction() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(
        HipApiTableRegistrationCallbackProvider<
            TableType>::wasRegistrationCallbackInvoked(),
        "Snapshot is not initialized");
    return hipApiTableHasEntry<Func>(ApiTable);
  }

  /// \returns the function inside the snapshot associated with the
  /// pointer-to-member accessor \c Func
  template <auto Func> const auto &getFunction() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        (tableSupportsFunction<Func>()),
        "The passed function is not inside the table."));
    return *(ApiTable.*Func);
  }
};

using HipCompilerApiTableSnapshot =
    HipApiTableSnapshot<ROCPROFILER_HIP_COMPILER_TABLE>;
using HipRuntimeApiTableSnapshot =
    HipApiTableSnapshot<ROCPROFILER_HIP_RUNTIME_TABLE>;

template <rocprofiler_intercept_table_t TableType>
class HipApiTableWrapperInstaller final
    : public HipApiTableRegistrationCallbackProvider<TableType> {
private:
  template <typename... Tuples>
  explicit HipApiTableWrapperInstaller(llvm::Error &Err,
                                    const Tuples &...WrapperSpecs)
      : HipApiTableRegistrationCallbackProvider<TableType>(
            [&](typename HipApiTableEnumInfo<TableType>::ApiTableType &Table) {
              (installWrapperEntry(Table, WrapperSpecs), ...);
            },
            Err){};

  /// Installs a wrapper for an entry inside an extension table of \p Table
  /// \p WrapperSpec is a 4-entry tuple, containing the pointer to member
  /// accessor for the extension table inside <tt>::HsaApiTable</tt>, pointer
  /// to member accessor function for the entry inside the extension, reference
  /// to where the underlying function entry will be saved to, and a function
  /// pointer to the wrapper being installed. Reports a fatal error if
  /// the entry is not present in the table
  template <auto Func>
  void installWrapperEntry(
      typename HipApiTableEnumInfo<TableType>::ApiTableType &Table,
      const std::tuple<decltype(Func), auto *&, auto &> &WrapperSpec) {
    auto &[ExtEntry, UnderlyingStoreLocation, WrapperFunc] = WrapperSpec;
    if (!tableHasEntry<ExtEntry>(Table)) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<hip::HipError>(llvm::formatv(
              "Failed to find entry inside the HSA API table at offset {0:x}.",
              static_cast<size_t>(&(Table.*ExtEntry)))));
    }
    UnderlyingStoreLocation = Table.*ExtEntry;
    Table.*ExtEntry = WrapperFunc;
  }

public:
  // Variadic template function to accept a variable-length list of tuples
  template <typename... Tuples>
  llvm::Expected<std::unique_ptr<HipApiTableWrapperInstaller>>
  requestWrapperInstallation(const Tuples &...WrapperSpecs) {
    llvm::Error Err = llvm::Error::success();
    auto Out = std::make_unique<HipApiTableWrapperInstaller>(Err, WrapperSpecs...);
    if (Err)
      return std::move(Err);
    return Out;
  }

  ~HipApiTableWrapperInstaller() override = default;
};

using HipCompilerApiTableWrapperInstaller =
    HipApiTableWrapperInstaller<ROCPROFILER_HIP_COMPILER_TABLE>;
using HipRuntimeApiTableWrapperInstaller =
    HipApiTableWrapperInstaller<ROCPROFILER_HIP_RUNTIME_TABLE>;

} // namespace luthier::rocprofiler

#endif
