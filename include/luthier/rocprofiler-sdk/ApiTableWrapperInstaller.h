//===-- ApiTableWrapperInstaller.h -------------------------------*- C++-*-===//
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
/// Defines the \c ApiTableWrapperInstaller template class which provides a way
/// to install wrappers around individual entries in an API table when it
/// is registered with rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_API_TABLE_WRAPPER_INSTALLER_H
#define LUTHIER_ROCPROFILER_API_TABLE_WRAPPER_INSTALLER_H
#include "luthier/hip/HipError.h"
#include "luthier/rocprofiler-sdk/ApiTableRegistrationCallbackProvider.h"

namespace luthier::rocprofiler {

/// \brief Used to install wrapper functions around entries inside the HSA
/// API table when it is registered with rocprofiler-sdk
class HsaApiTableWrapperInstaller final
    : public ApiTableRegistrationCallbackProvider<ROCPROFILER_HSA_TABLE> {
private:
  template <typename... Tuples>
  explicit HsaApiTableWrapperInstaller(llvm::Error &Err,
                                       const Tuples &...WrapperSpecs)
      : ApiTableRegistrationCallbackProvider(
            [&](::HsaApiTable &Table, uint64_t, uint64_t) {
              (installWrapperEntry(Table, WrapperSpecs), ...);
            },
            Err){};

  /// Installs a wrapper for an entry inside an extension table of
  /// \p Table
  /// \p WrapperSpec is a 3-entry tuple, containing the pointer to
  /// member accessor for the extension table inside
  /// <tt>::HsaApiTable</tt>, reference to where the
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
        typename hsa::ApiTableInfo<ExtTableType>::PointerToMemberRootAccessor;
    if (!hsa::apiTableHasEntry<ExtTableRootAccessor>(Table)) {
      LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<hsa::HsaError>(
          llvm::formatv("Failed to find entry inside the HSA API "
                        "table at offset {0:x}.",
                        static_cast<size_t>(&(Table.*ExtTableRootAccessor)))));
    }
    if (!hsa::apiTableHasEntry<FuncEntry>(Table.*ExtTableRootAccessor)) {
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

template <rocprofiler_intercept_table_t TableType,
          typename =
              std::enable_if_t<TableType == ROCPROFILER_HIP_COMPILER_TABLE ||
                               TableType == ROCPROFILER_HIP_RUNTIME_TABLE>>
class HipApiTableWrapperInstaller final
    : public ApiTableRegistrationCallbackProvider<TableType> {
private:
  template <typename... Tuples>
  explicit HipApiTableWrapperInstaller(llvm::Error &Err,
                                       const Tuples &...WrapperSpecs)
      : ApiTableRegistrationCallbackProvider<TableType>(
            [&](typename ApiTableEnumInfo<TableType>::ApiTableType &Table) {
              (installWrapperEntry(Table, WrapperSpecs), ...);
            },
            Err){};

  /// Installs a wrapper for an entry inside an extension table of \p Table
  /// \p WrapperSpec is a 3-entry tuple, containing the pointer to member
  /// accessor for the extension table inside the HIP API table, reference
  /// to where the underlying function entry will be saved to, and a function
  /// pointer to the wrapper being installed. Reports a fatal error if
  /// the entry is not present in the table
  template <auto Func>
  void installWrapperEntry(
      typename ApiTableEnumInfo<TableType>::ApiTableType &Table,
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
    auto Out =
        std::make_unique<HipApiTableWrapperInstaller>(Err, WrapperSpecs...);
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