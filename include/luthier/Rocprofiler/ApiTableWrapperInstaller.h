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
/// Defines a set of classes to provides a way to install wrappers around
/// individual entries in an API tables of HIP and HSA libraries when they
/// are registered with rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_API_TABLE_WRAPPER_INSTALLER_H
#define LUTHIER_ROCPROFILER_API_TABLE_WRAPPER_INSTALLER_H
#include "luthier/Rocprofiler/ApiTableRegistrationCallbackProvider.h"
#include "luthier/HIP/HipError.h"

namespace luthier::rocprofiler {

/// \brief Used to install wrapper functions around entries inside the HSA
/// API table when it is registered with rocprofiler-sdk
/// \tparam HsaApiTableType Type of the HSA API table to install wrappers for
template <typename HsaApiTableType>
class HsaApiTableWrapperInstaller final
    : public ApiTableRegistrationCallbackProvider<ROCPROFILER_HSA_TABLE> {

  /// Installs a wrapper for an entry inside the \p Table
  /// \param Table the table entry inside the \c ::HsaApiTable to install
  /// the wrapper function
  /// \param WrapperSpec A 3-entry tuple, containing a) the pointer-to
  /// member-accessor for the \p Table b) Pointer to a function pointer where
  /// the underlying function entry will be saved to, and c) a function
  /// pointer of the wrapper to be installed
  /// \note Reports a fatal error if the target entry is not inside the \p Table
  template <typename FuncType>
  static void installWrapperEntry(HsaApiTableType &Table,
                                  FuncType HsaApiTableType::*FuncEntry,
                                  FuncType &UnderlyingStoreLocation,
                                  FuncType WrapperFunc) {
    if (!hsa::apiTableHasEntry(Table, FuncEntry)) {
      LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<hsa::HsaError>(
          llvm::formatv("Failed to find function entry inside the HSA "
                        "extension table.")));
    }
    UnderlyingStoreLocation = Table.*FuncEntry;
    Table.*FuncEntry = WrapperFunc;
  }

public:
  /// On initialization, requests a callback from rocprofiler-sdk to install
  /// function wrappers for select entries in the \c HsaApiTableType of the \c
  /// ::HsaApiTable as specified by the \p WrapperSpecs
  /// \note Must only be called before rocprofiler-sdk has finished
  /// configuration
  /// \tparam Tuples Variadic tuple type for different entries to be wrapped
  /// in the target table
  /// \param Err an external \c llvm::Error that will hold any errors
  /// encountered in the constructor
  /// \param WrapperSpecs a variadic set of 3-entry tuples, with each tuple
  /// specifying an entry inside the target API table to be wrapped
  /// \sa installWrapperEntry
  template <typename... Tuples>
  explicit HsaApiTableWrapperInstaller(llvm::Error &Err,
                                       const Tuples &...WrapperSpecs)
      : ApiTableRegistrationCallbackProvider(
            [=, this](llvm::ArrayRef<::HsaApiTable *> Tables,
                      uint64_t LibVersion, uint64_t LibInstance) -> void {
              if (LibInstance != 0) {
                LUTHIER_REPORT_FATAL_ON_ERROR(llvm::make_error<hsa::HsaError>(
                    "Multiple instances of the HSA library"));
              }
              constexpr auto RootAccessor = hsa::ApiTableInfo<
                  HsaApiTableType>::PointerToMemberRootAccessor;
              if (!hsa::apiTableHasEntry<HsaApiTableType>(*Tables[0])) {
                LUTHIER_REPORT_FATAL_ON_ERROR(
                    llvm::make_error<hsa::HsaError>(llvm::formatv(
                        "Captured HSA table doesn't support extension {0}",
                        hsa::ApiTableInfo<HsaApiTableType>::Name)));
              }
              (installWrapperEntry(
                   *(Tables[0]->*RootAccessor), std::get<0>(WrapperSpecs),
                   std::get<1>(WrapperSpecs), std::get<2>(WrapperSpecs)),
               ...);
            },
            Err){};

  ~HsaApiTableWrapperInstaller() override = default;
};

template <rocprofiler_intercept_table_t TableType,
          typename =
              std::enable_if_t<TableType == ROCPROFILER_HIP_COMPILER_TABLE ||
                               TableType == ROCPROFILER_HIP_RUNTIME_TABLE>>
class HipApiTableWrapperInstaller final
    : public ApiTableRegistrationCallbackProvider<TableType> {
private:
  /// Installs a wrapper for an entry inside an extension table of \p Table
  /// \p WrapperSpec is a 3-entry tuple, containing the pointer to member
  /// accessor for the extension table inside the HIP API table, reference
  /// to where the underlying function entry will be saved to, and a function
  /// pointer to the wrapper being installed. Reports a fatal error if
  /// the entry is not present in the table
  template <typename FuncType>
  static void installWrapperEntry(
      hip::ApiTableEnumInfo<TableType>::ApiTableType &Table,
      FuncType hip::ApiTableEnumInfo<TableType>::ApiTableType::*ExtEntry,
      FuncType &UnderlyingStoreLocation, FuncType WrapperFunc) {
    if (!hip::apiTableHasEntry(Table, ExtEntry)) {
      LUTHIER_REPORT_FATAL_ON_ERROR(
          llvm::make_error<hip::HipError>(llvm::formatv(
              "Failed to find entry inside the HIP API table at offset {0:x}.",
              reinterpret_cast<size_t>(&(Table.*ExtEntry)) -
                  reinterpret_cast<size_t>(&Table))));
    }
    UnderlyingStoreLocation = Table.*ExtEntry;
    Table.*ExtEntry = WrapperFunc;
  }

public:
  template <typename... Tuples>
  explicit HipApiTableWrapperInstaller(llvm::Error &Err,
                                       const Tuples &...WrapperSpecs)
      : ApiTableRegistrationCallbackProvider<TableType>(
            [=](llvm::ArrayRef<
                    typename hip::ApiTableEnumInfo<TableType>::ApiTableType *>
                    Tables,
                uint64_t LibVersion, uint64_t LibInstance) {
              LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
                  LibInstance == 0,
                  "Multiple instances of the HIP library registered"));
              (installWrapperEntry(*Tables[0], std::get<0>(WrapperSpecs),
                                   std::get<1>(WrapperSpecs),
                                   std::get<2>(WrapperSpecs)),
               ...);
            },
            Err){};

  ~HipApiTableWrapperInstaller() override = default;
};

using HipCompilerApiTableWrapperInstaller =
    HipApiTableWrapperInstaller<ROCPROFILER_HIP_COMPILER_TABLE>;
using HipRuntimeApiTableWrapperInstaller =
    HipApiTableWrapperInstaller<ROCPROFILER_HIP_RUNTIME_TABLE>;

} // namespace luthier::rocprofiler

#endif