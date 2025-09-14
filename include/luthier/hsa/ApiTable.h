//===-- ApiTable.h ----------------------------------------------*- C++ -*-===//
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
/// Defines a set of API Table containers for HSA with automatic bounds
/// checking for table.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_API_TABLE_H
#define LUTHIER_HSA_API_TABLE_H
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/GenericLuthierError.h"
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// \brief Queries whether the HSA API \p Table struct provided by the HSA
/// runtime includes the pointer-to-member \p Entry (i.e., whether the
/// currently running HSA runtime supports a function or an extension table)
/// \details the implementation of the HSA standard by AMD (i.e. ROCr) provides
/// a set of core functionality from the base HSA standard, along with a set of
/// vendor-specific extensions added by AMD. Each extension has a corresponding
/// table in \c ::HsaApiTable (e.g., <tt>::HsaApiTable::core_</tt>). Each
/// extension table is itself a \c struct and each API inside the extension has
/// a corresponding function pointer field in the table (e.g., \c hsa_init has
/// the entry \c &::CoreApiTable::hsa_init_fn in the Core API table). The \c
/// ::HsaApiTable and its tables are forward-compatible i.e. newly added tables
/// and functions are added below the existing ones. This means that each entry
/// remains at a fixed offset in a table, ensuring a tool compiled against an
/// older version of HSA can also run against a newer version of HSA without any
/// interference from the newly added functionality. \n To check if an entry is
/// present in the HSA API table during runtime, a fixed-size \c version field
/// is provided at the very beginning of \c ::HsaApiTable (i.e., <tt>offsetof ==
/// 0</tt>) and also its extension tables. Tools can use the
/// \c ::ApiTableVersion::minor_id field to obtain the size of each
/// table in the active HSA runtime. Since table entries remain at a
/// fixed offset, a tool can confirm the currently running HSA runtime supports
/// its required extension or function by calling this function,
/// which makes sure the corresponding offset of the entry is smaller than the
/// size of the table.
/// \example To check if the HSA API \p Table has the core extension, one can
/// use the following:
/// \code{.cpp}
/// apiTableHasEntry<&::HsaApiTable::core_>(Table);
/// \endcode
/// \example To check if the \c ::CoreApiTable has the \c hsa_iterate_agents_fn
/// field, one can use the following:
/// \code{.cpp}
/// apiTableHasEntry<&::CoreApiTable::hsa_iterate_agents_fn>(CoreApiTable, );
/// \endcode
/// \param Table an HSA API Table or one of its sub-tables, likely obtained from
/// rocprofiler-sdk
/// \return \c true if \p Table contains <tt>Entry</tt>, \c false otherwise
/// \sa <hsa/hsa_api_trace.h> in ROCr
template <auto Entry> bool apiTableHasEntry(const auto &Table) {
  return reinterpret_cast<size_t>(&(Table.*Entry)) -
             reinterpret_cast<size_t>(&Table) <
         Table.version.minor_id;
}

/// Same as <tt>apiTableHasEntry(const auto &)</tt> but with the \p Entry
/// argument not hard coded and instead passed as a function argument
/// \sa apiTableHasEntry(const auto &)
template <typename HsaTableType, typename EntryType>
bool apiTableHasEntry(const HsaTableType &Table,
                      const EntryType HsaTableType::*Entry) {
  return reinterpret_cast<size_t>(&(Table.*Entry)) -
             reinterpret_cast<size_t>(&Table) <
         Table.version.minor_id;
}

/// \brief Struct containing \c constexpr compile-time info regarding individual
/// tables inside <tt>::HsaApiTable</tt>. Used to provide convenience accessors
/// using the table's type. If a table is not present here, simply define a
/// specialization of this template.
template <typename ApiTableType> struct ApiTableInfo;

template <> struct ApiTableInfo<::CoreApiTable> {
  static constexpr auto Name = "core";
  static constexpr auto PointerToMemberRootAccessor = &::HsaApiTable::core_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::core;
};

template <> struct ApiTableInfo<::AmdExtTable> {
  static constexpr auto Name = "amd";
  static constexpr auto PointerToMemberRootAccessor = &::HsaApiTable::amd_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::amd_ext;
};

template <> struct ApiTableInfo<::FinalizerExtTable> {
  static constexpr auto Name = "finalizer";
  static constexpr auto PointerToMemberRootAccessor =
      &::HsaApiTable::finalizer_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::finalizer_ext;
};

template <> struct ApiTableInfo<::ImageExtTable> {
  static constexpr auto Name = "image";
  static constexpr auto PointerToMemberRootAccessor =
      &::HsaApiTable::image_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::image_ext;
};

template <> struct ApiTableInfo<::ToolsApiTable> {
  static constexpr auto Name = "tools";
  static constexpr auto PointerToMemberRootAccessor = &::HsaApiTable::tools_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::tools;
};

template <> struct ApiTableInfo<::PcSamplingExtTable> {
  static constexpr auto Name = "pc sampling";
  static constexpr auto PointerToMemberRootAccessor =
      &::HsaApiTable::pc_sampling_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::pc_sampling_ext;
};

/// \brief An HSA API table container which provides bounds checking over
/// the entries inside the API table
/// \tparam ApiTableType Type of the HSA API table e.g. \c ::CoreApiTable
/// \sa apiTableHasEntry
template <typename ApiTableType> class ApiTableContainer {
private:
  const ApiTableType &ApiTable{};

public:
  explicit ApiTableContainer(const ApiTableType &ApiTable)
      : ApiTable(ApiTable) {};

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
        tableSupportsFunction<Func>(),
        "The passed function is not inside the table."));
    return *(ApiTable.*Func);
  }

  /// Obtains the function \c Func from the table snapshot and calls
  /// it with the passed \p Args and returns the results of the function call
  template <auto ApiTableType::*Func, typename... ArgTypes>
  auto callFunction(ArgTypes... Args) const {
    return getFunction<Func>()(Args...);
  }
};

/// \brief A static mapping struct between the HSA extension
/// enumerator and its corresponding latest table version type and major/minor
/// versions.
template <hsa_extension_t ExtType> struct ExtensionApiTableInfo;

template <> struct ExtensionApiTableInfo<HSA_EXTENSION_FINALIZER> {
  using TableType = hsa_ext_finalizer_1_00_pfn_t;
  static constexpr auto MajorVer = HSA_FINALIZER_API_TABLE_MAJOR_VERSION;
  static constexpr auto StepVer = HSA_FINALIZER_API_TABLE_STEP_VERSION;
};

template <> struct ExtensionApiTableInfo<HSA_EXTENSION_IMAGES> {
  using TableType = hsa_ext_images_1_pfn_t;
  static constexpr auto MajorVer = HSA_IMAGE_API_TABLE_MAJOR_VERSION;
  static constexpr auto StepVer = HSA_IMAGE_API_TABLE_STEP_VERSION;
};

template <> struct ExtensionApiTableInfo<HSA_EXTENSION_AMD_LOADER> {
  using TableType = hsa_ven_amd_loader_1_03_pfn_t;
  static constexpr auto MajorVer = 1;
  static constexpr auto StepVer = 3;
};

template <> struct ExtensionApiTableInfo<HSA_EXTENSION_AMD_PC_SAMPLING> {
  using TableType = hsa_ven_amd_pc_sampling_1_00_pfn_t;
  static constexpr auto MajorVer = HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION;
  static constexpr auto StepVer = HSA_PC_SAMPLING_API_TABLE_STEP_VERSION;
};

/// TODO: Check if a "compatibility checker" struct between extension table
/// versions is desired

} // namespace luthier::hsa

#endif