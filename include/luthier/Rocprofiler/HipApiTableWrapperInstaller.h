//===-- HipApiTableWrapperInstaller.h ---------------------------*- C++ -*-===//
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
/// Defines the class used to install wrapper functions around entries inside
/// a HIP API table when it is registered with rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_HIP_API_TABLE_WRAPPER_INSTALLER_H
#define LUTHIER_ROCPROFILER_HIP_API_TABLE_WRAPPER_INSTALLER_H
#include "luthier/HIP/ApiTable.h"
#include "luthier/HIP/HipError.h"
#include "luthier/Rocprofiler/ApiTableRegistrationCallbackProvider.h"
#include "luthier/Rocprofiler/HipApiTableEnumInfo.h"

namespace luthier::rocprofiler {

template <rocprofiler_intercept_table_t TableType,
          typename =
              std::enable_if_t<TableType == ROCPROFILER_HIP_COMPILER_TABLE ||
                               TableType == ROCPROFILER_HIP_RUNTIME_TABLE>>
class HipApiTableWrapperInstaller final
    : public ApiTableRegistrationCallbackProvider<TableType> {
private:
  /// Installs a wrapper for an entry inside the HIP API \p Table
  /// \param Table the HIP API table to install the wrapper function into
  /// \param ExtEntry pointer-to-member accessor selecting the function entry
  /// inside \p Table to wrap
  /// \param UnderlyingStoreLocation reference to where the original function
  /// entry will be saved before it is overwritten
  /// \param WrapperFunc the wrapper function pointer to install in place of the
  /// original entry
  /// \note Reports a fatal error if the target entry is not inside the \p Table
  template <typename FuncType>
  static void installWrapperEntry(
      ApiTableEnumInfo<TableType>::ApiTableType &Table,
      FuncType ApiTableEnumInfo<TableType>::ApiTableType::*ExtEntry,
      FuncType &UnderlyingStoreLocation, FuncType WrapperFunc) {
    if (!hip::apiTableHasEntry(Table, ExtEntry)) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_HIP_ERROR(llvm::formatv(
          "Failed to find entry inside the HIP API table at offset {0:x}.",
          reinterpret_cast<size_t>(&(Table.*ExtEntry)) -
              reinterpret_cast<size_t>(&Table))));
    }
    UnderlyingStoreLocation = Table.*ExtEntry;
    Table.*ExtEntry = WrapperFunc;
  }

public:
  template <typename... Tuples>
  explicit HipApiTableWrapperInstaller(const DynamicLibrary &RocprofilerLib,
                                       llvm::Error &Err,
                                       const Tuples &...WrapperSpecs)
      : ApiTableRegistrationCallbackProvider<TableType>(
            [=](llvm::ArrayRef<
                    typename ApiTableEnumInfo<TableType>::ApiTableType *>
                    Tables,
                uint64_t LibVersion, uint64_t LibInstance) {
              if (Tables.empty())
                LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
                    "No tables were passed to the callback"));
              /// Re-install the wrappers on every (re)registration. When the
              /// application finalizes and re-initializes the runtime,
              /// rocprofiler-sdk invokes this callback again with
              /// \c LibInstance > 0 and hands over a fresh table populated with
              /// the runtime's pristine entries, so the wrappers must be
              /// re-applied. This is safe because the callback fires during
              /// runtime initialization, before the table is in use.
              (installWrapperEntry(*Tables[0], std::get<0>(WrapperSpecs),
                                   std::get<1>(WrapperSpecs),
                                   std::get<2>(WrapperSpecs)),
               ...);
            },
            RocprofilerLib, Err) {};

  ~HipApiTableWrapperInstaller() override = default;
};

using HipCompilerApiTableWrapperInstaller =
    HipApiTableWrapperInstaller<ROCPROFILER_HIP_COMPILER_TABLE>;
using HipRuntimeApiTableWrapperInstaller =
    HipApiTableWrapperInstaller<ROCPROFILER_HIP_RUNTIME_TABLE>;

} // namespace luthier::rocprofiler

#endif
