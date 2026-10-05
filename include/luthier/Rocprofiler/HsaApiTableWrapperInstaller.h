//===-- HsaApiTableWrapperInstaller.h ---------------------------*- C++ -*-===//
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
/// the HSA API table when it is registered with rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_HSA_API_TABLE_WRAPPER_INSTALLER_H
#define LUTHIER_ROCPROFILER_HSA_API_TABLE_WRAPPER_INSTALLER_H
#include "luthier/HSA/ApiTable.h"
#include "luthier/Rocprofiler/ApiTableRegistrationCallbackProvider.h"
#include "luthier/Rocprofiler/HsaApiTableEnumInfo.h"

namespace luthier::rocprofiler {

/// \brief Used to install wrapper functions around entries inside the HSA
/// API table when it is registered with rocprofiler-sdk
/// \tparam HsaApiTableType Type of the HSA API table to install wrappers for
template <typename HsaApiTableType>
class HsaApiTableWrapperInstaller final
    : public ApiTableRegistrationCallbackProvider<ROCPROFILER_HSA_TABLE> {

  /// Installs a wrapper for an entry inside the \p Table
  /// \param Table the sub-table inside the \c ::HsaApiTable to install the
  /// wrapper function into
  /// \param FuncEntry pointer-to-member accessor selecting the function entry
  /// inside \p Table to wrap
  /// \param UnderlyingStoreLocation reference to where the original function
  /// entry will be saved before it is overwritten
  /// \param WrapperFunc the wrapper function pointer to install in place of the
  /// original entry
  /// \note Reports a fatal error if the target entry is not inside the \p Table
  template <typename FuncType>
  static void installWrapperEntry(HsaApiTableType &Table,
                                  FuncType HsaApiTableType::*FuncEntry,
                                  FuncType &UnderlyingStoreLocation,
                                  FuncType WrapperFunc) {
    if (!hsa::apiTableHasEntry(Table, FuncEntry)) {
      LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_HSA_ERROR(
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
  /// \param RocprofilerLib the rocprofiler-sdk library of the target
  /// application; Must outlive this object
  /// \param Err an external \c llvm::Error that will hold any errors
  /// encountered in the constructor
  /// \param WrapperSpecs a variadic set of 3-entry tuples, with each tuple
  /// specifying an entry inside the target API table to be wrapped
  /// \sa installWrapperEntry
  template <typename... Tuples>
  explicit HsaApiTableWrapperInstaller(const DynamicLibrary &RocprofilerLib,
                                       llvm::Error &Err,
                                       const Tuples &...WrapperSpecs)
      : ApiTableRegistrationCallbackProvider(
            [=, this](llvm::ArrayRef<::HsaApiTable *> Tables,
                      uint64_t LibVersion, uint64_t LibInstance) -> void {
              if (Tables.empty())
                LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_MAKE_ROCPROFILER_ERROR(
                    "No tables were passed to the callback"));
              /// Re-install the wrappers on every (re)registration. When the
              /// application finalizes and re-initializes HSA, rocprofiler-sdk
              /// invokes this callback again with \c LibInstance > 0 and hands
              /// over a fresh table populated with the runtime's pristine
              /// entries, so the wrappers must be re-applied. This is safe
              /// because the callback fires during \c hsa_init, before the
              /// table is in use.
              constexpr auto RootAccessor = hsa::ApiTableInfo<
                  HsaApiTableType>::PointerToMemberRootAccessor;
              if (!hsa::apiTableHasEntry<HsaApiTableType>(*Tables[0])) {
                LUTHIER_REPORT_FATAL_ON_ERROR(
                    LUTHIER_MAKE_HSA_ERROR(llvm::formatv(
                        "Captured HSA table doesn't support extension {0}",
                        hsa::ApiTableInfo<HsaApiTableType>::Name)));
              }
              /// The entry being within the table's size does not guarantee
              /// the runtime populated the sub-table pointer; an optional
              /// extension the runtime doesn't provide leaves it null.
              if ((Tables[0]->*RootAccessor) == nullptr) {
                LUTHIER_REPORT_FATAL_ON_ERROR(
                    LUTHIER_MAKE_HSA_ERROR(llvm::formatv(
                        "Captured HSA table's {0} extension pointer is null",
                        hsa::ApiTableInfo<HsaApiTableType>::Name)));
              }
              (installWrapperEntry(
                   *(Tables[0]->*RootAccessor), std::get<0>(WrapperSpecs),
                   std::get<1>(WrapperSpecs), std::get<2>(WrapperSpecs)),
               ...);
            },
            RocprofilerLib, Err) {};

  ~HsaApiTableWrapperInstaller() override = default;
};

} // namespace luthier::rocprofiler

#endif
