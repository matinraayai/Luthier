//===-- HipCompilerApiTableInterceptor.hpp --------------------------------===//
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
/// Defines the <tt>HipCompilerApiTableInterceptor</tt>,
/// in charge of capturing the HIP Compiler API Table using rocprofiler-sdk
/// and providing the captured table to other components of Luthier so that
/// they can install wrapper functions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HIP_HIP_COMPILER_API_TABLE_INTERCEPTOR_HPP
#define LUTHIER_HIP_HIP_COMPILER_API_TABLE_INTERCEPTOR_HPP
#include "HipRuntimeApiInterceptor.hpp"
#include "common/Singleton.hpp"
#include "luthier/rocprofiler-sdk/RocprofilerSDKError.h"
#include <rocprofiler-sdk/intercept_table.h>

namespace luthier::hip {

class HipCompilerApiTableInterceptor {
protected:
  /// Pointer to the HIP Compiler API table, provided by rocprofiler-sdk.
  /// The HIP runtime uses this table to call HIP __register* functions
  ::HipCompilerDispatchTable *RuntimeApiTable{nullptr};

  /// Saved copy of the "real" API table
  ::HipCompilerDispatchTable SavedApiTable{};

  /// Mutex to protect the \c RegistrationCallbacks field
  std::mutex CBMutex;

  /// A set of callbacks invoked when rocprofiler-sdk provides the API table
  llvm::SmallVector<std::function<void(::HipCompilerDispatchTable &)>>
      RegistrationCallbacks;

  HipCompilerApiTableInterceptor() = default;

  virtual ~HipCompilerApiTableInterceptor() = default;

  bool isTableCaptured() const { return RuntimeApiTable != nullptr; }

  [[nodiscard]] const ::HipCompilerDispatchTable &getHipCompilerTable() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        isTableCaptured(), "HIP Compiler API table is not captured"));
    return SavedApiTable;
  }

  llvm::Error addRegistrationCallback(
      const std::function<void(::HipCompilerDispatchTable &)> &CB) {
    std::lock_guard Lock(CBMutex);
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        !isTableCaptured(), "HIP Compiler API table is already captured"));
    RegistrationCallbacks.emplace_back(CB);
    return llvm::Error::success();
  }
};

template <size_t Idx>
class ROCPROFILER_HIDDEN_API UniqueHipCompilerApiTableInterceptor final
    : public HipCompilerApiTableInterceptor,
      public Singleton<UniqueHipCompilerApiTableInterceptor<Idx>> {
private:
  /// API table registration callback for rocprofiler-sdk
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data) {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        UniqueHipCompilerApiTableInterceptor::isInitialized(),
        "HIP API Table Interceptor number {0} is not initialized", Idx));
    auto &I = UniqueHipCompilerApiTableInterceptor::instance();
    std::lock_guard Lock(I.CBMutex);
    /// Check for errors
    LUTHIER_REPORT_FATAL_ON_ERROR(
        LUTHIER_ERROR_CHECK(NumTables == 1,
                            "Expected HIP to register only a single compiler "
                            "API table, instead got {0}",
                            NumTables));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        Type == ROCPROFILER_HIP_COMPILER_TABLE,
        "Expected to get HIP Compiler API table, but the API table type is {0}",
        Type));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        LibInstance == 0, "Multiple instances of HIP Compiler API table."));
    auto *Table = static_cast<HipCompilerDispatchTable *>(Tables[0]);
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        Table != nullptr, "HIP Compiler API table is nullptr"));
    I.RuntimeApiTable = Table;
    std::memcpy(&I.SavedApiTable, I.RuntimeApiTable, I.RuntimeApiTable->size);
    /// Invoke registered callbacks
    for (const auto &CB : I.RegistrationCallbacks) {
      CB(*I.RuntimeApiTable);
    }
  }

public:
  static llvm::Expected<std::unique_ptr<UniqueHipCompilerApiTableInterceptor>>
  initialize() {
    auto Out = std::make_unique<UniqueHipCompilerApiTableInterceptor>();
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ROCPROFILER_SUCCESS_CHECK(
        rocprofiler_at_intercept_table_registration(
            UniqueHipCompilerApiTableInterceptor::apiRegistrationCallback,
            ROCPROFILER_HIP_COMPILER_TABLE, nullptr)));
    return std::move(Out);
  }

  ~UniqueHipCompilerApiTableInterceptor() override = default;
};

template <size_t Idx>
ROCPROFILER_HIDDEN_API UniqueHipCompilerApiTableInterceptor<Idx>
    *Singleton<UniqueHipCompilerApiTableInterceptor<Idx>>::Instance{nullptr};

} // namespace luthier::hip

#endif
