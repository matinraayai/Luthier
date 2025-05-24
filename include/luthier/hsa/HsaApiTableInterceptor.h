//===-- HsaApiTableInterceptor.h ---------------------------------*- C++-*-===//
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
/// Defines the <tt>luthier::hsa::HsaApiTableInterceptor</tt>,
/// in charge of capturing the HSA API Table using rocprofiler-sdk and providing
/// the captured table to any component that requests it so they can install
/// wrapper functions.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HSA_API_TABLE_INTERCEPTOR_H
#define LUTHIER_HSA_HSA_API_TABLE_INTERCEPTOR_H
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/LuthierError.h"
#include "luthier/hsa/HsaError.h"
#include "luthier/rocprofiler-sdk/RocprofilerSDKError.h"
#include <hsa/hsa_api_trace.h>
#include <rocprofiler-sdk/intercept_table.h>

namespace luthier::hsa {

template <typename CallbackType> class HsaApiTableInterceptor {
private:
  /// Keeps track of whether the registration callback has been invoked by
  /// rocprofiler-sdk
  bool WasRegCallbackInvoked{false};

  CallbackType CB;

  /// API table registration callback for rocprofiler-sdk
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data) {
    /// Check for errors
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        NumTables == 1,
        "Expected HSA to register only a single API table, instead got {0}",
        NumTables));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        Type == ROCPROFILER_HSA_TABLE,
        "Expected to get HSA API table, but the API table type is {0}", Type));
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        LibInstance == 0, "Multiple instances of HSA library."));
    auto *Table = static_cast<HsaApiTable *>(Tables[0]);
    LUTHIER_REPORT_FATAL_ON_ERROR(
        LUTHIER_ERROR_CHECK(Table != nullptr, "HSA API table is nullptr"));

    auto &Interceptor = *static_cast<HsaApiTableInterceptor *>(Data);
    Interceptor.CB(*Table);
    Interceptor.WasRegCallbackInvoked = true;
  }

  explicit HsaApiTableInterceptor(CallbackType CB) : CB(CB) {};

public:
  static llvm::Expected<std::unique_ptr<HsaApiTableInterceptor>>
  requestApiTable(CallbackType CB) {
    auto Out = std::make_unique<HsaApiTableInterceptor>(CB);
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ROCPROFILER_SUCCESS_CHECK(
        rocprofiler_at_intercept_table_registration(
            HsaApiTableInterceptor::apiRegistrationCallback,
            ROCPROFILER_HSA_TABLE, Out.get())));
    return std::move(Out);
  }

  ~HsaApiTableInterceptor() {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ERROR_CHECK(
        WasRegCallbackInvoked,
        "HSA Api interceptor has been destroyed before rocprofiler-sdk "
        "performed the api table registration callback"));
  }

  [[nodiscard]] bool wasRegistrationCallbackInvoked() const {
    return WasRegCallbackInvoked;
  }
};

} // namespace luthier::hsa

#endif
