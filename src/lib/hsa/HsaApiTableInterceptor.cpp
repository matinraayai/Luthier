//===-- HsaApiTableInterceptor.cpp ----------------------------------------===//
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
/// Implements the HsaApiTableInterceptor class.
//===----------------------------------------------------------------------===//
#include <luthier/common/ErrorCheck.h>
#include <luthier/hsa/HsaApiTableInterceptor.h>
#include <luthier/rocprofiler/RocprofilerError.h>
#include <rocprofiler-sdk/registration.h>

namespace luthier::hsa {

void HsaApiTableInterceptor::apiRegistrationCallback(
    rocprofiler_intercept_table_t Type, uint64_t LibVersion,
    uint64_t LibInstance, void **Tables, uint64_t NumTables, void *Data) {
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
  auto *Table = static_cast<HsaApiTable *>(Tables[0]);
  if (Table == nullptr) {
    LUTHIER_REPORT_FATAL_ON_ERROR(
        llvm::make_error<rocprofiler::RocprofilerError>(
            "HSA API table is nullptr"));
  }

  auto &Interceptor = *static_cast<HsaApiTableInterceptor *>(Data);
  LUTHIER_REPORT_FATAL_ON_ERROR(Interceptor.Callback(*Table));
  Interceptor.WasRegCallbackInvoked = true;
}
llvm::Expected<std::unique_ptr<HsaApiTableInterceptor>>
HsaApiTableInterceptor::requestApiTable(CallbackType CB) {
  auto Out = std::make_unique<HsaApiTableInterceptor>(std::move(CB));
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
      rocprofiler_at_intercept_table_registration(
          HsaApiTableInterceptor::apiRegistrationCallback,
          ROCPROFILER_HSA_TABLE, Out.get()),
      "Failed to request HSA API tables from rocprofiler-sdk"));
  return std::move(Out);
}

HsaApiTableInterceptor::~HsaApiTableInterceptor() {
  int RocprofilerFiniStatus;
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_ROCPROFILER_CALL_ERROR_CHECK(
      rocprofiler_is_finalized(&RocprofilerFiniStatus),
      "Failed to check if rocprofiler is finalized."));
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
    WasRegCallbackInvoked || RocprofilerFiniStatus != 0,
    "HSA Api interceptor has been destroyed before rocprofiler-sdk "
    "performed the api table registration callback"));
}

} // namespace luthier::hsa