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
/// the captured table via a callback to any other component that requested it.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HSA_API_TABLE_INTERCEPTOR_H
#define LUTHIER_HSA_HSA_API_TABLE_INTERCEPTOR_H
#include <hsa/hsa_api_trace.h>
#include <luthier/common/GenericLuthierError.h>
#include <luthier/rocprofiler/RocprofilerError.h>
#include <rocprofiler-sdk/intercept_table.h>

namespace luthier::hsa {

class HsaApiTableInterceptor {
private:
  /// Keeps track of whether the registration callback has been invoked by
  /// rocprofiler-sdk
  bool WasRegCallbackInvoked{false};

  typedef std::function<llvm::Error(HsaApiTable &)> CallbackType;
  /// Callback invoked when the API table has been passed down to us
  /// by rocprofiler-sdk
  CallbackType Callback;

  /// API table registration callback for used by rocprofiler-sdk
  static void apiRegistrationCallback(rocprofiler_intercept_table_t Type,
                                      uint64_t LibVersion, uint64_t LibInstance,
                                      void **Tables, uint64_t NumTables,
                                      void *Data);

  explicit HsaApiTableInterceptor(CallbackType Callback)
      : Callback(std::move(Callback)) {};

public:
  static llvm::Expected<std::unique_ptr<HsaApiTableInterceptor>>
  requestApiTable(CallbackType CB);

  ~HsaApiTableInterceptor();

  [[nodiscard]] bool wasRegistrationCallbackInvoked() const {
    return WasRegCallbackInvoked;
  }
};

} // namespace luthier::hsa

#endif
