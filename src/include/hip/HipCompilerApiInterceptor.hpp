//===-- HipCompilerApiInterceptor.hpp - Luthier's HIP API Interceptor -----===//
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
///
/// \file
/// This file contains Luthier's HIP API Interceptor Singleton, implemented
/// using the rocprofiler-sdk API for capturing HIP compiler and runtime API
/// tables.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_HIP_HIP_COMPILER_API_INTERCEPT_HPP
#define LUTHIER_HIP_HIP_COMPILER_API_INTERCEPT_HPP

#include "common/ROCmLibraryApiInterceptor.hpp"
#include "common/Singleton.hpp"
#include <functional>
#include <llvm/ADT/DenseSet.h>
#include <luthier/hip/TraceApi.h>
#include <luthier/types.h>

namespace luthier::hip {

class HipCompilerApiInterceptor final
    : public ROCmLibraryApiInterceptor<ApiEvtID, ApiEvtArgs,
                                       HipCompilerDispatchTable,
                                       HipCompilerDispatchTable>,
      public Singleton<HipCompilerApiInterceptor> {

protected:
  llvm::Error installWrapper(ApiEvtID ApiID) override;

  llvm::Error uninstallWrapper(ApiEvtID ApiID) override;

public:
  HipCompilerApiInterceptor() = default;
  ~HipCompilerApiInterceptor() override {
    if (RuntimeApiTable != nullptr)
      *RuntimeApiTable = SavedRuntimeApiTable;
    SavedRuntimeApiTable = {};
    Singleton<HipCompilerApiInterceptor>::~Singleton();
  }

  llvm::Error initializeInterceptor(HipCompilerDispatchTable &Table) override {
    std::unique_lock Lock(InterceptorMutex);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        RuntimeApiTable == nullptr, "Interceptor is already initialized."));
    RuntimeApiTable = &Table;
    SavedRuntimeApiTable = Table;
    for (const auto &[ApiID, CBs] : InterceptedApiIDCallbacks) {
      LUTHIER_RETURN_ON_ERROR(installWrapper(ApiID));
    }
    return llvm::Error::success();
  }
};

} // namespace luthier::hip

#endif
