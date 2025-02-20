//===-- HipRuntimeApiInterceptor.hpp - HIP Runtime API Interceptor --------===//
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
/// This file defines Luthier's HIP Runtime API Interceptor Singleton, used
/// to install wrappers over the HIP dispatch API table using rocprofiler-sdk
/// and providing callbacks to Luthier sub-systems.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HIP_HIP_RUNTIME_API_INTERCEPTOR_HPP
#define LUTHIER_HIP_HIP_RUNTIME_API_INTERCEPTOR_HPP

#include "common/ROCmLibraryApiInterceptor.hpp"
#include "common/Singleton.hpp"
#include <functional>
#include <llvm/ADT/DenseSet.h>
#include <luthier/hip/TraceApi.h>
#include <luthier/types.h>

namespace luthier::hip {

class HipRuntimeApiInterceptor
    : public ROCmLibraryApiInterceptor<ApiEvtID, ApiEvtArgs, HipDispatchTable,
                                       HipDispatchTable>,
      public Singleton<HipRuntimeApiInterceptor> {
protected:
  llvm::Error installWrapper(ApiEvtID ApiID) override;

  llvm::Error uninstallWrapper(ApiEvtID ApiID) override;

public:
  HipRuntimeApiInterceptor() = default;

  ~HipRuntimeApiInterceptor() {
    if (RuntimeApiTable != nullptr)
      *RuntimeApiTable = SavedRuntimeApiTable;
    SavedRuntimeApiTable = {};
    Singleton<HipRuntimeApiInterceptor>::~Singleton();
  }

  llvm::Error initializeInterceptor(HipDispatchTable &Table) override {
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
