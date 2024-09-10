//===-- HipRuntimeApiInterceptor.hpp - HIP Runtime API Interceptor --------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file contains Luthier's HIP Runtime API Interceptor Singleton,
/// implemented using the rocprofiler-sdk API for capturing the HIP runtime API
/// tables and installing wrappers around them.
//===----------------------------------------------------------------------===//

#ifndef LUTHIER_HIP_HIP_RUNTIME_API_INTERCEPTOR_HPP
#define LUTHIER_HIP_HIP_RUNTIME_API_INTERCEPTOR_HPP

#include <functional>
#include <llvm/ADT/DenseSet.h>

#include "common/Error.hpp"
#include "common/ROCmLibraryApiInterceptor.hpp"
#include "common/Singleton.hpp"
#include <luthier/hip/TraceApi.h>
#include <luthier/types.h>

namespace luthier::hip {

class HipRuntimeApiInterceptor
    : public ROCmLibraryApiInterceptor<luthier::hip::ApiEvtID,
                                       luthier::hip::ApiEvtArgs,
                                       HipDispatchTable, HipDispatchTable>,
      public Singleton<HipRuntimeApiInterceptor> {

public:
  HipRuntimeApiInterceptor() = default;
  ~HipRuntimeApiInterceptor() {
    if (RuntimeApiTable != nullptr)
      *RuntimeApiTable = SavedRuntimeApiTable;
    SavedRuntimeApiTable = {};
    Singleton<HipRuntimeApiInterceptor>::~Singleton();
  }

  llvm::Error enableUserCallback(luthier::hip::ApiEvtID Op);

  llvm::Error disableUserCallback(luthier::hip::ApiEvtID Op);

  llvm::Error enableInternalCallback(luthier::hip::ApiEvtID Op);

  llvm::Error disableInternalCallback(luthier::hip::ApiEvtID Op);

  llvm::Error captureApiTable(HipDispatchTable *Table) {
    RuntimeApiTable = Table;
    SavedRuntimeApiTable = *Table;
    Status = API_TABLE_CAPTURED;
    return llvm::Error::success();
  }
};

} // namespace luthier::hip

#endif
