//===-- hsa.cpp -----------------------------------------------------------===//
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
/// \file hsa.cpp
/// Implements a set of commonly used functionality regarding the global state
/// of the HSA runtime.
//===----------------------------------------------------------------------===//
#include "luthier/HSA/hsa.h"
#include "luthier/HSA/Executable.h"
#include "luthier/HSA/ExecutableSymbol.h"
#include "luthier/HSA/HsaError.h"
#include <llvm/ADT/StringExtras.h>

namespace luthier::hsa {

llvm::Error init(const ApiTableContainer<::CoreApiTable> &CoreApi) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(CoreApi.callFunction<hsa_init>(),
                                      "Failed to initialize HSA");
}

llvm::Error shutdown(const ApiTableContainer<::CoreApiTable> &CoreApi) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(CoreApi.callFunction<hsa_shut_down>(),
                                      "Failed to shutdown the HSA runtime");
}
} // namespace luthier::hsa