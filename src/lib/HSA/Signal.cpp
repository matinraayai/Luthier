//===-- Signal.cpp ----------------------------------------------------------===//
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
///
/// \file
/// Implements a set of commonly used functionality for the \c hsa_signal_t
/// handle in HSA.
//===----------------------------------------------------------------------===//
#include "luthier/HSA/Signal.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSA/HsaError.h"
#include <cstdint>
#include <llvm/Support/FormatVariadic.h>

namespace luthier::hsa {

llvm::Expected<hsa_signal_t>
signalCreate(const ApiTableContainer<::CoreApiTable> &CoreApi,
            const hsa_signal_value_t InitialValue) {
  hsa_signal_t Signal;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_signal_create>(InitialValue, 0, nullptr,
                                              &Signal),
      "Failed to create an HSA signal"));
  return Signal;
}

llvm::Error signalDestroy(const ApiTableContainer<::CoreApiTable> &CoreApi,
                         const hsa_signal_t Signal) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_signal_destroy>(Signal),
      llvm::formatv("Failed to destroy signal {0:x}", Signal.handle));
}

hsa_signal_value_t
signalWait(const ApiTableContainer<::CoreApiTable> &CoreApi,
          const hsa_signal_t Signal, const hsa_signal_condition_t Condition,
          const hsa_signal_value_t CompareValue) {
  return CoreApi.callFunction<hsa_signal_wait_scacquire>(
      Signal, Condition, CompareValue, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
}

} // namespace luthier::hsa
