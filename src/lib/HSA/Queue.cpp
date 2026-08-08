//===-- Queue.cpp -----------------------------------------------------------===//
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
/// Implements a set of commonly used functionality for the \c hsa_queue_t
/// handle in HSA.
//===----------------------------------------------------------------------===//
#include "luthier/HSA/Queue.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/HSA/HsaError.h"
#include <cstdint>
#include <llvm/Support/FormatVariadic.h>

namespace luthier::hsa {

llvm::Expected<hsa_queue_t *>
queueCreate(const ApiTableContainer<::CoreApiTable> &CoreApi,
            const hsa_agent_t Agent, const uint32_t Size,
            const hsa_queue_type32_t Type) {
  hsa_queue_t *Queue;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_queue_create>(Agent, Size, Type, nullptr,
                                             nullptr, UINT32_MAX, UINT32_MAX,
                                             &Queue),
      llvm::formatv("Failed to create a queue of size {0} on agent {1:x}",
                    Size, Agent.handle)));
  return Queue;
}

llvm::Error queueDestroy(const ApiTableContainer<::CoreApiTable> &CoreApi,
                         hsa_queue_t *const Queue) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_queue_destroy>(Queue),
      "Failed to destroy an HSA queue");
}

} // namespace luthier::hsa
