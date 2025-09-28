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
///
/// \file
/// Implements a set of commonly used functionality regarding the global state
/// of the HSA runtime.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/hsa.h"
#include "luthier/hsa/Executable.h"
#include "luthier/hsa/ExecutableSymbol.h"
#include "luthier/hsa/HsaError.h"
#include <llvm/ADT/StringExtras.h>

namespace luthier::hsa {

llvm::Error init(const ApiTableContainer<::CoreApiTable> &CoreApi) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(CoreApi.callFunction<hsa_init>(),
                                      "Failed to initialize HSA");
}

llvm::Error getGpuAgents(const ApiTableContainer<::CoreApiTable> &CoreApi,
                         llvm::SmallVectorImpl<hsa_agent_t> &Agents) {
  auto ReturnGpuAgentsCallback = [](hsa_agent_t Agent, void *Data) {
    auto AgentList = static_cast<llvm::SmallVector<hsa_agent_t> *>(Data);
    hsa_device_type_t DevType = HSA_DEVICE_TYPE_CPU;

    const hsa_status_t Status =
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DevType);

    if (Status != HSA_STATUS_SUCCESS)
      return Status;
    if (DevType == HSA_DEVICE_TYPE_GPU) {
      AgentList->emplace_back(Agent);
    }
    return Status;
  };
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<hsa_iterate_agents>(ReturnGpuAgentsCallback,
                                               &Agents),
      "Failed to iterate over all HSA agents attached to the system");
}

llvm::Error shutdown(const ApiTableContainer<::CoreApiTable> &CoreApi) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(CoreApi.callFunction<hsa_shut_down>(),
                                      "Failed to shutdown the HSA runtime");
}
} // namespace luthier::hsa