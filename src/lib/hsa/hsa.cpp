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
#include <llvm/ADT/StringExtras.h>
#include <luthier/common/ErrorCheck.h>
#include <luthier/hsa/HsaError.h>
#include <luthier/hsa/hsa.h>

namespace luthier::hsa {

llvm::Error init(const decltype(hsa_init) &HsaInitFn) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(HsaInitFn(), "Failed to initialize HSA");
}

llvm::Error getGpuAgents(const decltype(hsa_iterate_agents) &HsaIterateAgentsFn,
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
      HsaIterateAgentsFn(ReturnGpuAgentsCallback, &Agents),
      "Failed to iterate over all HSA agents attached to the system");
}

llvm::Expected<std::vector<hsa_executable_t>> getAllExecutables(
    const decltype(hsa_ven_amd_loader_iterate_executables) &IterateExecFn) {
  typedef std::vector<hsa_executable_t> OutType;
  OutType Out;
  auto Iterator = [](hsa_executable_t Exec, void *Data) {
    // Remove executables with nullptr handles
    // This is a workaround for an HSA issue explained here:
    // https://github.com/ROCm/ROCR-Runtime/issues/206
    if (Exec.handle != 0) {
      auto Out = static_cast<OutType *>(Data);
      Out->emplace_back(Exec);
    }
    return HSA_STATUS_SUCCESS;
  };
  return LUTHIER_HSA_CALL_ERROR_CHECK(IterateExecFn(Iterator, &Out),
                                      "Failed to iterate over HSA executables");
}

llvm::Expected<llvm::ArrayRef<uint8_t>> convertToHostEquivalent(
    const decltype(hsa_ven_amd_loader_query_host_address) &QueryHostFn,
    const llvm::ArrayRef<uint8_t> Code) {
  llvm::Expected<const unsigned char *> CodeStartHostAddressOrErr =
      queryHostAddress(QueryHostFn, Code.data());
  LUTHIER_RETURN_ON_ERROR(CodeStartHostAddressOrErr.takeError());
  return llvm::ArrayRef{*CodeStartHostAddressOrErr, Code.size()};
}

llvm::Expected<llvm::StringRef> convertToHostEquivalent(
    const decltype(hsa_ven_amd_loader_query_host_address) &QueryHostFn,
    const llvm::StringRef Code) {
  llvm::Expected<llvm::ArrayRef<uint8_t>> HostAccessibleCodeOrErr =
      convertToHostEquivalent(QueryHostFn, llvm::arrayRefFromStringRef(Code));
  LUTHIER_RETURN_ON_ERROR(HostAccessibleCodeOrErr.takeError());
  return llvm::toStringRef(*HostAccessibleCodeOrErr);
}

llvm::Error shutdown(const decltype(hsa_shut_down) &HsaShutdownFn) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(HsaShutdownFn(),
                                      "Failed to shutdown the HSA runtime");
}
} // namespace luthier::hsa