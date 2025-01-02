//===-- hsa.cpp - Top Level HSA API Wrapper -------------------------------===//
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
/// This file contains the implementation of HSA API wrappers concerned with
/// the global status of the HSA runtime (e.g. agents attached to the device).
//===----------------------------------------------------------------------===//
#include "hsa/hsa.hpp"
#include <llvm/ADT/StringExtras.h>

namespace luthier::hsa {

llvm::Error init() {
  const auto &CoreTable =
      hsa::HsaRuntimeInterceptor::instance().getSavedApiTableContainer().core;
  return LUTHIER_HSA_SUCCESS_CHECK(CoreTable.hsa_init_fn());
}

llvm::Error getGpuAgents(llvm::SmallVectorImpl<GpuAgent> &Agents) {
  const auto &CoreTable =
      hsa::HsaRuntimeInterceptor::instance().getSavedApiTableContainer().core;
  auto ReturnGpuAgentsCallback = [](hsa_agent_t Agent, void *Data) {
    auto AgentMap = reinterpret_cast<llvm::SmallVector<GpuAgent> *>(Data);
    hsa_device_type_t DevType = HSA_DEVICE_TYPE_CPU;

    hsa_status_t Status =
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DevType);

    if (Status != HSA_STATUS_SUCCESS)
      return Status;
    if (DevType == HSA_DEVICE_TYPE_GPU) {
      AgentMap->emplace_back(Agent);
    }
    return Status;
  };
  return LUTHIER_HSA_SUCCESS_CHECK(
      CoreTable.hsa_iterate_agents_fn(ReturnGpuAgentsCallback, &Agents));
}

llvm::Expected<std::vector<Executable>> getAllExecutables() {
  const auto &LoaderApi =
      hsa::HsaRuntimeInterceptor::instance().getHsaVenAmdLoaderTable();
  typedef std::vector<Executable> OutType;
  OutType Out;
  auto Iterator = [](hsa_executable_t Exec, void *Data) {
    // Remove executables with nullptr handles
    // This is a workaround for an HSA issue explained here:
    // https://github.com/ROCm/ROCR-Runtime/issues/206
    if (Exec.handle != 0) {
      auto Out = reinterpret_cast<OutType *>(Data);
      Out->emplace_back(Exec);
    }
    return HSA_STATUS_SUCCESS;
  };
  return LUTHIER_HSA_SUCCESS_CHECK(
      LoaderApi.hsa_ven_amd_loader_iterate_executables(Iterator, &Out));
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
convertToHostEquivalent(llvm::ArrayRef<uint8_t> Code) {
  auto CodeStartHostAddress = queryHostAddress(Code.data());
  LUTHIER_RETURN_ON_ERROR(CodeStartHostAddress.takeError());
  return llvm::ArrayRef<uint8_t>{*CodeStartHostAddress, Code.size()};
}

llvm::Expected<llvm::StringRef> convertToHostEquivalent(llvm::StringRef Code) {
  auto Out = convertToHostEquivalent(llvm::arrayRefFromStringRef(Code));
  LUTHIER_RETURN_ON_ERROR(Out.takeError());
  return llvm::toStringRef(*Out);
}

llvm::Error shutdown() {
  const auto &CoreTable =
      hsa::HsaRuntimeInterceptor::instance().getSavedApiTableContainer().core;
  return LUTHIER_HSA_SUCCESS_CHECK(CoreTable.hsa_shut_down_fn());
}
} // namespace luthier::hsa