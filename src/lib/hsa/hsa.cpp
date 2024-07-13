//===-- hsa.cpp - Top Level HSA API Wrapper -------------------------------===//
//
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
  const auto &CoreTable = hsa::Interceptor::instance().getSavedHsaTables().core;
  return LUTHIER_HSA_SUCCESS_CHECK(CoreTable.hsa_init_fn());
}

llvm::Error getGpuAgents(llvm::SmallVectorImpl<GpuAgent> &agents) {
  const auto &CoreTable = hsa::Interceptor::instance().getSavedHsaTables().core;
  auto ReturnGpuAgentsCallback = [](hsa_agent_t agent, void *data) {
    auto AgentMap = reinterpret_cast<llvm::SmallVector<GpuAgent> *>(data);
    hsa_device_type_t DevType = HSA_DEVICE_TYPE_CPU;

    hsa_status_t Stat =
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &DevType);

    if (Stat != HSA_STATUS_SUCCESS)
      return Stat;
    if (DevType == HSA_DEVICE_TYPE_GPU) {
      AgentMap->emplace_back(agent);
    }
    return Stat;
  };
  return LUTHIER_HSA_SUCCESS_CHECK(
      CoreTable.hsa_iterate_agents_fn(ReturnGpuAgentsCallback, &agents));
}

llvm::Expected<std::vector<Executable>> getAllExecutables() {
  const auto &LoaderApi =
      hsa::Interceptor::instance().getHsaVenAmdLoaderTable();
  typedef std::vector<Executable> OutType;
  OutType Out;
  auto Iterator = [](hsa_executable_t exec, void *data) {
    if (exec.handle != 0) {
      auto Out = reinterpret_cast<OutType *>(data);
      Out->emplace_back(exec);
    }
    return HSA_STATUS_SUCCESS;
  };
  return LUTHIER_HSA_SUCCESS_CHECK(
      LoaderApi.hsa_ven_amd_loader_iterate_executables(Iterator, &Out));
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
convertToHostEquivalent(llvm::ArrayRef<uint8_t> code) {
  auto CodeStartHostAddress = queryHostAddress(code.data());
  LUTHIER_RETURN_ON_ERROR(CodeStartHostAddress.takeError());
  return llvm::ArrayRef<uint8_t>{*CodeStartHostAddress, code.size()};
}

llvm::Expected<llvm::StringRef> convertToHostEquivalent(llvm::StringRef Code) {
  auto Out = convertToHostEquivalent(llvm::arrayRefFromStringRef(Code));
  LUTHIER_RETURN_ON_ERROR(Out.takeError());
  return llvm::toStringRef(*Out);
}
llvm::Error shutdown() {
  const auto &CoreTable = hsa::Interceptor::instance().getSavedHsaTables().core;
  return LUTHIER_HSA_SUCCESS_CHECK(CoreTable.hsa_shut_down_fn());
}
} // namespace luthier::hsa