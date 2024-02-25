#include "hsa.hpp"

namespace luthier::hsa {

llvm::Error getGpuAgents(llvm::SmallVectorImpl<GpuAgent> &agents) {
  const auto &coreTable = HsaInterceptor::instance().getSavedHsaTables().core;
  auto returnGpuAgentsCallback = [](hsa_agent_t agent, void *data) {
    auto agentMap = reinterpret_cast<llvm::SmallVector<GpuAgent> *>(data);
    hsa_device_type_t devType = HSA_DEVICE_TYPE_CPU;

    hsa_status_t stat =
        hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &devType);

    if (stat != HSA_STATUS_SUCCESS)
      return stat;
    if (devType == HSA_DEVICE_TYPE_GPU) {
      agentMap->emplace_back(agent);
    }
    return stat;
  };
  return LUTHIER_HSA_SUCCESS_CHECK(
      coreTable.hsa_iterate_agents_fn(returnGpuAgentsCallback, &agents));
}

llvm::Error getAllExecutables(llvm::SmallVectorImpl<Executable> &executables) {
  const auto &loaderApi = HsaInterceptor::instance().getHsaVenAmdLoaderTable();
  auto iterator = [](hsa_executable_t exec, void *data) {
    auto out = reinterpret_cast<llvm::SmallVectorImpl<Executable> *>(data);
    out->emplace_back(exec);
    return HSA_STATUS_SUCCESS;
  };
  return LUTHIER_HSA_SUCCESS_CHECK(
      loaderApi.hsa_ven_amd_loader_iterate_executables(iterator, &executables));
}

} // namespace luthier::hsa