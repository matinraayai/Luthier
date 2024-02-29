#include "hsa.hpp"
#include <llvm/ADT/StringExtras.h>

namespace luthier::hsa {

llvm::Error getGpuAgents(llvm::SmallVectorImpl<GpuAgent> &agents) {
  const auto &CoreTable = HsaInterceptor::instance().getSavedHsaTables().core;
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

llvm::Error getAllExecutables(llvm::SmallVectorImpl<Executable> &Executables) {
  const auto &LoaderApi = HsaInterceptor::instance().getHsaVenAmdLoaderTable();
  auto Iterator = [](hsa_executable_t exec, void *data) {
    auto Out = reinterpret_cast<llvm::SmallVectorImpl<Executable> *>(data);
    Out->emplace_back(exec);
    return HSA_STATUS_SUCCESS;
  };
  return LUTHIER_HSA_SUCCESS_CHECK(
      LoaderApi.hsa_ven_amd_loader_iterate_executables(Iterator, &Executables));
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
convertToHostEquivalent(llvm::ArrayRef<uint8_t> code) {
  auto CodeStartHostAddress = queryHostAddress(code.data());
  LUTHIER_RETURN_ON_ERROR(CodeStartHostAddress.takeError());
  return llvm::ArrayRef<uint8_t>{*CodeStartHostAddress, code.size()};
}

llvm::Expected<llvm::StringRef>
convertToHostEquivalent(llvm::StringRef Code) {
  auto Out = convertToHostEquivalent(llvm::arrayRefFromStringRef(Code));
  LUTHIER_RETURN_ON_ERROR(Out.takeError());
  return llvm::toStringRef(*Out);
}

} // namespace luthier::hsa