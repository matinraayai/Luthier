#ifndef HSA_HPP
#define HSA_HPP
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include <llvm/ADT/SmallVector.h>

namespace luthier::hsa {

llvm::Error getGpuAgents(llvm::SmallVectorImpl<GpuAgent> &Agents);

llvm::Error getAllExecutables(llvm::SmallVectorImpl<Executable> &Executables);

template <typename T> llvm::Expected<T *> queryHostAddress(T *DeviceAddress) {
  const auto &LoaderApi = Interceptor::instance().getHsaVenAmdLoaderTable();
  const T *HostAddress;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(LoaderApi.hsa_ven_amd_loader_query_host_address(
          DeviceAddress, (reinterpret_cast<const void **>(&HostAddress)))));
  return HostAddress;
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
convertToHostEquivalent(llvm::ArrayRef<uint8_t> Code);

llvm::Expected<llvm::StringRef> convertToHostEquivalent(llvm::StringRef Code);

} // namespace luthier::hsa

#endif