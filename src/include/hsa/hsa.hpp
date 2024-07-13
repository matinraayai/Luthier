//===-- hsa.hpp - Top Level HSA API Wrapper -------------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of HSA API wrappers concerned with
/// the global status of the HSA runtime (e.g. agents attached to the device).
//===----------------------------------------------------------------------===//

#ifndef HSA_HSA_HPP
#define HSA_HSA_HPP
#include "hsa/hsa_agent.hpp"
#include "hsa/hsa_executable.hpp"
#include <llvm/ADT/SmallVector.h>

namespace luthier::hsa {

/// Initializes the HSA runtime if already not initialized, and increments
/// a reference counter.\n
/// Must only be used by unit tests
/// \return an \c llvm::Error indicating any HSA issues encountered
/// \sa hsa_init
llvm::Error init();

/// Queries the <tt>GpuAgent</tt>s attached to the device
/// \param [out] Agents <tt>GpuAgent</tt>s attached to the device
/// \return an \c llvm::Error indicating any HSA issues encountered
/// \sa hsa_iterate_agents
llvm::Error getGpuAgents(llvm::SmallVectorImpl<GpuAgent> &Agents);

/// \return All <tt>hsa::Executable</tt>s currently loaded into the HSA runtime
/// \sa hsa_ven_amd_loader_iterate_executables
llvm::Expected<std::vector<Executable>> getAllExecutables();

/// Queries the host-equivalent address of the given \p DeviceAddress \n
/// \tparam T Pointer type of the address
/// \param DeviceAddress Address residing on the device
/// \return The host equivalent address of the <tt>DeviceAddress</tt>, or an
/// \c llvm::Error indicating any HSA errors encountered
/// \sa hsa_ven_amd_loader_query_host_address
template <typename T> llvm::Expected<T *> queryHostAddress(T *DeviceAddress) {
  const auto &LoaderApi = Interceptor::instance().getHsaVenAmdLoaderTable();
  const T *HostAddress;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(LoaderApi.hsa_ven_amd_loader_query_host_address(
          DeviceAddress, (reinterpret_cast<const void **>(&HostAddress)))));
  return HostAddress;
}

/// Convenience version of <tt>hsa::queryHostAddress(T *)</tt> for locating
/// device code on the host
/// \param Code Device code on the device
/// \return an \c llvm::ArrayRef<uint8_t> pointing to the code accessible on
/// host memory, or an \c llvm::Error indicating any HSA errors encountered
llvm::Expected<llvm::ArrayRef<uint8_t>>
convertToHostEquivalent(llvm::ArrayRef<uint8_t> Code);

/// Convenience version of <tt>hsa::queryHostAddress(T *)</tt> for locating
/// device code on the host
/// \param Code Device code on the device
/// \return an \c llvm::StringRef pointing to the code accessible on
/// host memory, or an \c llvm::Error indicating any HSA errors encountered
llvm::Expected<llvm::StringRef> convertToHostEquivalent(llvm::StringRef Code);

/// Decreases the reference count of the HSA runtime instance; Shuts down the
/// HSA runtime if the counter reaches zero \n
/// Must only be used by unit tests
/// \return an \c llvm::Error indicating any HSA issues encountered
/// \sa hsa_shutdown
llvm::Error shutdown();

} // namespace luthier::hsa

#endif