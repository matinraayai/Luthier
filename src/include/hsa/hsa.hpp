//===-- hsa.hpp - Top Level HSA API Wrapper -------------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file contains the definition of HSA API wrappers concerned with
/// the global status of the HSA runtime (e.g. agents attached to the device).
//===----------------------------------------------------------------------===//

#ifndef HSA_HSA_HPP
#define HSA_HSA_HPP
#include "hsa/Executable.hpp"
#include "hsa/GpuAgent.hpp"
#include <llvm/ADT/SmallVector.h>

namespace luthier::hsa {

/// Initializes the HSA runtime if already not initialized, and increments
/// the internal HSA runtime reference counter.
/// \warning Must only be used by unit tests
/// \return an \c llvm::Error indicating any HSA issues encountered
/// \sa hsa_init
llvm::Error init();

/// Queries the <tt>GpuAgent</tt>s attached to the device
/// \param [out] Agent <tt>GpuAgent</tt>s attached to the device
/// \return an \c llvm::Error indicating any HSA issues encountered
/// \sa hsa_iterate_agents
llvm::Error getGpuAgents(llvm::SmallVectorImpl<GpuAgent> &Agent);

/// \return All <tt>hsa::Executable</tt>s currently loaded into the HSA runtime
/// \sa hsa_ven_amd_loader_iterate_executables
llvm::Expected<std::vector<Executable>> getAllExecutables();

/// Queries the host-accessible address of the given \p DeviceAddress \n
/// \tparam T Pointer type of the address
/// \param DeviceAddress Address residing on the device
/// \return The host accessible address of the <tt>DeviceAddress</tt>, or an
/// \c llvm::Error indicating any HSA errors encountered
/// \sa hsa_ven_amd_loader_query_host_address
template <typename T> llvm::Expected<T *> queryHostAddress(T *DeviceAddress) {
  const auto &LoaderApi =
      HsaRuntimeInterceptor::instance().getHsaVenAmdLoaderTable();
  const T *HostAddress;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(LoaderApi.hsa_ven_amd_loader_query_host_address(
          DeviceAddress, (reinterpret_cast<const void **>(&HostAddress)))));
  return HostAddress;
}

/// Convenience version of <tt>hsa::queryHostAddress(T *)</tt> for locating
/// device code on the host-accessible memory
/// \param Code an \c llvm::ArrayRef<uint8_t> encapsulating the GPU code region
/// residing on the device
/// \return an \c llvm::ArrayRef<uint8_t> pointing to the code accessible on
/// host memory, or an \c llvm::Error indicating any HSA errors encountered
llvm::Expected<llvm::ArrayRef<uint8_t>>
convertToHostEquivalent(llvm::ArrayRef<uint8_t> Code);

/// Convenience version of <tt>hsa::queryHostAddress(T *)</tt> for locating
/// device code on the host-accessible memory
/// \param Code an \c llvm::StringRef encapsulating the GPU code region
/// residing on the device
/// \return an \c llvm::StringRef pointing to the code accessible on
/// host memory, or an \c llvm::Error indicating any HSA errors encountered
llvm::Expected<llvm::StringRef> convertToHostEquivalent(llvm::StringRef Code);

/// Decreases the reference count of the HSA runtime instance; Shuts down the
/// HSA runtime if the counter reaches zero
/// \warning Must only be used by unit tests
/// \return an \c llvm::Error indicating any HSA issues encountered
/// \sa hsa_shutdown
llvm::Error shutdown();

} // namespace luthier::hsa

#endif
