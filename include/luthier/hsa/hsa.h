//===-- hsa.h ----------------------------------------------------*- C++-*-===//
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
/// Defines a set of commonly used functionality regarding the global state
/// of the HSA runtime.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HSA_H
#define LUTHIER_HSA_HSA_H
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// Initializes the HSA runtime if already not initialized, and increments
/// the internal HSA runtime reference counter
/// \param HsaInitFn the underlying \c hsa_init function used to carry out this
/// operation
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa hsa_init
llvm::Error init(const decltype(hsa_init) &HsaInitFn);

/// Queries the GPU <tt>hsa_agent_t</tt>s attached to the host
/// \param [in] HsaIterateAgentsFn the underlying \c hsa_iterate_agents function
/// used to carry out this operation
/// \param [out] Agent the list of <tt>hsa_agent_t</tt>s attached to the host
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa hsa_iterate_agents
llvm::Error getGpuAgents(const decltype(hsa_iterate_agents) &HsaIterateAgentsFn,
                         llvm::SmallVectorImpl<hsa_agent_t> &Agent);

/// Queries all the \c hsa_executable_t handles currently loaded into HSA
/// \param IterateExecFn the underlying
/// \c hsa_ven_amd_loader_iterate_executables function used to carry out the
/// operation
/// \return Expects all <tt>hsa_executable_t</tt>s currently loaded into the
/// HSA runtime on success
/// \sa hsa_ven_amd_loader_iterate_executables
llvm::Expected<std::vector<hsa_executable_t>> getAllExecutables(
    const decltype(hsa_ven_amd_loader_iterate_executables) &IterateExecFn);

/// Queries the host-accessible address of the given \p DeviceAddress \n
/// \tparam T Pointer type of the address
/// \param QueryHostFn the underlying \c hsa_ven_amd_loader_query_host_address
/// function used to carry out the operation
/// \param DeviceAddress Address residing on the device
/// \return Expects the host accessible address of the
/// <tt>DeviceAddress</tt> on success
/// \sa hsa_ven_amd_loader_query_host_address
template <typename T>
llvm::Expected<T *> queryHostAddress(
    const decltype(hsa_ven_amd_loader_query_host_address) &QueryHostFn,
    T *DeviceAddress) {
  const T *HostAddress;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(QueryHostFn(
      DeviceAddress, (reinterpret_cast<const void **>(&HostAddress)))));
  return HostAddress;
}

/// Convenience version of <tt>hsa::queryHostAddress(const
/// decltype(hsa_ven_amd_loader_query_host_address) &, T *)</tt> for locating
/// device code on the host-accessible memory
/// \param QueryHostFn the underlying \c hsa_ven_amd_loader_query_host_address
/// function used to carry out the operation
/// \param Code \c llvm::ArrayRef<uint8_t> encapsulating the GPU code region
/// residing on the device
/// \return Expects an \c llvm::ArrayRef<uint8_t> pointing to the code
/// accessible on host memory on success
llvm::Expected<llvm::ArrayRef<uint8_t>> convertToHostEquivalent(
    const decltype(hsa_ven_amd_loader_query_host_address) &QueryHostFn,
    llvm::ArrayRef<uint8_t> Code);

/// Convenience version of <tt>hsa::queryHostAddress(T *)</tt> for locating
/// device code on the host-accessible memory
/// \param Code an \c llvm::StringRef encapsulating the GPU code region
/// residing on the device
/// \return an \c llvm::StringRef pointing to the code accessible on
/// host memory, or an \c llvm::Error indicating any HSA errors encountered
llvm::Expected<llvm::StringRef> convertToHostEquivalent(llvm::StringRef Code);

/// Decreases the reference count of the HSA runtime instance; Shuts down the
/// HSA runtime if the counter reaches zero
/// \param HsaShutdownFn the underlying \c hsa_shut_down function used to
/// carry out the operation
/// \return an \c llvm::Error indicating the success or failure of the operation
llvm::Error shutdown(const decltype(hsa_shut_down) &HsaShutdownFn);

} // namespace luthier::hsa

#endif