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
#include "luthier/HSA/ApiTable.h"
#include <llvm/ADT/SmallVector.h>

namespace luthier::hsa {

/// Initializes the HSA runtime if already not initialized, and increments
/// the internal HSA runtime reference counter
/// \param CoreApi the \c ::CoreApiTable used to perform the required HSA
/// calls
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa ::hsa_init
llvm::Error init(const ApiTableContainer<::CoreApiTable> &CoreApi);

/// Queries the GPU <tt>hsa_agent_t</tt>s attached to the host
/// \param [in] CoreApi the \c ::CoreApiTable used to perform the required HSA
/// calls
/// \param [out] Agents the list of <tt>hsa_agent_t</tt>s of type GPU attached
/// to the host
/// \return \c llvm::Error indicating the success or failure of the operation
/// \sa hsa_iterate_agents
llvm::Error getGpuAgents(const ApiTableContainer<::CoreApiTable> &CoreApi,
                         llvm::SmallVectorImpl<hsa_agent_t> &Agents);

/// Queries all the \c hsa_executable_t handles currently loaded into HSA
/// \param LoaderApi the HSA Loader API container used to perform required
/// HSA calls
/// \return Expects all <tt>hsa_executable_t</tt>s currently loaded into the
/// HSA runtime on success
/// \sa hsa_ven_amd_loader_iterate_executables
llvm::Expected<std::vector<hsa_executable_t>> getAllExecutables(
    const ExtensionTableContainer<HSA_EXTENSION_AMD_LOADER> &LoaderApi);

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
    const ExtensionTableContainer<HSA_EXTENSION_AMD_LOADER> &LoaderApi,
    T *DeviceAddress) {
  const T *HostAddress;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApi.QueryHostFn(DeviceAddress,
                            reinterpret_cast<const void **>(&HostAddress)),
      llvm::formatv(
          "Failed to query the host address associated with address {0:x}.",
          DeviceAddress)));
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
    const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
        &LoaderApi,
    llvm::ArrayRef<uint8_t> Code);

/// Convenience version of <tt>hsa::queryHostAddress(T *)</tt> for locating
/// device code on the host-accessible memory
/// \param QueryHostFn the underlying \c hsa_ven_amd_loader_query_host_address
/// function used to carry out the operation
/// \param Code \c llvm::StringRef encapsulating the GPU code region
/// residing on the device
/// \return Expects a \c llvm::StringRef pointing to the code accessible on
/// host memory
llvm::Expected<llvm::StringRef> convertToHostEquivalent(
    const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
        &LoaderApi,
    llvm::StringRef Code);

/// \return Expects the executable, loaded code object and the executable
/// symbol of the device \p Address on success; \c llvm::Error is returned
/// if the address is not part of an executable or if an HSA issue is
/// encountered
[[nodiscard]] llvm::Expected<std::tuple<
    hsa_executable_t, hsa_loaded_code_object_t, hsa_executable_symbol_t>>
getExecutableDefinition(
    const ApiTableContainer<::CoreApiTable> &CoreApi,
    const ExtensionTableContainer<HSA_EXTENSION_AMD_LOADER> &LoaderApi,
    uint64_t Address);

/// Decreases the reference count of the HSA runtime instance; Shuts down the
/// HSA runtime if the internal counter reaches zero
/// \param CoreApi the HSA Core API table container for dispatching HSA
/// functions
/// \return \c llvm::Error indicating the success or failure of the operation
llvm::Error shutdown(const ApiTableContainer<::CoreApiTable> &CoreApi);

} // namespace luthier::hsa

#endif