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
/// Defines and implements a set of commonly used functionality regarding
/// the global state of the HSA runtime.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_HSA_H
#define LUTHIER_HSA_HSA_H
#include "luthier/HSA/ApiTable.h"
#include "luthier/HSA/Executable.h"
#include "luthier/HSA/ExecutableSymbol.h"
#include "luthier/HSA/HsaError.h"
#include "luthier/HSA/LoadedCodeObject.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>

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

/// Obtains a list of all the \c hsa_executable_t handles currently loaded into
/// HSA
/// \param LoaderApi the HSA Loader API container used to perform required
/// HSA calls
/// \return Expects all <tt>hsa_executable_t</tt>s currently loaded into the
/// HSA runtime on success
/// \sa hsa_ven_amd_loader_iterate_executables
template <typename ExtensionApiTableType = hsa_ven_amd_loader_1_03_pfn_t>
llvm::Expected<std::vector<hsa_executable_t>>
getAllExecutables(const ExtensionApiTableType &LoaderApi) {
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
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApi.hsa_ven_amd_loader_iterate_executables(Iterator, &Out),
      "Failed to iterate over HSA executables");
}

/// Queries the host-accessible address of the given \p DeviceAddress \n
/// \tparam T Pointer type of the address
/// \param LoaderApi the HSA Loader API container used to perform required
/// HSA calls
/// \param DeviceAddress Address residing on the device
/// \return Expects the host accessible address of the
/// <tt>DeviceAddress</tt> on success
/// \sa hsa_ven_amd_loader_query_host_address
template <typename T,
          typename ExtensionApiTableType = hsa_ven_amd_loader_1_00_pfn_t>
llvm::Expected<T *> queryHostAddress(const ExtensionApiTableType &LoaderApi,
                                     T *DeviceAddress) {
  const T *HostAddress;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApi.hsa_ven_amd_loader_query_host_address(
          DeviceAddress, reinterpret_cast<const void **>(&HostAddress)),
      llvm::formatv(
          "Failed to query the host address associated with address {0:x}.",
          DeviceAddress)));
  return HostAddress;
}

/// Convenience version of <tt>hsa::queryHostAddress(const
/// decltype(hsa_ven_amd_loader_query_host_address) &, T *)</tt> for locating
/// device code on the host-accessible memory
/// \param LoaderApi the HSA Loader API container used to perform required
/// HSA calls
/// \param Code \c llvm::ArrayRef<uint8_t> encapsulating the GPU code region
/// residing on the device
/// \return Expects a \c llvm::ArrayRef<uint8_t> pointing to the code
/// accessible on host memory on success
template <typename ExtensionApiTableType = hsa_ven_amd_loader_1_00_pfn_t>
llvm::Expected<llvm::ArrayRef<uint8_t>>
convertToHostEquivalent(const ExtensionApiTableType &LoaderApi,
                        llvm::ArrayRef<uint8_t> Code) {
  llvm::Expected<const unsigned char *> CodeStartHostAddressOrErr =
      queryHostAddress(LoaderApi, Code.data());
  LUTHIER_RETURN_ON_ERROR(CodeStartHostAddressOrErr.takeError());
  return llvm::ArrayRef{*CodeStartHostAddressOrErr, Code.size()};
}

/// Convenience version of <tt>hsa::queryHostAddress(T *)</tt> for locating
/// device code on the host-accessible memory
/// \param LoaderApi the HSA Loader API container used to perform required
/// HSA calls
/// \param Code \c llvm::StringRef encapsulating the GPU code region
/// residing on the device
/// \return Expects a \c llvm::StringRef pointing to the code accessible on
/// host memory
template <typename ExtensionApiTableType = hsa_ven_amd_loader_1_00_pfn_t>
llvm::Expected<llvm::StringRef>
convertToHostEquivalent(const ExtensionApiTableType &LoaderApi,
                        llvm::StringRef Code) {
  llvm::Expected<llvm::ArrayRef<uint8_t>> HostAccessibleCodeOrErr =
      convertToHostEquivalent(LoaderApi, llvm::arrayRefFromStringRef(Code));
  LUTHIER_RETURN_ON_ERROR(HostAccessibleCodeOrErr.takeError());
  return llvm::toStringRef(*HostAccessibleCodeOrErr);
}

/// \return Expects the executable, loaded code object and the executable
/// symbol of the device \p Address on success; \c llvm::Error is returned
/// if the address is not part of an executable or if an HSA issue is
/// encountered
template <typename ExtensionApiTableType = hsa_ven_amd_loader_1_01_pfn_t>
[[nodiscard]] llvm::Expected<std::tuple<
    hsa_executable_t, hsa_loaded_code_object_t, hsa_executable_symbol_t>>
getExecutableDefinition(const ApiTableContainer<::CoreApiTable> &CoreApi,
                        const ExtensionApiTableType &LoaderApi,
                        uint64_t Address) {
  hsa_executable_t Executable;
  // Check which executable this kernel object (address) belongs to
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApi.hsa_ven_amd_loader_query_executable(
          reinterpret_cast<const void *>(Address), &Executable),
      llvm::formatv(
          "Failed to get the executable associated with address {0:x}.",
          Address)));
  // Iterate over the LCOs and find the symbol that matches the kernel
  // descriptor address
  llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(
      hsa::executableGetLoadedCodeObjects(LoaderApi, Executable, LCOs));

  for (const auto &LCO : LCOs) {
    // Get the agent of the current LCO
    llvm::Expected<hsa_agent_t> AgentOrErr =
        hsa::loadedCodeObjectGetAgent(LoaderApi, LCO);
    LUTHIER_RETURN_ON_ERROR(AgentOrErr.takeError());
    // Find the first executable that matches the address
    llvm::Expected<std::optional<hsa_executable_symbol_t>>
        SymbolIfPresentOrErr = hsa::executableFindFirstAgentSymbol(
            CoreApi, Executable, *AgentOrErr,
            [&](hsa_executable_symbol_t S) -> llvm::Expected<bool> {
              llvm::Expected<uint64_t> SymAddrOrErr =
                  hsa::executableSymbolGetAddress(CoreApi, S);
              LUTHIER_RETURN_ON_ERROR(SymAddrOrErr.takeError());
              llvm::Expected<size_t> SymSizeOrErr =
                  hsa::executableSymbolGetSymbolSize(CoreApi, S);
              LUTHIER_RETURN_ON_ERROR(SymSizeOrErr.takeError());
              return *SymAddrOrErr <= Address &&
                     Address < *SymAddrOrErr + *SymSizeOrErr;
            });
    LUTHIER_RETURN_ON_ERROR(SymbolIfPresentOrErr.takeError());
    if (SymbolIfPresentOrErr->has_value()) {
      return std::make_tuple(Executable, LCO, **SymbolIfPresentOrErr);
    }
  }
  return llvm::make_error<hsa::HsaError>(llvm::formatv(
      "Failed to find the executable definition of address {0:x}", Address));
}

/// Decreases the reference count of the HSA runtime instance; Shuts down the
/// HSA runtime if the internal counter reaches zero
/// \param CoreApi the HSA Core API table container for dispatching HSA
/// functions
/// \return \c llvm::Error indicating the success or failure of the operation
llvm::Error shutdown(const ApiTableContainer<::CoreApiTable> &CoreApi);

} // namespace luthier::hsa

#endif