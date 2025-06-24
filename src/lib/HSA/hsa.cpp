//===-- hsa.cpp -----------------------------------------------------------===//
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
/// Implements a set of commonly used functionality regarding the global state
/// of the HSA runtime.
//===----------------------------------------------------------------------===//
#include <llvm/ADT/StringExtras.h>
#include <luthier/Common/ErrorCheck.h>
#include <luthier/Common/GenericLuthierError.h>
#include <luthier/HSA/Executable.h>
#include <luthier/HSA/ExecutableSymbol.h>
#include <luthier/HSA/HsaError.h>
#include <luthier/HSA/LoadedCodeObject.h>
#include <luthier/HSA/hsa.h>

namespace luthier::hsa {

llvm::Error init(const decltype(hsa_init) &HsaInitFn) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(HsaInitFn(), "Failed to initialize HSA");
}

llvm::Error getGpuAgents(const decltype(hsa_iterate_agents) &HsaIterateAgentsFn,
                         llvm::SmallVectorImpl<hsa_agent_t> &Agents) {
  auto ReturnGpuAgentsCallback = [](hsa_agent_t Agent, void *Data) {
    auto AgentList = static_cast<llvm::SmallVector<hsa_agent_t> *>(Data);
    hsa_device_type_t DevType = HSA_DEVICE_TYPE_CPU;

    const hsa_status_t Status =
        hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DevType);

    if (Status != HSA_STATUS_SUCCESS)
      return Status;
    if (DevType == HSA_DEVICE_TYPE_GPU) {
      AgentList->emplace_back(Agent);
    }
    return Status;
  };
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      HsaIterateAgentsFn(ReturnGpuAgentsCallback, &Agents),
      "Failed to iterate over all HSA agents attached to the system");
}

llvm::Expected<std::vector<hsa_executable_t>> getAllExecutables(
    const decltype(hsa_ven_amd_loader_iterate_executables) &IterateExecFn) {
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
  return LUTHIER_HSA_CALL_ERROR_CHECK(IterateExecFn(Iterator, &Out),
                                      "Failed to iterate over HSA executables");
}

llvm::Expected<llvm::ArrayRef<uint8_t>> convertToHostEquivalent(
    const decltype(hsa_ven_amd_loader_query_host_address) &QueryHostFn,
    const llvm::ArrayRef<uint8_t> Code) {
  llvm::Expected<const unsigned char *> CodeStartHostAddressOrErr =
      queryHostAddress(QueryHostFn, Code.data());
  LUTHIER_RETURN_ON_ERROR(CodeStartHostAddressOrErr.takeError());
  return llvm::ArrayRef{*CodeStartHostAddressOrErr, Code.size()};
}

llvm::Expected<llvm::StringRef> convertToHostEquivalent(
    const decltype(hsa_ven_amd_loader_query_host_address) &QueryHostFn,
    const llvm::StringRef Code) {
  llvm::Expected<llvm::ArrayRef<uint8_t>> HostAccessibleCodeOrErr =
      convertToHostEquivalent(QueryHostFn, llvm::arrayRefFromStringRef(Code));
  LUTHIER_RETURN_ON_ERROR(HostAccessibleCodeOrErr.takeError());
  return llvm::toStringRef(*HostAccessibleCodeOrErr);
}

llvm::Expected<std::tuple<hsa_executable_t, hsa_loaded_code_object_t,
                          hsa_executable_symbol_t>>
getExecutableDefinition(
    uint64_t Address,
    const decltype(hsa_ven_amd_loader_query_executable)
        &HsaVenAmdLoaderQueryExecutableFn,
    const decltype(hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
        &HsaVenAmdLoaderExecutableIterateLoadedCodeObjectsFn,
    const decltype(hsa_ven_amd_loader_loaded_code_object_get_info)
        &HsaVenAmdLoaderLoadedCodeObjectGetInfoFn,
    const decltype(hsa_executable_iterate_agent_symbols) &SymbolIterFn,
    const decltype(hsa_executable_symbol_get_info)
        &HsaExecutableSymbolGetInfoFn) {
  hsa_executable_t Executable;
  // Check which executable this kernel object (address) belongs to
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      HsaVenAmdLoaderQueryExecutableFn(reinterpret_cast<const void *>(Address),
                                       &Executable),
      llvm::formatv(
          "Failed to get the executable associated with address {0:x}.",
          Address)));
  // Iterate over the LCOs and find the symbol that matches the kernel
  // descriptor address
  llvm::SmallVector<hsa_loaded_code_object_t, 1> LCOs;
  LUTHIER_RETURN_ON_ERROR(hsa::executableGetLoadedCodeObjects(
      Executable, HsaVenAmdLoaderExecutableIterateLoadedCodeObjectsFn, LCOs));

  for (const auto &LCO : LCOs) {
    // Get the agent of the current LCO
    llvm::Expected<hsa_agent_t> AgentOrErr = hsa::loadedCodeObjectGetAgent(
        LCO, HsaVenAmdLoaderLoadedCodeObjectGetInfoFn);
    LUTHIER_RETURN_ON_ERROR(AgentOrErr.takeError());
    // Find the first executable that matches the address
    llvm::Expected<std::optional<hsa_executable_symbol_t>>
        SymbolIfPresentOrErr = hsa::executableFindFirstAgentSymbol(
            Executable, SymbolIterFn, *AgentOrErr,
            [&](hsa_executable_symbol_t S) -> llvm::Expected<bool> {
              llvm::Expected<uint64_t> SymAddrOrErr =
                  hsa::executableSymbolGetAddress(S,
                                                  HsaExecutableSymbolGetInfoFn);
              LUTHIER_RETURN_ON_ERROR(SymAddrOrErr.takeError());
              llvm::Expected<size_t> SymSizeOrErr =
                  hsa::executableSymbolGetSymbolSize(
                      S, HsaExecutableSymbolGetInfoFn);
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

llvm::Error shutdown(const decltype(hsa_shut_down) &HsaShutdownFn) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(HsaShutdownFn(),
                                      "Failed to shutdown the HSA runtime");
}
} // namespace luthier::hsa