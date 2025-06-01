//===-- Executable.cpp ----------------------------------------------------===//
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
/// Implements a set of commonly used functionality for the \c
/// hsa_executable_t in HSA.
//===----------------------------------------------------------------------===//
#include <luthier/hsa/Executable.h>
#include <luthier/hsa/HsaError.h>

namespace luthier::hsa {

llvm::Expected<hsa_executable_t> executableCreate(
    const decltype(hsa_executable_create_alt) &HsaCreateExecutableCreateAltFn,
    const hsa_profile_t Profile,
    const hsa_default_float_rounding_mode_t DefaultFloatRoundingMode) {
  hsa_executable_t Exec;
  if (const hsa_status_t Status = HsaCreateExecutableCreateAltFn(
          Profile, DefaultFloatRoundingMode, "", &Exec);
      Status != HSA_STATUS_SUCCESS) {
    return llvm::make_error<HsaError>(
        "Failed to create a new executable handle", Status);
  }
  return Exec;
}

llvm::Expected<hsa_loaded_code_object_t> executableLoadAgentCodeObject(
    const hsa_executable_t Exec,
    const decltype(hsa_executable_load_agent_code_object)
        &HsaExecutableLoadAgentCodeObjectFn,
    const hsa_code_object_reader_t Reader, const hsa_agent_t Agent,
    const llvm::StringRef LoaderOptions) {
  hsa_loaded_code_object_t LCO;
  if (const hsa_status_t Status = HsaExecutableLoadAgentCodeObjectFn(
          Exec, Agent, Reader, LoaderOptions.data(), &LCO);
      Status != HSA_STATUS_SUCCESS) {
    return llvm::make_error<HsaError>(
        llvm::formatv("Failed to load agent code object from code object "
                      "reader {0:x} to executable {1:x} for agent {2:x}",
                      Reader.handle, Exec.handle, Agent.handle),
        Status);
  }
  return LCO;
}

llvm::Error executableDefineExternalAgentGlobalVariable(
    const hsa_executable_t Exec,
    const decltype(hsa_executable_agent_global_variable_define)
        &HsaExecutableAgentGlobalVariableDefineFn,
    const hsa_agent_t Agent, const llvm::StringRef SymbolName,
    const void *Address) {
  if (const hsa_status_t Status = HsaExecutableAgentGlobalVariableDefineFn(
          Exec, Agent, SymbolName.data(), const_cast<void *>(Address));
      Status != HSA_STATUS_SUCCESS) {
    return llvm::make_error<HsaError>(
        llvm::formatv("Failed to define an external global variable named {0} "
                      "with address "
                      "{1:x} on agent {2:x} in executable {3:x}",
                      SymbolName, Address, Agent.handle, Exec.handle),
        Status);
  }
  return llvm::Error::success();
}

llvm::Error
executableFreeze(const hsa_executable_t Exec,
                 const decltype(hsa_executable_freeze) &HsaExecutableFreezeFn) {
  if (const hsa_status_t Status = HsaExecutableFreezeFn(Exec, "");
      Status != HSA_STATUS_SUCCESS) {
    return llvm::make_error<HsaError>(
        llvm::formatv("Failed to freeze the executable {0:x}", Exec.handle),
        Status);
  }
  return llvm::Error::success();
}

llvm::Error executableDestroy(
    const hsa_executable_t Exec,
    const decltype(hsa_executable_destroy) &HsaExecutableDestroyFn) {
  if (const hsa_status_t Status = HsaExecutableDestroyFn(Exec);
      Status != HSA_STATUS_SUCCESS) {
    return llvm::make_error<HsaError>(
        llvm::formatv("Failed to destroy executable {0:x}", Exec.handle),
        Status);
  }
  return llvm::Error::success();
}

llvm::Expected<hsa_profile_t> executableGetProfile(
    const hsa_executable_t Exec,
    const decltype(hsa_executable_get_info) &HsaExecutableGetInfoFn) {
  hsa_profile_t Out;
  if (const hsa_status_t Status =
          HsaExecutableGetInfoFn(Exec, HSA_EXECUTABLE_INFO_PROFILE, &Out);
      Status != HSA_STATUS_SUCCESS) {
    return llvm::make_error<HsaError>(
        "Failed to get the profile of executable {0:x}", Exec.handle, Status);
  }
  return Out;
}

llvm::Expected<hsa_executable_state_t> executableGetState(
    const hsa_executable_t Exec,
    const decltype(hsa_executable_get_info) &HsaExecutableGetInfoFn) {
  hsa_executable_state_t Out;
  if (const hsa_status_t Status =
          HsaExecutableGetInfoFn(Exec, HSA_EXECUTABLE_INFO_STATE, &Out);
      Status != HSA_STATUS_SUCCESS) {
    return llvm::make_error<HsaError>(
        "Failed to get the state of executable {0:x}", Exec.handle);
  }
  return Out;
}

llvm::Error executableGetLoadedCodeObjects(
    const hsa_executable_t Exec,
    const decltype(hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
        &HsaVenAmdLoaderExecutableIterateLoadedCodeObjectsFn,
    llvm::SmallVectorImpl<hsa_loaded_code_object_t> &LCOs) {
  auto Iterator = [](hsa_executable_t, hsa_loaded_code_object_t LCO,
                     void *Data) -> hsa_status_t {
    auto Out =
        static_cast<llvm::SmallVectorImpl<hsa_loaded_code_object_t> *>(Data);
    if (!Out)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    Out->emplace_back(LCO);
    return HSA_STATUS_SUCCESS;
  };
  if (const hsa_status_t Status =
          HsaVenAmdLoaderExecutableIterateLoadedCodeObjectsFn(Exec, Iterator,
                                                              &LCOs);
      Status != HSA_STATUS_SUCCESS) {
    return llvm::make_error<HsaError>(llvm::formatv(
        "Failed to iterate over the code objects of executable {0:x}", Exec));
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<hsa_executable_symbol_t>>
executableGetSymbolByName(const hsa_executable_t Exec,
                          const decltype(hsa_executable_get_symbol_by_name)
                              &HsaExecutableGetSymbolByNameFn,
                          const llvm::StringRef Name, const hsa_agent_t Agent) {
  hsa_executable_symbol_t Symbol;

  const hsa_status_t Status =
      HsaExecutableGetSymbolByNameFn(Exec, Name.str().c_str(), &Agent, &Symbol);
  if (Status == HSA_STATUS_SUCCESS)
    return Symbol;
  if (Status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME)
    return std::nullopt;
  return llvm::make_error<HsaError>(
      llvm::formatv("Failed to query the symbol name {0} for agent {1:x} "
                    "inside executable {2:x}",
                    Name, Agent.handle, Exec.handle),
      Status);
}

llvm::Error executableIterateSymbols(
    const hsa_executable_t Exec,
    const decltype(hsa_executable_iterate_agent_symbols) &SymbolIterFn,
    const hsa_agent_t Agent,
    const std::function<bool(hsa_executable_symbol_t, llvm::Error &)>
        &Callback) {

  struct CBDataType {
    decltype(Callback) CB;
    llvm::Error Err;
  } CBData{Callback, llvm::Error::success()};

  auto CTypeCB = [](hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t S,
                    void *D) -> hsa_status_t {
    auto *Data = static_cast<CBDataType *>(D);
    if (!Data) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    if (const bool ContinueIter = Data->CB(S, Data->Err);
        ContinueIter != false || Data->Err)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t Out = SymbolIterFn(Exec, Agent, CTypeCB, &CBData);
      Out == HSA_STATUS_SUCCESS || Out == HSA_STATUS_INFO_BREAK)
    return std::move(CBData.Err);
  return llvm::make_error<HsaError>(
      llvm::formatv("Failed to iterate over the executable symbols of agent "
                    "{0:x} inside executable {1:x}",
                    Agent.handle, Exec.handle));
}

} // namespace luthier::hsa
