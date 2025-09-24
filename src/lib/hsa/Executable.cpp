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
#include "luthier/hsa/Executable.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/hsa/HsaError.h"

namespace luthier::hsa {

llvm::Expected<hsa_executable_t> executableCreate(
    const ApiTableContainer<::CoreApiTable> &CoreApi,
    const hsa_profile_t Profile,
    const hsa_default_float_rounding_mode_t DefaultFloatRoundingMode) {
  hsa_executable_t Exec;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_create_alt_fn>(
          Profile, DefaultFloatRoundingMode, "", &Exec),
      "Failed to create a new executable handle"));
  return Exec;
}

llvm::Expected<hsa_loaded_code_object_t> executableLoadAgentCodeObject(
    const ApiTableContainer<::CoreApiTable> &CoreApi,
    const hsa_executable_t Exec, const hsa_code_object_reader_t Reader,
    const hsa_agent_t Agent, const llvm::StringRef LoaderOptions) {
  hsa_loaded_code_object_t LCO;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<
          &::CoreApiTable::hsa_executable_load_agent_code_object_fn>(
          Exec, Agent, Reader, LoaderOptions.data(), &LCO),
      llvm::formatv("Failed to load agent code object from code object "
                    "reader {0:x} to executable {1:x} for agent {2:x}",
                    Reader.handle, Exec.handle, Agent.handle)));
  return LCO;
}

llvm::Error executableDefineExternalAgentGlobalVariable(
    const ApiTableContainer<::CoreApiTable> &CoreApi,
    const hsa_executable_t Exec, const hsa_agent_t Agent,
    const llvm::StringRef SymbolName, const void *Address) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<
          &::CoreApiTable::hsa_executable_agent_global_variable_define_fn>(
          Exec, Agent, SymbolName.data(), const_cast<void *>(Address)),
      llvm::formatv("Failed to define an external global variable named {0} "
                    "with address "
                    "{1:x} on agent {2:x} in executable {3:x}",
                    SymbolName, Address, Agent.handle, Exec.handle)));
  return llvm::Error::success();
}

llvm::Error executableFreeze(const ApiTableContainer<::CoreApiTable> &CoreApi,
                             const hsa_executable_t Exec) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_freeze_fn>(Exec, ""),
      llvm::formatv("Failed to freeze the executable {0:x}", Exec.handle));
}

llvm::Error executableDestroy(const ApiTableContainer<::CoreApiTable> &CoreApi,
                              const hsa_executable_t Exec) {
  return LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_destroy_fn>(Exec),
      llvm::formatv("Failed to destroy executable {0:x}", Exec.handle));
}

llvm::Expected<hsa_profile_t>
executableGetProfile(const ApiTableContainer<::CoreApiTable> &CoreApi,
                     const hsa_executable_t Exec) {
  hsa_profile_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_get_info_fn>(
          Exec, HSA_EXECUTABLE_INFO_PROFILE, &Out),
      llvm::formatv("Failed to get the profile of executable {0:x}",
                    Exec.handle)));
  return Out;
}

llvm::Expected<hsa_executable_state_t>
executableGetState(const ApiTableContainer<::CoreApiTable> &CoreApi,
                   const hsa_executable_t Exec) {
  hsa_executable_state_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_get_info_fn>(
          Exec, HSA_EXECUTABLE_INFO_STATE, &Out),
      llvm::formatv("Failed to get the state of executable {0:x}",
                    Exec.handle)));
  return Out;
}

llvm::Expected<std::optional<hsa_executable_symbol_t>>
executableGetSymbolByName(const ApiTableContainer<::CoreApiTable> &CoreApi,
                          const hsa_executable_t Exec,
                          const llvm::StringRef Name, const hsa_agent_t Agent) {
  hsa_executable_symbol_t Symbol;

  const hsa_status_t Status =
      CoreApi
          .callFunction<&::CoreApiTable::hsa_executable_get_symbol_by_name_fn>(
              Exec, Name.str().c_str(), &Agent, &Symbol);
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

llvm::Error executableIterateAgentSymbols(
    const ApiTableContainer<::CoreApiTable> &CoreApi,
    const hsa_executable_t Exec, const hsa_agent_t Agent,
    const std::function<llvm::Error(hsa_executable_symbol_t)> &Callback) {

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
    Data->Err = Data->CB(S);
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t Out =
          CoreApi.callFunction<
              &::CoreApiTable::hsa_executable_iterate_agent_symbols_fn>(
              Exec, Agent, CTypeCB, &CBData);
      Out == HSA_STATUS_SUCCESS || Out == HSA_STATUS_INFO_BREAK)
    return std::move(CBData.Err);
  return llvm::make_error<HsaError>(
      llvm::formatv("Failed to iterate over the executable symbols of agent "
                    "{0:x} inside executable {1:x}",
                    Agent.handle, Exec.handle));
}

llvm::Expected<std::optional<hsa_executable_symbol_t>>
executableFindFirstAgentSymbol(
    const ApiTableContainer<::CoreApiTable> &CoreApi, hsa_executable_t Exec,
    hsa_agent_t Agent,
    const std::function<llvm::Expected<bool>(hsa_executable_symbol_t)>
        &Predicate) {
  struct CallbackDataType {
    decltype(Predicate) CB;
    std::optional<hsa_executable_symbol_t> Symbol;
    llvm::Error Err;
  } CBData{Predicate, std::nullopt, llvm::Error::success()};

  auto Iterator = [](hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t S,
                     void *D) -> hsa_status_t {
    auto *Data = static_cast<CallbackDataType *>(D);
    if (!Data) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    llvm::Expected<bool> Res = Data->CB(S);
    Data->Err = Res.takeError();
    if (Data->Err)
      return HSA_STATUS_INFO_BREAK;
    if (*Res) {
      Data->Symbol = S;
      return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
  };

  if (const hsa_status_t Out =
          CoreApi.callFunction<
              &::CoreApiTable::hsa_executable_iterate_agent_symbols_fn>(
              Exec, Agent, Iterator, &CBData);
      Out == HSA_STATUS_SUCCESS || Out == HSA_STATUS_INFO_BREAK) {
    LUTHIER_RETURN_ON_ERROR(CBData.Err);
    return CBData.Symbol;
  }

  return llvm::make_error<HsaError>(
      llvm::formatv("Failed to iterate over the executable symbols of agent "
                    "{0:x} inside executable {1:x}",
                    Agent.handle, Exec.handle));
}

} // namespace luthier::hsa