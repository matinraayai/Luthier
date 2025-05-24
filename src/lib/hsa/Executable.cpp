//===-- Executable.cpp - HSA Executable Wrapper Implementation ------------===//
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
/// This file implements a set of commonly used functionality for the \c
/// hsa_executable_t in HSA.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/Executable.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/hsa/HsaError.h"

namespace luthier::hsa {

llvm::Expected<hsa_executable_t> createExecutable(
    decltype(hsa_executable_create_alt) &HsaCreateExecutableCreateAltFn,
    hsa_profile_t Profile,
    hsa_default_float_rounding_mode_t DefaultFloatRoundingMode) {
  hsa_executable_t Exec;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaCreateExecutableCreateAltFn(
          Profile, DefaultFloatRoundingMode, "", &Exec)));
  return Exec;
}

llvm::Expected<hsa_loaded_code_object_t> loadAgentCodeObjectIntoExec(
    hsa_executable_t Exec,
    const decltype(hsa_executable_load_agent_code_object)
        &HsaExecutableLoadAgentCodeObjectFn,
    hsa_code_object_reader_t Reader, hsa_agent_t Agent,
    llvm::StringRef LoaderOptions) {
  hsa_loaded_code_object_t LCO;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableLoadAgentCodeObjectFn(
          Exec, Agent, Reader, LoaderOptions.data(), &LCO)));
  return LCO;
}

llvm::Error defineExternalAgentGlobalVariableInExec(
    hsa_executable_t Exec,
    const decltype(hsa_executable_agent_global_variable_define)
        &HsaExecutableAgentGlobalVariableDefineFn,
    hsa_agent_t Agent, llvm::StringRef SymbolName, const void *Address) {
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableAgentGlobalVariableDefineFn(
          Exec, Agent, SymbolName.data(), const_cast<void *>(Address))));
  return llvm::Error::success();
}

llvm::Error
freezeExec(hsa_executable_t Exec,
           const decltype(hsa_executable_freeze) &HsaExecutableFreezeFn) {
  return LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableFreezeFn(Exec, ""));
}

llvm::Error
destroyExec(hsa_executable_t Exec,
            const decltype(hsa_executable_destroy) &HsaExecutableDestroyFn) {
  return LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableDestroyFn(Exec));
}

llvm::Expected<hsa_profile_t> getExecProfile(
    hsa_executable_t Exec,
    const decltype(hsa_executable_get_info) &HsaExecutableGetInfoFn) {
  hsa_profile_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaExecutableGetInfoFn(Exec, HSA_EXECUTABLE_INFO_PROFILE, &Out)));
  return Out;
}

llvm::Expected<hsa_executable_state_t>
getExecState(hsa_executable_t Exec,
             const decltype(hsa_executable_get_info) &HsaExecutableGetInfoFn) {
  hsa_executable_state_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaExecutableGetInfoFn(Exec, HSA_EXECUTABLE_INFO_STATE, &Out)));
  return Out;
}

llvm::Error getExecLoadedCodeObjects(
    hsa_executable_t Exec,
    const decltype(hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
        &HsaVenAmdLoaderExecutableIterateLoadedCodeObjectsFn,
    llvm::SmallVectorImpl<hsa_loaded_code_object_t> &LCOs) {
  auto Iterator = [](hsa_executable_t Exec, hsa_loaded_code_object_t LCO,
                     void *Data) -> hsa_status_t {
    auto Out =
        static_cast<llvm::SmallVectorImpl<hsa_loaded_code_object_t> *>(Data);
    Out->emplace_back(LCO);
    return HSA_STATUS_SUCCESS;
  };
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaVenAmdLoaderExecutableIterateLoadedCodeObjectsFn(Exec, Iterator,
                                                          &LCOs)));
  return llvm::Error::success();
}

llvm::Expected<std::optional<hsa_executable_symbol_t>>
lookupExecutableSymbolByName(hsa_executable_t Exec,
                             const decltype(hsa_executable_get_symbol_by_name)
                                 &HsaExecutableGetSymbolByNameFn,
                             llvm::StringRef Name, hsa_agent_t Agent) {
  hsa_executable_symbol_t Symbol;

  auto Status =
      HsaExecutableGetSymbolByNameFn(Exec, Name.str().c_str(), &Agent, &Symbol);
  if (Status == HSA_STATUS_SUCCESS)
    return Symbol;
  if (Status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME)
    return std::nullopt;
  return LUTHIER_HSA_SUCCESS_CHECK(Status);
}

llvm::Error iterateSymbolsOfExecutable(
    hsa_executable_t Exec,
    decltype(hsa_executable_iterate_agent_symbols) &SymbolIterFn,
    hsa_agent_t Agent,
    const std::function<bool(hsa_executable_symbol_t, llvm::Error &)>
        &Callback) {

  struct CBDataType {
    decltype(Callback) CB;
    llvm::Error Err;
  } CBData{Callback, llvm::Error::success()};

  auto CTypeCB = [](hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t S,
                    void *D) -> hsa_status_t {
    auto *Data = static_cast<CBDataType *>(D);
    bool ContinueIter = Data->CB(S, Data->Err);
    if (ContinueIter != false || Data->Err)
      return HSA_STATUS_INFO_BREAK;
    else {
      return HSA_STATUS_SUCCESS;
    }
  };

  hsa_status_t Out = SymbolIterFn(Exec, Agent, CTypeCB, &CBData);
  if (Out == HSA_STATUS_SUCCESS || Out == HSA_STATUS_INFO_BREAK)
    return std::move(CBData.Err);
  else
    return LUTHIER_HSA_SUCCESS_CHECK(Out);
}

} // namespace luthier::hsa
