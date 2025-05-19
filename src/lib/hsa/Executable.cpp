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
/// This file implements the \c Executable class under the \c luthier::hsa
/// namespace.
//===----------------------------------------------------------------------===//
#include "hsa/Executable.hpp"
#include "hsa/CodeObjectReader.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include "luthier/hsa/HsaError.h"

namespace luthier::hsa {

llvm::Expected<Executable> Executable::create(
    decltype(hsa_executable_create_alt) *HsaCreateExecutableCreateAltFn,
    hsa_profile_t Profile,
    hsa_default_float_rounding_mode_t DefaultFloatRoundingMode) {
  hsa_executable_t Exec;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaCreateExecutableCreateAltFn(
          Profile, DefaultFloatRoundingMode, "", &Exec)));
  return Executable{Exec};
}

llvm::Expected<LoadedCodeObject> Executable::loadAgentCodeObject(
    const decltype(hsa_executable_load_agent_code_object)
        *HsaExecutableLoadAgentCodeObjectFn,
    const CodeObjectReader &Reader, const GpuAgent &Agent,
    llvm::StringRef LoaderOptions) {
  hsa_loaded_code_object_t LCO;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaExecutableLoadAgentCodeObjectFn(asHsaType(), Agent.asHsaType(),
                                         Reader.asHsaType(),
                                         LoaderOptions.data(), &LCO)));
  return LoadedCodeObject{LCO};
}

llvm::Error Executable::defineExternalAgentGlobalVariable(
    const decltype(hsa_executable_agent_global_variable_define)
        *HsaExecutableAgentGlobalVariableDefineFn,
    const hsa::GpuAgent &Agent, llvm::StringRef SymbolName,
    const void *Address) {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_agent_global_variable_define_fn(
          asHsaType(), Agent.asHsaType(), SymbolName.data(),
          const_cast<void *>(Address))));
  return llvm::Error::success();
}

llvm::Error Executable::freeze(
    const decltype(hsa_executable_freeze) *HsaExecutableFreezeFn) {
  return LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableFreezeFn(asHsaType(), ""));
}

llvm::Error Executable::destroy(
    const decltype(hsa_executable_destroy) *HsaExecutableDestroyFn) {
  return LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableDestroyFn(asHsaType()));
}

Executable::Executable(hsa_executable_t Exec)
    : HandleType<hsa_executable_t>(Exec) {}

llvm::Expected<hsa_profile_t> Executable::getProfile(
    const decltype(hsa_executable_get_info) *HsaExecutableGetInfoFn) const {
  hsa_profile_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaExecutableGetInfoFn(asHsaType(), HSA_EXECUTABLE_INFO_PROFILE, &Out)));
  return Out;
}

llvm::Expected<hsa_executable_state_t> Executable::getState(
    const decltype(hsa_executable_get_info) *HsaExecutableGetInfoFn) const {
  hsa_executable_state_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaExecutableGetInfoFn(asHsaType(), HSA_EXECUTABLE_INFO_STATE, &Out)));
  return Out;
}

llvm::Error Executable::getLoadedCodeObjects(
    const decltype(hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
        *HsaVenAmdLoaderExecutableIterateLoadedCodeObjectsFn,
    llvm::SmallVectorImpl<hsa::LoadedCodeObject> &LCOs) const {
  auto Iterator = [](hsa_executable_t Exec, hsa_loaded_code_object_t LCO,
                     void *Data) -> hsa_status_t {
    auto Out =
        static_cast<llvm::SmallVectorImpl<hsa::LoadedCodeObject> *>(Data);
    Out->emplace_back(LCO);
    return HSA_STATUS_SUCCESS;
  };
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaVenAmdLoaderExecutableIterateLoadedCodeObjectsFn(this->asHsaType(),
                                                          Iterator, &LCOs)));
  return llvm::Error::success();
}

llvm::Expected<std::optional<ExecutableSymbol>>
Executable::getExecutableSymbolByName(
    const decltype(hsa_executable_get_symbol_by_name)
        *HsaExecutableGetSymbolByNameFn,
    llvm::StringRef Name, const GpuAgent &Agent) const {
  hsa_executable_symbol_t Symbol;
  hsa_agent_t HsaAgent = Agent.asHsaType();

  auto Status = HsaExecutableGetSymbolByNameFn(
      this->asHsaType(), Name.str().c_str(), &HsaAgent, &Symbol);
  if (Status == HSA_STATUS_SUCCESS)
    return ExecutableSymbol(Symbol);
  if (Status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME)
    return std::nullopt;
  return LUTHIER_HSA_SUCCESS_CHECK(Status);
}

} // namespace luthier::hsa
