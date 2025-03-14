//===-- ExecutableImpl.cpp ------------------------------------------------===//
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
/// This file implements the \c hsa::ExecutableImpl class, the concrete
/// implementation of the \c hsa::Executable interface.
//===----------------------------------------------------------------------===//
#include "hsa/ExecutableImpl.hpp"
#include "hsa/CodeObjectReaderImpl.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
#include "hsa/ExecutableSymbolImpl.hpp"
#include "hsa/GpuAgentImpl.hpp"
#include "hsa/LoadedCodeObjectImpl.hpp"
#include "luthier/hsa/HsaError.h"

namespace luthier::hsa {

char ExecutableImpl::ID = 0;

llvm::Error ExecutableImpl::create(
    hsa_profile_t Profile,
    hsa_default_float_rounding_mode_t DefaultFloatRoundingMode) {
  hsa_executable_t Exec;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      hsa_executable_create_alt(Profile, DefaultFloatRoundingMode, "", &Exec)));
  *this = ExecutableImpl{Exec};
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<hsa::LoadedCodeObject>>
ExecutableImpl::loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                                    const hsa::GpuAgent &Agent,
                                    llvm::StringRef LoaderOptions) {
  hsa_loaded_code_object_t LCO;

  auto *AgentImpl = llvm::dyn_cast<hsa::GpuAgentImpl>(&Agent);
  auto *ReaderImpl = llvm::dyn_cast<hsa::CodeObjectReaderImpl>(&Reader);

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      AgentImpl != nullptr,
      "Passed hsa::GpuAgent handle is not a hsa::GpuAgentImpl."));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      ReaderImpl != nullptr, "Passed hsa::CodeObjectReader handle is not a "
                             "hsa::CodeObjectReaderImpl."));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_load_agent_code_object_fn(
          asHsaType(), AgentImpl->asHsaType(), ReaderImpl->asHsaType(),
          LoaderOptions.data(), &LCO)));
  LUTHIER_RETURN_ON_ERROR(
      ExecutableBackedObjectsCache::instance()
          .cacheExecutableOnLoadedCodeObjectCreation(*this));

  return std::make_unique<LoadedCodeObjectImpl>(LCO);
}

llvm::Error
ExecutableImpl::defineExternalAgentGlobalVariable(const hsa::GpuAgent &Agent,
                                                  llvm::StringRef SymbolName,
                                                  const void *Address) {
  auto *AgentImpl = llvm::dyn_cast<hsa::GpuAgentImpl>(&Agent);

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      AgentImpl != nullptr,
      "Passed hsa::GpuAgent handle is not a hsa::GpuAgentImpl."));

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_agent_global_variable_define_fn(
          asHsaType(), AgentImpl->asHsaType(), SymbolName.data(),
          const_cast<void *>(Address))));
  return llvm::Error::success();
}

llvm::Error ExecutableImpl::freeze() {
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_freeze_fn(asHsaType(), "")));
  return ExecutableBackedObjectsCache::instance()
      .cacheExecutableOnExecutableFreeze(*this);
}

llvm::Expected<bool> ExecutableImpl::validate() {
  uint32_t Result;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_validate_alt_fn(asHsaType(), "",
                                                        &Result)));
  return Result == 0;
}

llvm::Error ExecutableImpl::destroy() {
  LUTHIER_RETURN_ON_ERROR(ExecutableBackedObjectsCache::instance()
                              .invalidateExecutableOnExecutableDestroy(*this));
  return LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_destroy_fn(asHsaType()));
}

llvm::Expected<hsa_profile_t> ExecutableImpl::getProfile() {
  hsa_profile_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_PROFILE, &Out)));
  return Out;
}

llvm::Expected<hsa_executable_state_t> ExecutableImpl::getState() const {
  hsa_executable_state_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_STATE, &Out)));
  return Out;
}

llvm::Expected<hsa_default_float_rounding_mode_t>
ExecutableImpl::getRoundingMode() {
  hsa_default_float_rounding_mode_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &Out)));
  return Out;
}

llvm::Error ExecutableImpl::getLoadedCodeObjects(
    llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObject>> &LCOs) const {
  auto Iterator = [](hsa_executable_t Exec, hsa_loaded_code_object_t LCO,
                     void *Data) -> hsa_status_t {
    auto Out = reinterpret_cast<
        llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObject>> *>(Data);
    Out->emplace_back(new hsa::LoadedCodeObjectImpl(LCO));
    return HSA_STATUS_SUCCESS;
  };
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable()
          .hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
              this->asHsaType(), Iterator, &LCOs)));
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<ExecutableSymbol>>
ExecutableImpl::getExecutableSymbolByName(llvm::StringRef Name,
                                          const hsa::GpuAgent &Agent) {
  auto *AgentImpl = llvm::dyn_cast<hsa::GpuAgentImpl>(&Agent);

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      AgentImpl != nullptr,
      "Passed hsa::GpuAgent handle is not a hsa::GpuAgentImpl."));

  hsa_executable_symbol_t Symbol;
  hsa_agent_t HsaAgent = AgentImpl->asHsaType();

  auto Status = getApiTable().core.hsa_executable_get_symbol_by_name_fn(
      this->asHsaType(), Name.str().c_str(), &HsaAgent, &Symbol);
  if (Status == HSA_STATUS_SUCCESS)
    return std::make_unique<ExecutableSymbolImpl>(Symbol);
  else if (Status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME)
    return nullptr;
  else
    return LUTHIER_HSA_SUCCESS_CHECK(Status);
}

} // namespace luthier::hsa
