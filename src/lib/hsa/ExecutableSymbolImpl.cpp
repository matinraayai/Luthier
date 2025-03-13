//===-- ExecutableSymbolImpl.cpp ------------------------------------------===//
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
/// This file implements the \c hsa::ExecutableSymbolImpl class.
//===----------------------------------------------------------------------===//
#include "hsa/ExecutableSymbolImpl.hpp"
#include "hsa/GpuAgentImpl.hpp"
#include "luthier/hsa/HsaError.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-hsa-executable-symbol"

namespace luthier::hsa {

char ExecutableSymbolImpl::ID = 0;

llvm::Expected<hsa_symbol_kind_t> ExecutableSymbolImpl::getType() const {
  hsa_symbol_kind_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &Out)));
  return Out;
}

llvm::Expected<llvm::StringRef> ExecutableSymbolImpl::getName() const {
  uint32_t NameLength;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &NameLength)));
  std::string Out(NameLength, '\0');
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME, &Out.front())));
  return Out;
}

llvm::Expected<size_t> ExecutableSymbolImpl::getSize() const {
  uint32_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &Out)));
  return Out;
}

llvm::Expected<luthier::address_t> ExecutableSymbolImpl::getAddress() const {
  auto SymbolType = getType();
  LUTHIER_RETURN_ON_ERROR(SymbolType.takeError());
  luthier::address_t Out;
  auto InfoQueried = *SymbolType == HSA_SYMBOL_KIND_VARIABLE
                         ? HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS
                         : HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(asHsaType(),
                                                           InfoQueried, &Out)));

  return Out;
}

llvm::Expected<std::unique_ptr<GpuAgent>>
ExecutableSymbolImpl::getAgent() const {
  hsa_agent_t Agent;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_symbol_get_info_fn(
          this->asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &Agent)));
  return std::make_unique<hsa::GpuAgentImpl>(Agent);
}

} // namespace luthier::hsa
