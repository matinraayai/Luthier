//===-- ExecutableSymbol.cpp - HSA Executable Symbol Wrapper --------------===//
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
/// Implements the \c ExecutableSymbol class.
//===----------------------------------------------------------------------===//
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/GpuAgent.hpp"
#include "luthier/hsa/HsaError.h"

namespace luthier::hsa {

llvm::Expected<hsa_symbol_kind_t>
ExecutableSymbol::getType(const decltype(hsa_executable_symbol_get_info)
                              *HsaExecutableSymbolGetInfoFn) const {
  hsa_symbol_kind_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableSymbolGetInfoFn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &Out)));
  return Out;
}

llvm::Expected<llvm::StringRef>
ExecutableSymbol::getName(const decltype(hsa_executable_symbol_get_info)
                              *HsaExecutableSymbolGetInfoFn) const {
  uint32_t NameLength;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableSymbolGetInfoFn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &NameLength)));
  std::string Out(NameLength, '\0');
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableSymbolGetInfoFn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME, &Out.front())));
  return Out;
}

llvm::Expected<size_t>
ExecutableSymbol::getSize(const decltype(hsa_executable_symbol_get_info)
                              *HsaExecutableSymbolGetInfoFn) const {
  uint32_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableSymbolGetInfoFn(
          asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &Out)));
  return Out;
}

llvm::Expected<uint64_t>
ExecutableSymbol::getAddress(const decltype(hsa_executable_symbol_get_info)
                                 *HsaExecutableSymbolGetInfoFn) const {
  auto SymbolType = getType(HsaExecutableSymbolGetInfoFn);
  LUTHIER_RETURN_ON_ERROR(SymbolType.takeError());
  uint64_t Out;
  auto InfoQueried = *SymbolType == HSA_SYMBOL_KIND_VARIABLE
                         ? HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS
                         : HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      HsaExecutableSymbolGetInfoFn(asHsaType(), InfoQueried, &Out)));

  return Out;
}

llvm::Expected<GpuAgent>
ExecutableSymbol::getAgent(const decltype(hsa_executable_symbol_get_info)
                               *HsaExecutableSymbolGetInfoFn) const {
  hsa_agent_t Agent;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(HsaExecutableSymbolGetInfoFn(
          this->asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &Agent)));
  return hsa::GpuAgent(Agent);
}

} // namespace luthier::hsa
