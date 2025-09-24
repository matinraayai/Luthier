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
/// Implements a set of commonly used functionality for the
/// \c hsa_executable_symbol_t handle in HSA.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/ExecutableSymbol.h"
#include "luthier/common/ErrorCheck.h"
#include "luthier/hsa/HsaError.h"
#include <llvm/Support/Error.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier::hsa {

llvm::Expected<hsa_symbol_kind_t>
executableSymbolGetType(const ApiTableContainer<::CoreApiTable> &CoreApi,
                        const hsa_executable_symbol_t Symbol) {
  hsa_symbol_kind_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_symbol_get_info_fn>(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &Out),
      llvm::formatv("Failed to get the type of executable symbol {0:x}",
                    Symbol.handle)));
  return Out;
}

llvm::Expected<std::string>
executableSymbolGetName(const ApiTableContainer<::CoreApiTable> &CoreApi,
                        const hsa_executable_symbol_t Symbol) {
  uint32_t NameLength;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_symbol_get_info_fn>(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &NameLength),
      llvm::formatv("Failed to get the length of the name of the executable "
                    "symbol {0:x}",
                    Symbol.handle)));

  std::string Out(NameLength, '\0');
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_symbol_get_info_fn>(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &Out.front()),
      llvm::formatv("Failed to get the name of the executable symbol {0:x}",
                    Symbol.handle)));
  return Out;
}

llvm::Expected<size_t>
executableSymbolGetSymbolSize(const ApiTableContainer<::CoreApiTable> &CoreApi,
                              const hsa_executable_symbol_t Symbol) {
  uint32_t Out;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_symbol_get_info_fn>(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &Out),
      llvm::formatv("Failed to get the size of executable symbol {0:x}",
                    Symbol.handle)));
  return Out;
}

llvm::Expected<uint64_t>
executableSymbolGetAddress(const ApiTableContainer<::CoreApiTable> &CoreApi,
                           const hsa_executable_symbol_t Symbol) {
  llvm::Expected<hsa_symbol_kind_t> SymbolTypeOrErr =
      executableSymbolGetType(CoreApi, Symbol);
  LUTHIER_RETURN_ON_ERROR(SymbolTypeOrErr.takeError());

  uint64_t Out;
  const hsa_executable_symbol_info_t InfoQueried =
      *SymbolTypeOrErr == HSA_SYMBOL_KIND_VARIABLE
          ? HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS
          : HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_symbol_get_info_fn>(
          Symbol, InfoQueried, &Out),
      llvm::formatv("Failed to get the address of the executable symbol {0:x}",
                    Symbol.handle)));

  return Out;
}

llvm::Expected<hsa_agent_t>
executableSymbolGetAgent(const ApiTableContainer<::CoreApiTable> &CoreApi,
                         const hsa_executable_symbol_t Symbol) {
  hsa_agent_t Agent;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      CoreApi.callFunction<&::CoreApiTable::hsa_executable_symbol_get_info_fn>(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &Agent),
      llvm::formatv("Failed to get the agent of executable symbol {0:x}",
                    Symbol.handle)));
  return Agent;
}

} // namespace luthier::hsa