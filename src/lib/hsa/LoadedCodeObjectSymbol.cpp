//===-- LoadedCodeObjectSymbol.cpp - Loaded Code Object Symbol ------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file implements the \c LoadedCodeObjectSymbol under the \c luthier::hsa
/// namespace.
//===----------------------------------------------------------------------===//

#include "common/error.hpp"
#include "common/object_utils.hpp"
#include "hsa/Executable.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/hsa_loaded_code_object.hpp"
#include <luthier/LoadedCodeObjectSymbol.h>

namespace luthier {

llvm::Expected<hsa::LoadedCodeObjectSymbol>
luthier::hsa::LoadedCodeObjectSymbol::fromExecutableSymbol(
    hsa_executable_symbol_t Symbol) {
  return llvm::Expected<LoadedCodeObjectSymbol>(llvm::Error());
}

llvm::Expected<hsa_agent_t>
luthier::hsa::LoadedCodeObjectSymbol::getAgent() const {
  auto Agent = hsa::LoadedCodeObject(BackingLCO).getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());
  return Agent->asHsaType();
}

llvm::Expected<hsa_executable_t>
hsa::LoadedCodeObjectSymbol::getExecutable() const {
  auto Exec = hsa::LoadedCodeObject(BackingLCO).getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());
  return Exec->asHsaType();
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
hsa::LoadedCodeObjectSymbol::getSymbolContentsOnDevice() const {
  auto LCOWrapper = hsa::LoadedCodeObject(BackingLCO);
  auto StorageELF = LCOWrapper.getStorageELF();

  LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());

  auto LoadedMemory = LCOWrapper.getLoadedMemory();
  LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

  auto SymbolElfAddress = Symbol->getAddress();
  LUTHIER_RETURN_ON_ERROR(SymbolElfAddress.takeError());

  auto SymbolSize = Symbol->getSize();

  auto SymbolLMA = getSymbolLMA(StorageELF->getELFFile(), *Symbol);
  LUTHIER_RETURN_ON_ERROR(SymbolLMA.takeError());

  return llvm::ArrayRef<uint8_t>{
      reinterpret_cast<const uint8_t *>(*SymbolLMA + LoadedMemory->data()),
      SymbolSize};
}

llvm::Expected<std::optional<hsa_executable_symbol_t>>
hsa::LoadedCodeObjectSymbol::getExecutableSymbol() {
  hsa_executable_symbol_t Out;
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(hsa_agent_t, Agent, this->getAgent());
  auto Exec = this->getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());
  auto Name = getName();
  LUTHIER_RETURN_ON_ERROR(Name.takeError());

  auto Status = hsa::HsaRuntimeInterceptor::instance()
                    .getSavedApiTableContainer()
                    .core.hsa_executable_get_symbol_by_name_fn(
                        *Exec, Name->data(), &Agent, &Out);
  if (Status == HSA_STATUS_SUCCESS)
    return Out;
  else if (Status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME)
    return std::nullopt;
  else
    return LUTHIER_HSA_ERROR_CHECK(Status, HSA_STATUS_SUCCESS);
}

} // namespace luthier
