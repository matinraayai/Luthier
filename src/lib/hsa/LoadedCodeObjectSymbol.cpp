//===-- LoadedCodeObjectSymbol.cpp - Loaded Code Object Symbol ------------===//
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
/// This file implements the \c LoadedCodeObjectSymbol under the \c luthier::hsa
/// namespace.
//===----------------------------------------------------------------------===//

#include "common/ObjectUtils.hpp"
#include "hsa/Executable.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include "luthier/hsa/HsaError.h"
#include <luthier/hsa/LoadedCodeObjectSymbol.h>

namespace luthier {

hsa::LoadedCodeObjectSymbol::LoadedCodeObjectSymbol(
    hsa_loaded_code_object_t LCO, const llvm::object::ELFSymbolRef Symbol,
    hsa::LoadedCodeObjectSymbol::SymbolKind Kind,
    std::optional<hsa_executable_symbol_t> ExecutableSymbol)
    : BackingLCO(LCO), Symbol(Symbol), Kind(Kind),
      ExecutableSymbol(ExecutableSymbol) {};

llvm::Expected<const hsa::LoadedCodeObjectSymbol &>
luthier::hsa::LoadedCodeObjectSymbol::fromExecutableSymbol(
    hsa_executable_symbol_t Symbol) {
  auto &LCOSymbolCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectSymbolCache();
  std::lock_guard Lock(LCOSymbolCache.CacheMutex);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      LCOSymbolCache.ExecToLCOSymbolMap.contains(Symbol),
      "Failed to find the cached entry for symbol {0:x}.", Symbol.handle));
  return *LCOSymbolCache.ExecToLCOSymbolMap.at(Symbol);
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
hsa::LoadedCodeObjectSymbol::getLoadedSymbolContents() const {
  auto LoadedAddress = getLoadedSymbolAddress();
  LUTHIER_RETURN_ON_ERROR(LoadedAddress.takeError());

  auto SymbolSize = Symbol.getSize();

  return llvm::ArrayRef<uint8_t>{
      reinterpret_cast<const uint8_t *>(*LoadedAddress), SymbolSize};
}

llvm::Expected<luthier::address_t>
hsa::LoadedCodeObjectSymbol::getLoadedSymbolAddress() const {
  auto LCOWrapper = hsa::LoadedCodeObject(BackingLCO);
  auto StorageELF = LCOWrapper.getStorageELF();

  LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());

  auto LoadedMemory = LCOWrapper.getLoadedMemory();
  LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

  auto SymbolElfAddress = Symbol.getAddress();
  LUTHIER_RETURN_ON_ERROR(SymbolElfAddress.takeError());

  auto SymbolLMO = getLoadedMemoryOffset(StorageELF->getELFFile(), Symbol);
  LUTHIER_RETURN_ON_ERROR(SymbolLMO.takeError());

  return reinterpret_cast<luthier::address_t>(*SymbolLMO +
                                              LoadedMemory->data());
}

std::optional<hsa_executable_symbol_t>
hsa::LoadedCodeObjectSymbol::getExecutableSymbol() const {
  return ExecutableSymbol;
}

llvm::Expected<llvm::StringRef> hsa::LoadedCodeObjectSymbol::getName() const {
  return Symbol.getName();
}

size_t hsa::LoadedCodeObjectSymbol::getSize() const { return Symbol.getSize(); }

uint8_t hsa::LoadedCodeObjectSymbol::getBinding() const {
  return Symbol.getBinding();
}

const hsa::LoadedCodeObjectSymbol *
hsa::LoadedCodeObjectSymbol::fromLoadedAddress(
    luthier::address_t LoadedAddress) {
  auto &LCOSymbolCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectSymbolCache();
  std::lock_guard Lock(LCOSymbolCache.CacheMutex);
  if (LCOSymbolCache.LoadedAddressToSymbolMap.contains(LoadedAddress))
    return LCOSymbolCache.LoadedAddressToSymbolMap.at(LoadedAddress);
  else
    return nullptr;
}

} // namespace luthier
