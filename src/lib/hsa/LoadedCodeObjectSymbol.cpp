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
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/LoadedCodeObject.hpp"
#include "luthier/hsa/HsaError.h"
#include "luthier/hsa/LoadedCodeObjectKernel.h"
#include <luthier/hsa/LoadedCodeObjectSymbol.h>

namespace luthier {

hsa::LoadedCodeObjectSymbol::LoadedCodeObjectSymbol(
    hsa_loaded_code_object_t LCO,
    std::shared_ptr<luthier::AMDGCNObjectFile> StorageELF,
    llvm::object::ELFSymbolRef Symbol, SymbolKind Kind,
    std::optional<hsa_executable_symbol_t> ExecutableSymbol)
    : BackingLCO(LCO), StorageELF(std::move(StorageELF)), Symbol(Symbol),
      Kind(Kind), ExecutableSymbol(ExecutableSymbol) {};

llvm::Expected<std::unique_ptr<hsa::LoadedCodeObjectSymbol>>
luthier::hsa::LoadedCodeObjectSymbol::fromExecutableSymbol(
    hsa_executable_symbol_t Symbol) {
  llvm::Expected<luthier::address_t> LoadedAddressOrErr =
      hsa::ExecutableSymbol(Symbol).getAddress();
  LUTHIER_RETURN_ON_ERROR(LoadedAddressOrErr.takeError());
  return fromLoadedAddress(*LoadedAddressOrErr);
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

llvm::Expected<std::unique_ptr<hsa::LoadedCodeObjectSymbol>>
hsa::LoadedCodeObjectSymbol::fromLoadedAddress(
    luthier::address_t LoadedAddress) {
  hsa_executable_t Executable;
  const auto &LoaderTable =
      hsa::HsaRuntimeInterceptor::instance().getHsaVenAmdLoaderTable();

  // Check which executable this kernel object (address) belongs to
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(LoaderTable.hsa_ven_amd_loader_query_executable(
          reinterpret_cast<const void *>(LoadedAddress), &Executable)));
  llvm::SmallVector<LoadedCodeObject, 4> LCOs;
  LUTHIER_RETURN_ON_ERROR(
      hsa::Executable(Executable).getLoadedCodeObjects(LCOs));

  for (const auto &LCO : LCOs) {
    llvm::SmallVector<std::unique_ptr<LoadedCodeObjectSymbol>> Symbols;
    LUTHIER_RETURN_ON_ERROR(LCO.getLoadedCodeObjectSymbols(Symbols));
    for (auto &S : Symbols) {
      llvm::Expected<luthier::address_t> SLoadedAddrOrErr =
          S->getLoadedSymbolAddress();
      LUTHIER_RETURN_ON_ERROR(SLoadedAddrOrErr.takeError());
      if (*SLoadedAddrOrErr == LoadedAddress)
        return std::move(S);
      if (auto *KernelSymbol =
              llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(S.get())) {
        llvm::Expected<const hsa::KernelDescriptor *> KDAddress =
            KernelSymbol->getKernelDescriptor();
        LUTHIER_RETURN_ON_ERROR(KDAddress.takeError());
        if (reinterpret_cast<luthier::address_t>(*KDAddress) == LoadedAddress) {
          return std::move(S);
        }
      }
    }
  }
  return LUTHIER_CREATE_ERROR("Failed to find the Loaded Code Object symbol "
                              "associated with loaded address {0:x}.",
                              LoadedAddress);
}

void hsa::LoadedCodeObjectSymbol::print(llvm::raw_ostream &OS) const {
  OS << "Loaded Code Object Symbol:\n";
  auto Name = getName();
  if (Name.takeError())
    OS << "\tFailed to get name.\n";
  else
    OS << "\tName: " << *Name << "\n";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void hsa::LoadedCodeObjectSymbol::dump() const {
  print(llvm::dbgs());
}
#endif

} // namespace luthier
