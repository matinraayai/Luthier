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
#include "luthier/HSA/LoadedCodeObjectSymbol.h"
#include "../../../include/luthier/HSATooling/LoadedCodeObjectCache.h"
#include "luthier/HSA/Agent.h"
#include "luthier/HSA/Executable.h"
#include "luthier/HSA/ExecutableSymbol.h"
#include "luthier/HSA/HsaError.h"
#include "luthier/HSA/LoadedCodeObject.h"
#include "luthier/HSA/LoadedCodeObjectKernel.h"

namespace luthier {

hsa::LoadedCodeObjectSymbol::LoadedCodeObjectSymbol(
    hsa_loaded_code_object_t LCO, luthier::object::AMDGCNObjectFile &StorageELF,
    llvm::object::ELFSymbolRef Symbol, SymbolKind Kind,
    std::optional<hsa_executable_symbol_t> ExecutableSymbol)
    : BackingLCO(LCO), StorageELF(StorageELF), Symbol(Symbol), Kind(Kind),
      ExecutableSymbol(ExecutableSymbol) {};

llvm::Expected<std::unique_ptr<hsa::LoadedCodeObjectSymbol>>
luthier::hsa::LoadedCodeObjectSymbol::fromExecutableSymbol(
    const ApiTableContainer<::CoreApiTable> &CoreApi,
    const hsa_ven_amd_loader_1_03_pfn_t &LoaderApi,
    hsa_executable_symbol_t Symbol) {
  llvm::Expected<luthier::address_t> LoadedAddressOrErr =
      hsa::executableSymbolGetAddress(CoreApi, Symbol);
  LUTHIER_RETURN_ON_ERROR(LoadedAddressOrErr.takeError());
  return fromLoadedAddress(CoreApi, LoaderApi, *LoadedAddressOrErr);
}

llvm::Expected<std::unique_ptr<hsa::LoadedCodeObjectSymbol>>
luthier::hsa::LoadedCodeObjectSymbol::fromLoadedAddress(
    const ApiTableContainer<::CoreApiTable> &CoreApi,
    const hsa_ven_amd_loader_1_03_pfn_t &LoaderApi,
    luthier::address_t LoadedAddress) {
  hsa_executable_t Executable;

  // Check which executable this kernel object (address) belongs to
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_CALL_ERROR_CHECK(
      LoaderApi.hsa_ven_amd_loader_query_executable(
          reinterpret_cast<const void *>(LoadedAddress), &Executable),
      llvm::formatv("Failed to get the executable associated with the loaded "
                    "address {0:x}",
                    LoadedAddress)));
  llvm::SmallVector<hsa_loaded_code_object_t, 4> LCOs;
  LUTHIER_RETURN_ON_ERROR(
      executableGetLoadedCodeObjects(LoaderApi, Executable, LCOs));

  auto &COC = hsa::LoadedCodeObjectCache::instance();

  for (const auto &LCO : LCOs) {
    llvm::SmallVector<std::unique_ptr<LoadedCodeObjectSymbol>> Symbols;
    LUTHIER_RETURN_ON_ERROR(COC.getLoadedCodeObjectSymbols(LCO, Symbols));
    for (auto &S : Symbols) {
      llvm::Expected<luthier::address_t> SLoadedAddrOrErr =
          S->getLoadedSymbolAddress(LoaderApi);
      LUTHIER_RETURN_ON_ERROR(SLoadedAddrOrErr.takeError());
      if (*SLoadedAddrOrErr == LoadedAddress)
        return std::move(S);
      if (auto *KernelSymbol =
              llvm::dyn_cast<hsa::LoadedCodeObjectKernel>(S.get())) {
        llvm::Expected<const hsa::KernelDescriptor *> KDAddress =
            KernelSymbol->getKernelDescriptor(CoreApi);
        LUTHIER_RETURN_ON_ERROR(KDAddress.takeError());
        if (reinterpret_cast<luthier::address_t>(*KDAddress) == LoadedAddress) {
          return std::move(S);
        }
      }
    }
  }
  return llvm::make_error<hsa::HsaError>(
      llvm::formatv("Failed to find the Loaded Code Object symbol "
                    "associated with loaded address {0:x}.",
                    LoadedAddress),
      std::nullopt, std::source_location::current(),
      HsaError::StackTraceInitializer());
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
hsa::LoadedCodeObjectSymbol::getLoadedSymbolContents(
    const hsa_ven_amd_loader_1_03_pfn_t &VenLoaderTable) const {
  auto LoadedAddress = getLoadedSymbolAddress(VenLoaderTable);
  LUTHIER_RETURN_ON_ERROR(LoadedAddress.takeError());

  auto SymbolSize = Symbol.getSize();

  return llvm::ArrayRef<uint8_t>{
      reinterpret_cast<const uint8_t *>(*LoadedAddress), SymbolSize};
}

llvm::Expected<luthier::address_t>
hsa::LoadedCodeObjectSymbol::getLoadedSymbolAddress(
    const hsa_ven_amd_loader_1_03_pfn_t &VenLoaderTable) const {

  auto LoadedMemory =
      loadedCodeObjectGetLoadedMemory(VenLoaderTable, BackingLCO);
  LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

  auto SymbolElfAddress = Symbol.getAddress();
  LUTHIER_RETURN_ON_ERROR(SymbolElfAddress.takeError());

  auto SymbolLMO = object::getLoadOffset(Symbol);
  LUTHIER_RETURN_ON_ERROR(SymbolLMO.takeError());

  LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      SymbolLMO->has_value(), "Symbol doesn't have a load address"));

  return reinterpret_cast<luthier::address_t>(**SymbolLMO +
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
