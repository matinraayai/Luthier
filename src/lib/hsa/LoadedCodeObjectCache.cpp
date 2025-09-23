//===-- LoadedCodeObjectCache.cpp ----------------------------------===//
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
/// This file implements the \c LoadedCodeObjectCache Singleton.
//===----------------------------------------------------------------------===//
#include "luthier/hsa/LoadedCodeObjectCache.h"
#include "hsa/hsa.h"
#include "luthier/hsa/Executable.h"
#include "luthier/hsa/ExecutableSymbol.h"
#include "luthier/hsa/LoadedCodeObject.h"
#include "luthier/object/AMDGCNObjectFile.h"
#include <llvm/Object/ELFObjectFile.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-hsa-exec-cache"

namespace object = llvm::object;

namespace luthier {

template <>
hsa::LoadedCodeObjectCache *Singleton<hsa::LoadedCodeObjectCache>::Instance{
    nullptr};

namespace hsa {

decltype(hsa_executable_load_agent_code_object)
    *LoadedCodeObjectCache::UnderlyingHsaExecutableLoadAgentCodeObjectFn =
        nullptr;

decltype(hsa_executable_destroy)
    *LoadedCodeObjectCache::UnderlyingHsaExecutableDestroyFn = nullptr;

hsa_status_t LoadedCodeObjectCache::hsaExecutableLoadAgentCodeObjectWrapper(
    hsa_executable_t Executable, hsa_agent_t Agent,
    hsa_code_object_reader_t CodeObjectReader, const char *Options,
    hsa_loaded_code_object_t *LoadedCodeObject) {

  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
      UnderlyingHsaExecutableLoadAgentCodeObjectFn != nullptr,
      "Underlying hsa_executable_load_agent_code_object of "
      "LoadedCodeObjectCache is nullptr"));

  hsa_loaded_code_object_t LCO;
  /// Call the underlying function
  hsa_status_t Out = UnderlyingHsaExecutableLoadAgentCodeObjectFn(
      Executable, Agent, CodeObjectReader, Options, &LCO);

  /// If the caller of the wrapper requested to get the LCO handle, return it
  if (LoadedCodeObject != nullptr)
    *LoadedCodeObject = LCO;

  /// Return if the loader is not initialized or we encountered an error
  /// executing the underlying function
  if (!isInitialized() || Out != HSA_STATUS_SUCCESS)
    return Out;

  auto &COC = instance();

  llvm::ArrayRef<uint8_t> StorageMemory;
  LUTHIER_REPORT_FATAL_ON_ERROR(hsa::loadedCodeObjectGetStorageMemory(
                                    COC.VenLoaderSnapshot.getTable(), LCO)
                                    .moveInto(StorageMemory));

  auto StorageCopy =
      std::make_unique<llvm::SmallVector<uint8_t>>(StorageMemory);

  auto ParsedElfOrErr =
      object::AMDGCNObjectFile::createAMDGCNObjectFile(*StorageCopy);
  LUTHIER_REPORT_FATAL_ON_ERROR(ParsedElfOrErr.takeError());

  std::lock_guard Lock(COC.CacheMutex);
  {
    COC.LCOCache.insert({LCO, LCOCacheEntry{std::move(StorageCopy),
                                            std::move(*ParsedElfOrErr)}});
  }

  return Out;
}

hsa_status_t LoadedCodeObjectCache::hsaExecutableDestroyWrapper(
    hsa_executable_t Executable) {
  /// Check if the underlying function is not nullptr
  LUTHIER_REPORT_FATAL_ON_ERROR(
      LUTHIER_GENERIC_ERROR_CHECK(UnderlyingHsaExecutableDestroyFn != nullptr,
                                  "Underlying hsa_executable_destroy of "
                                  "LoadedCodeObjectCache is nullptr"));
  llvm::SmallVector<hsa_loaded_code_object_t, 2> LCOs;
  if (isInitialized()) {
    /// Remove the LCOs of the executable from the cache before it is destroyed
    auto &COC = instance();
    auto &VenTable = COC.VenLoaderSnapshot.getTable();
    LUTHIER_REPORT_FATAL_ON_ERROR(
        hsa::executableGetLoadedCodeObjects(VenTable, Executable, LCOs));
  }

  hsa_status_t Out = UnderlyingHsaExecutableDestroyFn(Executable);
  if (Out != HSA_STATUS_SUCCESS)
    return Out;

  if (isInitialized()) {
    /// Remove the LCOs of the executable from the cache before it is destroyed
    auto &COC = instance();
    std::lock_guard Lock(COC.CacheMutex);
    for (hsa_loaded_code_object_t LCO : LCOs) {
      COC.LCOCache.erase(LCO);
    }
  }
  return Out;
}

LoadedCodeObjectCache::LoadedCodeObjectCache(
    const rocprofiler::HsaApiTableSnapshot<::CoreApiTable>
        &CoreApiTableSnapshot,
    const rocprofiler::HsaExtensionTableSnapshot<HSA_EXTENSION_AMD_LOADER>
        &VenLoaderSnapshot,
    const amdgpu::hsamd::MetadataParser &MDParser, llvm::Error &Err)
    : CoreApiTableSnapshot(CoreApiTableSnapshot), MDParser(MDParser),
      VenLoaderSnapshot(VenLoaderSnapshot) {
  llvm::ErrorAsOutParameter EAO(Err);
  HsaWrapperInstaller = std::make_unique<
      rocprofiler::HsaApiTableWrapperInstaller<::CoreApiTable>>(
      Err,
      std::make_tuple(&::CoreApiTable::hsa_executable_load_agent_code_object_fn,
                      &UnderlyingHsaExecutableLoadAgentCodeObjectFn,
                      hsaExecutableLoadAgentCodeObjectWrapper),
      std::make_tuple(&::CoreApiTable::hsa_executable_destroy_fn,
                      &UnderlyingHsaExecutableDestroyFn,
                      hsaExecutableDestroyWrapper));
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
LoadedCodeObjectCache::getAssociatedCodeObject(
    hsa_loaded_code_object_t LCO) const {
  llvm::Expected<LCOCacheEntry &> EntryOrErr =
      getOrCreateLoadedCodeObjectEntry(LCO);
  LUTHIER_RETURN_ON_ERROR(EntryOrErr.takeError());
  return *EntryOrErr->CodeObject;
}

llvm::Expected<luthier::object::AMDGCNObjectFile &>
LoadedCodeObjectCache::getAssociatedObjectFile(
    hsa_loaded_code_object_t LCO) const {
  llvm::Expected<LCOCacheEntry &> EntryOrErr =
      getOrCreateLoadedCodeObjectEntry(LCO);
  LUTHIER_RETURN_ON_ERROR(EntryOrErr.takeError());
  return *EntryOrErr->ParsedELF;
}

bool LoadedCodeObjectCache::LoadedCodeObjectCache::isCached(
    hsa_loaded_code_object_t LCO) {
  std::lock_guard Lock(CacheMutex);
  return LCOCache.contains(LCO);
}

llvm::Error LoadedCodeObjectCache::getKernelSymbols(
    hsa_loaded_code_object_t LCO,
    llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
    const {

  llvm::Expected<luthier::object::AMDGCNObjectFile &> StorageElfOrErr =
      getAssociatedObjectFile(LCO);
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());

  llvm::StringMap<llvm::object::ELFSymbolRef> FuncSymbolsOfThisLCO;
  llvm::StringMap<llvm::object::ELFSymbolRef> KDSymbolsOfThisLCO;

  for (const llvm::object::ELFSymbolRef &Symbol : StorageElfOrErr->symbols()) {
    uint8_t Type = Symbol.getELFType();
    llvm::Expected<llvm::StringRef> SymbolNameOrErr = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolNameOrErr.takeError());
    uint64_t Size = Symbol.getSize();
    if (Type == llvm::ELF::STT_FUNC)
      FuncSymbolsOfThisLCO.insert({*SymbolNameOrErr, Symbol});
    else if ((Type == llvm::ELF::STT_OBJECT &&
              SymbolNameOrErr->ends_with(".kd") && Size == 64) ||
             (Type == llvm::ELF::STT_AMDGPU_HSA_KERNEL && Size == 64)) {
      // Kernel Descriptor Symbol
      KDSymbolsOfThisLCO.insert({*SymbolNameOrErr, Symbol});
      LLVM_DEBUG(llvm::dbgs()
                 << llvm::formatv("\tSymbol {0} is a kernel descriptor.\n",
                                  *SymbolNameOrErr));
    }
  }

  // Cache the LCO and Kernel Symbols Metadata
  std::unique_ptr<llvm::msgpack::Document> MD;
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr->getMetadataDocument().moveInto(MD));
  auto KernelsMDOrErr = MDParser.parseAllKernelsMetadata(*MD);
  LUTHIER_RETURN_ON_ERROR(KernelsMDOrErr.takeError());

  // Construct the kernel symbols and cache them
  for (auto &[NameWithKDAtTheEnd, KernelMD] : *KernelsMDOrErr) {

    LLVM_DEBUG(llvm::dbgs() << "Creating the kernel symbols.\n";);
    llvm::StringRef NameWithoutKD =
        llvm::StringRef(NameWithKDAtTheEnd)
            .substr(0, NameWithKDAtTheEnd.rfind(".kd"));
    // Find the KD symbol
    auto KDSymbolIter = KDSymbolsOfThisLCO.find(NameWithKDAtTheEnd);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        KDSymbolIter != KDSymbolsOfThisLCO.end(),
        llvm::formatv("Failed to find kernel {0} inside the list of "
                      "kernel descriptor symbols of LCO {1:x}.",
                      NameWithKDAtTheEnd, LCO.handle)));
    // Find the kernel function symbol
    auto KFuncSymbolIter = FuncSymbolsOfThisLCO.find(NameWithoutKD);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
        KFuncSymbolIter != FuncSymbolsOfThisLCO.end(),
        llvm::formatv("Failed to find kernel function {0} inside the function "
                      "symbols of LCO {1:x}.",
                      NameWithoutKD, LCO.handle)));

    // Construct the Kernel LCO Symbol
    auto KernelSymbol = LoadedCodeObjectKernel::create(
        CoreApiTableSnapshot.getTable(), VenLoaderSnapshot.getTable(), LCO,
        *StorageElfOrErr, std::move(KernelMD), KFuncSymbolIter->second,
        KDSymbolIter->second);
    LUTHIER_RETURN_ON_ERROR(KernelSymbol.takeError());
    Out.push_back(std::move(*KernelSymbol));
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObjectCache::getVariableSymbols(
    hsa_loaded_code_object_t LCO,
    llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
    const {
  llvm::Expected<luthier::object::AMDGCNObjectFile &> StorageElfOrErr =
      getAssociatedObjectFile(LCO);
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());

  for (const llvm::object::ELFSymbolRef &Symbol : StorageElfOrErr->symbols()) {
    auto Type = Symbol.getELFType();
    auto Binding = Symbol.getBinding();
    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
    LLVM_DEBUG(llvm::dbgs()
               << llvm::formatv("\tSymbol Name: {0}, Binding: {1}, Type:{2}\n ",
                                *SymbolName, Binding, Type));
    if (Type == llvm::ELF::STT_OBJECT && !SymbolName->ends_with(".kd")) {
      // Variable Symbol
      auto VarSymbolOrErr = std::move(LoadedCodeObjectVariable::create(
          CoreApiTableSnapshot.getTable(), VenLoaderSnapshot.getTable(), LCO,
          *StorageElfOrErr, Symbol));
      LUTHIER_RETURN_ON_ERROR(VarSymbolOrErr.takeError());
      Out.push_back(std::move(*VarSymbolOrErr));
    }
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObjectCache::getDeviceFunctionSymbols(
    hsa_loaded_code_object_t LCO,
    llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
    const {
  llvm::Expected<luthier::object::AMDGCNObjectFile &> StorageElfOrErr =
      getAssociatedObjectFile(LCO);
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());

  llvm::StringMap<llvm::object::ELFSymbolRef> FuncSymbolsOfThisLCO;
  llvm::StringMap<llvm::object::ELFSymbolRef> KDSymbolsOfThisLCO;

  for (const llvm::object::ELFSymbolRef &Symbol : StorageElfOrErr->symbols()) {
    auto Type = Symbol.getELFType();
    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
    auto Size = Symbol.getSize();
    if (Type == llvm::ELF::STT_FUNC)
      FuncSymbolsOfThisLCO.insert({*SymbolName, Symbol});
    else if ((Type == llvm::ELF::STT_OBJECT && SymbolName->ends_with(".kd") &&
              Size == 64) ||
             (Type == llvm::ELF::STT_AMDGPU_HSA_KERNEL && Size == 64)) {
      // Kernel Descriptor Symbol
      KDSymbolsOfThisLCO.insert({*SymbolName, Symbol});
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "\tSymbol {0} is a kernel descriptor.\n", *SymbolName));
    }
  }

  for (auto &[FuncSymbolName, FuncSymbol] : FuncSymbolsOfThisLCO) {
    if (!KDSymbolsOfThisLCO.contains((FuncSymbolName + ".kd").str())) {
      auto DevFuncSymOrErr = LoadedCodeObjectDeviceFunction::create(
          LCO, *StorageElfOrErr, FuncSymbol);
      LUTHIER_RETURN_ON_ERROR(DevFuncSymOrErr.takeError());
      Out.push_back(std::move(*DevFuncSymOrErr));
    }
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObjectCache::getExternalSymbols(
    hsa_loaded_code_object_t LCO,
    llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
    const {
  llvm::Expected<luthier::object::AMDGCNObjectFile &> StorageElfOrErr =
      getAssociatedObjectFile(LCO);
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());

  llvm::StringMap<llvm::object::ELFSymbolRef> FuncSymbolsOfThisLCO;
  llvm::StringMap<llvm::object::ELFSymbolRef> KDSymbolsOfThisLCO;

  for (const llvm::object::ELFSymbolRef &Symbol : StorageElfOrErr->symbols()) {
    auto Type = Symbol.getELFType();
    auto Binding = Symbol.getBinding();
    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());

    if (Type == llvm::ELF::STT_NOTYPE && Binding == llvm::ELF::STB_GLOBAL &&
        *SymbolName != "UNDEF") {
      auto ExternSymbol = LoadedCodeObjectExternSymbol::create(
          CoreApiTableSnapshot.getTable(), VenLoaderSnapshot.getTable(), LCO,
          *StorageElfOrErr, Symbol);
      LUTHIER_RETURN_ON_ERROR(ExternSymbol.takeError());

      Out.push_back(std::move(*ExternSymbol));
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "\tSymbol {0} is an external symbol.\n", *SymbolName));
    }
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObjectCache::getLoadedCodeObjectSymbols(
    hsa_loaded_code_object_t LCO,
    llvm::SmallVectorImpl<std::unique_ptr<LoadedCodeObjectSymbol>> &Out) const {
  LUTHIER_RETURN_ON_ERROR(getKernelSymbols(LCO, Out));
  LUTHIER_RETURN_ON_ERROR(getDeviceFunctionSymbols(LCO, Out));
  LUTHIER_RETURN_ON_ERROR(getVariableSymbols(LCO, Out));
  LUTHIER_RETURN_ON_ERROR(getExternalSymbols(LCO, Out));

  return llvm::Error::success();
}

llvm::Expected<LoadedCodeObjectCache::LCOCacheEntry &>
LoadedCodeObjectCache::getOrCreateLoadedCodeObjectEntry(
    hsa_loaded_code_object_t LCO) const {
  std::lock_guard Lock(CacheMutex);
  auto LCOEntry = LCOCache.find(LCO);
  /// If not already cached, try querying the storage ELF from the loader
  /// API
  if (LCOEntry == LCOCache.end()) {
    llvm::ArrayRef<uint8_t> LCOStorageMemory;
    LUTHIER_RETURN_ON_ERROR(hsa::loadedCodeObjectGetStorageMemory(
                                VenLoaderSnapshot.getTable(), LCO)
                                .moveInto(LCOStorageMemory));

    try {
      /// TODO: Install a signal handler to treat segfaults encountered
      /// here as exceptions
      auto StorageCopy =
          std::make_unique<llvm::SmallVector<uint8_t>>(LCOStorageMemory);

      auto ParsedElfOrErr =
          object::AMDGCNObjectFile::createAMDGCNObjectFile(*StorageCopy);
      LUTHIER_REPORT_FATAL_ON_ERROR(ParsedElfOrErr.takeError());

      LCOEntry = LCOCache
                     .insert({LCO, LCOCacheEntry{std::move(StorageCopy),
                                                 std::move(*ParsedElfOrErr)}})
                     .first;
    } catch (...) {
      return llvm::make_error<GenericLuthierError>(
          "Failed to obtain the loaded code object's storage memory");
    }
  }
  return LCOEntry->second;
}

llvm::Expected<std::unique_ptr<LoadedCodeObjectSymbol>>
LoadedCodeObjectCache::getLoadedCodeObjectSymbolByName(
    hsa_loaded_code_object_t LCO, llvm::StringRef Name) const {
  llvm::Expected<luthier::object::AMDGCNObjectFile &> StorageElfOrErr =
      getAssociatedObjectFile(LCO);
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());

  auto OptElfSymbolOrErr = StorageElfOrErr->lookupSymbol(Name);
  LUTHIER_RETURN_ON_ERROR(OptElfSymbolOrErr.takeError());
  if (!OptElfSymbolOrErr->has_value())
    return nullptr;
  else {
    uint8_t SymbolType = (**OptElfSymbolOrErr).getELFType();
    size_t SymbolSize = (**OptElfSymbolOrErr).getSize();
    uint8_t SymbolBinding = (**OptElfSymbolOrErr).getBinding();
    if ((SymbolType == llvm::ELF::STT_OBJECT && Name.ends_with(".kd") &&
         SymbolSize == 64) ||
        (SymbolType == llvm::ELF::STT_AMDGPU_HSA_KERNEL && SymbolSize == 64)) {
      // Find the Kernel function symbol
      auto KDFuncOrErr =
          StorageElfOrErr->lookupSymbol(Name.substr(0, Name.rfind(".kd")));
      LUTHIER_RETURN_ON_ERROR(KDFuncOrErr.takeError());
      LUTHIER_RETURN_ON_ERROR(LUTHIER_GENERIC_ERROR_CHECK(
          KDFuncOrErr->has_value(),
          llvm::formatv("Failed to find the kernel function for {0}.", Name)));
      std::unique_ptr<llvm::msgpack::Document> MD;
      LUTHIER_RETURN_ON_ERROR(
          StorageElfOrErr->getMetadataDocument().moveInto(MD));
      auto KernelsMDOrErr = MDParser.parseKernelMetadata(*MD, Name);
      LUTHIER_RETURN_ON_ERROR(KernelsMDOrErr.takeError());

      return LoadedCodeObjectKernel::create(
          CoreApiTableSnapshot.getTable(), VenLoaderSnapshot.getTable(), LCO,
          *StorageElfOrErr, std::move(*KernelsMDOrErr), **KDFuncOrErr,
          **OptElfSymbolOrErr);
    } else if (SymbolType == llvm::ELF::STT_FUNC) {
      // Find the KD symbol (if available)
      auto KDSymOrErr = StorageElfOrErr->lookupSymbol((Name + ".kd").str());
      LUTHIER_RETURN_ON_ERROR(KDSymOrErr.takeError());
      if (KDSymOrErr->has_value()) {
        std::unique_ptr<llvm::msgpack::Document> MD;
        LUTHIER_RETURN_ON_ERROR(
            StorageElfOrErr->getMetadataDocument().moveInto(MD));
        auto KernelsMDOrErr =
            MDParser.parseKernelMetadata(*MD, (Name + ".kd").str());
        LUTHIER_RETURN_ON_ERROR(KernelsMDOrErr.takeError());

        return LoadedCodeObjectKernel::create(
            CoreApiTableSnapshot.getTable(), VenLoaderSnapshot.getTable(), LCO,
            *StorageElfOrErr, std::move(*KernelsMDOrErr), **OptElfSymbolOrErr,
            **KDSymOrErr);
      } else {
        return LoadedCodeObjectDeviceFunction::create(LCO, *StorageElfOrErr,
                                                      **OptElfSymbolOrErr);
      }
    } else if (SymbolType == llvm::ELF::STT_OBJECT) {
      return LoadedCodeObjectVariable::create(
          CoreApiTableSnapshot.getTable(), VenLoaderSnapshot.getTable(), LCO,
          *StorageElfOrErr, **OptElfSymbolOrErr);
    } else if (SymbolType == llvm::ELF::STT_NOTYPE &&
               SymbolBinding == llvm::ELF::STB_GLOBAL && Name != "UNDEF") {
      return LoadedCodeObjectExternSymbol::create(
          CoreApiTableSnapshot.getTable(), VenLoaderSnapshot.getTable(), LCO,
          *StorageElfOrErr, **OptElfSymbolOrErr);
    } else {
      return nullptr;
    }
  }
  return nullptr;
}
} // namespace hsa
} // namespace luthier
