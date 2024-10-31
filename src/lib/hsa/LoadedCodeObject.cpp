//===-- LoadedCodeObject.cpp - HSA Loaded Code Object Wrapper -------------===//
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
/// This file implements the \c LoadedCodeObject class under the \c luthier::hsa
/// namespace.
//===----------------------------------------------------------------------===//

#include "hsa/LoadedCodeObject.hpp"

#include "hsa/Executable.hpp"
#include "hsa/ExecutableBackedObjectsCache.hpp"
#include "hsa/ExecutableSymbol.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/HsaRuntimeInterceptor.hpp"
#include <llvm/ADT/StringMap.h>

namespace object = llvm::object;

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-loaded-code-object"

namespace luthier::hsa {

LoadedCodeObject::LoadedCodeObject(hsa_loaded_code_object_t LCO)
    : HandleType<hsa_loaded_code_object_t>(LCO) {}

llvm::Expected<Executable> LoadedCodeObject::getExecutable() const {
  hsa_executable_t Exec;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_EXECUTABLE, &Exec)));

  return Executable(Exec);
}

llvm::Expected<GpuAgent> LoadedCodeObject::getAgent() const {
  hsa_agent_t Agent;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(), HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
          &Agent)));
  return GpuAgent(Agent);
}

llvm::Expected<llvm::object::ELF64LEObjectFile &>
LoadedCodeObject::getStorageELF() const {
  auto &LCOCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectCache();
  std::lock_guard Lock(LCOCache.ExecutableCacheMutex);
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ERROR_CHECK(LCOCache.StorageELFOfLCOs.contains(asHsaType()),
                          "LCO {0:x} is not cached.", hsaHandle()));
  return *LCOCache.StorageELFOfLCOs.at(asHsaType());
}

llvm::Expected<long> LoadedCodeObject::getLoadDelta() const {
  long LoadDelta;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA, &LoadDelta)));
  return LoadDelta;
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
LoadedCodeObject::getLoadedMemory() const {
  luthier::address_t LoadBase;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &LoadBase)));

  uint64_t LoadSize;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &LoadSize)));

  return llvm::ArrayRef<uint8_t>{reinterpret_cast<uint8_t *>(LoadBase),
                                 LoadSize};
}

llvm::Expected<std::string> LoadedCodeObject::getUri() const {
  unsigned int UriLength;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH, &UriLength)));

  std::string URI;
  URI.resize(UriLength);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(), HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI,
          URI.data())));

  return URI;
}

llvm::Expected<ISA> LoadedCodeObject::getISA() const {
  auto &LCOCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectCache();
  std::lock_guard Lock(LCOCache.ExecutableCacheMutex);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      LCOCache.isCached(*this), "LCO {0:x} is not cached.", hsaHandle()));
  return ISA(LCOCache.ISAOfLCOs.at(asHsaType()));
}

llvm::Expected<const hsa::md::Metadata &>
LoadedCodeObject::getMetadata() const {
  auto &LCOCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectCache();
  std::lock_guard Lock(LCOCache.ExecutableCacheMutex);
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      LCOCache.isCached(*this), "LCO {0:x} is not cached.", hsaHandle()));
  return LCOCache.MetadataOfLCOs.at(this->asHsaType());
}

llvm::Error LoadedCodeObject::getKernelSymbols(
    llvm::SmallVectorImpl<const hsa::LoadedCodeObjectSymbol *> &Out) const {
  auto &LCOCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectCache();
  std::lock_guard Lock(LCOCache.ExecutableCacheMutex);

  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      LCOCache.isCached(*this), "LCO {0:x} is not cached.", hsaHandle()));
  // Retrieve kernel symbol names associated with this LCO
  const auto &KernelSymbolsOfThisLCO =
      LCOCache.KernelSymbolsOfLCOs.at(asHsaType());
  for (const auto &KernelNameAndSymbol : KernelSymbolsOfThisLCO) {

    // Append the LCO kernel symbol to the output vector
    Out.push_back(
        llvm::dyn_cast<LoadedCodeObjectSymbol>(KernelNameAndSymbol.second));
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getVariableSymbols(
    llvm::SmallVectorImpl<const hsa::LoadedCodeObjectSymbol *> &Out) const {
  auto &LCOCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectCache();
  std::lock_guard Lock(LCOCache.ExecutableCacheMutex);
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      LCOCache.isCached(*this), "LCO {0:x} is not cached.", hsaHandle()));
  // Retrieve variable symbols associated with this LCO
  const auto &VariableSymbolsOfThisLCO =
      LCOCache.VariableSymbolsOfLCOs.at(asHsaType());

  for (const auto &NameAndVariableSymbol : VariableSymbolsOfThisLCO) {
    Out.push_back(NameAndVariableSymbol.second);
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getDeviceFunctionSymbols(
    llvm::SmallVectorImpl<const hsa::LoadedCodeObjectSymbol *> &Out) const {
  auto &LCOCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectCache();
  std::lock_guard Lock(LCOCache.ExecutableCacheMutex);
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      LCOCache.isCached(*this), "LCO {0:x} is not cached.", hsaHandle()));
  const auto &DeviceFuncSymbolsOfThisLCO =
      LCOCache.DeviceFuncSymbolsOfLCOs.at(asHsaType());

  // Device Functions
  for (const auto &NameAndDeviceFuncSymbol : DeviceFuncSymbolsOfThisLCO) {
    Out.push_back(NameAndDeviceFuncSymbol.second);
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getExternalSymbols(
    llvm::SmallVectorImpl<const hsa::LoadedCodeObjectSymbol *> &Out) const {
  auto &LCOCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectCache();
  std::lock_guard Lock(LCOCache.ExecutableCacheMutex);
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      LCOCache.isCached(*this), "LCO {0:x} is not cached.", hsaHandle()));
  const auto &ExternSymbolsOfThisLCO =
      LCOCache.ExternSymbolsOfLCOs.at(asHsaType());

  for (const auto &NameAndExternSymbol : ExternSymbolsOfThisLCO) {
    Out.push_back(NameAndExternSymbol.second);
  }

  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getLoadedCodeObjectSymbols(
    llvm::SmallVectorImpl<const LoadedCodeObjectSymbol *> &Out) const {
  auto &LCOCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectCache();
  std::lock_guard Lock(LCOCache.ExecutableCacheMutex);
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      LCOCache.isCached(*this), "LCO {0:x} is not cached.", hsaHandle()));
  // Retrieve symbol information associated with this LCO
  const auto &KernelDescSymbolsOfThisLCO =
      LCOCache.KernelSymbolsOfLCOs.at(asHsaType());
  const auto &VariableSymbolsOfThisLCO =
      LCOCache.VariableSymbolsOfLCOs.at(asHsaType());
  const auto &DeviceFuncSymbolsOfThisLCO =
      LCOCache.DeviceFuncSymbolsOfLCOs.at(asHsaType());

  Out.reserve(Out.size() + KernelDescSymbolsOfThisLCO.size() +
              VariableSymbolsOfThisLCO.size() +
              DeviceFuncSymbolsOfThisLCO.size());

  LUTHIER_RETURN_ON_ERROR(this->getKernelSymbols(Out));
  LUTHIER_RETURN_ON_ERROR(this->getVariableSymbols(Out));
  LUTHIER_RETURN_ON_ERROR(this->getDeviceFunctionSymbols(Out));
  return llvm::Error::success();
}

llvm::Expected<const LoadedCodeObjectSymbol *>
LoadedCodeObject::getLoadedCodeObjectSymbolByName(llvm::StringRef Name) const {
  auto &LCOCache =
      ExecutableBackedObjectsCache::instance().getLoadedCodeObjectCache();
  std::lock_guard Lock(LCOCache.ExecutableCacheMutex);
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      LCOCache.isCached(*this), "LCO {0:x} is not cached.", hsaHandle()));

  const auto &KernelSymbolsOfThisLCO =
      LCOCache.KernelSymbolsOfLCOs.at(asHsaType());
  const auto &VariableSymbolsOfThisLCO =
      LCOCache.VariableSymbolsOfLCOs.at(asHsaType());
  const auto &DeviceFuncSymbolsOfThisLCO =
      LCOCache.DeviceFuncSymbolsOfLCOs.at(asHsaType());
  const auto &ExternSymbolsOfThisLCO =
      LCOCache.ExternSymbolsOfLCOs.at(asHsaType());

  if (KernelSymbolsOfThisLCO.contains(Name)) {
    return KernelSymbolsOfThisLCO.at(Name);
  } else if (VariableSymbolsOfThisLCO.contains(Name)) {
    return VariableSymbolsOfThisLCO.at(Name);
  } else if (DeviceFuncSymbolsOfThisLCO.contains(Name)) {
    return DeviceFuncSymbolsOfThisLCO.at(Name);
  } else if (ExternSymbolsOfThisLCO.contains(Name)) {
    return ExternSymbolsOfThisLCO.at(Name);
  } else {
    auto NameWithKD = (Name + ".kd").str();
    if (KernelSymbolsOfThisLCO.contains(Name))
      return KernelSymbolsOfThisLCO.at(Name);
    else
      return nullptr;
  }
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
LoadedCodeObject::getStorageMemory() const {
  luthier::address_t StorageBase;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
          &StorageBase)));

  uint64_t StorageSize;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
          &StorageSize)));

  return llvm::ArrayRef<uint8_t>{reinterpret_cast<uint8_t *>(StorageBase),
                                 StorageSize};
}

} // namespace luthier::hsa
