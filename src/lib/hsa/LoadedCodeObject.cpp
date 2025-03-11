//===-- LoadedCodeObject.cpp - HSA Loaded Code Object Wrapper -------------===//
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
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
      LCOCache.isCached(*this), "LCO {0:x} is not cached.", hsaHandle()));
  return *LCOCache.CachedLCOs.at(this->asHsaType()).ElfObjectFile;
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
  auto StorageElfOrErr = getStorageELF();
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());
  auto ElfISAOrErr = getELFObjectFileISA(*StorageElfOrErr);
  LUTHIER_RETURN_ON_ERROR(ElfISAOrErr.takeError());
  return hsa::ISA::fromLLVM(std::get<0>(*ElfISAOrErr),
                            std::get<1>(*ElfISAOrErr),
                            std::get<2>(*ElfISAOrErr));
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
    llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
    const {
  llvm::Expected<luthier::AMDGCNObjectFile &> StorageElfOrErr = getStorageELF();
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());

  llvm::StringMap<llvm::object::ELFSymbolRef> FuncSymbolsOfThisLCO;
  llvm::StringMap<llvm::object::ELFSymbolRef> KDSymbolsOfThisLCO;

  for (const object::ELFSymbolRef &Symbol : StorageElfOrErr->symbols()) {
    auto Type = Symbol.getELFType();
    auto Binding = Symbol.getBinding();
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

  // Cache the LCO and Kernel Symbols Metadata
  std::shared_ptr<hsa::md::Metadata> MetaData;
  LUTHIER_RETURN_ON_ERROR(
      parseNoteMetaData(*StorageElfOrErr).moveInto(MetaData));

  // Construct the kernel symbols and cache them
  for (auto &KernelMD : MetaData->Kernels) {

    LLVM_DEBUG(llvm::dbgs() << "Creating the kernel symbols.\n";);

    auto &NameWithKDAtTheEnd = KernelMD.Symbol;
    llvm::StringRef NameWithoutKD =
        llvm::StringRef(NameWithKDAtTheEnd)
            .substr(0, NameWithKDAtTheEnd.rfind(".kd"));
    // Find the KD symbol
    auto KDSymbolIter = KDSymbolsOfThisLCO.find(NameWithKDAtTheEnd);
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ERROR_CHECK(KDSymbolIter != KDSymbolsOfThisLCO.end(),
                            "Failed to find kernel {0} inside the list of "
                            "kernel descriptor symbols of LCO {1:x}.",
                            NameWithKDAtTheEnd, hsaHandle()));
    // Find the kernel function symbol
    auto KFuncSymbolIter = FuncSymbolsOfThisLCO.find(NameWithoutKD);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        KFuncSymbolIter != FuncSymbolsOfThisLCO.end(),
        "Failed to find kernel function {0} inside the function "
        "symbols of LCO {1:x}.",
        NameWithoutKD, hsaHandle()));

    // Construct the Kernel LCO Symbol
    auto KernelSymbol = LoadedCodeObjectKernel::create(
        asHsaType(), *StorageElfOrErr, MetaData, KFuncSymbolIter->second,
        KDSymbolIter->second);
    LUTHIER_RETURN_ON_ERROR(KernelSymbol.takeError());
    Out.push_back(std::move(*KernelSymbol));
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getVariableSymbols(
    llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
    const {
  llvm::Expected<luthier::AMDGCNObjectFile &> StorageElfOrErr = getStorageELF();
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());

  for (const object::ELFSymbolRef &Symbol : StorageElfOrErr->symbols()) {
    auto Type = Symbol.getELFType();
    auto Binding = Symbol.getBinding();
    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
    auto Size = Symbol.getSize();
    LLVM_DEBUG(llvm::dbgs()
               << llvm::formatv("\tSymbol Name: {0}, Binding: {1}, Type: {2}\n",
                                *SymbolName, Binding, Type));
    if (Type == llvm::ELF::STT_OBJECT && !SymbolName->ends_with(".kd")) {
      // Variable Symbol
      auto VarSymbolOrErr = std::move(
          LoadedCodeObjectVariable::create(asHsaType(), *StorageElfOrErr, Symbol));
      LUTHIER_RETURN_ON_ERROR(VarSymbolOrErr.takeError());
      Out.push_back(std::move(*VarSymbolOrErr));
    }
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getDeviceFunctionSymbols(
    llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
    const {
  llvm::Expected<luthier::AMDGCNObjectFile &> StorageElfOrErr = getStorageELF();
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());

  llvm::StringMap<llvm::object::ELFSymbolRef> FuncSymbolsOfThisLCO;
  llvm::StringMap<llvm::object::ELFSymbolRef> KDSymbolsOfThisLCO;

  for (const object::ELFSymbolRef &Symbol : StorageElfOrErr->symbols()) {
    auto Type = Symbol.getELFType();
    auto Binding = Symbol.getBinding();
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
          asHsaType(), *StorageElfOrErr, FuncSymbol);
      LUTHIER_RETURN_ON_ERROR(DevFuncSymOrErr.takeError());
      Out.push_back(std::move(*DevFuncSymOrErr));
    }
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getExternalSymbols(
    llvm::SmallVectorImpl<std::unique_ptr<hsa::LoadedCodeObjectSymbol>> &Out)
    const {
  llvm::Expected<luthier::AMDGCNObjectFile &> StorageElfOrErr = getStorageELF();
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());

  llvm::StringMap<llvm::object::ELFSymbolRef> FuncSymbolsOfThisLCO;
  llvm::StringMap<llvm::object::ELFSymbolRef> KDSymbolsOfThisLCO;

  for (const object::ELFSymbolRef &Symbol : StorageElfOrErr->symbols()) {
    auto Type = Symbol.getELFType();
    auto Binding = Symbol.getBinding();
    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
    auto Size = Symbol.getSize();
    if (Type == llvm::ELF::STT_NOTYPE && Binding == llvm::ELF::STB_GLOBAL &&
        *SymbolName != "UNDEF") {
      auto ExternSymbol = LoadedCodeObjectExternSymbol::create(
          asHsaType(), *StorageElfOrErr, Symbol);
      LUTHIER_RETURN_ON_ERROR(ExternSymbol.takeError());

      Out.push_back(std::move(*ExternSymbol));
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "\tSymbol {0} is an external symbol.\n", *SymbolName));
    }
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getLoadedCodeObjectSymbols(
    llvm::SmallVectorImpl<std::unique_ptr<LoadedCodeObjectSymbol>> &Out) const {
  LUTHIER_RETURN_ON_ERROR(getKernelSymbols(Out));
  LUTHIER_RETURN_ON_ERROR(getDeviceFunctionSymbols(Out));
  LUTHIER_RETURN_ON_ERROR(getVariableSymbols(Out));
  LUTHIER_RETURN_ON_ERROR(getExternalSymbols(Out));

  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<LoadedCodeObjectSymbol>>
LoadedCodeObject::getLoadedCodeObjectSymbolByName(llvm::StringRef Name) const {
  llvm::Expected<luthier::AMDGCNObjectFile &> StorageElfOrErr = getStorageELF();
  LUTHIER_RETURN_ON_ERROR(StorageElfOrErr.takeError());

  auto OptElfSymbolOrErr = lookupSymbolByName(*StorageElfOrErr, Name);
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
      auto KDFuncOrErr = lookupSymbolByName(*StorageElfOrErr,
                                            Name.substr(0, Name.rfind(".kd")));
      LUTHIER_RETURN_ON_ERROR(KDFuncOrErr.takeError());
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
          KDFuncOrErr->has_value(),
          "Failed to find the kernel function for {0}.", Name));
      std::shared_ptr<hsa::md::Metadata> MetaData;
      LUTHIER_RETURN_ON_ERROR(
          parseNoteMetaData(*StorageElfOrErr).moveInto(MetaData));

      return LoadedCodeObjectKernel::create(asHsaType(), *StorageElfOrErr,
                                            MetaData, **KDFuncOrErr,
                                            **OptElfSymbolOrErr);
    } else if (SymbolType == llvm::ELF::STT_FUNC) {
      // Find the KD symbol (if available)
      auto KDSymOrErr =
          lookupSymbolByName(*StorageElfOrErr, (Name + ".kd").str());
      LUTHIER_RETURN_ON_ERROR(KDSymOrErr.takeError());
      if (KDSymOrErr->has_value()) {
        std::shared_ptr<hsa::md::Metadata> MetaData;
        LUTHIER_RETURN_ON_ERROR(
            parseNoteMetaData(*StorageElfOrErr).moveInto(MetaData));

        return LoadedCodeObjectKernel::create(asHsaType(), *StorageElfOrErr,
                                              MetaData, **OptElfSymbolOrErr,
                                              **KDSymOrErr);
      } else {
        return LoadedCodeObjectDeviceFunction::create(
            asHsaType(), *StorageElfOrErr, **OptElfSymbolOrErr);
      }
    } else if (SymbolType == llvm::ELF::STT_OBJECT) {
      return LoadedCodeObjectVariable::create(asHsaType(), *StorageElfOrErr,
                                              **OptElfSymbolOrErr);
    } else if (SymbolType == llvm::ELF::STT_NOTYPE &&
               SymbolBinding == llvm::ELF::STB_GLOBAL && Name != "UNDEF") {
      return LoadedCodeObjectExternSymbol::create(asHsaType(), *StorageElfOrErr,
                                                  **OptElfSymbolOrErr);
    } else {
      return nullptr;
    }
  }
  return nullptr;
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
