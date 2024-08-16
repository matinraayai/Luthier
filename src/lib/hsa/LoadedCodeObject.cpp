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

#include "common/object_utils.hpp"
#include "hsa/Executable.hpp"
#include "hsa/GpuAgent.hpp"
#include "hsa/hsa_executable_symbol.hpp"
#include "hsa/hsa_intercept.hpp"
#include "hsa/hsa_platform.hpp"
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
  std::lock_guard Lock(getCacheMutex());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(StorageELFOfLCOs.contains(hsaHandle())));
  return *StorageELFOfLCOs.at(hsaHandle());
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
  std::lock_guard Lock(getCacheMutex());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(StorageELFOfLCOs.contains(hsaHandle())));
  return ISA(ISAOfLCOs.at(hsaHandle()));
}

llvm::Expected<const hsa::md::Metadata &>
LoadedCodeObject::getMetadata() const {
  std::lock_guard Lock(getCacheMutex());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  return MetadataOfLCOs.at(this->hsaHandle());
}

llvm::Error LoadedCodeObject::getKernelSymbols(
    llvm::SmallVectorImpl<hsa::LoadedCodeObjectSymbol> &Out) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  // Retrieve kernel symbol names associated with this LCO
  const auto &KernelSymbolsOfThisLCO = KernelSymbolsOfLCOs.at(hsaHandle());
  for (const auto &[KernelName, KernelSymbol] : KernelSymbolsOfThisLCO) {
    // Append the LCO kernel symbol to the output vector
    Out.push_back(KernelSymbol);
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getVariableSymbols(
    llvm::SmallVectorImpl<hsa::LoadedCodeObjectSymbol> &Out) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  // Retrieve variable symbols associated with this LCO
  const auto &VariableSymbolsOfThisLCO = VariableSymbolsOfLCOs.at(hsaHandle());

  for (const auto &[Name, VariableSymbol] : VariableSymbolsOfThisLCO) {
    Out.push_back(VariableSymbol);
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getDeviceFunctionSymbols(
    llvm::SmallVectorImpl<hsa::LoadedCodeObjectSymbol> &Out) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  const auto &DeviceFuncSymbolsOfThisLCO =
      DeviceFuncSymbolsOfLCOs.at(hsaHandle());

  // Device Functions
  for (const auto &[Name, DeviceFuncSymbol] : DeviceFuncSymbolsOfThisLCO) {
    Out.push_back(DeviceFuncSymbol);
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getExternalSymbols(
    llvm::SmallVectorImpl<hsa::LoadedCodeObjectSymbol> &Out) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  const auto &ExternSymbolsOfThisLCO = ExternSymbolsOfLCOs.at(hsaHandle());

  for (const auto &[Name, ExternSymbol] : ExternSymbolsOfThisLCO) {
    Out.push_back(ExternSymbol);
  }

  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getLoadedCodeObjectSymbols(
    llvm::SmallVectorImpl<LoadedCodeObjectSymbol> &Out) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  // Retrieve symbol information associated with this LCO
  const auto &KernelDescSymbolsOfThisLCO = KernelSymbolsOfLCOs.at(hsaHandle());
  const auto &VariableSymbolsOfThisLCO = VariableSymbolsOfLCOs.at(hsaHandle());
  const auto &DeviceFuncSymbolsOfThisLCO =
      DeviceFuncSymbolsOfLCOs.at(hsaHandle());

  Out.reserve(Out.size() + KernelDescSymbolsOfThisLCO.size() +
              VariableSymbolsOfThisLCO.size() +
              DeviceFuncSymbolsOfThisLCO.size());

  LUTHIER_RETURN_ON_ERROR(this->getKernelSymbols(Out));
  LUTHIER_RETURN_ON_ERROR(this->getVariableSymbols(Out));
  LUTHIER_RETURN_ON_ERROR(this->getDeviceFunctionSymbols(Out));
  return llvm::Error::success();
}

llvm::Expected<std::optional<LoadedCodeObjectSymbol>>
LoadedCodeObject::getLoadedCodeObjectSymbolByName(llvm::StringRef Name) const {
  std::lock_guard Lock(getCacheMutex());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  const auto &KernelSymbolsOfThisLCO = KernelSymbolsOfLCOs.at(hsaHandle());
  const auto &VariableSymbolsOfThisLCO = VariableSymbolsOfLCOs.at(hsaHandle());
  const auto &DeviceFuncSymbolsOfThisLCO =
      DeviceFuncSymbolsOfLCOs.at(hsaHandle());
  const auto &ExternSymbolsOfThisLCO = ExternSymbolsOfLCOs.at(hsaHandle());

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
      return std::nullopt;
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

llvm::DenseSet<decltype(hsa_loaded_code_object_t::handle)>
    LoadedCodeObject::CachedLCOs{};

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               std::unique_ptr<llvm::object::ELF64LEObjectFile>>
    LoadedCodeObject::StorageELFOfLCOs{};

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle), hsa_isa_t>
    LoadedCodeObject::ISAOfLCOs;

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<LoadedCodeObjectKernel>>
    LoadedCodeObject::KernelSymbolsOfLCOs{};

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<LoadedCodeObjectDeviceFunction>>
    LoadedCodeObject::DeviceFuncSymbolsOfLCOs{};

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<LoadedCodeObjectVariable>>
    LoadedCodeObject::VariableSymbolsOfLCOs{};

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<LoadedCodeObjectExternSymbol>>
    LoadedCodeObject::ExternSymbolsOfLCOs{};

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle), hsa::md::Metadata>
    LoadedCodeObject::MetadataOfLCOs{};

llvm::Error LoadedCodeObject::cache() const {
  std::lock_guard Lock(getCacheMutex());
  // Cache the Storage ELF
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("Caching LCO with Handle {0:x}.\n",
                                           this->hsaHandle()));
  auto StorageMemory = this->getStorageMemory();
  LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());
  auto StorageELF = getAMDGCNObjectFile(*StorageMemory);
  LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());
  auto &CachedELF =
      *(StorageELFOfLCOs.insert({this->hsaHandle(), std::move(*StorageELF)})
            .first->second);

  // Cache the ISA of the ELF
  auto LLVMISA = getELFObjectFileISA(CachedELF);
  LUTHIER_RETURN_ON_ERROR(LLVMISA.takeError());
  auto ISA = hsa::ISA::fromLLVM(std::get<0>(*LLVMISA), std::get<1>(*LLVMISA),
                                std::get<2>(*LLVMISA));
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());
  ISAOfLCOs.insert({this->hsaHandle(), ISA->asHsaType()});
  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("ISA of LCO {0:x}: {1}.\n",
                                           this->hsaHandle(),
                                           llvm::cantFail(ISA->getName())));
  // Cache the ELF Symbols
  // We can cache the variable and extern symbols right away, but we
  // need to wait until the end of the iteration to distinguish between
  // kernels and device function
  auto &VariableELFSymbolsOfThisLCO =
      VariableSymbolsOfLCOs.insert({hsaHandle(), {}}).first->second;
  auto &ExternSymbolsOfThisLCO =
      ExternSymbolsOfLCOs.insert({hsaHandle(), {}}).first->second;

  llvm::StringMap<const llvm::object::ELFSymbolRef *> KDSymbolsOfThisLCO;
  llvm::StringMap<const llvm::object::ELFSymbolRef *> FuncSymbolsOfThisLCO;

  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("Caching symbols for LCO {0:x}:\n",
                                           this->hsaHandle()));

  for (const object::ELFSymbolRef &Symbol : CachedELF.symbols()) {
    auto Type = Symbol.getELFType();
    auto Binding = Symbol.getBinding();
    auto SymbolName = Symbol.getName();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
    auto Size = Symbol.getSize();
    LLVM_DEBUG(llvm::dbgs()
               << llvm::formatv("\tSymbol Name: {0}, Binding: {1}, Type: {2}\n",
                                *SymbolName, Binding, Type));
    if (Type == llvm::ELF::STT_FUNC)
      FuncSymbolsOfThisLCO.insert({*SymbolName, &Symbol});
    else if (Type == llvm::ELF::STT_OBJECT) {
      // Kernel Descriptor Symbol
      if (SymbolName->ends_with(".kd") && Size == 64) {
        KDSymbolsOfThisLCO.insert({*SymbolName, &Symbol});
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "\tSymbol {0} is a kernel descriptor.\n", *SymbolName));
      }
      // Variable Symbol
      else {
        VariableELFSymbolsOfThisLCO.insert(
            {*SymbolName,
             LoadedCodeObjectVariable(this->asHsaType(), &Symbol)});
      }
    } else if (Type == llvm::ELF::STT_AMDGPU_HSA_KERNEL && Size == 64) {
      KDSymbolsOfThisLCO.insert({*SymbolName, &Symbol});
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "\tSymbol {0} is a kernel descriptor.\n", *SymbolName));
    } else if (Type == llvm::ELF::STT_NOTYPE) {
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "\tSymbol {0} is an external symbol.\n", *SymbolName));
      ExternSymbolsOfThisLCO.insert(
          {*SymbolName,
           LoadedCodeObjectExternSymbol(this->asHsaType(), &Symbol)});
    }
  }

  // Cache the LCO and Kernel Symbols Metadata
  auto MetaData = parseNoteMetaData(CachedELF);
  LUTHIER_RETURN_ON_ERROR(MetaData.takeError());
  auto &LCOCachedMetaData =
      MetadataOfLCOs.insert({this->hsaHandle(), *MetaData}).first->getSecond();

  auto &KernelSymbolsOfThisLCO =
      KernelSymbolsOfLCOs.insert({hsaHandle(), {}}).first->second;

  // Construct the kernel symbols and cache them
  for (auto &KernelMD : LCOCachedMetaData.Kernels) {
    auto &NameWithKDAtTheEnd = KernelMD.Symbol;
    llvm::StringRef NameWithoutKD =
        llvm::StringRef(NameWithKDAtTheEnd)
            .substr(0, NameWithKDAtTheEnd.rfind(".kd"));
    // Find the KD symbol
    auto KDSymbolIter = KDSymbolsOfThisLCO.find(NameWithKDAtTheEnd);
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ASSERTION(KDSymbolIter != KDSymbolsOfThisLCO.end()));
    // Find the kernel function symbol
    auto KFuncSymbolIter = FuncSymbolsOfThisLCO.find(NameWithoutKD);
    LUTHIER_RETURN_ON_ERROR(
        LUTHIER_ASSERTION(KFuncSymbolIter != FuncSymbolsOfThisLCO.end()));
    // Construct the Kernel LCO Symbol
    KernelSymbolsOfThisLCO.insert(
        {NameWithoutKD,
         LoadedCodeObjectKernel(this->asHsaType(), KFuncSymbolIter->second,
                                KDSymbolIter->second, KernelMD)});
    // Remove the kernel function symbol from the map so that it doesn't
    // get counted as a device function in the later step
    FuncSymbolsOfThisLCO.erase(KFuncSymbolIter);

    LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                   "Metadata for kernel symbol {0}:\n", KernelMD.Symbol));
    LLVM_DEBUG(llvm::dbgs() << KernelMD << "\n");
  }

  // Finally, construct the device function LCO symbols
  auto &DeviceFuncSymbolsOfThisLCO =
      DeviceFuncSymbolsOfLCOs.insert({hsaHandle(), {}}).first->second;
  for (const auto &[Name, FuncSymbol] : FuncSymbolsOfThisLCO) {
    DeviceFuncSymbolsOfThisLCO.insert(
        {Name, LoadedCodeObjectDeviceFunction(this->asHsaType(), FuncSymbol)});
  }

  CachedLCOs.insert(hsaHandle());
  return llvm::Error::success();
}

bool LoadedCodeObject::isCached() const {
  std::lock_guard Lock(getCacheMutex());
  return CachedLCOs.contains(this->hsaHandle());
}

llvm::Error LoadedCodeObject::invalidate() const {
  std::lock_guard Lock(getCacheMutex());
  StorageELFOfLCOs.erase(this->hsaHandle());
  ISAOfLCOs.erase(this->hsaHandle());
  KernelSymbolsOfLCOs.erase(this->hsaHandle());
  DeviceFuncSymbolsOfLCOs.erase(this->hsaHandle());
  DeviceFuncSymbolsOfLCOs.erase(this->hsaHandle());
  VariableSymbolsOfLCOs.erase(this->hsaHandle());
  ExternSymbolsOfLCOs.erase(this->hsaHandle());
  MetadataOfLCOs.erase(this->hsaHandle());
  return llvm::Error::success();
}

} // namespace luthier::hsa
