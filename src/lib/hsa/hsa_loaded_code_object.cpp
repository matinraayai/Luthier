#include "hsa/hsa_loaded_code_object.hpp"

#include "common/object_utils.hpp"
#include "hsa/hsa_agent.hpp"
#include "hsa/hsa_executable.hpp"
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

llvm::Error LoadedCodeObject::getExecutableSymbols(
    llvm::SmallVectorImpl<ExecutableSymbol> &Out) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  // Retrieve symbol information associated with this LCO
  const auto &KernelDescSymbolsOfThisLCO =
      KernelDescSymbolsOfLCOs.at(hsaHandle());
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

llvm::Expected<std::optional<ExecutableSymbol>>
LoadedCodeObject::getExecutableSymbolByName(llvm::StringRef Name) const {
  std::lock_guard Lock(getCacheMutex());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  const auto &KernelDescSymbolsOfThisLCO =
      KernelDescSymbolsOfLCOs.at(hsaHandle());
  const auto &KernelFuncSymbolsOfThisLCO =
      KernelFuncSymbolsOfLCOs.at(hsaHandle());
  const auto &VariableSymbolsOfThisLCO = VariableSymbolsOfLCOs.at(hsaHandle());
  const auto &DeviceFuncSymbolsOfThisLCO =
      DeviceFuncSymbolsOfLCOs.at(hsaHandle());

  if (KernelDescSymbolsOfThisLCO.contains(Name)) {
    return constructKernelSymbolUsingName(Name);
  } else if (KernelFuncSymbolsOfThisLCO.contains(Name)) {
    auto NameWithKD = (Name + ".kd").str();
    return constructKernelSymbolUsingName(NameWithKD);
  } else if (VariableSymbolsOfThisLCO.contains(Name)) {
    return constructVariableSymbolUsingName(Name);
  } else if (DeviceFuncSymbolsOfThisLCO.contains(Name)) {
    return constructDeviceFunctionSymbolUsingName(Name);
  } else {
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

llvm::Expected<std::optional<hsa_executable_symbol_t>>
LoadedCodeObject::getHSASymbolHandleByNameFromExecutable(
    llvm::StringRef Name) const {
  hsa_executable_symbol_t Out;
  auto Agent = this->getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());
  auto AgentHsaHandle = Agent->asHsaType();
  auto Exec = this->getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());
  auto Status = getApiTable().core.hsa_executable_get_symbol_by_name_fn(
      Exec->asHsaType(), Name.data(), &AgentHsaHandle, &Out);
  if (Status == HSA_STATUS_SUCCESS)
    return Out;
  else if (Status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME)
    return std::nullopt;
  else
    return LUTHIER_HSA_ERROR_CHECK(Status, HSA_STATUS_SUCCESS);
}

llvm::DenseSet<decltype(hsa_loaded_code_object_t::handle)>
    LoadedCodeObject::CachedLCOs{};

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               std::unique_ptr<llvm::object::ELF64LEObjectFile>>
    LoadedCodeObject::StorageELFOfLCOs{};

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle), hsa_isa_t>
    LoadedCodeObject::ISAOfLCOs;

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<object::ELFSymbolRef>>
    LoadedCodeObject::KernelDescSymbolsOfLCOs{};
llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<object::ELFSymbolRef>>
    LoadedCodeObject::KernelFuncSymbolsOfLCOs{};
llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<object::ELFSymbolRef>>
    LoadedCodeObject::DeviceFuncSymbolsOfLCOs{};
llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<object::ELFSymbolRef>>
    LoadedCodeObject::VariableSymbolsOfLCOs{};
llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle), hsa::md::Metadata>
    LoadedCodeObject::MetadataOfLCOs{};

llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<hsa::md::Kernel::Metadata *>>
    LoadedCodeObject::KernelSymbolsMetadataOfLCOs{};

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
  auto &KernelDescELFSymbolsOfThisLCO =
      KernelDescSymbolsOfLCOs.insert({hsaHandle(), {}}).first->second;
  auto &KernelFuncELFSymbolsOfThisLCO =
      KernelFuncSymbolsOfLCOs.insert({hsaHandle(), {}}).first->second;
  auto &VariableELFSymbolsOfThisLCO =
      VariableSymbolsOfLCOs.insert({hsaHandle(), {}}).first->second;
  auto &DeviceFuncELFSymbolsOfThisLCO =
      DeviceFuncSymbolsOfLCOs.insert({hsaHandle(), {}}).first->second;

  LLVM_DEBUG(llvm::dbgs() << llvm::formatv("Caching symbols for LCO {0:x}:\n",
                                           this->hsaHandle()));
  for (const object::ELFSymbolRef &Symbol : CachedELF.symbols()) {
    auto Type = Symbol.getELFType();
    auto Binding = Symbol.getBinding();
    auto SymbolName = Symbol.getName();
    auto Size = Symbol.getSize();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
    LLVM_DEBUG(llvm::dbgs()
               << llvm::formatv("\tSymbol Name: {0}, Binding: {1}, Type: {2}\n",
                                *SymbolName, Binding, Type));
    if (Binding == llvm::ELF::STB_GLOBAL) {
      // Kernel Function Symbol
      if (Type == llvm::ELF::STT_FUNC) {
        KernelFuncELFSymbolsOfThisLCO.insert({*SymbolName, Symbol});
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "\tSymbol {0} is a kernel function.\n", *SymbolName));
      } else if (Type == llvm::ELF::STT_OBJECT) {
        // Kernel Descriptor Symbol
        if (SymbolName->ends_with(".kd") && Size == 64) {
          KernelDescELFSymbolsOfThisLCO.insert({*SymbolName, Symbol});
          LLVM_DEBUG(llvm::dbgs()
                     << llvm::formatv("\tSymbol {0} is a kernel descriptor.\n",
                                      *SymbolName));
        }
        // Variable Symbol
        else {
          VariableELFSymbolsOfThisLCO.insert({*SymbolName, Symbol});
        }
      } else if (Type == llvm::ELF::STT_AMDGPU_HSA_KERNEL && Size == 64) {
        KernelDescELFSymbolsOfThisLCO.insert({*SymbolName, Symbol});
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "\tSymbol {0} is a kernel descriptor.\n", *SymbolName));
      } else if (Type == llvm::ELF::STT_NOTYPE) {
        LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                       "\tSymbol {0} is an external symbol.\n", *SymbolName));
        VariableELFSymbolsOfThisLCO.insert({*SymbolName, Symbol});
      }
    } else if (Binding == llvm::ELF::STB_LOCAL && Type == llvm::ELF::STT_FUNC) {
      DeviceFuncELFSymbolsOfThisLCO.insert({*SymbolName, Symbol});
      LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                     "\tSymbol {0} is a device function.\n", *SymbolName));
    }
  }
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(KernelDescELFSymbolsOfThisLCO.size() ==
                        KernelFuncELFSymbolsOfThisLCO.size()));
  LLVM_DEBUG(
      llvm::dbgs()
      << "Number of kernel function symbols and kernel descriptors match.\n");

  // Cache the LCO and Kernel Symbols Metadata
  auto MetaData = parseNoteMetaData(CachedELF);
  LUTHIER_RETURN_ON_ERROR(MetaData.takeError());
  auto &LCOCachedMetaData =
      MetadataOfLCOs.insert({this->hsaHandle(), *MetaData}).first->getSecond();
  auto &LCOKernelMetaDataMap =
      KernelSymbolsMetadataOfLCOs
          .insert(
              {this->hsaHandle(), llvm::StringMap<md::Kernel::Metadata *>{}})
          .first->getSecond();
  for (auto &KernelMD : LCOCachedMetaData.Kernels) {
    LCOKernelMetaDataMap.insert({KernelMD.Symbol, &KernelMD});
    LLVM_DEBUG(llvm::dbgs() << llvm::formatv(
                   "Metadata for kernel symbol {0}:\n", KernelMD.Symbol));
    LLVM_DEBUG(llvm::dbgs() << KernelMD << "\n");
  }

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
      KernelFuncELFSymbolsOfThisLCO.size() == LCOKernelMetaDataMap.size()));
  LLVM_DEBUG(llvm::dbgs() << "All kernels' Metadata was found\n");

  CachedLCOs.insert(hsaHandle());
  return llvm::Error::success();
}
llvm::Error LoadedCodeObject::invalidate() const {
  std::lock_guard Lock(getCacheMutex());
  StorageELFOfLCOs.erase(this->hsaHandle());
  ISAOfLCOs.erase(this->hsaHandle());
  KernelDescSymbolsOfLCOs.erase(this->hsaHandle());
  KernelFuncSymbolsOfLCOs.erase(this->hsaHandle());
  DeviceFuncSymbolsOfLCOs.erase(this->hsaHandle());
  VariableSymbolsOfLCOs.erase(this->hsaHandle());
  MetadataOfLCOs.erase(this->hsaHandle());
  KernelSymbolsMetadataOfLCOs.erase(this->hsaHandle());
  return llvm::Error::success();
}

bool LoadedCodeObject::isCached() const {
  std::lock_guard Lock(getCacheMutex());
  return CachedLCOs.contains(this->hsaHandle());
}
llvm::Expected<const hsa::md::Metadata &>
LoadedCodeObject::getMetadata() const {
  std::lock_guard Lock(getCacheMutex());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  return MetadataOfLCOs.at(this->hsaHandle());
}
llvm::Error LoadedCodeObject::getKernelSymbols(
    llvm::SmallVectorImpl<hsa::ExecutableSymbol> &Out) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  // Retrieve kernel symbol names associated with this LCO
  const auto &KernelDescSymbolsOfThisLCO =
      KernelDescSymbolsOfLCOs.at(hsaHandle());
  // Construct hsa::ExecutableSymbols from the cached ELF information
  for (const auto &NameWithKDAtTheEnd : KernelDescSymbolsOfThisLCO.keys()) {
    // Append the HSA symbol to the output vector
    auto KernelHSASymbol = constructKernelSymbolUsingName(NameWithKDAtTheEnd);
    LUTHIER_RETURN_ON_ERROR(KernelHSASymbol.takeError());
    Out.push_back(*KernelHSASymbol);
  }
  return llvm::Error::success();
}

llvm::Error LoadedCodeObject::getVariableSymbols(
    llvm::SmallVectorImpl<hsa::ExecutableSymbol> &Out) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  // Retrieve variable symbols associated with this LCO
  const auto &VariableSymbolsOfThisLCO = VariableSymbolsOfLCOs.at(hsaHandle());

  // Construct hsa::ExecutableSymbols from the cached ELF information
  for (const auto &Name : VariableSymbolsOfThisLCO.keys()) {
    auto VariableHSASymbol = constructVariableSymbolUsingName(Name);
    LUTHIER_RETURN_ON_ERROR(VariableHSASymbol.takeError());
    if (VariableHSASymbol->has_value())
      Out.push_back(**VariableHSASymbol);
  }
  return llvm::Error::success();
}

llvm::Expected<hsa::ExecutableSymbol>
LoadedCodeObject::constructKernelSymbolUsingName(
    llvm::StringRef NameWithKDAtTheEnd) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  // Retrieve kernel symbol information associated with this LCO
  const auto &KernelDescSymbolsOfThisLCO =
      KernelDescSymbolsOfLCOs.at(hsaHandle());
  const auto &KernelFuncSymbolsOfThisLCO =
      KernelFuncSymbolsOfLCOs.at(hsaHandle());
  const auto &KernelMetaDataOfThisLCO =
      KernelSymbolsMetadataOfLCOs.at(hsaHandle());

  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
      KernelDescSymbolsOfThisLCO.contains(NameWithKDAtTheEnd)));

  const auto &KDSymbol = KernelDescSymbolsOfThisLCO.at(NameWithKDAtTheEnd);
  auto NameWithoutKDAtTheEnd =
      NameWithKDAtTheEnd.substr(0, NameWithKDAtTheEnd.rfind(".kd"));

  // Check if other maps also have this entry
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
      KernelFuncSymbolsOfThisLCO.contains(NameWithoutKDAtTheEnd)));
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(KernelMetaDataOfThisLCO.contains(NameWithKDAtTheEnd)));

  // Get the function symbol, hsa symbol handle, and the Metadata associated
  // with this kernel
  auto &KFuncSymbol = KernelFuncSymbolsOfThisLCO.at(NameWithoutKDAtTheEnd);
  auto SymbolHandle =
      getHSASymbolHandleByNameFromExecutable(NameWithKDAtTheEnd);
  LUTHIER_RETURN_ON_ERROR(SymbolHandle.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(SymbolHandle->has_value()));
  auto KernelMD = KernelMetaDataOfThisLCO.at(NameWithKDAtTheEnd);

  // Append the HSA symbol to the output vector
  return hsa::ExecutableSymbol{**SymbolHandle, this->asHsaType(), &KDSymbol,
                               &KFuncSymbol, KernelMD};
}

llvm::Expected<std::optional<hsa::ExecutableSymbol>>
LoadedCodeObject::constructVariableSymbolUsingName(llvm::StringRef Name) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  // Retrieve variable symbols associated with this LCO
  const auto &VariableSymbolsOfThisLCO = VariableSymbolsOfLCOs.at(hsaHandle());
  // Check if the Name is in the variable Map
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(VariableSymbolsOfThisLCO.contains(Name)));
  // Get the ELF Symbol
  auto &VariableELFSymbol = VariableSymbolsOfThisLCO.at(Name);
  // Get the HSA handle to the symbol
  auto VariableSymbolHandle = getHSASymbolHandleByNameFromExecutable(Name);
  LUTHIER_RETURN_ON_ERROR(VariableSymbolHandle.takeError());

  // If the variable has no ELF type associated with it, then it is external
  // to this LCO
  if (VariableELFSymbol.getELFType() == llvm::ELF::STT_NOTYPE) {
    // if no HSA handle is found, then the external symbol hasn't been defined
    // yet, and we return nothing
    if (!VariableSymbolHandle->has_value()) {
      return std::nullopt;
    }
    // if HSA is reporting a handle associated with this symbol, then it
    // has been defined, either externally
    // (for example using hsa_executable_agent_global_variable_define), or
    // in another LCO inside the current executable
    else {
      // Check if other LCOs in the current executable have this variable
      // defined
      auto Exec = this->getExecutable();
      LUTHIER_RETURN_ON_ERROR(Exec.takeError());
      auto LCOs = Exec->getLoadedCodeObjects();
      LUTHIER_RETURN_ON_ERROR(LCOs.takeError());
      for (const auto &LCO : *LCOs) {
        if (LCO != *this) {
          const auto &VariableSymbolsOfOtherLCO =
              VariableSymbolsOfLCOs.at(LCO.hsaHandle());
          if (VariableSymbolsOfOtherLCO.contains(Name)) {
            const auto &VariableSymbolELFInOtherLCO =
                VariableSymbolsOfOtherLCO.at(Name);
            if (VariableSymbolELFInOtherLCO.getELFType() !=
                llvm::ELF::STT_NOTYPE) {
              return hsa::ExecutableSymbol{**VariableSymbolHandle,
                                           LCO.asHsaType(), &VariableELFSymbol};
            }
          }
        }
      }
      // We have checked all other LCOs for the parent executable and didn't
      // find any LCOs that has defined this symbol, hence this is symbol
      // is external and doesn't have a parent LCO
      return hsa::ExecutableSymbol{
          **VariableSymbolHandle, {0}, &VariableELFSymbol};
    }
  }
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(VariableSymbolHandle->has_value()));

  return hsa::ExecutableSymbol{**VariableSymbolHandle, this->asHsaType(),
                               &VariableELFSymbol};
}

llvm::Expected<hsa::ExecutableSymbol>
LoadedCodeObject::constructDeviceFunctionSymbolUsingName(
    llvm::StringRef Name) const {
  std::lock_guard Lock(getCacheMutex());
  // Ensure this LCO is cached before doing anything
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(this->isCached()));
  const auto &DeviceFuncSymbolsOfThisLCO =
      DeviceFuncSymbolsOfLCOs.at(hsaHandle());
  // Check if the name is found in the device functions of this LCO
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(DeviceFuncSymbolsOfThisLCO.contains(Name)));
  auto &DeviceFunctionELFSymbol = DeviceFuncSymbolsOfThisLCO.at(Name);
  // return the device function
  return ExecutableSymbol::createDeviceFunctionSymbol(this->asHsaType(),
                                                      DeviceFunctionELFSymbol);
}
llvm::Error LoadedCodeObject::getDeviceFunctionSymbols(
    llvm::SmallVectorImpl<hsa::ExecutableSymbol> &Out) const {
  const auto &DeviceFuncSymbolsOfThisLCO =
      DeviceFuncSymbolsOfLCOs.at(hsaHandle());

  // Device Functions
  for (const auto &Name : DeviceFuncSymbolsOfThisLCO.keys()) {
    auto DeviceFunctionHSASymbol = constructDeviceFunctionSymbolUsingName(Name);
    LUTHIER_RETURN_ON_ERROR(DeviceFunctionHSASymbol.takeError());
    Out.push_back(*DeviceFunctionHSASymbol);
  }
  return llvm::Error::success();
}

} // namespace luthier::hsa
