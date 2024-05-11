#include "hsa_loaded_code_object.hpp"

#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_intercept.hpp"
#include "hsa_platform.hpp"
#include "object_utils.hpp"
#include <llvm/ADT/StringMap.h>

namespace object = llvm::object;

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

llvm::Expected<hsa_ven_amd_loader_code_object_storage_type_t>
LoadedCodeObject::getStorageType() const {
  hsa_ven_amd_loader_code_object_storage_type_t StorageType;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
          &StorageType)));
  return StorageType;
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

llvm::Expected<int> LoadedCodeObject::getStorageFile() const {
  int FD;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
          this->asHsaType(),
          HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE,
          &FD)));
  return FD;
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

llvm::Expected<llvm::object::ELF64LEObjectFile &>
LoadedCodeObject::getStorageELF() const {
  std::lock_guard Lock(getCacheMutex());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(StorageELFOfLCOs.contains(hsaHandle())));
  return *StorageELFOfLCOs.at(hsaHandle());
}

llvm::Expected<ISA> LoadedCodeObject::getISA() const {
  std::lock_guard Lock(getCacheMutex());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(StorageELFOfLCOs.contains(hsaHandle())));
  return ISA(ISAOfLCOs.at(hsaHandle()));
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
    LoadedCodeObject::KernelDescSymbols{};
llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<object::ELFSymbolRef>>
    LoadedCodeObject::KernelFuncSymbols{};
llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<object::ELFSymbolRef>>
    LoadedCodeObject::DeviceFuncSymbols{};
llvm::DenseMap<decltype(hsa_loaded_code_object_t::handle),
               llvm::StringMap<object::ELFSymbolRef>>
    LoadedCodeObject::VariableSymbols{};

llvm::Error LoadedCodeObject::cache() const {
  std::lock_guard Lock(getCacheMutex());
  // Cache the Storage ELF
  llvm::outs() << "getting the memory and stuff\n";
  auto StorageMemory = this->getStorageMemory();
  LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());
  auto StorageELF = getAMDGCNObjectFile(*StorageMemory);
  LUTHIER_RETURN_ON_ERROR(StorageELF.takeError());
  llvm::outs() << "Caching the elf worked\n";
  auto &CachedELF =
      *(StorageELFOfLCOs.insert({this->hsaHandle(), std::move(*StorageELF)})
            .first->second);
  llvm::outs() << "Getting the ISA\n";
  // Cache the ISA of the ELF
  llvm::Triple TT = CachedELF.makeTriple();
  std::optional<llvm::StringRef> CPU = CachedELF.tryGetCPUName();
  LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(CPU.has_value()));
  llvm::SubtargetFeatures Features;
  LUTHIER_RETURN_ON_ERROR(CachedELF.getFeatures().moveInto(Features));
  auto ISA = hsa::ISA::fromLLVM(TT, *CPU, Features);
  LUTHIER_RETURN_ON_ERROR(ISA.takeError());
  ISAOfLCOs.insert({this->hsaHandle(), ISA->asHsaType()});
  llvm::outs() << "ISA works\n";
  // Cache the ELF Symbols
  auto &KernelDescSymbolsOfLCO =
      KernelDescSymbols.insert({hsaHandle(), {}}).first->second;
  auto &KernelFuncSymbolsOfLCO =
      KernelFuncSymbols.insert({hsaHandle(), {}}).first->second;
  auto &VariableSymbolsOfLCO =
      VariableSymbols.insert({hsaHandle(), {}}).first->second;
  auto &DeviceFuncSymbolsOfLCO =
      DeviceFuncSymbols.insert({hsaHandle(), {}}).first->second;

  for (const object::ELFSymbolRef &Symbol : CachedELF.symbols()) {
    auto Type = Symbol.getELFType();
    auto Binding = Symbol.getBinding();
    auto SymbolName = Symbol.getName();
    auto Size = Symbol.getSize();
    LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
    llvm::outs() << "Symbol Name : " << *SymbolName << "\n";
    llvm::outs() << "Binding: " << int(Binding) << "\n";
    llvm::outs() << "Type: " << int(Type) << "\n";
    if (Binding == llvm::ELF::STB_GLOBAL) {
      // Kernel Function Symbol
      if (Type == llvm::ELF::STT_FUNC) {
        KernelFuncSymbolsOfLCO.insert({*SymbolName, Symbol});
        llvm::outs() << "Found kernel function\n";
      } else if (Type == llvm::ELF::STT_OBJECT) {
        // Kernel Descriptor Symbol
        if (SymbolName->ends_with(".kd") && Size == 64) {
          KernelDescSymbolsOfLCO.insert({*SymbolName, Symbol});
          llvm::outs() << "found a kernel!\n";
        }
        // Variable Symbol
        else {
          VariableSymbolsOfLCO.insert({*SymbolName, Symbol});
        }
      } else if (Type == llvm::ELF::STT_AMDGPU_HSA_KERNEL && Size == 64) {
        KernelDescSymbolsOfLCO.insert({*SymbolName, Symbol});
      } else if (Type == llvm::ELF::STT_NOTYPE) {
        llvm::outs() << "Found an external Symbol\n";
        VariableSymbolsOfLCO.insert({*SymbolName, Symbol});
      }
    } else if (Binding == llvm::ELF::STB_LOCAL && Type == llvm::ELF::STT_FUNC) {
      DeviceFuncSymbolsOfLCO.insert({*SymbolName, Symbol});
      llvm::outs() << "Found device function\n";
    }
  }
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(KernelDescSymbols.size() == KernelFuncSymbols.size()));
  CachedLCOs.insert(hsaHandle());
  return llvm::Error::success();
}
llvm::Error LoadedCodeObject::invalidate() const {
  std::lock_guard Lock(getCacheMutex());
  StorageELFOfLCOs.erase(this->hsaHandle());
  ISAOfLCOs.erase(this->hsaHandle());
  KernelDescSymbols.erase(this->hsaHandle());
  KernelFuncSymbols.erase(this->hsaHandle());
  DeviceFuncSymbols.erase(this->hsaHandle());
  VariableSymbols.erase(this->hsaHandle());
  return llvm::Error::success();
}
llvm::Expected<std::vector<ExecutableSymbol>>
LoadedCodeObject::getExecutableSymbols() const {
  std::lock_guard Lock(getCacheMutex());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(StorageELFOfLCOs.contains(hsaHandle())));
  auto Agent = this->getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());
  auto Exec = this->getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());
  const auto &KernelDescSymbolsOfLCO = KernelDescSymbols.at(hsaHandle());
  const auto &KernelFuncSymbolsOfLCO = KernelFuncSymbols.at(hsaHandle());
  const auto &VariableSymbolsOfLCO = VariableSymbols.at(hsaHandle());
  const auto &DeviceFuncSymbolsOfLCO = DeviceFuncSymbols.at(hsaHandle());

  std::vector<ExecutableSymbol> Out;
  Out.reserve(KernelDescSymbolsOfLCO.size() + VariableSymbolsOfLCO.size() +
              DeviceFuncSymbolsOfLCO.size());
  // Kernels
  for (const auto &[NameWithKD, KDSymbol] : KernelDescSymbolsOfLCO) {
    llvm::outs() << "Returning Kernel name: " << NameWithKD << "\n";
    auto NameWithoutKD = NameWithKD.substr(0, NameWithKD.rfind(".kd"));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
        KernelFuncSymbols.at(hsaHandle()).contains(NameWithoutKD)));
    auto KFuncSymbol = KernelFuncSymbolsOfLCO.at(NameWithoutKD);
    auto KernelSymbolHandle = getSymbolByNameFromExecutable(NameWithKD);
    LUTHIER_RETURN_ON_ERROR(KernelSymbolHandle.takeError());
    Out.emplace_back(*KernelSymbolHandle, this->asHsaType(), KDSymbol,
                     KFuncSymbol);
  }
  // Variables
  for (const auto &[Name, Sym] : VariableSymbolsOfLCO) {
    llvm::outs() << "Returning Variable name: " << Name << "\n";
    auto VariableSymbolHandle = getSymbolByNameFromExecutable(Name);
    LUTHIER_RETURN_ON_ERROR(VariableSymbolHandle.takeError());
    Out.emplace_back(*VariableSymbolHandle, this->asHsaType(), Sym);
  }
  // Device Functions
  for (const auto &[Name, Sym] : DeviceFuncSymbolsOfLCO) {
    llvm::outs() << "Returning Device function name: " << Name << "\n";
    auto Symbol =
        ExecutableSymbol::createDeviceFunctionSymbol(this->asHsaType(), Sym);
    LUTHIER_RETURN_ON_ERROR(Symbol.takeError());
    Out.push_back(*Symbol);
  }
  return Out;
}
bool LoadedCodeObject::isCached() const {
  std::lock_guard Lock(getCacheMutex());
  return CachedLCOs.contains(this->hsaHandle());
}
llvm::Expected<std::optional<ExecutableSymbol>>
LoadedCodeObject::getExecutableSymbolByName(llvm::StringRef Name) const {
  std::lock_guard Lock(getCacheMutex());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(StorageELFOfLCOs.contains(hsaHandle())));
  auto Agent = this->getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());
  auto Exec = this->getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());
  const auto &KernelDescSymbolsOfLCO = KernelDescSymbols.at(hsaHandle());
  const auto &KernelFuncSymbolsOfLCO = KernelFuncSymbols.at(hsaHandle());
  const auto &VariableSymbolsOfLCO = VariableSymbols.at(hsaHandle());
  const auto &DeviceFuncSymbolsOfLCO = DeviceFuncSymbols.at(hsaHandle());
  // Check if name is a kernel
  if (KernelDescSymbolsOfLCO.contains(Name)) {
    auto NameWithoutKD = Name.substr(0, Name.rfind(".kd"));
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
        KernelFuncSymbols.at(hsaHandle()).contains(NameWithoutKD)));
    auto KDescSymbol = KernelDescSymbolsOfLCO.at(Name);
    auto KFuncSymbol = KernelFuncSymbolsOfLCO.at(NameWithoutKD);
    auto KernelSymbolHandle = getSymbolByNameFromExecutable(Name);
    LUTHIER_RETURN_ON_ERROR(KernelSymbolHandle.takeError());
    return ExecutableSymbol{*KernelSymbolHandle, this->asHsaType(), KDescSymbol,
                            KFuncSymbol};
  } else if (KernelFuncSymbolsOfLCO.contains(Name)) {
    auto NameWithKD = (Name + ".kd").str();
    auto KDescSymbol = KernelDescSymbolsOfLCO.at(NameWithKD);
    auto KFuncSymbol = KernelFuncSymbolsOfLCO.at(Name);
    auto KernelSymbolHandle = getSymbolByNameFromExecutable(NameWithKD);
    LUTHIER_RETURN_ON_ERROR(KernelSymbolHandle.takeError());
    return ExecutableSymbol{*KernelSymbolHandle, this->asHsaType(), KDescSymbol,
                            KFuncSymbol};

  } else if (VariableSymbolsOfLCO.contains(Name)) {
    auto VarSym = VariableSymbolsOfLCO.at(Name);
    auto VariableSymbolHandle = getSymbolByNameFromExecutable(Name);
    LUTHIER_RETURN_ON_ERROR(VariableSymbolHandle.takeError());
    return ExecutableSymbol{*VariableSymbolHandle, this->asHsaType(), VarSym};
  } else if (DeviceFuncSymbolsOfLCO.contains(Name)) {
    auto DevFuncSym = DeviceFuncSymbolsOfLCO.at(Name);
    auto Symbol = ExecutableSymbol::createDeviceFunctionSymbol(
        this->asHsaType(), DevFuncSym);
    LUTHIER_RETURN_ON_ERROR(Symbol.takeError());
    return *Symbol;
  } else
    return std::nullopt;
}

llvm::Expected<hsa_executable_symbol_t>
LoadedCodeObject::getSymbolByNameFromExecutable(llvm::StringRef Name) const {
  hsa_executable_symbol_t Out;
  auto Agent = this->getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());
  auto AgentHsaHandle = Agent->asHsaType();
  auto Exec = this->getExecutable();
  LUTHIER_RETURN_ON_ERROR(Exec.takeError());
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_get_symbol_by_name_fn(
          Exec->asHsaType(), Name.data(), &AgentHsaHandle, &Out)));
  return Out;
}

} // namespace luthier::hsa
