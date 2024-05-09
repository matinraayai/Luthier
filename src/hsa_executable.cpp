#include "hsa_executable.hpp"

#include <llvm/BinaryFormat/ELF.h>

#include <unordered_set>

#include "error.hpp"
#include "hsa_agent.hpp"
#include "hsa_code_object_reader.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_loaded_code_object.hpp"
#include "object_utils.hpp"

namespace luthier::hsa {

llvm::Expected<Executable>
Executable::create(hsa_profile_t Profile,
                   hsa_default_float_rounding_mode_t DefaultFloatRoundingMode,
                   const char *Options) {
  hsa_executable_t Exec;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(hsa_executable_create_alt(
      Profile, DefaultFloatRoundingMode, Options, &Exec)));

  return Executable{Exec};
}

llvm::Error Executable::freeze(const char *Options) {
  return LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_freeze_fn(asHsaType(), Options));
}

Executable::Executable(hsa_executable_t Exec)
    : HandleType<hsa_executable_t>(Exec) {}

llvm::Expected<hsa_profile_t> Executable::getProfile() {
  hsa_profile_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_PROFILE, &Out)));
  return Out;
}

llvm::Expected<hsa_executable_state_t> Executable::getState() const {
  hsa_executable_state_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_STATE, &Out)));
  return Out;
}

llvm::Expected<hsa_default_float_rounding_mode_t>
Executable::getRoundingMode() {
  hsa_default_float_rounding_mode_t Out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &Out)));
  return Out;
}

llvm::Expected<std::vector<ExecutableSymbol>>
Executable::getAgentSymbols(const GpuAgent &Agent) const {
  std::vector<ExecutableSymbol> Out;
  auto Iterator = [](hsa_executable_t Exec, hsa_agent_t Agent,
                     hsa_executable_symbol_t Symbol, void *Data) {
    auto Out = reinterpret_cast<std::vector<ExecutableSymbol> *>(Data);
    Out->push_back(ExecutableSymbol::fromHandle(Symbol));
    return HSA_STATUS_SUCCESS;
  };

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_iterate_agent_symbols_fn(
          asHsaType(), Agent.asHsaType(), Iterator, &Out)));

  auto LoadedCodeObjects = getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LoadedCodeObjects.takeError());

  for (const auto &LCO : *LoadedCodeObjects) {
    auto LCOAgent = LCO.getAgent();
    LUTHIER_RETURN_ON_ERROR(LCOAgent.takeError());
    if (*LCOAgent == Agent) {
      auto LoadedMemory = LCO.getLoadedMemory();
      LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

      auto HostElf = LCO.getStorageELF();
      LUTHIER_RETURN_ON_ERROR(HostElf.takeError());

      for (llvm::object::ELFSymbolRef ElfSymbol : HostElf->symbols()) {
        auto SymbolType = ElfSymbol.getELFType();
        auto Binding = ElfSymbol.getBinding();
        // Indirect functions have a local binding
        if (SymbolType == llvm::ELF::STT_FUNC &&
            Binding == llvm::ELF::STB_LOCAL) {
          auto SymbolName = ElfSymbol.getName();
          LUTHIER_RETURN_ON_ERROR(SymbolName.takeError());
          auto LoadedAddress = getSymbolLMA(HostElf->getELFFile(), ElfSymbol);
          LUTHIER_RETURN_ON_ERROR(LoadedAddress.takeError());
          Out.emplace_back(
              std::string(*SymbolName),
              arrayRefFromStringRef(
                  toStringRef(*LoadedMemory)
                      .substr(*LoadedAddress, ElfSymbol.getSize())),
              LCO);
        }
      }
    }
  }
  return Out;
}

llvm::Expected<std::vector<ExecutableSymbol>>
Executable::getProgramSymbols(const GpuAgent &Agent) const {
  std::vector<ExecutableSymbol> Out;
  auto Iterator = [](hsa_executable_t Exec, hsa_executable_symbol_t Symbol,
                     void *Data) {
    auto Out = reinterpret_cast<std::vector<ExecutableSymbol> *>(Data);
    Out->push_back(ExecutableSymbol::fromHandle(Symbol));
    return HSA_STATUS_SUCCESS;
  };

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_iterate_program_symbols_fn(
          asHsaType(), Iterator, &Out)));
  return Out;
}

llvm::Expected<std::optional<ExecutableSymbol>>
Executable::getAgentSymbolByName(const GpuAgent &Agent,
                                 llvm::StringRef Name) const {
  hsa_executable_symbol_t Symbol;
  hsa_agent_t HsaAgent = Agent.asHsaType();

  auto Status = getApiTable().core.hsa_executable_get_symbol_by_name_fn(
      this->asHsaType(), Name.str().c_str(), &HsaAgent, &Symbol);
  if (Status == HSA_STATUS_SUCCESS)
    return ExecutableSymbol::fromHandle(Symbol);
  else if (Status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME) {
    // Try again, this time with ".kd" attached to the name
    Status = getApiTable().core.hsa_executable_get_symbol_by_name_fn(
        this->asHsaType(), (Name + ".kd").str().c_str(), &HsaAgent, &Symbol);
    if (Status == HSA_STATUS_SUCCESS)
      return ExecutableSymbol::fromHandle(Symbol);
    // Possible indirect function symbol
    else if (Status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME) {
      auto LoadedCodeObjects = getLoadedCodeObjects();
      LUTHIER_RETURN_ON_ERROR(LoadedCodeObjects.takeError());
      for (const auto &LCO : *LoadedCodeObjects) {
        auto StorageMemory = LCO.getStorageMemory();
        LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());
        auto LoadedMemory = LCO.getLoadedMemory();
        LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());

        auto HostELF = LCO.getStorageELF();
        LUTHIER_RETURN_ON_ERROR(HostELF.takeError());

        auto ELFSymbol = luthier::getSymbolByName(*HostELF, Name);
        LUTHIER_RETURN_ON_ERROR(ELFSymbol.takeError());

        if (ELFSymbol->has_value()) {
          auto Type = ELFSymbol.get()->getELFType();
          auto Binding = ELFSymbol.get()->getBinding();

          if (!(Type == llvm::ELF::STT_FUNC &&
                Binding == llvm::ELF::STB_LOCAL)) {
            // This is not an indirect function, return
            return std::nullopt;
          }

          auto LoadedAddress = getSymbolLMA(HostELF->getELFFile(), **ELFSymbol);
          LUTHIER_RETURN_ON_ERROR(LoadedAddress.takeError());

          return ExecutableSymbol{
              std::string(Name),
              arrayRefFromStringRef(
                  toStringRef(*LoadedMemory)
                      .substr(*LoadedAddress, ELFSymbol.get()->getSize())),
              LCO};
        }
      }
    }
  }
  return std::nullopt;
}

llvm::Expected<std::vector<LoadedCodeObject>>
Executable::getLoadedCodeObjects() const {
  std::vector<LoadedCodeObject> LoadedCodeObjects;
  auto Iterator = [](hsa_executable_t Exec, hsa_loaded_code_object_t LCO,
                     void *Data) -> hsa_status_t {
    auto Out = reinterpret_cast<std::vector<LoadedCodeObject> *>(Data);
    Out->emplace_back(LCO);
    return HSA_STATUS_SUCCESS;
  };
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable()
          .hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
              this->asHsaType(), Iterator, &LoadedCodeObjects)));
  return LoadedCodeObjects;
}

llvm::Expected<llvm::DenseSet<hsa::GpuAgent>> Executable::getAgents() const {
  auto LoadedCodeObjects = getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LoadedCodeObjects.takeError());
  llvm::DenseSet<hsa::GpuAgent> Agents;
  for (const auto &LCO : *LoadedCodeObjects) {
    hsa_agent_t Agent;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
            LCO.asHsaType(), HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
            &Agent)));
    Agents.insert(hsa::GpuAgent{Agent});
  }
  return Agents;
}

llvm::Expected<hsa::LoadedCodeObject>
Executable::loadAgentCodeObject(const hsa::CodeObjectReader &Reader,
                                const hsa::GpuAgent &Agent,
                                llvm::StringRef Options) {
  hsa_loaded_code_object_t LCO;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_load_agent_code_object_fn(
          asHsaType(), Agent.asHsaType(), Reader.asHsaType(), nullptr, &LCO)));
  return hsa::LoadedCodeObject(LCO);
}

llvm::Expected<hsa::LoadedCodeObject>
Executable::loadProgramCodeObject(const CodeObjectReader &Reader,
                                  llvm::StringRef Options) {
  hsa_loaded_code_object_t LCO;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_load_program_code_object_fn(
          asHsaType(), Reader.asHsaType(), Options.data(), &LCO)));
  return hsa::LoadedCodeObject(LCO);
}

llvm::Error Executable::defineExternalProgramGlobalVariable(
    const ExecutableSymbol &Symbol) {
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(hsa_symbol_kind_t, SymbolType,
                                   Symbol.getType());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(SymbolType == HSA_SYMBOL_KIND_VARIABLE));

  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(luthier::address_t, VariableAddress,
                                   Symbol.getVariableAddress());
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(std::string, VariableName, Symbol.getName());

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_global_variable_define_fn(
          asHsaType(), VariableName.c_str(),
          reinterpret_cast<void *>(VariableAddress))));
  return llvm::Error::success();
}

llvm::Error
Executable::defineExternalAgentGlobalVariable(const ExecutableSymbol &Symbol) {
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(hsa_symbol_kind_t, SymbolType,
                                   Symbol.getType());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(SymbolType == HSA_SYMBOL_KIND_VARIABLE));

  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(luthier::address_t, VariableAddress,
                                   Symbol.getVariableAddress());
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(std::string, VariableName, Symbol.getName());

  auto Agent = Symbol.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_agent_global_variable_define_fn(
          asHsaType(), Agent->asHsaType(), VariableName.c_str(),
          reinterpret_cast<void *>(
              *reinterpret_cast<uint64_t *>(VariableAddress)))));
  return llvm::Error::success();
}

llvm::Error
Executable::defineAgentReadOnlyVariable(const hsa::ExecutableSymbol &Symbol) {
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(hsa_symbol_kind_t, SymbolType,
                                   Symbol.getType());
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_ASSERTION(SymbolType == HSA_SYMBOL_KIND_VARIABLE));

  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(luthier::address_t, VariableAddress,
                                   Symbol.getVariableAddress());
  LUTHIER_RETURN_ON_MOVE_INTO_FAIL(std::string, VariableName, Symbol.getName());

  auto Agent = Symbol.getAgent();
  LUTHIER_RETURN_ON_ERROR(Agent.takeError());

  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_readonly_variable_define_fn(
          asHsaType(), Agent->asHsaType(), VariableName.c_str(),
          reinterpret_cast<void *>(VariableAddress))));
  return llvm::Error::success();
}

llvm::Error Executable::destroy() {
  return LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_destroy_fn(asHsaType()));
}
llvm::Expected<bool> Executable::validate(llvm::StringRef Options) {
  uint32_t Result;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_validate_alt_fn(
          asHsaType(), Options.data(), &Result)));
  return Result == 0;
}
llvm::Error Executable::cache() const { return llvm::Error::success(); }

llvm::Error Executable::invalidate() const { return llvm::Error::success(); }

} // namespace luthier::hsa
