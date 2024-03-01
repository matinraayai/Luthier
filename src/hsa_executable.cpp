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

llvm::Expected<Executable> Executable::create(
    hsa_profile_t profile,
    hsa_default_float_rounding_mode_t default_float_rounding_mode,
    const char *options) {
  hsa_executable_t executable;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(hsa_executable_create_alt(
      profile, default_float_rounding_mode, options, &executable)));

  return Executable{executable};
}

llvm::Error Executable::freeze(const char *options) {
  return LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_freeze_fn(asHsaType(), options));
}

Executable::Executable(hsa_executable_t executable)
    : HandleType<hsa_executable_t>(executable) {}

llvm::Expected<hsa_profile_t> Executable::getProfile() {
  hsa_profile_t out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_PROFILE, &out)));
  return out;
}

llvm::Expected<hsa_executable_state_t> Executable::getState() {
  hsa_executable_state_t out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_STATE, &out)));
  return out;
}

llvm::Expected<hsa_default_float_rounding_mode_t>
Executable::getRoundingMode() {
  hsa_default_float_rounding_mode_t out;
  LUTHIER_RETURN_ON_ERROR(
      LUTHIER_HSA_SUCCESS_CHECK(getApiTable().core.hsa_executable_get_info_fn(
          asHsaType(), HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &out)));
  return out;
}

llvm::Expected<std::vector<ExecutableSymbol>>
Executable::getSymbols(const GpuAgent &agent) const {
  std::vector<ExecutableSymbol> Out;
  auto Iterator = [](hsa_executable_t exec, hsa_agent_t agent,
                     hsa_executable_symbol_t symbol, void *data) {
    auto out = reinterpret_cast<std::vector<ExecutableSymbol> *>(data);
    out->emplace_back(symbol, agent, exec);
    return HSA_STATUS_SUCCESS;
  };
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_iterate_agent_symbols_fn(
          asHsaType(), agent.asHsaType(), Iterator, &Out)));

  std::unordered_set<std::string> kernelNames;
  for (const auto &s : Out) {
    auto type = s.getType();
    LUTHIER_RETURN_ON_ERROR(type.takeError());
    if (*type == HSA_SYMBOL_KIND_KERNEL) {
      auto sName = s.getName();
      LUTHIER_RETURN_ON_ERROR(sName.takeError());
      kernelNames.insert(*sName);
      kernelNames.insert(sName->substr(0, sName->find(".kd")));
    }
  }
  auto LoadedCodeObjects = getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(LoadedCodeObjects.takeError());
  for (const auto &lco : *LoadedCodeObjects) {
    auto LCOAgent = lco.getAgent();
    LUTHIER_RETURN_ON_ERROR(LCOAgent.takeError());
    if (*LCOAgent == agent) {
      auto LCOStorageType = lco.getStorageType();
      LUTHIER_RETURN_ON_ERROR(LCOStorageType.takeError());
      LUTHIER_RETURN_ON_ERROR(LUTHIER_ASSERTION(
          *LCOStorageType ==
          HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY));
      auto StorageMemory = lco.getStorageMemory();
      LUTHIER_RETURN_ON_ERROR(StorageMemory.takeError());
      auto LoadedMemory = lco.getLoadedMemory();
      LUTHIER_RETURN_ON_ERROR(LoadedMemory.takeError());
      auto hostElfOrError = getELFObjectFileBase(*StorageMemory);
      LUTHIER_RETURN_ON_ERROR(hostElfOrError.takeError());
      auto HostElf = hostElfOrError->get();

      auto Syms = HostElf->symbols();
      for (llvm::object::ELFSymbolRef ElfSymbol : Syms) {
        auto typeOrError = ElfSymbol.getELFType();
        auto nameOrError = ElfSymbol.getName();
        LUTHIER_RETURN_ON_ERROR(nameOrError.takeError());
        auto name = std::string(nameOrError.get());
        auto addressOrError = ElfSymbol.getAddress();
        LUTHIER_RETURN_ON_ERROR(addressOrError.takeError());
        if (typeOrError == llvm::ELF::STT_FUNC &&
            !kernelNames.contains(name)) {
          Out.emplace_back(
              name,
              arrayRefFromStringRef(
                  toStringRef(*LoadedMemory)
                      .substr(addressOrError.get(), ElfSymbol.getSize())),
              agent.asHsaType(), this->asHsaType());
        }
      }
    }
  }
  return Out;
}

llvm::Expected<std::optional<ExecutableSymbol>>
Executable::getSymbolByName(const luthier::hsa::GpuAgent &agent,
                            const std::string &name) const {
  hsa_executable_symbol_t Symbol;
  hsa_agent_t HsaAgent = agent.asHsaType();

  auto Status = getApiTable().core.hsa_executable_get_symbol_by_name_fn(
      this->asHsaType(), name.c_str(), &HsaAgent, &Symbol);
  if (Status == HSA_STATUS_SUCCESS)
    return ExecutableSymbol{Symbol, agent.asHsaType(), this->asHsaType()};
  else if (Status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME) {
    auto LoadedCodeObjects = getLoadedCodeObjects();
    LUTHIER_RETURN_ON_ERROR(LoadedCodeObjects.takeError());
    for (const auto &lco : *LoadedCodeObjects) {
      auto storageMemory = lco.getStorageMemory();
      LUTHIER_RETURN_ON_ERROR(storageMemory.takeError());
      auto loadedMemory = lco.getLoadedMemory();
      LUTHIER_RETURN_ON_ERROR(loadedMemory.takeError());

      auto hostElfOrError = getELFObjectFileBase(*storageMemory);
      LUTHIER_RETURN_ON_ERROR(hostElfOrError.takeError());

      auto hostElf = hostElfOrError->get();
      // TODO: Replace this with a hash lookup
      auto Syms = hostElf->symbols();
      for (llvm::object::ELFSymbolRef elfSymbol : Syms) {
        auto nameOrError = elfSymbol.getName();
        LUTHIER_RETURN_ON_ERROR(nameOrError.takeError());
        if (nameOrError.get() == name) {
          auto addressOrError = elfSymbol.getAddress();
          LUTHIER_RETURN_ON_ERROR(addressOrError.takeError());

          return ExecutableSymbol{
              name,
              arrayRefFromStringRef(
                  toStringRef(*loadedMemory)
                      .substr(addressOrError.get(), elfSymbol.getSize())),
              agent.asHsaType(), this->asHsaType()};
        }
      }
    }
    return std::nullopt;
  } else {
    return std::nullopt;
  }
}

llvm::Expected<std::vector<LoadedCodeObject>>
Executable::getLoadedCodeObjects() const {
  std::vector<LoadedCodeObject> loadedCodeObjects;
  auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco,
                     void *data) -> hsa_status_t {
    auto out = reinterpret_cast<std::vector<LoadedCodeObject> *>(data);
    out->emplace_back(lco);
    return HSA_STATUS_SUCCESS;
  };
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getLoaderTable()
          .hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
              this->asHsaType(), iterator, &loadedCodeObjects)));
  return loadedCodeObjects;
}
llvm::Expected<std::vector<hsa::GpuAgent>> Executable::getAgents() const {
  auto loadedCodeObjects = getLoadedCodeObjects();
  LUTHIER_RETURN_ON_ERROR(loadedCodeObjects.takeError());
  std::vector<hsa::GpuAgent> agents;
  for (const auto &lco : *loadedCodeObjects) {
    hsa_agent_t agent;
    LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
        getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
            lco.asHsaType(), HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
            &agent)));
    agents.emplace_back(agent);
  }
  return agents;
}

llvm::Expected<hsa::LoadedCodeObject>
Executable::loadCodeObject(hsa::CodeObjectReader reader, hsa::GpuAgent agent) {
  hsa_loaded_code_object_t lco;
  LUTHIER_RETURN_ON_ERROR(LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_load_agent_code_object_fn(
          asHsaType(), agent.asHsaType(), reader.asHsaType(), nullptr, &lco)));
  return hsa::LoadedCodeObject(lco);
}

llvm::Error Executable::destroy() {
  return LUTHIER_HSA_SUCCESS_CHECK(
      getApiTable().core.hsa_executable_destroy_fn(asHsaType()));
}

} // namespace luthier::hsa
