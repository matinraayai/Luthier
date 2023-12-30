#include "hsa_executable.hpp"

#include <elfio/elf_types.hpp>
#include <unordered_set>

#include "code_view.hpp"
#include "error.h"
#include "hsa_agent.hpp"
#include "hsa_code_object_reader.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_loaded_code_object.hpp"

namespace luthier::hsa {

Executable Executable::create(hsa_profile_t profile, hsa_default_float_rounding_mode_t default_float_rounding_mode,
                              const char *options) {
    hsa_executable_t executable;
    LUTHIER_HSA_CHECK(hsa_executable_create_alt(profile, default_float_rounding_mode, options, &executable));

    return Executable{executable};
}

void Executable::freeze(const char *options) {
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_freeze_fn(asHsaType(), options));
}

Executable::Executable(hsa_executable_t executable) : HandleType<hsa_executable_t>(executable) {}

hsa_profile_t Executable::getProfile() {
    hsa_profile_t out;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_get_info_fn(asHsaType(), HSA_EXECUTABLE_INFO_PROFILE, &out));
    return out;
}

hsa_executable_state_t Executable::getState() {
    hsa_executable_state_t out;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_get_info_fn(asHsaType(), HSA_EXECUTABLE_INFO_STATE, &out));
    return out;
}

hsa_default_float_rounding_mode_t Executable::getRoundingMode() {
    hsa_default_float_rounding_mode_t out;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_get_info_fn(
        asHsaType(), HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &out));
    return out;
}

std::vector<ExecutableSymbol> Executable::getSymbols(const GpuAgent &agent) const {
    std::vector<ExecutableSymbol> out;
    auto iterator = [](hsa_executable_t exec, hsa_agent_t agent, hsa_executable_symbol_t symbol, void *data) {
        auto out = reinterpret_cast<std::vector<ExecutableSymbol> *>(data);
        out->emplace_back(symbol, agent, exec);
        return HSA_STATUS_SUCCESS;
    };
    LUTHIER_HSA_CHECK(
        getApiTable().core.hsa_executable_iterate_agent_symbols_fn(asHsaType(), agent.asHsaType(), iterator, &out));

    std::unordered_set<std::string> kernelNames;
    for (const auto &s: out) {
        if (s.getType() == HSA_SYMBOL_KIND_KERNEL) {
            std::string sName = s.getName();
            kernelNames.insert(sName);
            kernelNames.insert(sName.substr(0, sName.find(".kd")));
        }
    }
    for (const auto &lco: this->getLoadedCodeObjects()) {
        if (lco.getAgent() == agent) {
            if (lco.getStorageType() == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
                auto storageMemory = lco.getStorageMemory();
                auto loadedMemory = lco.getLoadedMemory();
                auto reader = code::ElfView::makeView(storageMemory);
                for (unsigned int j = 0; j < reader->getNumSymbols(); j++) {
                    auto info = reader->getSymbol(j);
                    if (info->getType() == ELFIO::STT_FUNC && !kernelNames.contains(info->getName())) {
                        auto storageAddressOffset =
                            info->getAddress() - reinterpret_cast<luthier_address_t>(storageMemory.data());
                        auto symbolSize = info->getSize();
                        out.emplace_back(info->getName(), loadedMemory.substr(storageAddressOffset, symbolSize),
                                         agent.asHsaType(), this->asHsaType());
                    }
                }
            }
        }
    }
    return out;
}

std::optional<ExecutableSymbol> Executable::getSymbolByName(const luthier::hsa::GpuAgent &agent,
                                                            const std::string &name) const {
    hsa_executable_symbol_t symbol;
    hsa_agent_t hsaAgent = agent.asHsaType();

    auto status =
        getApiTable().core.hsa_executable_get_symbol_by_name_fn(this->asHsaType(), name.c_str(), &hsaAgent, &symbol);
    if (status == HSA_STATUS_SUCCESS) return ExecutableSymbol{symbol, agent.asHsaType(), this->asHsaType()};
    else if (status == HSA_STATUS_ERROR_INVALID_SYMBOL_NAME) {
        for (const auto &lco: getLoadedCodeObjects()) {
            auto storageMemory = lco.getStorageMemory();
            auto s = code::ElfView::makeView(storageMemory)->getSymbol(name);
            if (s.has_value()) {
                auto loadedMemory = lco.getLoadedMemory();
                auto storageAddressOffset = s->getSection()->get_address();
                return ExecutableSymbol{s->getName(), loadedMemory.substr(storageAddressOffset, storageAddressOffset),
                                        agent.asHsaType(), this->asHsaType()};
            }
        }
        return std::nullopt;
    } else {
        return std::nullopt;
    }
}

std::vector<LoadedCodeObject> Executable::getLoadedCodeObjects() const {
    std::vector<LoadedCodeObject> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void *data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<LoadedCodeObject> *>(data);
        out->emplace_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
        this->asHsaType(), iterator, &loadedCodeObjects));
    return loadedCodeObjects;
}
std::vector<hsa::GpuAgent> Executable::getAgents() const {
    auto loadedCodeObjects = getLoadedCodeObjects();
    std::vector<hsa::GpuAgent> agents;
    for (const auto &lco: loadedCodeObjects) {
        hsa_agent_t agent;
        LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(
            lco.asHsaType(), HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT, &agent));
        agents.emplace_back(agent);
    }
    return agents;
}
hsa::LoadedCodeObject Executable::loadCodeObject(hsa::CodeObjectReader reader, hsa::GpuAgent agent) {
    hsa_loaded_code_object_t lco;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_load_agent_code_object_fn(asHsaType(), agent.asHsaType(),
                                                                                  reader.asHsaType(), nullptr, &lco));
    return hsa::LoadedCodeObject(lco);
}
void Executable::destroy() { LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_destroy_fn(asHsaType())); }

}// namespace luthier::hsa
