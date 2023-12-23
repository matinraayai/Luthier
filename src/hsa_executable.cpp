#include "hsa_executable.hpp"
#include "error.h"
#include "hsa_agent.hpp"

namespace luthier::hsa {

Executable::Executable(hsa_profile_t profile,
                       hsa_default_float_rounding_mode_t default_float_rounding_mode,
                       const char *options) : HandleType<hsa_executable_t>(([&]() {
                                                  hsa_executable_t executable;
                                                  LUTHIER_HSA_CHECK(hsa_executable_create_alt(profile,
                                                                                              default_float_rounding_mode,
                                                                                              options,
                                                                                              &executable));
                                                  return executable;
                                              })()) {}

Executable Executable::create(hsa_profile_t profile,
                              hsa_default_float_rounding_mode_t default_float_rounding_mode,
                              const char *options) {
    return {profile, default_float_rounding_mode, options};
}

hsa_status_t Executable::freeze(const char *options) {
    return getApiTable().core.hsa_executable_freeze_fn(asHsaType(), options);
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
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_get_info_fn(asHsaType(),
                                                                    HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &out));
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
    return out;
}

std::vector<LoadedCodeObject> Executable::getLoadedCodeObjects() const {
    std::vector<LoadedCodeObject> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void *data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<LoadedCodeObject> *>(data);
        out->emplace_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    LUTHIER_HSA_CHECK(getLoaderTable().hsa_ven_amd_loader_executable_iterate_loaded_code_objects(this->asHsaType(),
                                                                                                 iterator,
                                                                                                 &loadedCodeObjects));
    return loadedCodeObjects;
}
std::vector<hsa::GpuAgent> Executable::getAgents() const {
    auto loadedCodeObjects = getLoadedCodeObjects();
    std::vector<hsa::GpuAgent> agents;
    for (const auto &lco: loadedCodeObjects) {
        hsa_agent_t agent;
        LUTHIER_HSA_CHECK(
            getLoaderTable().hsa_ven_amd_loader_loaded_code_object_get_info(lco.asHsaType(),
                                                                            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
                                                                            &agent));
        agents.emplace_back(agent);
    }
    return agents;
}

}// namespace luthier::hsa
