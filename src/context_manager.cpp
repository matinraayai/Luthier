#include "context_manager.hpp"
#include "hsa_intercept.h"

hsa_status_t sibir::ContextManager::populateAgentInfo(hsa_agent_t agent, sibir::hsa_agent_entry_t& entry) {
    const auto& coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
    hsa_status_t status;

    // Get the name (architecture) of the agent
    std::string agentName;
    agentName.resize(64);

    coreApi.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_ISA, agentName.data());

    // Get the Isa name of the agent
    std::vector<std::string> supportedAgentIsaNames;

    auto getIsaNameCallback = [](hsa_isa_t isa, void* data){
        auto out = reinterpret_cast<std::vector<std::string>*>(data);
        auto coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
        hsa_status_t status = HSA_STATUS_ERROR;
        uint32_t isaNameSize;
        status = SIBIR_HSA_CHECK(coreApi.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME_LENGTH, &isaNameSize));
        if (status != HSA_STATUS_SUCCESS)
            return status;
        std::string isaName;
        isaName.resize(isaNameSize);
        status = SIBIR_HSA_CHECK(coreApi.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME, isaName.data()));
        if (status != HSA_STATUS_SUCCESS)
            return status;
        out->push_back(isaName);
        return HSA_STATUS_SUCCESS;
    };

    status = SIBIR_HSA_CHECK(coreApi.hsa_agent_iterate_isas_fn(agent, getIsaNameCallback, &supportedAgentIsaNames));

    if (status != HSA_STATUS_SUCCESS)
        return status;
    // Assert that there's only one supported ISA for the agent
    assert(supportedAgentIsaNames.size() == 1);

    entry.isa = supportedAgentIsaNames[0];

    return HSA_STATUS_SUCCESS;
}


hsa_status_t sibir::ContextManager::initGpuAgentsMap() {
    auto& coreTable = SibirHsaInterceptor::Instance().getSavedHsaTables().core;

    auto returnGpuAgentsCallback = [](hsa_agent_t agent, void* data) {
        auto agentMap = reinterpret_cast<std::unordered_map<decltype(hsa_agent_t::handle), hsa_agent_entry_t>*>(data);
        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

        hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

        if (stat != HSA_STATUS_SUCCESS)
            return stat;
        if (dev_type == HSA_DEVICE_TYPE_GPU) {
            hsa_agent_entry_t entry;
            sibir::ContextManager::populateAgentInfo(agent, entry);
            agentMap->insert({agent.handle, entry});
        }

        return stat;
    };

    return coreTable.hsa_iterate_agents_fn(returnGpuAgentsCallback, &agentsMap_);
}