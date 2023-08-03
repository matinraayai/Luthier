#include "context_manager.hpp"
#include "hsa_intercept.hpp"
#include <amd_comgr/amd_comgr.h>
#include <fmt/color.h>


amd_comgr_status_t iterateComgrMetaDataCallback(amd_comgr_metadata_node_t keyMetaDataNode,
                                                amd_comgr_metadata_node_t valueMetaDataNode,
                                                void *data) {
    auto metaDataMap = reinterpret_cast<std::unordered_map<std::string, std::any>*>(data);
    amd_comgr_metadata_kind_t kind;
    amd_comgr_metadata_node_t sonMetaData;
    size_t size;
    std::string key;
    std::string value;


    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(keyMetaDataNode, &size, nullptr));
    key.resize(size);
    fmt::println("Size of key: {}", size);
    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(keyMetaDataNode, &size, key.data()));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_kind(valueMetaDataNode, &kind));

    switch (kind) {
        case AMD_COMGR_METADATA_KIND_STRING: {
            fmt::println("Kind: Str");
            fmt::println("Size of value: {}", size);
            SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(valueMetaDataNode, &size, nullptr));
            value.resize(size);
            SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(valueMetaDataNode, &size, value.data()));
            if (!metaDataMap->contains(key)) {
                metaDataMap->insert({key, value});
            }
            else {
                std::any_cast<std::vector<std::string>>(metaDataMap->at(key)).push_back(value);
            }
            break;
        }
        case AMD_COMGR_METADATA_KIND_LIST: {
            fmt::println("Kind: List");
            SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_list_size(valueMetaDataNode, &size));
            metaDataMap->insert({key, std::vector<std::string>(size)});
            for (size_t i = 0; i < size; i++) {
                SIBIR_AMD_COMGR_CHECK(amd_comgr_index_list_metadata(valueMetaDataNode, i, &sonMetaData));
                SIBIR_AMD_COMGR_CHECK(iterateComgrMetaDataCallback(keyMetaDataNode, sonMetaData, data));
                SIBIR_AMD_COMGR_CHECK(amd_comgr_destroy_metadata(sonMetaData));
            }
            break;
        }
        case AMD_COMGR_METADATA_KIND_MAP: {
            fmt::println("Kind: Map");
            std::unordered_map<std::string, std::any> childMap;
            SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_map_size(valueMetaDataNode, &size));
            SIBIR_AMD_COMGR_CHECK(amd_comgr_iterate_map_metadata(valueMetaDataNode, iterateComgrMetaDataCallback, &childMap));
            metaDataMap->insert({key, childMap});
            break;
        }
        default:
            return AMD_COMGR_STATUS_ERROR;
    } // switch
    return AMD_COMGR_STATUS_SUCCESS;
};

std::shared_ptr<sibir::AgentMetaData> sibir::ContextManager::populateAgentInfo(hsa_agent_t agent) {
    const auto& coreApi = HsaInterceptor::Instance().getSavedHsaTables().core;
    // Get the name (architecture) of the agent
    std::string agentName;
    agentName.resize(64);

    coreApi.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_ISA, agentName.data());

    // Get the Isa name of the agent
    std::vector<std::string> supportedAgentIsaNames;

    auto getIsaNameCallback = [](hsa_isa_t isa, void* data) {
        auto out = reinterpret_cast<std::vector<std::string>*>(data);
        auto coreApi = HsaInterceptor::Instance().getSavedHsaTables().core;
        uint32_t isaNameSize;
        SIBIR_HSA_CHECK(coreApi.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME_LENGTH, &isaNameSize));
        std::string isaName;
        isaName.resize(isaNameSize);
        SIBIR_HSA_CHECK(coreApi.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME, isaName.data()));
        out->push_back(isaName);
        return HSA_STATUS_SUCCESS;
    };

    SIBIR_HSA_CHECK(coreApi.hsa_agent_iterate_isas_fn(agent, getIsaNameCallback, &supportedAgentIsaNames));

    // Assert that there's only one supported ISA for the agent
    assert(supportedAgentIsaNames.size() == 1);

    return std::make_shared<AgentMetaData>(supportedAgentIsaNames[0]);
}


hsa_status_t sibir::ContextManager::initGpuAgentsMap() {
    auto& coreTable = HsaInterceptor::Instance().getSavedHsaTables().core;

    auto returnGpuAgentsCallback = [](hsa_agent_t agent, void* data) {
        auto agentMap = reinterpret_cast<agent_meta_map_t*>(data);
        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

        hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

        if (stat != HSA_STATUS_SUCCESS)
            return stat;
        if (dev_type == HSA_DEVICE_TYPE_GPU) {
            auto entry = sibir::ContextManager::populateAgentInfo(agent);
            agentMap->insert({agent.handle, entry});
        }

        return stat;
    };

    return coreTable.hsa_iterate_agents_fn(returnGpuAgentsCallback, &agentsMetaDataMap_);
}

std::string sibir::ContextManager::getDemangledName(const char *mangledName) {
    amd_comgr_data_t mangledNameData;
    amd_comgr_data_t demangledNameData;
    std::string out;

    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_BYTES, &mangledNameData));

    size_t size = strlen(mangledName);
    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(mangledNameData, size, mangledName));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_demangle_symbol_name(mangledNameData, &demangledNameData));

    size_t demangledNameSize = 0;
    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_data(demangledNameData, &demangledNameSize, nullptr));

    out.resize(demangledNameSize);

    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_data(demangledNameData, &demangledNameSize, out.data()));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_release_data(mangledNameData));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_release_data(demangledNameData));

    return out;
}
