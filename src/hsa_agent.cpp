#include "hsa_agent.hpp"
#include "hsa_isa.hpp"

//
//amd_comgr_status_t iterateComgrMetaDataCallback(amd_comgr_metadata_node_t keyMetaDataNode,
//                                                amd_comgr_metadata_node_t valueMetaDataNode,
//                                                void *data) {
//    auto metaDataMap = reinterpret_cast<std::unordered_map<std::string, std::any>*>(data);
//    amd_comgr_metadata_kind_t kind;
//    amd_comgr_metadata_node_t sonMetaData;
//    size_t size;
//    std::string key;
//    std::string value;
//
//
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(keyMetaDataNode, &size, nullptr));
//    key.resize(size);
//    fmt::println("Size of key: {}", size);
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(keyMetaDataNode, &size, key.data()));
//
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_kind(valueMetaDataNode, &kind));
//
//    switch (kind) {
//        case AMD_COMGR_METADATA_KIND_STRING: {
//            fmt::println("Kind: Str");
//            fmt::println("Size of value: {}", size);
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(valueMetaDataNode, &size, nullptr));
//            value.resize(size);
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(valueMetaDataNode, &size, value.data()));
//            if (!metaDataMap->contains(key)) {
//                metaDataMap->insert({key, value});
//            }
//            else {
//                std::any_cast<std::vector<std::string>>(metaDataMap->at(key)).push_back(value);
//            }
//            break;
//        }
//        case AMD_COMGR_METADATA_KIND_LIST: {
//            fmt::println("Kind: List");
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_list_size(valueMetaDataNode, &size));
//            metaDataMap->insert({key, std::vector<std::string>(size)});
//            for (size_t i = 0; i < size; i++) {
//                LUTHIER_AMD_COMGR_CHECK(amd_comgr_index_list_metadata(valueMetaDataNode, i, &sonMetaData));
//                LUTHIER_AMD_COMGR_CHECK(iterateComgrMetaDataCallback(keyMetaDataNode, sonMetaData, data));
//                LUTHIER_AMD_COMGR_CHECK(amd_comgr_destroy_metadata(sonMetaData));
//            }
//            break;
//        }
//        case AMD_COMGR_METADATA_KIND_MAP: {
//            fmt::println("Kind: Map");
//            std::unordered_map<std::string, std::any> childMap;
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_map_size(valueMetaDataNode, &size));
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_iterate_map_metadata(valueMetaDataNode, iterateComgrMetaDataCallback, &childMap));
//            metaDataMap->insert({key, childMap});
//            break;
//        }
//        default:
//            return AMD_COMGR_STATUS_ERROR;
//    } // switch
//    return AMD_COMGR_STATUS_SUCCESS;
//};

//std::string luthier::ContextManager::getDemangledName(const char *mangledName) {
//    amd_comgr_data_t mangledNameData;
//    amd_comgr_data_t demangledNameData;
//    std::string out;
//
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_BYTES, &mangledNameData));
//
//    size_t size = strlen(mangledName);
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(mangledNameData, size, mangledName));
//
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_demangle_symbol_name(mangledNameData, &demangledNameData));
//
//    size_t demangledNameSize = 0;
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data(demangledNameData, &demangledNameSize, nullptr));
//
//    out.resize(demangledNameSize);
//
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data(demangledNameData, &demangledNameSize, out.data()));
//
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_release_data(mangledNameData));
//
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_release_data(demangledNameData));
//
//    return out;
//}

//luthier::AgentMetaData::AgentMetaData(hsa_agent_t agent) : agent_(agent) {
//        const auto& coreApi = HsaInterceptor::instance().getSavedHsaTables().core;
//        // Get the Isa name of the agent
//        std::vector<std::string> supportedAgentIsaNames;
//        auto getIsaNameCallback = [](hsa_isa_t isa, void* data) {
//            auto out = reinterpret_cast<std::vector<std::string>*>(data);
//            auto coreApi = HsaInterceptor::instance().getSavedHsaTables().core;
//            uint32_t isaNameSize;
//            LUTHIER_HSA_CHECK(coreApi.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME_LENGTH, &isaNameSize));
//            std::string isaName;
//            isaName.resize(isaNameSize);
//            LUTHIER_HSA_CHECK(coreApi.hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME, isaName.data()));
//            out->push_back(isaName);
//            return HSA_STATUS_SUCCESS;
//        };
//
//        LUTHIER_HSA_CHECK(coreApi.hsa_agent_iterate_isas_fn(agent, getIsaNameCallback, &supportedAgentIsaNames));
//
//        // For now assert that there's only one supported ISA for the agent
//        assert(supportedAgentIsaNames.size() == 1);
//
//        hsaMetaDataMap_.insert({HSA_AGENT_INFO_ISA, supportedAgentIsaNames});
//        isaName_ = supportedAgentIsaNames[0];
//        LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_isa_metadata(isaName_.c_str(), &metaDataRootNode_));
//};

namespace luthier::hsa {

std::vector<Isa> GpuAgent::getIsa() const {
    std::vector<Isa> supportedIsaList;
    auto iterator = [](hsa_isa_t isa, void* data) {
        auto supportedIsaList = reinterpret_cast<std::vector<Isa>*>(data);
        supportedIsaList->emplace_back(isa);
        return HSA_STATUS_SUCCESS;
    };

    LUTHIER_HSA_CHECK(getApiTable().core.hsa_agent_iterate_isas_fn(this->asHsaType(),
                                                                   iterator, &supportedIsaList));

    return supportedIsaList;
}
}

