#ifndef CONTEXT_MANAGER_HPP
#define CONTEXT_MANAGER_HPP
#include "error_and_log.hpp"
#include <any>
#include <hsa/hsa.h>
#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <fmt/color.h>

#define AGENT_COMGR_META_ACCESSOR(metaName, metaType, typeConvertor) \
    metaType get##metaName##fromComgrMeta() {                                 \
        std::string key{#metaName};\
        if (!amdComgrMetaDataMap_.contains(key)) {             \
            amd_comgr_metadata_node_t valueMetaData;           \
            SIBIR_AMD_COMGR_CHECK(amd_comgr_metadata_lookup(metaDataRootNode_, key.c_str(), &valueMetaData)); \
            std::string value;                                 \
            size_t size;                                       \
            SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(valueMetaData, &size, nullptr));      \
            value.resize(size);                                \
            SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(valueMetaData, &size, value.data())); \
            amdComgrMetaDataMap_.insert({key, typeConvertor(value.c_str())});                                     \
        }                                                      \
        return std::any_cast<metaType>(amdComgrMetaDataMap_.at(key));                                 \
    }


namespace sibir {

class AgentMetaData {
 private:
    std::unordered_map<std::string, std::any> amdComgrMetaDataMap_;
    std::unordered_map<std::string, std::any> hsaMetaDataMap_;
    std::string isaName_;
    amd_comgr_metadata_node_t metaDataRootNode_;


 public:
    explicit AgentMetaData(std::string isaName) : isaName_(std::move(isaName)) {
        SIBIR_AMD_COMGR_CHECK(amd_comgr_get_isa_metadata(isaName_.c_str(), &metaDataRootNode_));
    };
    AgentMetaData() = delete;

    ~AgentMetaData() {
        amdComgrMetaDataMap_.clear();
        hsaMetaDataMap_.clear();
    }

    std::string getIsaName() const { return isaName_; };

    AGENT_COMGR_META_ACCESSOR(AddressableNumSGPRs, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(AddressableNumVGPRs, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(Architecture, std::string, std::string)
    AGENT_COMGR_META_ACCESSOR(EUsPerCU, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(LDSBankCount, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(LocalMemorySize, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(MaxFlatWorkGroupSize, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(MaxWavesPerCU, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(TotalNumSGPRs, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(TotalNumVGPRs, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(VGPRAllocGranule, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(SGPRAllocGranule, int, std::atoi)
    AGENT_COMGR_META_ACCESSOR(Name, std::string, std::string)
    AGENT_COMGR_META_ACCESSOR(Os, std::string, std::string)
    AGENT_COMGR_META_ACCESSOR(Processor, std::string, std::string)
    AGENT_COMGR_META_ACCESSOR(TrapHandlerEnabled, bool, bool)
    AGENT_COMGR_META_ACCESSOR(Vendor, std::string, std::string)
    AGENT_COMGR_META_ACCESSOR(Version, std::string, std::string)
};
//typedef struct sibir_hsa_agent_info_entry_s {
//    // metadata information obtained from AMD Comgr library
//    struct  {
//        bool sramecc;
//        bool xnack;
//        unsigned int lDSBankCount; /
//        unsigned int localMemorySize; /
//        unsigned int maxFlatWorkGroupSize; /
//        unsigned int maxWavesPerCU; /
//        std::string name; /
//        std::string os; /
//        std::string processor; /
//        unsigned int sGPRAllocGranule; /
//        unsigned int vGPRAllocGranule; /
//        unsigned int totalNumSGPRs; /
//        unsigned int totalNumVGPRs; /
//        bool trapHandlerEnabled; /
//        std::string vendor;
//        std::string version;
//    } amd_comgr;
//    struct  {
//        std::string isaName;
//    } hsa;
//} sibir_hsa_agent_info_entry_t;
#undef AGENT_COMGR_META_ACCESSOR

class ContextManager {
 private:
    // Use the agent handle for hashing
    // The order in which the agents are put in the map is important, hence this should be an ordered map
    typedef std::map<decltype(hsa_agent_t::handle), std::shared_ptr<sibir::AgentMetaData>> agent_meta_map_t;
    agent_meta_map_t agentsMetaDataMap_{};

    ContextManager() {
        SIBIR_HSA_CHECK(initGpuAgentsMap());
    }
    ~ContextManager() {
        agentsMetaDataMap_.clear();
    }

    /**
     * Initializes the GPU agents the first time a disassembly request is sent
     * @return status of the agent query from HSA
     */
    hsa_status_t initGpuAgentsMap();

    static std::shared_ptr<AgentMetaData> populateAgentInfo(hsa_agent_t agent);

 public:
    ContextManager(const ContextManager &) = delete;
    ContextManager &operator=(const ContextManager &) = delete;

    static inline ContextManager &Instance() {
        static ContextManager instance;
        return instance;
    }


    inline std::shared_ptr<AgentMetaData> getHsaAgentInfo(hsa_agent_t agent) const {return agentsMetaDataMap_.at(agent.handle);}

    std::vector<hsa_agent_t> getHsaAgents() const {
        std::vector<hsa_agent_t> agents;
        agents.reserve(agentsMetaDataMap_.size());
        for (const auto& h: agentsMetaDataMap_)
            agents.push_back({h.first});
        return agents;
    }

    static std::string getDemangledName(const char *mangledName);
};
}






#endif //CONTEXT_MANAGER_HPP
