#ifndef CONTEXT_MANAGER_H_
#define CONTEXT_MANAGER_H_
#include "error_check.h"
#include <unordered_map>
#include <vector>
#include <hsa/hsa.h>

namespace sibir {
// Cache the agent information needed for disassembly
// Maybe we need more than just a single entry
typedef struct hsa_agent_entry_s {
    std::string isa;
} hsa_agent_entry_t;

class ContextManager {
 private:
    // Use the agent handle for hashing
    std::unordered_map<decltype(hsa_agent_t::handle), sibir::hsa_agent_entry_t> agentsMap_{};

    ContextManager() {
        SIBIR_HSA_CHECK(initGpuAgentsMap());
    }
    ~ContextManager() {
        agentsMap_.clear();
    }

    /**
     * Initializes the GPU agents the first time a disassembly request is sent
     * @return status of the agent query from HSA
     */
    hsa_status_t initGpuAgentsMap();

    static hsa_status_t populateAgentInfo(hsa_agent_t agent, hsa_agent_entry_t &entry);

 public:
    ContextManager(const ContextManager &) = delete;
    ContextManager &operator=(const ContextManager &) = delete;

    static inline ContextManager &Instance() {
        static ContextManager instance;
        return instance;
    }


    inline const sibir::hsa_agent_entry_t& getHsaAgentInfo(hsa_agent_t agent) const {return agentsMap_.at(agent.handle);}

    std::vector<hsa_agent_t> getHsaAgents() const {
        std::vector<hsa_agent_t> agents;
        agents.reserve(agentsMap_.size());
        for (const auto& h: agentsMap_)
            agents.push_back({h.first});
        return agents;
    }

    static std::string getDemangledName(const char *mangledName);
};
}






#endif //CONTEXT_MANAGER_H_
