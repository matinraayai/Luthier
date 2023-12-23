#ifndef CONTEXT_MANAGER_HPP
#define CONTEXT_MANAGER_HPP
#include "error.h"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_intercept.hpp"
#include <hsa/hsa.h>
#include <vector>

namespace luthier {
class ContextManager {
 private:
    std::vector<luthier::hsa::GpuAgent> agents_;

    ContextManager() {
        auto &coreTable = HsaInterceptor::instance().getSavedHsaTables().core;

        auto returnGpuAgentsCallback = [](hsa_agent_t agent, void *data) {
            auto agentMap = reinterpret_cast<std::vector<hsa::GpuAgent> *>(data);
            hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

            hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

            if (stat != HSA_STATUS_SUCCESS)
                return stat;
            if (dev_type == HSA_DEVICE_TYPE_GPU) {
                agentMap->push_back(luthier::hsa::GpuAgent(agent));
            }
            return stat;
        };

        LUTHIER_HSA_CHECK(coreTable.hsa_iterate_agents_fn(returnGpuAgentsCallback, &agents_));
    }
    ~ContextManager() = default;

 public:
    ContextManager(const ContextManager &) = delete;
    ContextManager &operator=(const ContextManager &) = delete;

    static inline ContextManager &Instance() {
        static ContextManager instance;
        return instance;
    }

    const std::vector<luthier::hsa::GpuAgent> &getHsaAgents() const { return agents_; };

    std::vector<luthier::hsa::Executable> getHsaExecutables() const {
        const auto &loaderApi = HsaInterceptor::instance().getHsaVenAmdLoaderTable();
        std::vector<luthier::hsa::Executable> out;
        auto iterator = [](hsa_executable_t exec, void *data) {
            auto out = reinterpret_cast<std::vector<luthier::hsa::Executable>*>(data);
            out->emplace_back(exec);
            return HSA_STATUS_SUCCESS;
        };
        LUTHIER_HSA_CHECK(loaderApi.hsa_ven_amd_loader_iterate_executables(iterator, &out));
        return out;
    }
};

}// namespace luthier

#endif//CONTEXT_MANAGER_HPP
