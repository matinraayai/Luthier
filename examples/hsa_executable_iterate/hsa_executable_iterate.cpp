#include <fstream>
#include <functional>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <map>
#include <mutex>
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_hsa.h>
#include <sibir.h>
#include <string>

void sibir_at_init() {
    std::cout << "Hi from Sibir!" << std::endl;
}


void sibir_at_term() {
    std::cout << "Bye from Sibir!" << std::endl;
}

inline void check_error(hsa_status_t err, const char* call_name) {
    if (err != HSA_STATUS_SUCCESS) {
        const char* err_msg = "Unknown Error";
        hsa_status_string(err, &err_msg);
        std::cerr << "HSA call " << call_name << "failed! Error code: " <<
            err_msg << std::endl << std::flush; \
    }
}

#define CHECK_HSA_CALL(call) check_error(call, #call)


static unsigned extractAqlBits(unsigned v, unsigned pos, unsigned width) {
    return (v >> pos) & ((1 << width) - 1);
};


/**
 * Returns the list of GPU HSA agents (devices) on the system
 * @param [out] agentList a vector of GPU agents
 * @return HSA_STATUS_SUCCESS
 */
hsa_status_t getGpuAgents(std::vector<hsa_agent_t>& agentList) {
    int i = 0;
    auto iterateAgentCallback = [](hsa_agent_t agent, void* data) {
        auto agent_list = reinterpret_cast<std::vector<hsa_agent_t>*>(data);
        hsa_status_t stat = HSA_STATUS_ERROR;
        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

        stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

        if (stat != HSA_STATUS_SUCCESS) {
            return stat;
        }
        if (dev_type == HSA_DEVICE_TYPE_GPU) {
            agent_list->push_back(agent);
        }
        return stat;
    };
    return hsa_iterate_agents(iterateAgentCallback, &agentList);
}


hsa_status_t getAllExecutableSymbols(const hsa_executable_t& executable,
                                     const std::vector<hsa_agent_t>& agentList,
                                     std::vector<hsa_executable_symbol_t>& symbols) {
    hsa_status_t out = HSA_STATUS_ERROR;
    for (auto agent : agentList) {
        auto iterCallback = [](hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data) {
            auto symbolVec = reinterpret_cast<std::vector<hsa_executable_symbol_t>*>(data);
            symbolVec->push_back(symbol);
            return HSA_STATUS_SUCCESS;
        };
        out = hsa_executable_iterate_agent_symbols(executable,
                                                   agent,
                                                   iterCallback, &symbols);
        if (out != HSA_STATUS_SUCCESS)
            return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
}

void sibir_at_hsa_event(hsa_api_args_t* args, sibir_api_phase_t phase, hsa_api_id_t api_id) {
    if (phase == SIBIR_API_PHASE_EXIT) {
        if (api_id == HSA_API_ID_hsa_executable_freeze) {
            auto executable = args->hsa_executable_freeze.executable;
            fprintf(stdout, "HSA Executable Freeze Callback\n");
            // Get the state of the executable (frozen or not frozen)
            hsa_executable_state_t e_state;
            CHECK_HSA_CALL(hsa_executable_get_info(executable, HSA_EXECUTABLE_INFO_STATE, &e_state));

            fprintf(stdout, "Is executable frozen: %s", (e_state == HSA_EXECUTABLE_STATE_FROZEN ? "yes" : "no"));

            std::vector<hsa_agent_t> agentList;
            std::vector<hsa_executable_symbol_t> symbols;
            getGpuAgents(agentList);
            getAllExecutableSymbols(executable, agentList, symbols);
            fprintf(stdout, "");
        }
    }

}

void sibir_at_hip_event(hip_api_args_t* args, sibir_api_phase_t phase, hip_api_id_t api_id) {
    fprintf(stdout, "<call to (%s)\t on %s> ",
            hip_api_name(api_id),
            phase == SIBIR_API_PHASE_ENTER ? "entry" : "exit"
            );
    switch (api_id) {
        case HIP_API_ID_hipMemcpy:
            fprintf(stdout, "dst(%p) src(%p) size(0x%x) kind(%u)",
                    args->hipMemcpy.dst,
                    args->hipMemcpy.src,
                    (uint32_t) (args->hipMemcpy.sizeBytes),
                    (uint32_t) (args->hipMemcpy.kind));
            break;
        case HIP_API_ID_hipMalloc:
            fprintf(stdout, "ptr(%p) size(0x%x)",
                    args->hipMalloc.ptr,
                    (uint32_t) (args->hipMalloc.size));
            break;
        case HIP_API_ID_hipFree:
            fprintf(stdout, "ptr(%p)", args->hipFree.ptr);
            break;
        case HIP_API_ID_hipLaunchKernel:
            fprintf(stdout, "kernel(\"%s\") stream(%p)",
                    hipKernelNameRefByPtr(args->hipLaunchKernel.function_address,
                                          args->hipLaunchKernel.stream),
                    args->hipModuleLaunchKernel.stream);
            break;
        default:
            break;
    }
    fprintf(stdout, "\n"); fflush(stdout);
}