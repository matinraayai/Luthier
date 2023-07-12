#include <sibir_impl.hpp>
#include <sibir.h>
#include <roctracer/roctracer.h>
#include "hsa_intercept.h"
#include <iomanip>
#include "code_object_manager.h"
#include "disassembler.h"

void sibir::hipStartupCallback(void* cb_data, sibir_api_phase_t phase, int api_id) {
//    static std::vector<hip___hipRegisterFatBinary_api_args_t*>
    static const void* lastSavedFatBinary{};
    if (phase == SIBIR_API_PHASE_EXIT) {
        if (api_id == HIP_PRIVATE_API_ID___hipRegisterFatBinary) {
            auto args = reinterpret_cast<hip___hipRegisterFatBinary_api_args_t*>(cb_data);
            lastSavedFatBinary = args->data;
        }
        else if (api_id == HIP_PRIVATE_API_ID___hipRegisterFunction) {
            auto args = reinterpret_cast<hip___hipRegisterFunction_api_args_t*>(cb_data);
            auto& coManager = SibirCodeObjectManager::Instance();

            // If the function doesn't have __sibir_wrap__ in its name then it belongs to the instrumented application
            // Give control to the user-defined callback
            if (std::string(args->deviceFunction).find("__sibir_wrap__") == std::string::npos)
                SibirHipInterceptor::Instance().SetCallback(sibir::hipApiCallback);
            else {
                coManager.registerFatBinary(lastSavedFatBinary);
                coManager.registerFunction(lastSavedFatBinary, args->deviceFunction, args->hostFunction, args->deviceName);
            }
        }
        else if (api_id == HIP_PRIVATE_API_ID___hipRegisterManagedVar) {
        }
        else if (api_id == HIP_PRIVATE_API_ID___hipRegisterSurface) {

        }
        else if (api_id == HIP_PRIVATE_API_ID___hipRegisterTexture) {

        }
        else if (api_id == HIP_PRIVATE_API_ID___hipRegisterVar) {

        }
    }
}


void __attribute__((constructor)) sibir::init() {
    std::cout << "Initializing Sibir...." << std::endl << std::flush;
    assert(SibirHipInterceptor::Instance().IsEnabled());
    sibir_at_init();
    SibirHipInterceptor::Instance().SetCallback(sibir::hipStartupCallback);
    SibirHsaInterceptor::Instance().SetCallback(sibir::hsaApiCallback);
}

__attribute__((destructor)) void sibir::finalize() {
    sibir_at_term();
    std::cout << "Sibir Terminated." << std::endl << std::flush;
}

const HsaApiTable* sibir_get_hsa_table() {
    return &SibirHsaInterceptor::Instance().getSavedHsaTables().root;
}

const hsa_ven_amd_loader_1_03_pfn_s* sibir_get_hsa_ven_amd_loader() {
    return &SibirHsaInterceptor::Instance().getHsaVenAmdLoaderTable();
}

void sibir::hipApiCallback(void* cb_data, sibir_api_phase_t phase, int api_id) {
    sibir_at_hip_event(cb_data, phase, api_id);
}

void sibir::hsaApiCallback(hsa_api_args_t *cb_data, sibir_api_phase_t phase, hsa_api_id_t api_id) {
    sibir_at_hsa_event(cb_data, phase, api_id);
}

std::vector<Instr> sibir_disassemble_kernel_object(uint64_t kernel_object) {
    return Disassembler::Instance().disassemble(kernel_object);
}

//void print_instructions(const std::vector<Inst>& isa) {
//
//        std::cout << "Decoded by rocdbg-api: " << instruction << " Instruction Size: " << instrSize << " Address: " << std::hex <<
//            curr_address << " Bytes: ";
//        for (std::byte &el: instBytes) {
//            std::cout << std::hex << std::setfill('0') << std::setw(1) << uint16_t(el) << " ";
//        }
//        std::cout << std::dec << std::endl;
////    InstPrinter printer{};
////    for (const auto& inst: isa)
////        std::cout << printer.print(inst) << std::endl;
//}

void* sibir_get_hip_function(const char* funcName) {
    return SibirHipInterceptor::Instance().GetHipFunction(funcName);
}


hsa_executable_t sibir_insert_call(const Instr *instr, const char *dev_func_name, sibir_ipoint_t point) {

    std::vector<hsa_agent_t> agents;

    auto& coreTable = SibirHsaInterceptor::Instance().getSavedHsaTables().core;

    auto queryAgentsCallback = [](hsa_agent_t agent, void* data) {
        auto agents = reinterpret_cast<std::vector<hsa_agent_t>*>(data);
        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

        hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

        if (stat != HSA_STATUS_SUCCESS)
            return stat;
        if (dev_type == HSA_DEVICE_TYPE_GPU)
            agents->push_back(agent);

        return stat;
    };

    SIBIR_HSA_CHECK(coreTable.hsa_iterate_agents_fn(queryAgentsCallback, &agents));

    return SibirCodeObjectManager::Instance().getInstrumentationFunction(dev_func_name, agents[0]);

}

extern "C" {

ROCTRACER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 49;

ROCTRACER_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                             uint64_t failed_tool_count, const char *const *failed_tool_names) {
    [](auto &&...) {}(runtime_version, failed_tool_count, failed_tool_names);
    return SibirHsaInterceptor::Instance().captureHsaApiTable(table);
}

ROCTRACER_EXPORT void OnUnload() {}
}