#include "sibir_impl.hpp"
#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "hsa_intercept.hpp"
#include "hip_intercept.hpp"
#include <roctracer/roctracer.h>
#include "sibir.h"
#include <fmt/color.h>
#include "log.hpp"

void sibir::impl::hipStartupCallback(void *cb_data, sibir_api_phase_t phase, int api_id) {
    SIBIR_LOG_FUNCTION_CALL_START
    static const void *lastSavedFatBinary{};
    if (phase == SIBIR_API_PHASE_EXIT) {
        if (api_id == HIP_PRIVATE_API_ID___hipRegisterFatBinary) {
            auto args = reinterpret_cast<hip___hipRegisterFatBinary_api_args_t *>(cb_data);
            lastSavedFatBinary = args->data;
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterFunction) {
            auto args = reinterpret_cast<hip___hipRegisterFunction_api_args_t *>(cb_data);
            auto &coManager = CodeObjectManager::Instance();

            // If the function doesn't have __sibir_wrap__ in its name then it belongs to the instrumented application
            // Give control to the user-defined callback
            if (std::string(args->deviceFunction).find("__sibir_wrap__") == std::string::npos)
                HipInterceptor::Instance().SetCallback(sibir::impl::hipApiCallback);
            else {
                coManager.registerFatBinary(lastSavedFatBinary);
                coManager.registerFunction(lastSavedFatBinary, args->deviceFunction, args->hostFunction, args->deviceName);
            }
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterManagedVar) {
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterSurface) {

        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterTexture) {

        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterVar) {
        }
    }
    SIBIR_LOG_FUNCTION_CALL_END
}

__attribute__((constructor)) void sibir::impl::init() {
    SIBIR_LOG_FUNCTION_CALL_START
    assert(HipInterceptor::Instance().IsEnabled());
    sibir_at_init();
    HipInterceptor::Instance().SetCallback(sibir::impl::hipStartupCallback);
    HsaInterceptor::Instance().SetCallback(sibir::impl::hsaApiCallback);
    SIBIR_LOG_FUNCTION_CALL_END
}

__attribute__((destructor)) void sibir::impl::finalize() {
    SIBIR_LOG_FUNCTION_CALL_START
    sibir_at_term();
    SIBIR_LOG_FUNCTION_CALL_END
}

const HsaApiTable *sibir_get_hsa_table() {
    return &sibir::HsaInterceptor::Instance().getSavedHsaTables().root;
}

const hsa_ven_amd_loader_1_03_pfn_s *sibir_get_hsa_ven_amd_loader() {
    return &sibir::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
}

void sibir::impl::hipApiCallback(void *cb_data, sibir_api_phase_t phase, int api_id) {
    sibir_at_hip_event(cb_data, phase, api_id);
}

void sibir::impl::hsaApiCallback(hsa_api_args_t *cb_data, sibir_api_phase_t phase, hsa_api_id_t api_id) {
    sibir_at_hsa_event(cb_data, phase, api_id);
}

std::vector<sibir::Instr> sibir_disassemble_kernel_object(uint64_t kernel_object) {
    return sibir::Disassembler::Instance().disassemble(kernel_object);
}

void *sibir_get_hip_function(const char *funcName) {
    return sibir::HipInterceptor::Instance().GetHipFunction(funcName);
}

void sibir_insert_call(sibir::Instr *instr, const char *dev_func_name, sibir_ipoint_t point) {

    auto agent = instr->getAgent();

    std::string instrumentationFunc = sibir::CodeObjectManager::Instance().getCodeObjectOfInstrumentationFunction(dev_func_name, agent);

    sibir::CodeGenerator::instrument(*instr, instrumentationFunc, point);
}

void sibir_enable_instrumented(hsa_kernel_dispatch_packet_t* dispatch_packet, const sibir_address_t func, bool flag) {
//    if (flag) {
//        auto instrumentedKd = sibir::CodeObjectManager::Instance().getInstrumentedFunctionOfKD(func);
//        dispatch_packet->kernel_object = reinterpret_cast<uint64_t>(instrumentedKd);
//    }
//    else
//        dispatch_packet->kernel_object = reinterpret_cast<uint64_t>(func);
}

extern "C" {

ROCTRACER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 49;

ROCTRACER_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                             uint64_t failed_tool_count, const char *const *failed_tool_names) {
    [](auto &&...) {}(runtime_version, failed_tool_count, failed_tool_names);
    return sibir::HsaInterceptor::Instance().captureHsaApiTable(table);
}

ROCTRACER_EXPORT void OnUnload() {}
}