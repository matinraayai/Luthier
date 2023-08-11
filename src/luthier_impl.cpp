#include "luthier_impl.hpp"
#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "hsa_intercept.hpp"
#include "hip_intercept.hpp"
#include <roctracer/roctracer.h>
#include "luthier.h"
#include <fmt/color.h>
#include "log.hpp"

void luthier::impl::hipStartupCallback(void *cb_data, luthier_api_phase_t phase, int api_id) {
    LUTHIER_LOG_FUNCTION_CALL_START
    static const void *lastSavedFatBinary{};
    if (phase == LUTHIER_API_PHASE_EXIT) {
        if (api_id == HIP_PRIVATE_API_ID___hipRegisterFatBinary) {
            auto args = reinterpret_cast<hip___hipRegisterFatBinary_api_args_t *>(cb_data);
            lastSavedFatBinary = args->data;
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterFunction) {
            auto args = reinterpret_cast<hip___hipRegisterFunction_api_args_t *>(cb_data);
            auto &coManager = CodeObjectManager::Instance();

            // If the function doesn't have __luthier_wrap__ in its name then it belongs to the instrumented application
            // Give control to the user-defined callback
            if (std::string(args->deviceFunction).find("__luthier_wrap__") == std::string::npos)
                HipInterceptor::Instance().SetCallback(luthier::impl::hipApiCallback);
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
    LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((constructor)) void luthier::impl::init() {
    LUTHIER_LOG_FUNCTION_CALL_START
    assert(HipInterceptor::Instance().IsEnabled());
    luthier_at_init();
    HipInterceptor::Instance().SetCallback(luthier::impl::hipStartupCallback);
    HsaInterceptor::Instance().SetCallback(luthier::impl::hsaApiCallback);
    LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((destructor)) void luthier::impl::finalize() {
    LUTHIER_LOG_FUNCTION_CALL_START
    luthier_at_term();
    LUTHIER_LOG_FUNCTION_CALL_END
}

const HsaApiTable *luthier_get_hsa_table() {
    return &luthier::HsaInterceptor::Instance().getSavedHsaTables().root;
}

const hsa_ven_amd_loader_1_03_pfn_s *luthier_get_hsa_ven_amd_loader() {
    return &luthier::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
}

void luthier::impl::hipApiCallback(void *cb_data, luthier_api_phase_t phase, int api_id) {
    luthier_at_hip_event(cb_data, phase, api_id);
}

void luthier::impl::hsaApiCallback(hsa_api_args_t *cb_data, luthier_api_phase_t phase, hsa_api_id_t api_id) {
    luthier_at_hsa_event(cb_data, phase, api_id);
}

std::vector<luthier::Instr> luthier_disassemble_kernel_object(uint64_t kernel_object) {
    return luthier::Disassembler::Instance().disassemble(kernel_object);
}

void *luthier_get_hip_function(const char *funcName) {
    return luthier::HipInterceptor::Instance().GetHipFunction(funcName);
}

void luthier_insert_call(luthier::Instr *instr, const char *dev_func_name, luthier_ipoint_t point) {

    auto agent = instr->getAgent();

    std::string instrumentationFunc = luthier::CodeObjectManager::Instance().getCodeObjectOfInstrumentationFunction(dev_func_name, agent);

    luthier::CodeGenerator::instrument(*instr, instrumentationFunc, point);
}

void luthier_enable_instrumented(hsa_kernel_dispatch_packet_t* dispatch_packet, const luthier_address_t func, bool flag) {
//    if (flag) {
//        auto instrumentedKd = luthier::CodeObjectManager::Instance().getInstrumentedFunctionOfKD(func);
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
    return luthier::HsaInterceptor::Instance().captureHsaApiTable(table);
}

ROCTRACER_EXPORT void OnUnload() {}
}