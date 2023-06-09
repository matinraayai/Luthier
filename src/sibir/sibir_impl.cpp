#include <sibir_impl.hpp>
#include <sibir.h>
#include <roctracer/roctracer.h>
#include "hsa_intercept.h"


void __attribute__((constructor)) Sibir::init() {
    std::cout << "Initializing Sibir...." << std::endl << std::flush;
    assert(SibirHipInterceptor::Instance().IsEnabled());
    sibir_at_init();
    SibirHipInterceptor::Instance().SetCallback(Sibir::hip_api_callback);
    SibirHsaInterceptor::Instance().SetCallback(Sibir::hsa_api_callback);
}

__attribute__((destructor)) void Sibir::finalize() {
    sibir_at_term();
    std::cout << "Sibir Terminated." << std::endl << std::flush;
}

const HsaApiTable* sibir_get_hsa_table() {
    return &SibirHsaInterceptor::Instance().getSavedHsaTables().root;
}

void Sibir::hip_api_callback(hip_api_args_t* cb_data, sibir_api_phase_t phase, hip_api_id_t api_id) {
    sibir_at_hip_event(cb_data, phase, api_id);
}

void Sibir::hsa_api_callback(hsa_api_args_t* cb_data, sibir_api_phase_t phase, hsa_api_id_t api_id) {
    sibir_at_hsa_event(cb_data, phase, api_id);
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