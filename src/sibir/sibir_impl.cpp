#include <sibir_impl.hpp>
#include <sibir.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer.h>


#define CHECK_ROCTRACER_CALL(call)                                         \
   do {                                                                    \
      int err = call;                                                      \
      if (err != 0) {                                                      \
         std::cerr << roctracer_error_string() << std::endl << std::flush; \
         abort();                                                          \
      }                                                                    \
   } while (0)


void __attribute__((constructor)) Sibir::init() {
    std::cout << "Initializing Sibir...." << std::endl << std::flush;
    assert(SibirHipInterceptor::Instance().IsEnabled());
    SibirHipInterceptor::Instance().SetCallback(Sibir::hip_api_callback);
//    CHECK_ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, Sibir::hip_api_callback, nullptr));
    CHECK_ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, Sibir::hsa_api_callback, nullptr));
    sibir_at_init();
}

__attribute__((destructor)) void Sibir::finalize() {
//    CHECK_ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
    CHECK_ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));
    sibir_at_term();
    std::cout << "Sibir Terminated." << std::endl << std::flush;
}

void Sibir::hip_api_callback(hip_api_args_t* cb_data, sibir_api_phase_t phase, hip_api_id_t api_id) {
    sibir_at_hip_event(cb_data, phase, api_id);
}

void Sibir::hsa_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
    (void)arg;
    auto* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
    sibir_at_hsa_event(cid, data);
}