#ifndef SIBIR_IMPL
#define SIBIR_IMPL
#include "hip_intercept.h"
#include "sibir_types.h"
#include "code_object_manager.h"

namespace Sibir {
    void init();
    void finalize();
    void hip_startup_callback(void* cb_data, sibir_api_phase_t phase, int api_id);
    void hip_api_callback(void* cb_data, sibir_api_phase_t phase, int api_id);
    void hsa_api_callback(hsa_api_args_t* cb_data, sibir_api_phase_t phase, hsa_api_id_t api_id);
};


#endif
