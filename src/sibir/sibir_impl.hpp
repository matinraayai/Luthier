#ifndef SIBIR_IMPL
#define SIBIR_IMPL
#include "hip_intercept.h"
#include "sibir_types.h"

namespace Sibir {
    void init();
    void finalize();
    void hip_api_callback(hip_api_args_t* cb_data, sibir_api_phase_t phase, hip_api_id_t api_id);
    void hsa_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
};


#endif
