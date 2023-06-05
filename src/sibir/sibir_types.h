#ifndef SIBIR_TYPES_H_
#define SIBIR_TYPES_H_
#include <hip/amd_detail/hip_prof_str.h>

enum sibir_api_phase_t {
    SIBIR_API_PHASE_ENTER,
    SIBIR_API_PHASE_EXIT
};

typedef decltype(hip_api_data_t::args) hip_api_args_t;

#endif
