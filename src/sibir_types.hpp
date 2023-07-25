#ifndef SIBIR_TYPES_HPP
#define SIBIR_TYPES_HPP

#include "hip_arg_types.h"
#include "hip_private_api.h"
#include <hip/amd_detail/hip_prof_str.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <roctracer/hsa_prof_str.h>

typedef uint64_t sibir_address_t;

enum sibir_api_phase_t {
    SIBIR_API_PHASE_ENTER,
    SIBIR_API_PHASE_EXIT
};

enum sibir_ipoint_t {
    SIBIR_IPOINT_BEFORE,
    SIBIR_IPOINT_AFTER
};

struct kernel_descriptor_t {
    uint8_t reserved0[16];
    int64_t kernel_code_entry_byte_offset;
    uint8_t reserved1[20];
    uint32_t compute_pgm_rsrc3;
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;
    uint8_t reserved2[6];
};

typedef decltype(hsa_api_data_t::args) hsa_api_args_t;

#endif
