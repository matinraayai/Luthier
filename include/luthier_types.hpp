#ifndef LUTHIER_TYPES_HPP
#define LUTHIER_TYPES_HPP

#include "hip_arg_types.h"
#include "hip_private_api.h"
#include <hip/amd_detail/hip_prof_str.h>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <roctracer/hsa_prof_str.h>
#include <hsa/amd_hsa_kernel_code.h>

typedef uint64_t luthier_address_t;

enum luthier_api_phase_t {
    LUTHIER_API_PHASE_ENTER,
    LUTHIER_API_PHASE_EXIT
};

enum luthier_ipoint_t {
    LUTHIER_IPOINT_BEFORE,
    LUTHIER_IPOINT_AFTER
};

//struct kernel_descriptor_t {
//    uint8_t reserved0[16];
//    int64_t kernel_code_entry_byte_offset;
//    uint8_t reserved1[20];
//    uint32_t compute_pgm_rsrc3;
//    amd_compute_pgm_rsrc_one32_t compute_pgm_rsrc1;
//    amd_compute_pgm_rsrc_two32_t compute_pgm_rsrc2;
//    uint16_t kernel_code_properties;
//    uint8_t reserved2[6];
//};

struct kernel_descriptor_t {
    uint32_t group_segment_fixed_size;
    uint32_t private_segment_fixed_size;
    uint32_t kernarg_size;
    uint8_t reserved0[4];
    int64_t kernel_code_entry_byte_offset;
    uint8_t reserved1[20];
    uint32_t compute_pgm_rsrc3; // GFX10+ and GFX90A+
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;
    uint8_t reserved2[6];
};


typedef decltype(hsa_api_data_t::args) hsa_api_args_t;

#endif
