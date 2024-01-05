/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
 */
////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////


#ifndef LUTHIER_TYPES_H
#define LUTHIER_TYPES_H

#include "hip_arg_types.h"
#include "hip_private_api.h"
#include <hip/amd_detail/hip_prof_str.h>
#include <hip/hip_runtime_api.h>
//#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

typedef uint64_t luthier_address_t;

typedef struct {
    uint64_t handle;
} luthier_instruction_t;

typedef struct {
    luthier_instruction_t* instructions;
    size_t num_instructions;
} luthier_instruction_list_t;

constexpr const char *LUTHIER_DEVICE_FUNCTION_WRAP = "__luthier_wrap__";

constexpr const char *LUTHIER_RESERVED_MANAGED_VAR = "__luthier_reserved";

enum luthier_status_t {
    LUTHIER_STATUS_SUCCESS = 0,
    LUTHIER_STATUS_ERROR = 1,
    LUTHIER_STATUS_INVALID_ARGUMENT = 2
};

enum luthier_api_evt_phase_t {
    LUTHIER_API_EVT_PHASE_ENTER,
    LUTHIER_API_EVT_PHASE_EXIT
};

enum luthier_ipoint_t {
    LUTHIER_IPOINT_BEFORE,
    LUTHIER_IPOINT_AFTER
};

struct kernel_descriptor_t {
    uint32_t group_segment_fixed_size;
    uint32_t private_segment_fixed_size;
    uint32_t kernarg_size;
    uint8_t reserved0[4];
    int64_t kernel_code_entry_byte_offset;
    uint8_t reserved1[20];
    uint32_t compute_pgm_rsrc3;// GFX10+ and GFX90A+
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;
    uint8_t reserved2[6];
};

// Taken from ROCr
struct luthier_hsa_aql_packet_t {
    union {
        struct {
            uint16_t header;
            struct {
                uint8_t user_data[62];
            } body;
        } packet;
        struct {
            uint16_t header;
            uint8_t format;
            uint8_t rest[61];
        } amd_vendor;
        hsa_kernel_dispatch_packet_t dispatch;
        hsa_barrier_and_packet_t barrier_and;
        hsa_barrier_or_packet_t barrier_or;
        hsa_agent_dispatch_packet_t agent;
    };
};

enum hsa_api_evt_id_t {
    HSA_API_ID_FIRST = 0,
    /* block: CoreApi API */
    HSA_API_ID_hsa_init = 0,
    HSA_API_ID_hsa_shut_down = 1,
    HSA_API_ID_hsa_system_get_info = 2,
    HSA_API_ID_hsa_system_extension_supported = 3,
    HSA_API_ID_hsa_system_get_extension_table = 4,
    HSA_API_ID_hsa_iterate_agents = 5,
    HSA_API_ID_hsa_agent_get_info = 6,
    HSA_API_ID_hsa_queue_create = 7,
    HSA_API_ID_hsa_soft_queue_create = 8,
    HSA_API_ID_hsa_queue_destroy = 9,
    HSA_API_ID_hsa_queue_inactivate = 10,
    HSA_API_ID_hsa_queue_load_read_index_scacquire = 11,
    HSA_API_ID_hsa_queue_load_read_index_relaxed = 12,
    HSA_API_ID_hsa_queue_load_write_index_scacquire = 13,
    HSA_API_ID_hsa_queue_load_write_index_relaxed = 14,
    HSA_API_ID_hsa_queue_store_write_index_relaxed = 15,
    HSA_API_ID_hsa_queue_store_write_index_screlease = 16,
    HSA_API_ID_hsa_queue_cas_write_index_scacq_screl = 17,
    HSA_API_ID_hsa_queue_cas_write_index_scacquire = 18,
    HSA_API_ID_hsa_queue_cas_write_index_relaxed = 19,
    HSA_API_ID_hsa_queue_cas_write_index_screlease = 20,
    HSA_API_ID_hsa_queue_add_write_index_scacq_screl = 21,
    HSA_API_ID_hsa_queue_add_write_index_scacquire = 22,
    HSA_API_ID_hsa_queue_add_write_index_relaxed = 23,
    HSA_API_ID_hsa_queue_add_write_index_screlease = 24,
    HSA_API_ID_hsa_queue_store_read_index_relaxed = 25,
    HSA_API_ID_hsa_queue_store_read_index_screlease = 26,
    HSA_API_ID_hsa_agent_iterate_regions = 27,
    HSA_API_ID_hsa_region_get_info = 28,
    HSA_API_ID_hsa_agent_get_exception_policies = 29,
    HSA_API_ID_hsa_agent_extension_supported = 30,
    HSA_API_ID_hsa_memory_register = 31,
    HSA_API_ID_hsa_memory_deregister = 32,
    HSA_API_ID_hsa_memory_allocate = 33,
    HSA_API_ID_hsa_memory_free = 34,
    HSA_API_ID_hsa_memory_copy = 35,
    HSA_API_ID_hsa_memory_assign_agent = 36,
    HSA_API_ID_hsa_signal_create = 37,
    HSA_API_ID_hsa_signal_destroy = 38,
    HSA_API_ID_hsa_signal_load_relaxed = 39,
    HSA_API_ID_hsa_signal_load_scacquire = 40,
    HSA_API_ID_hsa_signal_store_relaxed = 41,
    HSA_API_ID_hsa_signal_store_screlease = 42,
    HSA_API_ID_hsa_signal_wait_relaxed = 43,
    HSA_API_ID_hsa_signal_wait_scacquire = 44,
    HSA_API_ID_hsa_signal_and_relaxed = 45,
    HSA_API_ID_hsa_signal_and_scacquire = 46,
    HSA_API_ID_hsa_signal_and_screlease = 47,
    HSA_API_ID_hsa_signal_and_scacq_screl = 48,
    HSA_API_ID_hsa_signal_or_relaxed = 49,
    HSA_API_ID_hsa_signal_or_scacquire = 50,
    HSA_API_ID_hsa_signal_or_screlease = 51,
    HSA_API_ID_hsa_signal_or_scacq_screl = 52,
    HSA_API_ID_hsa_signal_xor_relaxed = 53,
    HSA_API_ID_hsa_signal_xor_scacquire = 54,
    HSA_API_ID_hsa_signal_xor_screlease = 55,
    HSA_API_ID_hsa_signal_xor_scacq_screl = 56,
    HSA_API_ID_hsa_signal_exchange_relaxed = 57,
    HSA_API_ID_hsa_signal_exchange_scacquire = 58,
    HSA_API_ID_hsa_signal_exchange_screlease = 59,
    HSA_API_ID_hsa_signal_exchange_scacq_screl = 60,
    HSA_API_ID_hsa_signal_add_relaxed = 61,
    HSA_API_ID_hsa_signal_add_scacquire = 62,
    HSA_API_ID_hsa_signal_add_screlease = 63,
    HSA_API_ID_hsa_signal_add_scacq_screl = 64,
    HSA_API_ID_hsa_signal_subtract_relaxed = 65,
    HSA_API_ID_hsa_signal_subtract_scacquire = 66,
    HSA_API_ID_hsa_signal_subtract_screlease = 67,
    HSA_API_ID_hsa_signal_subtract_scacq_screl = 68,
    HSA_API_ID_hsa_signal_cas_relaxed = 69,
    HSA_API_ID_hsa_signal_cas_scacquire = 70,
    HSA_API_ID_hsa_signal_cas_screlease = 71,
    HSA_API_ID_hsa_signal_cas_scacq_screl = 72,
    HSA_API_ID_hsa_isa_from_name = 73,
    HSA_API_ID_hsa_isa_get_info = 74,
    HSA_API_ID_hsa_isa_compatible = 75,
    HSA_API_ID_hsa_code_object_serialize = 76,
    HSA_API_ID_hsa_code_object_deserialize = 77,
    HSA_API_ID_hsa_code_object_destroy = 78,
    HSA_API_ID_hsa_code_object_get_info = 79,
    HSA_API_ID_hsa_code_object_get_symbol = 80,
    HSA_API_ID_hsa_code_symbol_get_info = 81,
    HSA_API_ID_hsa_code_object_iterate_symbols = 82,
    HSA_API_ID_hsa_executable_create = 83,
    HSA_API_ID_hsa_executable_destroy = 84,
    HSA_API_ID_hsa_executable_load_code_object = 85,
    HSA_API_ID_hsa_executable_freeze = 86,
    HSA_API_ID_hsa_executable_get_info = 87,
    HSA_API_ID_hsa_executable_global_variable_define = 88,
    HSA_API_ID_hsa_executable_agent_global_variable_define = 89,
    HSA_API_ID_hsa_executable_readonly_variable_define = 90,
    HSA_API_ID_hsa_executable_validate = 91,
    HSA_API_ID_hsa_executable_get_symbol = 92,
    HSA_API_ID_hsa_executable_symbol_get_info = 93,
    HSA_API_ID_hsa_executable_iterate_symbols = 94,
    HSA_API_ID_hsa_status_string = 95,
    HSA_API_ID_hsa_extension_get_name = 96,
    HSA_API_ID_hsa_system_major_extension_supported = 97,
    HSA_API_ID_hsa_system_get_major_extension_table = 98,
    HSA_API_ID_hsa_agent_major_extension_supported = 99,
    HSA_API_ID_hsa_cache_get_info = 100,
    HSA_API_ID_hsa_agent_iterate_caches = 101,
    HSA_API_ID_hsa_signal_silent_store_relaxed = 102,
    HSA_API_ID_hsa_signal_silent_store_screlease = 103,
    HSA_API_ID_hsa_signal_group_create = 104,
    HSA_API_ID_hsa_signal_group_destroy = 105,
    HSA_API_ID_hsa_signal_group_wait_any_scacquire = 106,
    HSA_API_ID_hsa_signal_group_wait_any_relaxed = 107,
    HSA_API_ID_hsa_agent_iterate_isas = 108,
    HSA_API_ID_hsa_isa_get_info_alt = 109,
    HSA_API_ID_hsa_isa_get_exception_policies = 110,
    HSA_API_ID_hsa_isa_get_round_method = 111,
    HSA_API_ID_hsa_wavefront_get_info = 112,
    HSA_API_ID_hsa_isa_iterate_wavefronts = 113,
    HSA_API_ID_hsa_code_object_get_symbol_from_name = 114,
    HSA_API_ID_hsa_code_object_reader_create_from_file = 115,
    HSA_API_ID_hsa_code_object_reader_create_from_memory = 116,
    HSA_API_ID_hsa_code_object_reader_destroy = 117,
    HSA_API_ID_hsa_executable_create_alt = 118,
    HSA_API_ID_hsa_executable_load_program_code_object = 119,
    HSA_API_ID_hsa_executable_load_agent_code_object = 120,
    HSA_API_ID_hsa_executable_validate_alt = 121,
    HSA_API_ID_hsa_executable_get_symbol_by_name = 122,
    HSA_API_ID_hsa_executable_iterate_agent_symbols = 123,
    HSA_API_ID_hsa_executable_iterate_program_symbols = 124,

    /* block: AmdExt API */
    HSA_API_ID_hsa_amd_coherency_get_type = 125,
    HSA_API_ID_hsa_amd_coherency_set_type = 126,
    HSA_API_ID_hsa_amd_profiling_set_profiler_enabled = 127,
    HSA_API_ID_hsa_amd_profiling_async_copy_enable = 128,
    HSA_API_ID_hsa_amd_profiling_get_dispatch_time = 129,
    HSA_API_ID_hsa_amd_profiling_get_async_copy_time = 130,
    HSA_API_ID_hsa_amd_profiling_convert_tick_to_system_domain = 131,
    HSA_API_ID_hsa_amd_signal_async_handler = 132,
    HSA_API_ID_hsa_amd_async_function = 133,
    HSA_API_ID_hsa_amd_signal_wait_any = 134,
    HSA_API_ID_hsa_amd_queue_cu_set_mask = 135,
    HSA_API_ID_hsa_amd_memory_pool_get_info = 136,
    HSA_API_ID_hsa_amd_agent_iterate_memory_pools = 137,
    HSA_API_ID_hsa_amd_memory_pool_allocate = 138,
    HSA_API_ID_hsa_amd_memory_pool_free = 139,
    HSA_API_ID_hsa_amd_memory_async_copy = 140,
    HSA_API_ID_hsa_amd_agent_memory_pool_get_info = 141,
    HSA_API_ID_hsa_amd_agents_allow_access = 142,
    HSA_API_ID_hsa_amd_memory_pool_can_migrate = 143,
    HSA_API_ID_hsa_amd_memory_migrate = 144,
    HSA_API_ID_hsa_amd_memory_lock = 145,
    HSA_API_ID_hsa_amd_memory_unlock = 146,
    HSA_API_ID_hsa_amd_memory_fill = 147,
    HSA_API_ID_hsa_amd_interop_map_buffer = 148,
    HSA_API_ID_hsa_amd_interop_unmap_buffer = 149,
    HSA_API_ID_hsa_amd_image_create = 150,
    HSA_API_ID_hsa_amd_pointer_info = 151,
    HSA_API_ID_hsa_amd_pointer_info_set_userdata = 152,
    HSA_API_ID_hsa_amd_ipc_memory_create = 153,
    HSA_API_ID_hsa_amd_ipc_memory_attach = 154,
    HSA_API_ID_hsa_amd_ipc_memory_detach = 155,
    HSA_API_ID_hsa_amd_signal_create = 156,
    HSA_API_ID_hsa_amd_ipc_signal_create = 157,
    HSA_API_ID_hsa_amd_ipc_signal_attach = 158,
    HSA_API_ID_hsa_amd_register_system_event_handler = 159,
    HSA_API_ID_hsa_amd_queue_intercept_create = 160,
    HSA_API_ID_hsa_amd_queue_intercept_register = 161,
    HSA_API_ID_hsa_amd_queue_set_priority = 162,
    HSA_API_ID_hsa_amd_memory_async_copy_rect = 163,
    HSA_API_ID_hsa_amd_runtime_queue_create_register = 164,
    HSA_API_ID_hsa_amd_memory_lock_to_pool = 165,
    HSA_API_ID_hsa_amd_register_deallocation_callback = 166,
    HSA_API_ID_hsa_amd_deregister_deallocation_callback = 167,
    HSA_API_ID_hsa_amd_signal_value_pointer = 168,
    HSA_API_ID_hsa_amd_svm_attributes_set = 169,
    HSA_API_ID_hsa_amd_svm_attributes_get = 170,
    HSA_API_ID_hsa_amd_svm_prefetch_async = 171,
    HSA_API_ID_hsa_amd_queue_cu_get_mask = 172,

    /* block: ImageExt API */
    HSA_API_ID_hsa_ext_image_get_capability = 173,
    HSA_API_ID_hsa_ext_image_data_get_info = 174,
    HSA_API_ID_hsa_ext_image_create = 175,
    HSA_API_ID_hsa_ext_image_import = 176,
    HSA_API_ID_hsa_ext_image_export = 177,
    HSA_API_ID_hsa_ext_image_copy = 178,
    HSA_API_ID_hsa_ext_image_clear = 179,
    HSA_API_ID_hsa_ext_image_destroy = 180,
    HSA_API_ID_hsa_ext_sampler_create = 181,
    HSA_API_ID_hsa_ext_sampler_destroy = 182,
    HSA_API_ID_hsa_ext_image_get_capability_with_layout = 183,
    HSA_API_ID_hsa_ext_image_data_get_info_with_layout = 184,
    HSA_API_ID_hsa_ext_image_create_with_layout = 185,
    HSA_API_ID_LAST = hsa_api_evt_id_t::HSA_API_ID_hsa_ext_image_create_with_layout,

    /* HSA Events */
    HSA_EVT_ID_FIRST = 1000,
    HSA_EVT_ID_hsa_queue_packet_submit = 1000,
    HSA_EVT_ID_LAST = HSA_EVT_ID_hsa_queue_packet_submit
};

typedef union {
    /* block: CoreApi API */
    struct {
    } hsa_init;
    struct {
    } hsa_shut_down;
    struct {
        hsa_system_info_t attribute;
        void *value;
    } hsa_system_get_info;
    struct {
        uint16_t extension;
        uint16_t version_major;
        uint16_t version_minor;
        bool *result;
    } hsa_system_extension_supported;
    struct {
        uint16_t extension;
        uint16_t version_major;
        uint16_t version_minor;
        void *table;
    } hsa_system_get_extension_table;
    struct {
        hsa_status_t (*callback)(hsa_agent_t agent, void *data);
        void *data;
    } hsa_iterate_agents;
    struct {
        hsa_agent_t agent;
        hsa_agent_info_t attribute;
        void *value;
    } hsa_agent_get_info;
    struct {
        hsa_agent_t agent;
        uint32_t size;
        hsa_queue_type32_t type;
        void (*callback)(hsa_status_t status, hsa_queue_t *source, void *data);
        void *data;
        uint32_t private_segment_size;
        uint32_t group_segment_size;
        hsa_queue_t **queue;
    } hsa_queue_create;
    struct {
        hsa_region_t region;
        uint32_t size;
        hsa_queue_type32_t type;
        uint32_t features;
        hsa_signal_t doorbell_signal;
        hsa_queue_t **queue;
    } hsa_soft_queue_create;
    struct {
        hsa_queue_t *queue;
    } hsa_queue_destroy;
    struct {
        hsa_queue_t *queue;
    } hsa_queue_inactivate;
    struct {
        const hsa_queue_t *queue;
    } hsa_queue_load_read_index_scacquire;
    struct {
        const hsa_queue_t *queue;
    } hsa_queue_load_read_index_relaxed;
    struct {
        const hsa_queue_t *queue;
    } hsa_queue_load_write_index_scacquire;
    struct {
        const hsa_queue_t *queue;
    } hsa_queue_load_write_index_relaxed;
    struct {
        const hsa_queue_t *queue;
        uint64_t value;
    } hsa_queue_store_write_index_relaxed;
    struct {
        const hsa_queue_t *queue;
        uint64_t value;
    } hsa_queue_store_write_index_screlease;
    struct {
        const hsa_queue_t *queue;
        uint64_t expected;
        uint64_t value;
    } hsa_queue_cas_write_index_scacq_screl;
    struct {
        const hsa_queue_t *queue;
        uint64_t expected;
        uint64_t value;
    } hsa_queue_cas_write_index_scacquire;
    struct {
        const hsa_queue_t *queue;
        uint64_t expected;
        uint64_t value;
    } hsa_queue_cas_write_index_relaxed;
    struct {
        const hsa_queue_t *queue;
        uint64_t expected;
        uint64_t value;
    } hsa_queue_cas_write_index_screlease;
    struct {
        const hsa_queue_t *queue;
        uint64_t value;
    } hsa_queue_add_write_index_scacq_screl;
    struct {
        const hsa_queue_t *queue;
        uint64_t value;
    } hsa_queue_add_write_index_scacquire;
    struct {
        const hsa_queue_t *queue;
        uint64_t value;
    } hsa_queue_add_write_index_relaxed;
    struct {
        const hsa_queue_t *queue;
        uint64_t value;
    } hsa_queue_add_write_index_screlease;
    struct {
        const hsa_queue_t *queue;
        uint64_t value;
    } hsa_queue_store_read_index_relaxed;
    struct {
        const hsa_queue_t *queue;
        uint64_t value;
    } hsa_queue_store_read_index_screlease;
    struct {
        hsa_agent_t agent;
        hsa_status_t (*callback)(hsa_region_t region, void *data);
        void *data;
    } hsa_agent_iterate_regions;
    struct {
        hsa_region_t region;
        hsa_region_info_t attribute;
        void *value;
    } hsa_region_get_info;
    struct {
        hsa_agent_t agent;
        hsa_profile_t profile;
        uint16_t *mask;
    } hsa_agent_get_exception_policies;
    struct {
        uint16_t extension;
        hsa_agent_t agent;
        uint16_t version_major;
        uint16_t version_minor;
        bool *result;
    } hsa_agent_extension_supported;
    struct {
        void *ptr;
        size_t size;
    } hsa_memory_register;
    struct {
        void *ptr;
        size_t size;
    } hsa_memory_deregister;
    struct {
        hsa_region_t region;
        size_t size;
        void **ptr;
    } hsa_memory_allocate;
    struct {
        void *ptr;
    } hsa_memory_free;
    struct {
        void *dst;
        const void *src;
        size_t size;
    } hsa_memory_copy;
    struct {
        void *ptr;
        hsa_agent_t agent;
        hsa_access_permission_t access;
    } hsa_memory_assign_agent;
    struct {
        hsa_signal_value_t initial_value;
        uint32_t num_consumers;
        const hsa_agent_t *consumers;
        hsa_signal_t *signal;
    } hsa_signal_create;
    struct {
        hsa_signal_t signal;
    } hsa_signal_destroy;
    struct {
        hsa_signal_t signal;
    } hsa_signal_load_relaxed;
    struct {
        hsa_signal_t signal;
    } hsa_signal_load_scacquire;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_store_relaxed;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_store_screlease;
    struct {
        hsa_signal_t signal;
        hsa_signal_condition_t condition;
        hsa_signal_value_t compare_value;
        uint64_t timeout_hint;
        hsa_wait_state_t wait_state_hint;
    } hsa_signal_wait_relaxed;
    struct {
        hsa_signal_t signal;
        hsa_signal_condition_t condition;
        hsa_signal_value_t compare_value;
        uint64_t timeout_hint;
        hsa_wait_state_t wait_state_hint;
    } hsa_signal_wait_scacquire;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_and_relaxed;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_and_scacquire;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_and_screlease;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_and_scacq_screl;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_or_relaxed;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_or_scacquire;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_or_screlease;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_or_scacq_screl;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_xor_relaxed;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_xor_scacquire;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_xor_screlease;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_xor_scacq_screl;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_exchange_relaxed;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_exchange_scacquire;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_exchange_screlease;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_exchange_scacq_screl;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_add_relaxed;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_add_scacquire;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_add_screlease;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_add_scacq_screl;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_subtract_relaxed;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_subtract_scacquire;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_subtract_screlease;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_subtract_scacq_screl;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t expected;
        hsa_signal_value_t value;
    } hsa_signal_cas_relaxed;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t expected;
        hsa_signal_value_t value;
    } hsa_signal_cas_scacquire;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t expected;
        hsa_signal_value_t value;
    } hsa_signal_cas_screlease;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t expected;
        hsa_signal_value_t value;
    } hsa_signal_cas_scacq_screl;
    struct {
        const char *name;
        hsa_isa_t *isa;
    } hsa_isa_from_name;
    struct {
        hsa_isa_t isa;
        hsa_isa_info_t attribute;
        uint32_t index;
        void *value;
    } hsa_isa_get_info;
    struct {
        hsa_isa_t code_object_isa;
        hsa_isa_t agent_isa;
        bool *result;
    } hsa_isa_compatible;
    struct {
        hsa_code_object_t code_object;
        hsa_status_t (*alloc_callback)(size_t size, hsa_callback_data_t data, void **address);
        hsa_callback_data_t callback_data;
        const char *options;
        void **serialized_code_object;
        size_t *serialized_code_object_size;
    } hsa_code_object_serialize;
    struct {
        void *serialized_code_object;
        size_t serialized_code_object_size;
        const char *options;
        hsa_code_object_t *code_object;
    } hsa_code_object_deserialize;
    struct {
        hsa_code_object_t code_object;
    } hsa_code_object_destroy;
    struct {
        hsa_code_object_t code_object;
        hsa_code_object_info_t attribute;
        void *value;
    } hsa_code_object_get_info;
    struct {
        hsa_code_object_t code_object;
        const char *symbol_name;
        hsa_code_symbol_t *symbol;
    } hsa_code_object_get_symbol;
    struct {
        hsa_code_symbol_t code_symbol;
        hsa_code_symbol_info_t attribute;
        void *value;
    } hsa_code_symbol_get_info;
    struct {
        hsa_code_object_t code_object;
        hsa_status_t (*callback)(hsa_code_object_t code_object, hsa_code_symbol_t symbol, void *data);
        void *data;
    } hsa_code_object_iterate_symbols;
    struct {
        hsa_profile_t profile;
        hsa_executable_state_t executable_state;
        const char *options;
        hsa_executable_t *executable;
    } hsa_executable_create;
    struct {
        hsa_executable_t executable;
    } hsa_executable_destroy;
    struct {
        hsa_executable_t executable;
        hsa_agent_t agent;
        hsa_code_object_t code_object;
        const char *options;
    } hsa_executable_load_code_object;
    struct {
        hsa_executable_t executable;
        const char *options;
    } hsa_executable_freeze;
    struct {
        hsa_executable_t executable;
        hsa_executable_info_t attribute;
        void *value;
    } hsa_executable_get_info;
    struct {
        hsa_executable_t executable;
        const char *variable_name;
        void *address;
    } hsa_executable_global_variable_define;
    struct {
        hsa_executable_t executable;
        hsa_agent_t agent;
        const char *variable_name;
        void *address;
    } hsa_executable_agent_global_variable_define;
    struct {
        hsa_executable_t executable;
        hsa_agent_t agent;
        const char *variable_name;
        void *address;
    } hsa_executable_readonly_variable_define;
    struct {
        hsa_executable_t executable;
        uint32_t *result;
    } hsa_executable_validate;
    struct {
        hsa_executable_t executable;
        const char *module_name;
        const char *symbol_name;
        hsa_agent_t agent;
        int32_t call_convention;
        hsa_executable_symbol_t *symbol;
    } hsa_executable_get_symbol;
    struct {
        hsa_executable_symbol_t executable_symbol;
        hsa_executable_symbol_info_t attribute;
        void *value;
    } hsa_executable_symbol_get_info;
    struct {
        hsa_executable_t executable;
        hsa_status_t (*callback)(hsa_executable_t exec, hsa_executable_symbol_t symbol, void *data);
        void *data;
    } hsa_executable_iterate_symbols;
    struct {
        hsa_status_t status;
        const char **status_string;
    } hsa_status_string;
    struct {
        uint16_t extension;
        const char **name;
    } hsa_extension_get_name;
    struct {
        uint16_t extension;
        uint16_t version_major;
        uint16_t *version_minor;
        bool *result;
    } hsa_system_major_extension_supported;
    struct {
        uint16_t extension;
        uint16_t version_major;
        size_t table_length;
        void *table;
    } hsa_system_get_major_extension_table;
    struct {
        uint16_t extension;
        hsa_agent_t agent;
        uint16_t version_major;
        uint16_t *version_minor;
        bool *result;
    } hsa_agent_major_extension_supported;
    struct {
        hsa_cache_t cache;
        hsa_cache_info_t attribute;
        void *value;
    } hsa_cache_get_info;
    struct {
        hsa_agent_t agent;
        hsa_status_t (*callback)(hsa_cache_t cache, void *data);
        void *data;
    } hsa_agent_iterate_caches;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_silent_store_relaxed;
    struct {
        hsa_signal_t signal;
        hsa_signal_value_t value;
    } hsa_signal_silent_store_screlease;
    struct {
        uint32_t num_signals;
        const hsa_signal_t *signals;
        uint32_t num_consumers;
        const hsa_agent_t *consumers;
        hsa_signal_group_t *signal_group;
    } hsa_signal_group_create;
    struct {
        hsa_signal_group_t signal_group;
    } hsa_signal_group_destroy;
    struct {
        hsa_signal_group_t signal_group;
        const hsa_signal_condition_t *conditions;
        const hsa_signal_value_t *compare_values;
        hsa_wait_state_t wait_state_hint;
        hsa_signal_t *signal;
        hsa_signal_value_t *value;
    } hsa_signal_group_wait_any_scacquire;
    struct {
        hsa_signal_group_t signal_group;
        const hsa_signal_condition_t *conditions;
        const hsa_signal_value_t *compare_values;
        hsa_wait_state_t wait_state_hint;
        hsa_signal_t *signal;
        hsa_signal_value_t *value;
    } hsa_signal_group_wait_any_relaxed;
    struct {
        hsa_agent_t agent;
        hsa_status_t (*callback)(hsa_isa_t isa, void *data);
        void *data;
    } hsa_agent_iterate_isas;
    struct {
        hsa_isa_t isa;
        hsa_isa_info_t attribute;
        void *value;
    } hsa_isa_get_info_alt;
    struct {
        hsa_isa_t isa;
        hsa_profile_t profile;
        uint16_t *mask;
    } hsa_isa_get_exception_policies;
    struct {
        hsa_isa_t isa;
        hsa_fp_type_t fp_type;
        hsa_flush_mode_t flush_mode;
        hsa_round_method_t *round_method;
    } hsa_isa_get_round_method;
    struct {
        hsa_wavefront_t wavefront;
        hsa_wavefront_info_t attribute;
        void *value;
    } hsa_wavefront_get_info;
    struct {
        hsa_isa_t isa;
        hsa_status_t (*callback)(hsa_wavefront_t wavefront, void *data);
        void *data;
    } hsa_isa_iterate_wavefronts;
    struct {
        hsa_code_object_t code_object;
        const char *module_name;
        const char *symbol_name;
        hsa_code_symbol_t *symbol;
    } hsa_code_object_get_symbol_from_name;
    struct {
        hsa_file_t file;
        hsa_code_object_reader_t *code_object_reader;
    } hsa_code_object_reader_create_from_file;
    struct {
        const void *code_object;
        size_t size;
        hsa_code_object_reader_t *code_object_reader;
    } hsa_code_object_reader_create_from_memory;
    struct {
        hsa_code_object_reader_t code_object_reader;
    } hsa_code_object_reader_destroy;
    struct {
        hsa_profile_t profile;
        hsa_default_float_rounding_mode_t default_float_rounding_mode;
        const char *options;
        hsa_executable_t *executable;
    } hsa_executable_create_alt;
    struct {
        hsa_executable_t executable;
        hsa_code_object_reader_t code_object_reader;
        const char *options;
        hsa_loaded_code_object_t *loaded_code_object;
    } hsa_executable_load_program_code_object;
    struct {
        hsa_executable_t executable;
        hsa_agent_t agent;
        hsa_code_object_reader_t code_object_reader;
        const char *options;
        hsa_loaded_code_object_t *loaded_code_object;
    } hsa_executable_load_agent_code_object;
    struct {
        hsa_executable_t executable;
        const char *options;
        uint32_t *result;
    } hsa_executable_validate_alt;
    struct {
        hsa_executable_t executable;
        const char *symbol_name;
        const hsa_agent_t *agent;
        hsa_executable_symbol_t *symbol;
    } hsa_executable_get_symbol_by_name;
    struct {
        hsa_executable_t executable;
        hsa_agent_t agent;
        hsa_status_t (*callback)(hsa_executable_t exec, hsa_agent_t agent, hsa_executable_symbol_t symbol, void *data);
        void *data;
    } hsa_executable_iterate_agent_symbols;
    struct {
        hsa_executable_t executable;
        hsa_status_t (*callback)(hsa_executable_t exec, hsa_executable_symbol_t symbol, void *data);
        void *data;
    } hsa_executable_iterate_program_symbols;

    /* block: AmdExt API */
    struct {
        hsa_agent_t agent;
        hsa_amd_coherency_type_t *type;
    } hsa_amd_coherency_get_type;
    struct {
        hsa_agent_t agent;
        hsa_amd_coherency_type_t type;
    } hsa_amd_coherency_set_type;
    struct {
        hsa_queue_t *queue;
        int enable;
    } hsa_amd_profiling_set_profiler_enabled;
    struct {
        bool enable;
    } hsa_amd_profiling_async_copy_enable;
    struct {
        hsa_agent_t agent;
        hsa_signal_t signal;
        hsa_amd_profiling_dispatch_time_t *time;
    } hsa_amd_profiling_get_dispatch_time;
    struct {
        hsa_signal_t signal;
        hsa_amd_profiling_async_copy_time_t *time;
    } hsa_amd_profiling_get_async_copy_time;
    struct {
        hsa_agent_t agent;
        uint64_t agent_tick;
        uint64_t *system_tick;
    } hsa_amd_profiling_convert_tick_to_system_domain;
    struct {
        hsa_signal_t signal;
        hsa_signal_condition_t cond;
        hsa_signal_value_t value;
        hsa_amd_signal_handler handler;
        void *arg;
    } hsa_amd_signal_async_handler;
    struct {
        void (*callback)(void *arg);
        void *arg;
    } hsa_amd_async_function;
    struct {
        uint32_t signal_count;
        hsa_signal_t *signals;
        hsa_signal_condition_t *conds;
        hsa_signal_value_t *values;
        uint64_t timeout_hint;
        hsa_wait_state_t wait_hint;
        hsa_signal_value_t *satisfying_value;
    } hsa_amd_signal_wait_any;
    struct {
        const hsa_queue_t *queue;
        uint32_t num_cu_mask_count;
        const uint32_t *cu_mask;
    } hsa_amd_queue_cu_set_mask;
    struct {
        hsa_amd_memory_pool_t memory_pool;
        hsa_amd_memory_pool_info_t attribute;
        void *value;
    } hsa_amd_memory_pool_get_info;
    struct {
        hsa_agent_t agent;
        hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void *data);
        void *data;
    } hsa_amd_agent_iterate_memory_pools;
    struct {
        hsa_amd_memory_pool_t memory_pool;
        size_t size;
        uint32_t flags;
        void **ptr;
    } hsa_amd_memory_pool_allocate;
    struct {
        void *ptr;
    } hsa_amd_memory_pool_free;
    struct {
        void *dst;
        hsa_agent_t dst_agent;
        const void *src;
        hsa_agent_t src_agent;
        size_t size;
        uint32_t num_dep_signals;
        const hsa_signal_t *dep_signals;
        hsa_signal_t completion_signal;
    } hsa_amd_memory_async_copy;
    struct {
        hsa_agent_t agent;
        hsa_amd_memory_pool_t memory_pool;
        hsa_amd_agent_memory_pool_info_t attribute;
        void *value;
    } hsa_amd_agent_memory_pool_get_info;
    struct {
        uint32_t num_agents;
        const hsa_agent_t *agents;
        const uint32_t *flags;
        const void *ptr;
    } hsa_amd_agents_allow_access;
    struct {
        hsa_amd_memory_pool_t src_memory_pool;
        hsa_amd_memory_pool_t dst_memory_pool;
        bool *result;
    } hsa_amd_memory_pool_can_migrate;
    struct {
        const void *ptr;
        hsa_amd_memory_pool_t memory_pool;
        uint32_t flags;
    } hsa_amd_memory_migrate;
    struct {
        void *host_ptr;
        size_t size;
        hsa_agent_t *agents;
        int num_agent;
        void **agent_ptr;
    } hsa_amd_memory_lock;
    struct {
        void *host_ptr;
    } hsa_amd_memory_unlock;
    struct {
        void *ptr;
        uint32_t value;
        size_t count;
    } hsa_amd_memory_fill;
    struct {
        uint32_t num_agents;
        hsa_agent_t *agents;
        int interop_handle;
        uint32_t flags;
        size_t *size;
        void **ptr;
        size_t *metadata_size;
        const void **metadata;
    } hsa_amd_interop_map_buffer;
    struct {
        void *ptr;
    } hsa_amd_interop_unmap_buffer;
    struct {
        hsa_agent_t agent;
        const hsa_ext_image_descriptor_t *image_descriptor;
        const hsa_amd_image_descriptor_t *image_layout;
        const void *image_data;
        hsa_access_permission_t access_permission;
        hsa_ext_image_t *image;
    } hsa_amd_image_create;
    struct {
        const void *ptr;
        hsa_amd_pointer_info_t *info;
        void *(*alloc)(size_t);
        uint32_t *num_agents_accessible;
        hsa_agent_t **accessible;
    } hsa_amd_pointer_info;
    struct {
        const void *ptr;
        void *userdata;
    } hsa_amd_pointer_info_set_userdata;
    struct {
        void *ptr;
        size_t len;
        hsa_amd_ipc_memory_t *handle;
    } hsa_amd_ipc_memory_create;
    struct {
        const hsa_amd_ipc_memory_t *handle;
        size_t len;
        uint32_t num_agents;
        const hsa_agent_t *mapping_agents;
        void **mapped_ptr;
    } hsa_amd_ipc_memory_attach;
    struct {
        void *mapped_ptr;
    } hsa_amd_ipc_memory_detach;
    struct {
        hsa_signal_value_t initial_value;
        uint32_t num_consumers;
        const hsa_agent_t *consumers;
        uint64_t attributes;
        hsa_signal_t *signal;
    } hsa_amd_signal_create;
    struct {
        hsa_signal_t signal;
        hsa_amd_ipc_signal_t *handle;
    } hsa_amd_ipc_signal_create;
    struct {
        const hsa_amd_ipc_signal_t *handle;
        hsa_signal_t *signal;
    } hsa_amd_ipc_signal_attach;
    struct {
        hsa_amd_system_event_callback_t callback;
        void *data;
    } hsa_amd_register_system_event_handler;
    struct {
        hsa_agent_t agent_handle;
        uint32_t size;
        hsa_queue_type32_t type;
        void (*callback)(hsa_status_t status, hsa_queue_t *source, void *data);
        void *data;
        uint32_t private_segment_size;
        uint32_t group_segment_size;
        hsa_queue_t **queue;
    } hsa_amd_queue_intercept_create;
    struct {
        hsa_queue_t *queue;
        hsa_amd_queue_intercept_handler callback;
        void *user_data;
    } hsa_amd_queue_intercept_register;
    struct {
        hsa_queue_t *queue;
        hsa_amd_queue_priority_t priority;
    } hsa_amd_queue_set_priority;
    struct {
        const hsa_pitched_ptr_t *dst;
        const hsa_dim3_t *dst_offset;
        const hsa_pitched_ptr_t *src;
        const hsa_dim3_t *src_offset;
        const hsa_dim3_t *range;
        hsa_dim3_t range__val;
        hsa_agent_t copy_agent;
        hsa_amd_copy_direction_t dir;
        uint32_t num_dep_signals;
        const hsa_signal_t *dep_signals;
        hsa_signal_t completion_signal;
    } hsa_amd_memory_async_copy_rect;
    struct {
        hsa_amd_runtime_queue_notifier callback;
        void *user_data;
    } hsa_amd_runtime_queue_create_register;
    struct {
        void *host_ptr;
        size_t size;
        hsa_agent_t *agents;
        int num_agent;
        hsa_amd_memory_pool_t pool;
        uint32_t flags;
        void **agent_ptr;
    } hsa_amd_memory_lock_to_pool;
    struct {
        void *ptr;
        hsa_amd_deallocation_callback_t callback;
        void *user_data;
    } hsa_amd_register_deallocation_callback;
    struct {
        void *ptr;
        hsa_amd_deallocation_callback_t callback;
    } hsa_amd_deregister_deallocation_callback;
    struct {
        hsa_signal_t signal;
        volatile hsa_signal_value_t **value_ptr;
    } hsa_amd_signal_value_pointer;
    struct {
        void *ptr;
        size_t size;
        hsa_amd_svm_attribute_pair_t *attribute_list;
        size_t attribute_count;
    } hsa_amd_svm_attributes_set;
    struct {
        void *ptr;
        size_t size;
        hsa_amd_svm_attribute_pair_t *attribute_list;
        size_t attribute_count;
    } hsa_amd_svm_attributes_get;
    struct {
        void *ptr;
        size_t size;
        hsa_agent_t agent;
        uint32_t num_dep_signals;
        const hsa_signal_t *dep_signals;
        hsa_signal_t completion_signal;
    } hsa_amd_svm_prefetch_async;
    struct {
        const hsa_queue_t *queue;
        uint32_t num_cu_mask_count;
        uint32_t *cu_mask;
    } hsa_amd_queue_cu_get_mask;

    /* block: ImageExt API */
    struct {
        hsa_agent_t agent;
        hsa_ext_image_geometry_t geometry;
        const hsa_ext_image_format_t *image_format;
        uint32_t *capability_mask;
    } hsa_ext_image_get_capability;
    struct {
        hsa_agent_t agent;
        const hsa_ext_image_descriptor_t *image_descriptor;
        hsa_access_permission_t access_permission;
        hsa_ext_image_data_info_t *image_data_info;
    } hsa_ext_image_data_get_info;
    struct {
        hsa_agent_t agent;
        const hsa_ext_image_descriptor_t *image_descriptor;
        const void *image_data;
        hsa_access_permission_t access_permission;
        hsa_ext_image_t *image;
    } hsa_ext_image_create;
    struct {
        hsa_agent_t agent;
        const void *src_memory;
        size_t src_row_pitch;
        size_t src_slice_pitch;
        hsa_ext_image_t dst_image;
        const hsa_ext_image_region_t *image_region;
    } hsa_ext_image_import;
    struct {
        hsa_agent_t agent;
        hsa_ext_image_t src_image;
        void *dst_memory;
        size_t dst_row_pitch;
        size_t dst_slice_pitch;
        const hsa_ext_image_region_t *image_region;
    } hsa_ext_image_export;
    struct {
        hsa_agent_t agent;
        hsa_ext_image_t src_image;
        const hsa_dim3_t *src_offset;
        hsa_ext_image_t dst_image;
        const hsa_dim3_t *dst_offset;
        const hsa_dim3_t *range;
    } hsa_ext_image_copy;
    struct {
        hsa_agent_t agent;
        hsa_ext_image_t image;
        const void *data;
        const hsa_ext_image_region_t *image_region;
    } hsa_ext_image_clear;
    struct {
        hsa_agent_t agent;
        hsa_ext_image_t image;
    } hsa_ext_image_destroy;
    struct {
        hsa_agent_t agent;
        const hsa_ext_sampler_descriptor_t *sampler_descriptor;
        hsa_ext_sampler_t *sampler;
    } hsa_ext_sampler_create;
    struct {
        hsa_agent_t agent;
        hsa_ext_sampler_t sampler;
    } hsa_ext_sampler_destroy;
    struct {
        hsa_agent_t agent;
        hsa_ext_image_geometry_t geometry;
        const hsa_ext_image_format_t *image_format;
        hsa_ext_image_data_layout_t image_data_layout;
        uint32_t *capability_mask;
    } hsa_ext_image_get_capability_with_layout;
    struct {
        hsa_agent_t agent;
        const hsa_ext_image_descriptor_t *image_descriptor;
        hsa_access_permission_t access_permission;
        hsa_ext_image_data_layout_t image_data_layout;
        size_t image_data_row_pitch;
        size_t image_data_slice_pitch;
        hsa_ext_image_data_info_t *image_data_info;
    } hsa_ext_image_data_get_info_with_layout;
    struct {
        hsa_agent_t agent;
        const hsa_ext_image_descriptor_t *image_descriptor;
        const void *image_data;
        hsa_access_permission_t access_permission;
        hsa_ext_image_data_layout_t image_data_layout;
        size_t image_data_row_pitch;
        size_t image_data_slice_pitch;
        hsa_ext_image_t *image;
    } hsa_ext_image_create_with_layout;
} hsa_api_args_t;

typedef union {
    struct {
        luthier_hsa_aql_packet_t *packets;
        uint64_t pkt_count;
        uint64_t user_pkt_index;
    } hsa_queue_packet_submit;
} hsa_evt_args_t;

typedef union {
    hsa_api_args_t api_args;
    hsa_evt_args_t evt_args;
} hsa_api_evt_args_t;

#endif
