//===-- ApiTable.h ----------------------------------------------*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
/// \file
/// Defines the set of API entries for rocprofiler-sdk.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_ROCPROFILER_ROCPROFILER_SDK_LIBRARY_H
#define LUTHIER_ROCPROFILER_ROCPROFILER_SDK_LIBRARY_H
#include "luthier/Common/DynamicLibraryFunctionEntry.h"
#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counter_config.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/counters.h>
#include <rocprofiler-sdk/experimental/registration.h>
#include <rocprofiler-sdk/experimental/thread-trace/agent.h>
#include <rocprofiler-sdk/experimental/thread-trace/dispatch.h>
#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/version.h>

namespace luthier {

#if ROCPROFILER_SDK_VERSION >= ROCPROFILER_SDK_COMPUTE_VERSION(1, 0, 0)
/* rocprofiler-sdk/agent.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_query_available_agents)
/* rocprofiler-sdk/buffer.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_create_buffer)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_destroy_buffer)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_flush_buffer)
/* rocprofiler-sdk/buffer_tracing.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_configure_buffer_tracing_service)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_iterate_buffer_tracing_kind_operations)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_iterate_buffer_tracing_kinds)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_iterate_buffer_tracing_record_args)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_query_buffer_tracing_kind_name)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_query_buffer_tracing_kind_operation_name)
/* rocprofiler-sdk/callback_tracing.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_configure_callback_tracing_service)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_iterate_callback_tracing_kind_operation_args)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_iterate_callback_tracing_kind_operations)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_iterate_callback_tracing_kinds)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_query_callback_tracing_kind_name)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_query_callback_tracing_kind_operation_name)
/* rocprofiler-sdk/context.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_context_is_active)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_context_is_valid)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_create_context)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_start_context)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_stop_context)
/* rocprofiler-sdk/counter_config.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_create_counter_config)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_destroy_counter_config)
/* rocprofiler-sdk/counters.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_iterate_agent_supported_counters)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_query_counter_info)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_query_record_counter_id)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_query_record_dimension_position)
/* rocprofiler-sdk/deprecated/counters.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_iterate_counter_dimensions)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_query_counter_instance_count)
/* rocprofiler-sdk/device_counting_service.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_configure_device_counting_service)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_sample_device_counting_service)
/* rocprofiler-sdk/dispatch_counting_service.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_configure_buffer_dispatch_counting_service)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_configure_callback_dispatch_counting_service)
/* rocprofiler-sdk/experimental/counters.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_load_counter_definition)
/* rocprofiler-sdk/experimental/registration.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_configure_attach)
/* rocprofiler-sdk/experimental/thread-trace/agent.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_configure_device_thread_trace_service)
/* rocprofiler-sdk/experimental/thread-trace/dispatch.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_configure_dispatch_thread_trace_service)
/* rocprofiler-sdk/experimental/thread-trace/trace_decoder.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_thread_trace_decoder_codeobj_load)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_thread_trace_decoder_codeobj_unload)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_thread_trace_decoder_create)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_thread_trace_decoder_destroy)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_thread_trace_decoder_info_string)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_trace_decode)
/* rocprofiler-sdk/external_correlation.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_configure_external_correlation_id_request_service)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_pop_external_correlation_id)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_push_external_correlation_id)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_query_external_correlation_id_request_kind_name)
/* rocprofiler-sdk/intercept_table.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_at_intercept_table_registration)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_query_intercept_table_name)
/* rocprofiler-sdk/internal_threading.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_assign_callback_thread)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_at_internal_thread_create)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_create_callback_thread)
/* rocprofiler-sdk/ompt.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_ompt_is_finalized)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_ompt_is_initialized)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_ompt_start_tool)
/* rocprofiler-sdk/pc_sampling.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_configure_pc_sampling_service)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_get_pc_sampling_instruction_not_issued_reason_name)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_get_pc_sampling_instruction_type_name)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_query_pc_sampling_agent_configurations)
/* rocprofiler-sdk/registration.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_configure)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_force_configure)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_is_finalized)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_is_initialized)
/* rocprofiler-sdk/rocprofiler.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_get_status_name)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_get_status_string)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_get_thread_id)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_get_timestamp)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_get_version)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_get_version_triplet)
#endif

#if ROCPROFILER_SDK_VERSION >= ROCPROFILER_SDK_COMPUTE_VERSION(1, 2, 0)
/* rocprofiler-sdk/experimental/registration.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_iterate_runtime_registration_info)
#endif

#if ROCPROFILER_SDK_VERSION >= ROCPROFILER_SDK_COMPUTE_VERSION(1, 3, 0)
/* rocprofiler-sdk/experimental/spm.h */
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_spm_configure_buffer_dispatch_service)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_spm_configure_callback_dispatch_service)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_spm_create_counter_config)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_spm_destroy_counter_config)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_spm_iterate_agent_supported_counters)
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(
    rocprofiler_spm_query_agent_configurations)
#endif

#if ROCPROFILER_SDK_VERSION >= ROCPROFILER_SDK_COMPUTE_VERSION(1, 0, 0) &&     \
    ROCPROFILER_SDK_VERSION < ROCPROFILER_SDK_COMPUTE_VERSION(1, 3, 0)
#include <rocprofiler-sdk/spm.h>
LUTHIER_DYNAMIC_LIBRARY_FUNCTION_ENTRY(rocprofiler_configure_spm_service)
#endif

} // namespace luthier

#endif