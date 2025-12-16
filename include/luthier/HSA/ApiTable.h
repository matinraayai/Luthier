//===-- ApiTable.h - HSA API Table Bounds Checking --------------*- C++ -*-===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// Defines a set of API Table containers for HSA with automatic bounds
/// checking for table.
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_API_TABLE_H
#define LUTHIER_HSA_API_TABLE_H
#include "luthier/HSA/HsaError.h"
#include "luthier/Common/ErrorCheck.h"
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <llvm/Support/Error.h>

namespace luthier::hsa {

/// \brief Contains compile-time information regarding each API function of the
/// HSA runtime.
/// \details This struct is primarily used by the \c ApiTableContainer class
/// to provide direct
template <auto ApiFunc> struct ApiInfo;

#define DEFINE_HSA_API_INFO(ApiTableName, HsaFunc)                             \
  template <> struct ApiInfo<HsaFunc> {                                        \
    using ApiTable = ApiTableName;                                             \
    static constexpr auto ApiName = #HsaFunc;                                  \
    static constexpr auto ApiTablePointerToMember = &ApiTable::HsaFunc##_fn;   \
    static constexpr auto ApiTableOffset = offsetof(ApiTable, HsaFunc##_fn);   \
  };                                                                           \
  template <> struct ApiInfo<&ApiTableName::HsaFunc##_fn> {                    \
    using ApiTable = ApiTableName;                                             \
    static constexpr auto ApiName = #HsaFunc;                                  \
    static constexpr auto ApiTablePointerToMember = &ApiTable::HsaFunc##_fn;   \
    static constexpr auto ApiTableOffset = offsetof(ApiTable, HsaFunc##_fn);   \
  };

DEFINE_HSA_API_INFO(::CoreApiTable, hsa_init)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_shut_down)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_system_get_info)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_system_extension_supported)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_system_get_extension_table)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_iterate_agents)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_agent_get_info)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_create)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_soft_queue_create)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_destroy)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_inactivate)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_load_read_index_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_load_read_index_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_load_write_index_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_load_write_index_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_store_write_index_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_store_write_index_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_cas_write_index_scacq_screl)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_cas_write_index_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_cas_write_index_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_cas_write_index_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_add_write_index_scacq_screl)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_add_write_index_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_add_write_index_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_add_write_index_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_store_read_index_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_queue_store_read_index_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_agent_iterate_regions)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_region_get_info)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_agent_get_exception_policies)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_agent_extension_supported)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_memory_register)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_memory_deregister)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_memory_allocate)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_memory_free)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_memory_copy)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_memory_assign_agent)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_create)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_destroy)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_load_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_load_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_store_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_store_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_wait_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_wait_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_and_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_and_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_and_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_and_scacq_screl)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_or_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_or_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_or_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_or_scacq_screl)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_xor_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_xor_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_xor_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_xor_scacq_screl)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_exchange_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_exchange_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_exchange_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_exchange_scacq_screl)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_add_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_add_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_add_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_add_scacq_screl)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_subtract_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_subtract_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_subtract_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_subtract_scacq_screl)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_cas_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_cas_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_cas_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_cas_scacq_screl)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_isa_from_name)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_isa_get_info)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_isa_compatible)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_object_serialize)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_object_deserialize)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_object_destroy)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_object_get_info)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_object_get_symbol)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_symbol_get_info)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_object_iterate_symbols)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_create)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_destroy)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_load_code_object)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_freeze)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_get_info)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_global_variable_define)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_agent_global_variable_define)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_readonly_variable_define)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_validate)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_get_symbol)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_symbol_get_info)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_iterate_symbols)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_status_string)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_extension_get_name)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_system_major_extension_supported)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_system_get_major_extension_table)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_agent_major_extension_supported)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_cache_get_info)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_agent_iterate_caches)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_silent_store_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_silent_store_screlease)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_group_create)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_group_destroy)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_group_wait_any_scacquire)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_signal_group_wait_any_relaxed)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_agent_iterate_isas)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_isa_get_info_alt)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_isa_get_exception_policies)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_isa_get_round_method)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_wavefront_get_info)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_isa_iterate_wavefronts)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_object_get_symbol_from_name)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_object_reader_create_from_file)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_object_reader_create_from_memory)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_code_object_reader_destroy)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_create_alt)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_load_program_code_object)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_load_agent_code_object)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_validate_alt)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_get_symbol_by_name)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_iterate_agent_symbols)
DEFINE_HSA_API_INFO(::CoreApiTable, hsa_executable_iterate_program_symbols)

DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_coherency_get_type)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_coherency_set_type)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_profiling_set_profiler_enabled)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_profiling_async_copy_enable)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_profiling_get_dispatch_time)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_profiling_get_async_copy_time)
DEFINE_HSA_API_INFO(::AmdExtTable,
                    hsa_amd_profiling_convert_tick_to_system_domain)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_signal_async_handler)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_async_function)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_signal_wait_any)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_queue_cu_set_mask)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_pool_get_info)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_agent_iterate_memory_pools)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_pool_allocate)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_pool_free)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_async_copy)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_async_copy_on_engine)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_copy_engine_status)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_agent_memory_pool_get_info)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_agents_allow_access)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_pool_can_migrate)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_migrate)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_lock)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_unlock)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_fill)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_interop_map_buffer)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_interop_unmap_buffer)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_image_create)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_pointer_info)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_pointer_info_set_userdata)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_ipc_memory_create)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_ipc_memory_attach)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_ipc_memory_detach)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_signal_create)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_ipc_signal_create)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_ipc_signal_attach)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_register_system_event_handler)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_queue_set_priority)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_async_copy_rect)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_memory_lock_to_pool)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_register_deallocation_callback)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_deregister_deallocation_callback)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_signal_value_pointer)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_svm_attributes_set)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_svm_attributes_get)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_svm_prefetch_async)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_spm_acquire)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_spm_release)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_spm_set_dest_buffer)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_queue_cu_get_mask)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_portable_export_dmabuf)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_portable_close_dmabuf)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_queue_intercept_create)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_queue_intercept_register)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_runtime_queue_create_register)

DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_get_capability)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_data_get_info)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_create)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_import)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_export)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_copy)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_clear)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_destroy)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_sampler_create)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_sampler_destroy)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_get_capability_with_layout)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_data_get_info_with_layout)
DEFINE_HSA_API_INFO(::ImageExtTable, hsa_ext_image_create_with_layout)

DEFINE_HSA_API_INFO(::FinalizerExtTable, hsa_ext_program_create)
DEFINE_HSA_API_INFO(::FinalizerExtTable, hsa_ext_program_destroy)
DEFINE_HSA_API_INFO(::FinalizerExtTable, hsa_ext_program_add_module)
DEFINE_HSA_API_INFO(::FinalizerExtTable, hsa_ext_program_iterate_modules)
DEFINE_HSA_API_INFO(::FinalizerExtTable, hsa_ext_program_get_info)
DEFINE_HSA_API_INFO(::FinalizerExtTable, hsa_ext_program_finalize)
// clang-format on

#if HSA_AMD_EXT_API_TABLE_MAJOR_VERSION >= 0x02
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_address_reserve)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_address_free)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_handle_create)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_handle_release)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_map)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_unmap)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_set_access)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_get_access)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_export_shareable_handle)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_import_shareable_handle)
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_retain_alloc_handle)
DEFINE_HSA_API_INFO(::AmdExtTable,
                    hsa_amd_vmem_get_alloc_properties_from_handle)
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x01
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_agent_set_async_scratch_limit)
#endif
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x02
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_queue_get_info)
#endif
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x03
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_vmem_address_reserve_align)
#endif
#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x04
DEFINE_HSA_API_INFO(::AmdExtTable, hsa_amd_enable_logging)
#endif
#endif

/// \brief Struct containing \c constexpr compile-time info regarding individual
/// tables inside <tt>::HsaApiTable</tt>. Used to provide convenience accessors
/// using the table's type. If a table is not present here, simply define a
/// specialization of this template.
template <typename ApiTableType> struct ApiTableInfo;

template <> struct ApiTableInfo<::CoreApiTable> {
  static constexpr auto Name = "core";
  static constexpr auto PointerToMemberRootAccessor = &::HsaApiTable::core_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::core;
  static constexpr auto MajorVer = HSA_CORE_API_TABLE_MAJOR_VERSION;
  static constexpr auto MinorVer = sizeof(::CoreApiTable);
  static constexpr auto StepVer = HSA_CORE_API_TABLE_STEP_VERSION;
  static constexpr auto TableOffset = offsetof(HsaApiTable, core_);
};

template <> struct ApiTableInfo<::AmdExtTable> {
  static constexpr auto Name = "amd";
  static constexpr auto PointerToMemberRootAccessor = &::HsaApiTable::amd_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::amd_ext;
  static constexpr auto MajorVer = HSA_AMD_EXT_API_TABLE_MAJOR_VERSION;
  static constexpr auto MinorVer = sizeof(::AmdExtTable);
  static constexpr auto StepVer = HSA_AMD_EXT_API_TABLE_STEP_VERSION;
  static constexpr auto TableOffset = offsetof(HsaApiTable, amd_ext_);
};

template <> struct ApiTableInfo<::FinalizerExtTable> {
  static constexpr auto Name = "finalizer";
  static constexpr auto PointerToMemberRootAccessor =
      &::HsaApiTable::finalizer_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::finalizer_ext;
  static constexpr auto MajorVer = HSA_FINALIZER_API_TABLE_MAJOR_VERSION;
  static constexpr auto MinorVer = sizeof(::FinalizerExtTable);
  static constexpr auto StepVer = HSA_FINALIZER_API_TABLE_STEP_VERSION;
  static constexpr auto TableOffset = offsetof(HsaApiTable, finalizer_ext_);
};

template <> struct ApiTableInfo<::ImageExtTable> {
  static constexpr auto Name = "image";
  static constexpr auto PointerToMemberRootAccessor =
      &::HsaApiTable::image_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::image_ext;
  static constexpr auto MajorVer = HSA_IMAGE_API_TABLE_MAJOR_VERSION;
  static constexpr auto MinorVer = sizeof(::ImageExtTable);
  static constexpr auto StepVer = HSA_IMAGE_API_TABLE_STEP_VERSION;
  static constexpr auto TableOffset = offsetof(HsaApiTable, image_ext_);
};

template <> struct ApiTableInfo<::ToolsApiTable> {
  static constexpr auto Name = "tools";
  static constexpr auto PointerToMemberRootAccessor = &::HsaApiTable::tools_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::tools;
  static constexpr auto MajorVer = HSA_TOOLS_API_TABLE_MAJOR_VERSION;
  static constexpr auto MinorVer = sizeof(::ToolsApiTable);
  static constexpr auto StepVer = HSA_TOOLS_API_TABLE_STEP_VERSION;
  static constexpr auto TableOffset = offsetof(HsaApiTable, tools_);
};

template <> struct ApiTableInfo<::PcSamplingExtTable> {
  static constexpr auto Name = "pc sampling";
  static constexpr auto PointerToMemberRootAccessor =
      &::HsaApiTable::pc_sampling_ext_;
  static constexpr auto PointerToMemberContainerAccessor =
      &::HsaApiTableContainer::pc_sampling_ext;
  static constexpr auto MajorVer = HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION;
  static constexpr auto MinorVer = sizeof(::PcSamplingExtTable);
  static constexpr auto StepVer = HSA_PC_SAMPLING_API_TABLE_STEP_VERSION;
  static constexpr auto TableOffset = offsetof(HsaApiTable, pc_sampling_ext_);
};

/// \brief A static mapping struct between the HSA extension enumerator and its
/// corresponding latest table version type and major/minor versions.
template <hsa_extension_t ExtType> struct ExtensionApiTableInfo;

template <> struct ExtensionApiTableInfo<HSA_EXTENSION_FINALIZER> {
  using TableType = hsa_ext_finalizer_1_00_pfn_t;
  static constexpr auto MajorVer = HSA_FINALIZER_API_TABLE_MAJOR_VERSION;
  static constexpr auto StepVer = HSA_FINALIZER_API_TABLE_STEP_VERSION;
};

template <> struct ExtensionApiTableInfo<HSA_EXTENSION_IMAGES> {
  using TableType = hsa_ext_images_1_pfn_t;
  static constexpr auto MajorVer = HSA_IMAGE_API_TABLE_MAJOR_VERSION;
  static constexpr auto StepVer = HSA_IMAGE_API_TABLE_STEP_VERSION;
};

template <> struct ExtensionApiTableInfo<HSA_EXTENSION_AMD_LOADER> {
  using TableType = hsa_ven_amd_loader_1_03_pfn_t;
  static constexpr auto MajorVer = 1;
  static constexpr auto StepVer = 3;
};

template <> struct ExtensionApiTableInfo<HSA_EXTENSION_AMD_PC_SAMPLING> {
  using TableType = hsa_ven_amd_pc_sampling_1_00_pfn_t;
  static constexpr auto MajorVer = HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION;
  static constexpr auto StepVer = HSA_PC_SAMPLING_API_TABLE_STEP_VERSION;
};

/// \brief Queries whether the HSA API \p Table struct provided by the HSA
/// runtime includes the pointer-to-member \p Entry (i.e., whether the
/// currently running HSA runtime supports a function or an extension table)
/// \details the implementation of the HSA standard by AMD (i.e. ROCr) provides
/// a set of core functionality from the base HSA standard, along with a set of
/// vendor-specific extensions added by AMD. Each extension has a corresponding
/// table in \c ::HsaApiTable (e.g., <tt>::HsaApiTable::core_</tt>). Each
/// extension table is itself a \c struct and each API inside the extension has
/// a corresponding function pointer field in the table (e.g., \c hsa_init has
/// the entry \c &::CoreApiTable::hsa_init_fn in the Core API table). The \c
/// ::HsaApiTable and its tables are forward-compatible i.e. newly added tables
/// and functions are added below the existing ones. This means that each entry
/// remains at a fixed offset in a table, ensuring a tool compiled against an
/// older version of HSA can also run against a newer version of HSA without any
/// interference from the newly added functionality. \n To check if an entry is
/// present in the HSA API table during runtime, a fixed-size \c version field
/// is provided at the very beginning of \c ::HsaApiTable (i.e., <tt>offsetof ==
/// 0</tt>) and also its extension tables. Tools can use the
/// \c ::ApiTableVersion::minor_id field to obtain the size of each
/// table in the active HSA runtime. Since table entries remain at a
/// fixed offset, a tool can confirm the currently running HSA runtime supports
/// its required extension or function by calling this function,
/// which makes sure the corresponding offset of the entry is smaller than the
/// size of the table.
/// \example To check if the HSA API \p Table has the core extension, one can
/// use the following:
/// \code{.cpp}
/// apiTableHasEntry<&::HsaApiTable::core_>(Table);
/// \endcode
/// \example To check if the \c ::CoreApiTable has the \c hsa_iterate_agents_fn
/// field, one can use the following:
/// \code{.cpp}
/// apiTableHasEntry<&::CoreApiTable::hsa_iterate_agents_fn>(CoreApiTable, );
/// \endcode
/// \param Table an HSA API Table or one of its sub-tables, likely obtained from
/// rocprofiler-sdk
/// \return \c true if \p Table contains <tt>Entry</tt>, \c false otherwise
/// \sa <hsa/hsa_api_trace.h> in ROCr
template <
    auto Entry, typename ApiTableType,
    typename = std::enable_if_t<!std::is_same_v<ApiTableType, ::HsaApiTable>>>
bool apiTableHasEntry(const ApiTableType &Table) {
  return ApiInfo<Entry>::ApiTableOffset < Table.version.minor_id;
}

/// Same as <tt>apiTableHasEntry(const auto &)</tt> but with the \p Entry
/// argument not hard coded and instead passed as a function argument
/// \sa apiTableHasEntry(const auto &)
template <typename HsaTableType, typename EntryType>
bool apiTableHasEntry(const HsaTableType &Table,
                      const EntryType HsaTableType::*Entry) {
  return reinterpret_cast<size_t>(&(Table.*Entry)) -
             reinterpret_cast<size_t>(&Table) <
         Table.version.minor_id;
}

template <typename ApiTableType>
bool apiTableHasEntry(const HsaApiTable &Table) {
  return ApiTableInfo<ApiTableType>::TableOffset < Table.version.minor_id;
}

/// \brief An HSA API table container which provides bounds checking over the
/// entries inside the API table
/// \tparam ApiTableType Type of the HSA API table e.g. \c ::CoreApiTable
/// \sa apiTableHasEntry
template <typename ApiTableType> class ApiTableContainer {
private:
  const ApiTableType &ApiTable{};

public:
  explicit ApiTableContainer(const ApiTableType &ApiTable)
      : ApiTable(ApiTable) {};

  /// \brief Checks if the \c Func is present in the API table snapshot
  /// \tparam Func pointer-to-member of the function
  /// entry inside the extension table being queried
  /// \return \c true if the function is available
  /// inside the API table, \c false otherwise.
  /// Reports a fatal error if the snapshot has not
  /// been initialized by rocprofiler-sdk
  template <auto Func> [[nodiscard]] bool tableSupportsFunction() const {
    return apiTableHasEntry<Func>(ApiTable);
  }

  /// \returns the function inside the snapshot
  /// associated with the pointer-to-member accessor
  /// \c Func
  template <auto Func> const auto &getFunction() const {
    LUTHIER_REPORT_FATAL_ON_ERROR(LUTHIER_HSA_ERROR_CHECK(
        tableSupportsFunction<Func>(), "The passed function is not inside the "
                                       "table."));
    return *(ApiTable.*ApiInfo<Func>::ApiTablePointerToMember);
  }

  /// Obtains the function \c Func from the table snapshot (if exists) and calls
  /// it with the passed \p Args and returns the results of the function call;
  ///
  template <auto Func, typename... ArgTypes>
  auto callFunction(ArgTypes... Args) const {
    return getFunction<Func>()(Args...);
  }
};

} // namespace luthier::hsa

#endif