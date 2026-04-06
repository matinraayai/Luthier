#ifndef LUTHIER_TEST_DIRECT_HSA_API_TABLE_H
#define LUTHIER_TEST_DIRECT_HSA_API_TABLE_H

#include "luthier/HSA/ApiTable.h"
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

namespace luthier::test {

/// Build a CoreApiTable populated with direct function pointers to the real
/// HSA runtime.  This allows the Luthier HSA wrapper functions (which expect
/// an ApiTableContainer<CoreApiTable>) to be used from a standalone test
/// binary without rocprofiler's API table interception mechanism.
inline ::CoreApiTable buildDirectCoreApiTable() {
  ::CoreApiTable T{};

  // Version — must match what the runtime expects.
  T.version.major_id = HSA_API_TABLE_MAJOR_VERSION;
  T.version.minor_id = sizeof(::CoreApiTable);
  T.version.step_id = 0;

  // Populate every function pointer with the real HSA symbol.
  // Only the functions that are actually used by the Luthier wrappers and
  // the fuzzer need to be non-null; the rest can stay nullptr.

  // --- System ---
  T.hsa_init_fn = &hsa_init;
  T.hsa_shut_down_fn = &hsa_shut_down;
  T.hsa_system_get_info_fn = &hsa_system_get_info;
  T.hsa_system_extension_supported_fn = &hsa_system_extension_supported;
  T.hsa_system_get_extension_table_fn = &hsa_system_get_extension_table;

  // --- Agents ---
  T.hsa_iterate_agents_fn = &hsa_iterate_agents;
  T.hsa_agent_get_info_fn = &hsa_agent_get_info;
  T.hsa_agent_iterate_regions_fn = &hsa_agent_iterate_regions;
  T.hsa_agent_get_exception_policies_fn = &hsa_agent_get_exception_policies;
  T.hsa_agent_extension_supported_fn = &hsa_agent_extension_supported;

  // --- Queues ---
  T.hsa_queue_create_fn = &hsa_queue_create;
  T.hsa_queue_destroy_fn = &hsa_queue_destroy;
  T.hsa_queue_inactivate_fn = &hsa_queue_inactivate;
  T.hsa_queue_load_read_index_scacquire_fn =
      &hsa_queue_load_read_index_scacquire;
  T.hsa_queue_load_read_index_relaxed_fn = &hsa_queue_load_read_index_relaxed;
  T.hsa_queue_load_write_index_scacquire_fn =
      &hsa_queue_load_write_index_scacquire;
  T.hsa_queue_load_write_index_relaxed_fn =
      &hsa_queue_load_write_index_relaxed;
  T.hsa_queue_store_write_index_relaxed_fn =
      &hsa_queue_store_write_index_relaxed;
  T.hsa_queue_store_write_index_screlease_fn =
      &hsa_queue_store_write_index_screlease;
  T.hsa_queue_add_write_index_scacq_screl_fn =
      &hsa_queue_add_write_index_scacq_screl;
  T.hsa_queue_add_write_index_scacquire_fn =
      &hsa_queue_add_write_index_scacquire;
  T.hsa_queue_add_write_index_relaxed_fn = &hsa_queue_add_write_index_relaxed;
  T.hsa_queue_add_write_index_screlease_fn =
      &hsa_queue_add_write_index_screlease;
  T.hsa_queue_store_read_index_relaxed_fn =
      &hsa_queue_store_read_index_relaxed;
  T.hsa_queue_store_read_index_screlease_fn =
      &hsa_queue_store_read_index_screlease;
  T.hsa_queue_cas_write_index_scacq_screl_fn =
      &hsa_queue_cas_write_index_scacq_screl;
  T.hsa_queue_cas_write_index_scacquire_fn =
      &hsa_queue_cas_write_index_scacquire;
  T.hsa_queue_cas_write_index_relaxed_fn = &hsa_queue_cas_write_index_relaxed;
  T.hsa_queue_cas_write_index_screlease_fn =
      &hsa_queue_cas_write_index_screlease;

  // --- Regions / Memory ---
  T.hsa_region_get_info_fn = &hsa_region_get_info;
  T.hsa_memory_allocate_fn = &hsa_memory_allocate;
  T.hsa_memory_free_fn = &hsa_memory_free;
  T.hsa_memory_copy_fn = &hsa_memory_copy;
  T.hsa_memory_assign_agent_fn = &hsa_memory_assign_agent;
  T.hsa_memory_register_fn = &hsa_memory_register;
  T.hsa_memory_deregister_fn = &hsa_memory_deregister;

  // --- Signals ---
  T.hsa_signal_create_fn = &hsa_signal_create;
  T.hsa_signal_destroy_fn = &hsa_signal_destroy;
  T.hsa_signal_load_relaxed_fn = &hsa_signal_load_relaxed;
  T.hsa_signal_load_scacquire_fn = &hsa_signal_load_scacquire;
  T.hsa_signal_store_relaxed_fn = &hsa_signal_store_relaxed;
  T.hsa_signal_store_screlease_fn = &hsa_signal_store_screlease;
  T.hsa_signal_wait_relaxed_fn = &hsa_signal_wait_relaxed;
  T.hsa_signal_wait_scacquire_fn = &hsa_signal_wait_scacquire;

  // --- ISA ---
  T.hsa_isa_from_name_fn = &hsa_isa_from_name;
  T.hsa_isa_get_info_fn = &hsa_isa_get_info;
  T.hsa_isa_compatible_fn = &hsa_isa_compatible;
  T.hsa_agent_iterate_isas_fn = &hsa_agent_iterate_isas;
  T.hsa_isa_get_info_alt_fn = &hsa_isa_get_info_alt;
  T.hsa_isa_get_exception_policies_fn = &hsa_isa_get_exception_policies;
  T.hsa_isa_get_round_method_fn = &hsa_isa_get_round_method;
  T.hsa_isa_iterate_wavefronts_fn = &hsa_isa_iterate_wavefronts;
  T.hsa_wavefront_get_info_fn = &hsa_wavefront_get_info;

  // --- Executables ---
  T.hsa_executable_create_alt_fn = &hsa_executable_create_alt;
  T.hsa_executable_destroy_fn = &hsa_executable_destroy;
  T.hsa_executable_freeze_fn = &hsa_executable_freeze;
  T.hsa_executable_get_info_fn = &hsa_executable_get_info;
  T.hsa_executable_get_symbol_by_name_fn = &hsa_executable_get_symbol_by_name;
  T.hsa_executable_symbol_get_info_fn = &hsa_executable_symbol_get_info;
  T.hsa_executable_iterate_symbols_fn = &hsa_executable_iterate_symbols;
  T.hsa_executable_iterate_agent_symbols_fn =
      &hsa_executable_iterate_agent_symbols;

  // --- Code Object Reader ---
  T.hsa_code_object_reader_create_from_memory_fn =
      &hsa_code_object_reader_create_from_memory;
  T.hsa_code_object_reader_destroy_fn = &hsa_code_object_reader_destroy;
  T.hsa_executable_load_agent_code_object_fn =
      &hsa_executable_load_agent_code_object;

  // --- Caches ---
  T.hsa_agent_iterate_caches_fn = &hsa_agent_iterate_caches;
  T.hsa_cache_get_info_fn = &hsa_cache_get_info;

  // --- Status ---
  T.hsa_status_string_fn = &hsa_status_string;

  // --- Extensions ---
  T.hsa_agent_major_extension_supported_fn =
      &hsa_agent_major_extension_supported;
  T.hsa_system_major_extension_supported_fn =
      &hsa_system_major_extension_supported;

  return T;
}

} // namespace luthier::test

#endif // LUTHIER_TEST_DIRECT_HSA_API_TABLE_H
