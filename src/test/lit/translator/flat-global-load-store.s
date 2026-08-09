// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:flat_rmw.kd \
// RUN:   -initial-execution-point=0:flat_rmw.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// FLAT semantics smoke test: global_load_dword + global_store_dword must
// lift through MIRToIRTranslator without assertion and emit proper typed
// load/store IR against ptr addrspace(1).

// CHECK: define {{.*}} @flat_rmw
// CHECK-DAG: load i32, ptr addrspace(1)
// CHECK-DAG: store i32 {{.*}}, ptr addrspace(1)

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  flat_rmw
  .p2align  8
  .type   flat_rmw,@function
flat_rmw:
  s_load_dwordx2 s[0:1], s[4:5], 0x0
  s_waitcnt lgkmcnt(0)
  v_mov_b32_e32 v0, s0
  v_mov_b32_e32 v1, s1
  global_load_dword v2, v[0:1], off
  s_waitcnt vmcnt(0)
  v_add_u32_e32 v2, 1, v2
  global_store_dword v[0:1], v2, off
  s_endpgm

flat_rmw_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel flat_rmw
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_queue_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_user_sgpr_flat_scratch_init 0
    .amdhsa_user_sgpr_private_segment_size 0
    .amdhsa_uses_dynamic_stack 0
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 6
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
