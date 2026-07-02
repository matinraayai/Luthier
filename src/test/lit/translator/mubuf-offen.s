// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mubuf_offen.kd \
// RUN:   -initial-execution-point=0:mubuf_offen.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// MUBUF semantics smoke test: BUFFER_LOAD_DWORD_OFFEN +
// BUFFER_STORE_DWORD_OFFEN. Exercises the v4i32-bitcast on srsrc
// (MUBUFSem.td fix). Must lift without translator assertion and emit
// struct.buffer.load/store intrinsics.

// CHECK: define {{.*}} @mubuf_offen
// CHECK-DAG: call {{.*}} @llvm.amdgcn.struct.buffer.load
// CHECK-DAG: call {{.*}} @llvm.amdgcn.struct.buffer.store

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mubuf_offen
  .p2align  8
  .type   mubuf_offen,@function
mubuf_offen:
  v_mov_b32_e32 v1, 0
  buffer_load_dword v0, v1, s[0:3], 0 offen offset:0
  s_waitcnt vmcnt(0)
  v_add_u32_e32 v0, 1, v0
  buffer_store_dword v0, v1, s[0:3], 0 offen offset:0
  s_endpgm

mubuf_offen_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mubuf_offen
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
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
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
