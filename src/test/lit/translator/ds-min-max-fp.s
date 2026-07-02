// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_minmax_kern.kd \
// RUN:   -initial-execution-point=0:ds_minmax_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// DSSem fix #4: DS_MIN_F32 / DS_MAX_F32 must lower to llvm.minnum.f32 /
// llvm.maxnum.f32 (IEEE minimumNumber / maximumNumber), not to a
// select(fcmp olt) which mishandles NaN.

// CHECK: define {{.*}} @ds_minmax_kern
// CHECK-DAG: call {{.*}}float @llvm.minnum.f32
// CHECK-DAG: call {{.*}}float @llvm.maxnum.f32
// FCmp + select form must NOT appear for these instructions.
// CHECK-NOT: fcmp olt
// CHECK-NOT: fcmp ogt

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_minmax_kern
  .p2align  8
  .type   ds_minmax_kern,@function
ds_minmax_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0x3f800000
  ds_min_f32 v0, v1
  ds_max_f32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm

ds_minmax_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_minmax_kern
    .amdhsa_group_segment_fixed_size 4
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
