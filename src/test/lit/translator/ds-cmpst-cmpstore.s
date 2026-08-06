// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_cmpst_kern.kd \
// RUN:   -initial-execution-point=0:ds_cmpst_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// DSSem fix #3: DS_CMPST_B32 (pre-GFX11, gfx908) uses pre-GFX11 operand
// ordering — cmp = data0, src = data1. The compare is against data0 and
// the replacement on success is data1. Distinguishable via two distinct
// constants in v1 (= data0 = cmp = 11) and v2 (= data1 = src = 22).

// CHECK: define {{.*}} @ds_cmpst_kern
// CHECK-DAG: call i32 @llvm.ssa.copy.i32(i32 11)
// CHECK-DAG: call i32 @llvm.ssa.copy.i32(i32 22)
// CHECK-DAG: icmp eq i32 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-DAG: select i1 %{{[0-9]+}}, i32 %{{[0-9]+}}

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_cmpst_kern
  .p2align  8
  .type   ds_cmpst_kern,@function
ds_cmpst_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 11
  v_mov_b32_e32 v2, 22
  ds_cmpst_b32 v0, v1, v2
  s_waitcnt lgkmcnt(0)
  s_endpgm

ds_cmpst_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_cmpst_kern
    .amdhsa_group_segment_fixed_size 4
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
