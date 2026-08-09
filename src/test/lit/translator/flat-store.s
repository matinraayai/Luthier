// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:flat_store_kern.kd \
// RUN:   -initial-execution-point=0:flat_store_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// FLAT stores (gfx9). Exercises FLAT_STORE_{BYTE, SHORT, DWORD, DWORDX2,
// DWORDX4} and the SHORT_D16_HI form (upper-half extract before store).

// CHECK: define {{.*}} @flat_store_kern

// Sub-DWORD stores truncate $vdata to i8/i16. DWORD/X2/X4 store directly.
// CHECK-DAG: store i8
// CHECK-DAG: store i16
// CHECK-DAG: store i32
// CHECK-DAG: store i64
// CHECK-DAG: store i128
// CHECK-DAG: trunc i32 {{.*}} to i16

// SHORT_D16_HI: lshr by 16, truncate to i16, store.
// CHECK-DAG: lshr i32 {{.*}}, 16

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  flat_store_kern
  .p2align  8
  .type   flat_store_kern,@function
flat_store_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v2, 42
  v_mov_b32_e32 v3, 7
  v_mov_b32_e32 v4, 1
  v_mov_b32_e32 v5, 1
  v_mov_b32_e32 v6, 1
  v_mov_b32_e32 v7, 1
  flat_store_byte  v[0:1], v2
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_store_short v[0:1], v2
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_store_short_d16_hi v[0:1], v2
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_store_dword v[0:1], v2
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_store_dwordx2 v[0:1], v[2:3]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_store_dwordx4 v[0:1], v[4:7]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  s_endpgm
flat_store_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel flat_store_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
