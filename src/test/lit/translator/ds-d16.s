// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_d16_kern.kd \
// RUN:   -initial-execution-point=0:ds_d16_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Group I coverage: D16 read variants pack a sub-i32 load into either the
// low or high 16 bits of vdst, preserving the other half. We exercise
// DS_READ_U8_D16, DS_READ_U8_D16_HI, DS_READ_U16_D16, DS_READ_I8_D16.

// CHECK: define {{.*}} @ds_d16_kern
// CHECK-DAG: load i8, ptr addrspace(3)
// CHECK-DAG: load i16, ptr addrspace(3)
// CHECK-DAG: zext i8 {{.*}} to i32
// CHECK-DAG: sext i8 {{.*}} to i16
// CHECK-DAG: shl i32 {{.*}}, 16

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_d16_kern
  .p2align  8
  .type   ds_d16_kern,@function
ds_d16_kern:
  v_mov_b32_e32 v0, 0
  ds_read_u8_d16 v1, v0
  ds_read_u8_d16_hi v2, v0
  ds_read_u16_d16 v3, v0
  ds_read_i8_d16 v4, v0
  s_waitcnt lgkmcnt(0)
  s_endpgm
ds_d16_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_d16_kern
    .amdhsa_group_segment_fixed_size 4
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 5
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
