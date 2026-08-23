// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_dual_kern.kd \
// RUN:   -initial-execution-point=0:ds_dual_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Group G coverage: dual-offset DS family. DS_READ2_B32 produces two loads
// at base+offset0*4 and base+offset1*4, packed into a v2i32. DS_WRITE2_B32
// stores two i32 values at the same stride. DS_READ2ST64_B32 uses stride 256
// instead of 4.

// CHECK: define {{.*}} @ds_dual_kern
// Two i32 loads at offset0*4 and offset1*4 (constants resolve to 0 and 4)
// CHECK-DAG: load i32, ptr addrspace(3)
// CHECK-DAG: insertelement <2 x i32>
// Two i32 stores from data0/data1
// CHECK-DAG: store i32 {{.*}} ptr addrspace(3)

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_dual_kern
  .p2align  8
  .type   ds_dual_kern,@function
ds_dual_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 11
  v_mov_b32_e32 v2, 22
  ds_read2_b32 v[3:4], v0 offset0:0 offset1:1
  s_waitcnt lgkmcnt(0)
  ds_write2_b32 v0, v1, v2 offset0:0 offset1:1
  s_waitcnt lgkmcnt(0)
  s_endpgm
ds_dual_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_dual_kern
    .amdhsa_group_segment_fixed_size 16
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
