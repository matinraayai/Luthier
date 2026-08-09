// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_addtid_kern.kd \
// RUN:   -initial-execution-point=0:ds_addtid_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// DS_{READ,WRITE}_ADDTID_B32 — per-lane LDS address from
//   byte_addr = (({OFFSET1,OFFSET0} + M0[15:0]) & 0xFFFF) + laneID * 4
// laneID comes from int_amdgcn_mbcnt_lo/hi. The load/store goes through
// ptr addrspace(3).

// CHECK: define {{.*}} @ds_addtid_kern
// CHECK-DAG: call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
// CHECK-DAG: call i32 @llvm.amdgcn.mbcnt.hi(i32 -1,
// 0xFFFF = 65535 mask appears twice: M0[15:0] extraction and 16-bit sum trunc.
// CHECK-DAG: and i32 {{.*}}, 65535
// laneID * 4
// CHECK-DAG: mul i32 {{.*}}, 4
// CHECK-DAG: load i32, ptr addrspace(3)
// CHECK-DAG: store i32 {{.*}} ptr addrspace(3)

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_addtid_kern
  .p2align  8
  .type   ds_addtid_kern,@function
ds_addtid_kern:
  v_mov_b32_e32 v0, 42
  ds_read_addtid_b32 v1
  s_waitcnt lgkmcnt(0)
  ds_write_addtid_b32 v0
  s_waitcnt lgkmcnt(0)
  s_endpgm
ds_addtid_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_addtid_kern
    .amdhsa_group_segment_fixed_size 256
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
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
