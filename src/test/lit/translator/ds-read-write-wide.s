// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_wide_kern.kd \
// RUN:   -initial-execution-point=0:ds_wide_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// DSSem fix #6: DS_READ_B96/B128 and DS_WRITE_B96/B128 should be modeled as
// a single load/store on i96/i128 (LLVM lowers to sequential dword traffic).
// Previously had empty `Semantic = []`.

// CHECK: define {{.*}} @ds_wide_kern
// CHECK-DAG: load i96, ptr addrspace(3)
// CHECK-DAG: load i128, ptr addrspace(3)
// CHECK-DAG: store {{.*}} ptr addrspace(3)

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_wide_kern
  .p2align  8
  .type   ds_wide_kern,@function
ds_wide_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 1
  v_mov_b32_e32 v2, 2
  v_mov_b32_e32 v3, 3
  v_mov_b32_e32 v4, 4
  ds_read_b96 v[5:7], v0
  s_waitcnt lgkmcnt(0)
  ds_write_b96 v0, v[1:3]
  s_waitcnt lgkmcnt(0)
  ds_read_b128 v[5:8], v0
  s_waitcnt lgkmcnt(0)
  ds_write_b128 v0, v[1:4]
  s_waitcnt lgkmcnt(0)
  s_endpgm

ds_wide_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_wide_kern
    .amdhsa_group_segment_fixed_size 64
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 9
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
