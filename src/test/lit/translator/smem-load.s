// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:smem_load.kd \
// RUN:   -initial-execution-point=0:smem_load.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// SMEM scalar load: S_LOAD_DWORD_IMM lifts to `load i32, ptr addrspace(4)`
// (sbase as global_ptr_ty + offset). S_BUFFER_LOAD_DWORD_IMM lifts to
// `call @llvm.amdgcn.s.buffer.load` with the cpol forwarded.

// CHECK: define {{.*}} @smem_load
// CHECK-DAG: load i32, ptr addrspace(1)
// CHECK-DAG: call i32 @llvm.amdgcn.s.buffer.load.i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  smem_load
  .p2align  8
  .type   smem_load,@function
smem_load:
  s_load_dword s4, s[0:1], 0x10
  s_buffer_load_dword s5, s[0:3], 0x10
  s_waitcnt lgkmcnt(0)
  s_endpgm

smem_load_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel smem_load
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel
