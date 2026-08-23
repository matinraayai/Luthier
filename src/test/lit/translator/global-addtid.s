// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:addtid_kern.kd \
// RUN:   -initial-execution-point=0:addtid_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// GLOBAL_LOAD/STORE_DWORD_ADDTID — per-thread-id global addressing:
// addr = [saddr +] offset + lane_id * 4. lane_id from mbcnt_hi(mbcnt_lo).

// CHECK: define {{.*}} @addtid_kern
// CHECK-DAG: call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
// CHECK-DAG: call i32 @llvm.amdgcn.mbcnt.hi(i32 -1,
// CHECK-DAG: mul i32 {{.*}}, 4
// CHECK-DAG: load i32, ptr addrspace(1)
// CHECK-DAG: store i32 {{.*}} ptr addrspace(1)

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .globl  addtid_kern
  .p2align  8
  .type   addtid_kern,@function
addtid_kern:
  v_mov_b32_e32 v0, 0
  global_load_addtid_b32 v1, off offset:64
  s_waitcnt vmcnt(0)
  global_store_addtid_b32 v0, off offset:128
  s_endpgm
addtid_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel addtid_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 4
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
