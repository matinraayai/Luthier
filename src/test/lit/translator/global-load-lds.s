// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:gld_lds_kern.kd \
// RUN:   -initial-execution-point=0:gld_lds_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// GLOBAL_LOAD_LDS_* — DRAM-to-LDS direct load. Modeled via
// int_amdgcn_global_load_lds(global_ptr, local_ptr, byte_size, off, cpol).

// CHECK: define {{.*}} @gld_lds_kern
// CHECK-DAG: call void @llvm.amdgcn.global.load.lds(

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  gld_lds_kern
  .p2align  8
  .type   gld_lds_kern,@function
gld_lds_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v2, 0
  s_mov_b32 m0, 0
  global_load_lds_dword v[0:1], off
  s_waitcnt vmcnt(0) lgkmcnt(0)
  s_endpgm
gld_lds_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel gld_lds_kern
    .amdhsa_group_segment_fixed_size 4
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 4
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
