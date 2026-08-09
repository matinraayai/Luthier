// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:fence_kern.kd \
// RUN:   -initial-execution-point=0:fence_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// GLOBAL_INV / GLOBAL_WB / GLOBAL_WBINV — gfx12 cache fences. Modeled as
// IR `fence syncscope(...) <ordering>` where scope decodes from cpol bits 3-4.

// CHECK: define {{.*}} @fence_kern
// CHECK-DAG: fence syncscope("wavefront") acquire
// CHECK-DAG: fence syncscope("wavefront") release
// CHECK-DAG: fence syncscope("wavefront") acq_rel

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
  .globl  fence_kern
  .p2align  8
  .type   fence_kern,@function
fence_kern:
  // Default cpol=0 → wavefront scope on gfx12.
  global_inv
  global_wb
  global_wbinv
  s_endpgm
fence_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel fence_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 4
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
