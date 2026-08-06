// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mubuf_fence.kd \
// RUN:   -initial-execution-point=0:mubuf_fence.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// BUFFER_INV / BUFFER_WBL2 — cache-control fences. Modeled as IR
// `fence syncscope(...) <ordering>`:
//   BUFFER_INV  -> acquire (invalidate)
//   BUFFER_WBL2 -> release (write-back)
// Scope decodes from cpol bits per subtarget (CDNA3 SC0/SC1 → agent etc.).

// CHECK: define {{.*}} @mubuf_fence
// CHECK-DAG: fence syncscope({{.*}}) acquire
// CHECK-DAG: fence syncscope({{.*}}) release

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  mubuf_fence
  .p2align  8
  .type   mubuf_fence,@function
mubuf_fence:
  buffer_inv sc1
  buffer_wbl2 sc1
  s_endpgm

mubuf_fence_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mubuf_fence
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 4
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
