// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mubuf_legacy_fence.kd \
// RUN:   -initial-execution-point=0:mubuf_legacy_fence.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// BUFFER_WBINVL1* — legacy gfx9 L1 write-back+invalidate. No cpol operand;
// modeled as `fence syncscope("agent") acq_rel` (L1 is an agent-local cache).

// CHECK: define {{.*}} @mubuf_legacy_fence
// CHECK: fence syncscope("agent") acq_rel

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mubuf_legacy_fence
  .p2align  8
  .type   mubuf_legacy_fence,@function
mubuf_legacy_fence:
  buffer_wbinvl1_vol
  s_endpgm

mubuf_legacy_fence_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mubuf_legacy_fence
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 4
  .end_amdhsa_kernel
