// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:vopc_class.kd \
// RUN:   -initial-execution-point=0:vopc_class.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// V_CMP_CLASS_F32 — FP class test. src1 is a bitmask of classes
// (NaN/Inf/zero/...). Result is per-lane i1, broadcast via Ballot to VCC.
// Was modeled with empty Semantic = [] (no IR emitted); now uses
// int_amdgcn_class.f32 with BallotWaveMask + ImplicitDef VCC.

// CHECK: define {{.*}} @vopc_class
// CHECK: call i1 @llvm.amdgcn.class.f32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  vopc_class
  .p2align  8
  .type   vopc_class,@function
vopc_class:
  v_cmp_class_f32_e32 vcc, v0, v1
  s_endpgm

vopc_class_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel vopc_class
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 4
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
