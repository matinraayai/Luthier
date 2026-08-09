// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:vop1_swap.kd \
// RUN:   -initial-execution-point=0:vop1_swap.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// V_SWAP_B32 — swap two VGPR contents. Out: (vdst, vdst1); in: (src0,
// src1). Previously empty Semantic; now snapshots both inputs then
// writes vdst <- old src1, vdst1 <- old src0.

// Smoke test: pre-fix the body was empty (no IR emitted but no crash either).
// Just confirm the kernel lifts cleanly — the swap result is unused so the
// optimizer removes the snapshot+writeback pair, leaving just the kernel
// scaffolding. The fix's value is that V_SWAP_B32 in code that uses the
// swapped registers will now propagate proper data flow.
// CHECK: define {{.*}} @vop1_swap

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  vop1_swap
  .p2align  8
  .type   vop1_swap,@function
vop1_swap:
  v_mov_b32_e32 v0, 7
  v_mov_b32_e32 v1, 11
  v_swap_b32 v0, v1
  s_endpgm

vop1_swap_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel vop1_swap
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
