// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:cpol_kern.kd \
// RUN:   -initial-execution-point=0:cpol_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// CPOL-derived SyncScope mapping (gfx7-gfx11, pre-CDNA3, pre-gfx12):
//   GLC=0 SLC=0  → syncscope("workgroup")
//   GLC=1 SLC=0  → syncscope("agent")
//   GLC=*  SLC=1 → syncscope = System (default; no syncscope qualifier shown)
//
// AtomicOrdering remains Monotonic — AMDGPU has no HW acquire/release; those
// are produced by SIMemoryLegalizer's surrounding fences.

// CHECK: define {{.*}} @cpol_kern

// No modifiers → GLC=0 SLC=0 → workgroup.
// CHECK-DAG: atomicrmw add {{.*}} syncscope("workgroup") monotonic
// CHECK-DAG: atomicrmw and {{.*}} syncscope("workgroup") monotonic

// glc-modified RTN → GLC=1 SLC=0 → agent.
// CHECK-DAG: atomicrmw add {{.*}} syncscope("agent") monotonic
// CHECK-DAG: atomicrmw xor {{.*}} syncscope("agent") monotonic

// slc-modified → SLC=1 → system (no syncscope qualifier — System is default).
// CHECK-DAG: atomicrmw or ptr {{.*}} monotonic, align

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  cpol_kern
  .p2align  8
  .type   cpol_kern,@function
cpol_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v2, 7
  // NoRet, no modifiers — GLC=0 SLC=0 → workgroup.
  flat_atomic_add  v[0:1], v2
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_atomic_and  v[0:1], v2
  s_waitcnt vmcnt(0) lgkmcnt(0)
  // RTN, glc — GLC=1 SLC=0 → agent.
  flat_atomic_add v3, v[0:1], v2 glc
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_atomic_xor v4, v[0:1], v2 glc
  s_waitcnt vmcnt(0) lgkmcnt(0)
  // NoRet, slc — SLC=1 → system.
  flat_atomic_or  v[0:1], v2 slc
  s_waitcnt vmcnt(0) lgkmcnt(0)
  s_endpgm
cpol_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel cpol_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 5
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
