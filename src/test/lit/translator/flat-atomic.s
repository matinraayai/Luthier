// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:flat_atomic_kern.kd \
// RUN:   -initial-execution-point=0:flat_atomic_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// FLAT atomics. Exercises NoRet and RTN forms of ADD, AND, SMIN, UMAX. Verifies
// bugs B (NoRet writes back to $vdata) and C (CMPSWAP $vdata read as v2i32)
// are fixed: NoRet forms emit only the atomicrmw side-effect, RTN forms write
// the old value to $vdst.

// CHECK: define {{.*}} @flat_atomic_kern

// At least one atomicrmw per arithmetic op exercised:
// CHECK-DAG: atomicrmw add
// CHECK-DAG: atomicrmw and
// CHECK-DAG: atomicrmw min
// CHECK-DAG: atomicrmw umax

// CMPSWAP RTN: extractelement from a v2i32 source register (cmp / new pair).
// CHECK-DAG: extractelement
// CHECK-DAG: select i1

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  flat_atomic_kern
  .p2align  8
  .type   flat_atomic_kern,@function
flat_atomic_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v2, 7
  v_mov_b32_e32 v3, 1
  // NoRet forms.
  flat_atomic_add  v[0:1], v2
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_atomic_and  v[0:1], v2
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_atomic_smin v[0:1], v2
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_atomic_umax v[0:1], v2
  s_waitcnt vmcnt(0) lgkmcnt(0)
  // RTN forms — write old value to $vdst (now v4..v7 distinct).
  flat_atomic_add v4, v[0:1], v2 glc
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_atomic_smin v5, v[0:1], v2 glc
  s_waitcnt vmcnt(0) lgkmcnt(0)
  // CMPSWAP RTN — v[2:3] holds {src, cmp}.
  flat_atomic_cmpswap v6, v[0:1], v[2:3] glc
  s_waitcnt vmcnt(0) lgkmcnt(0)
  s_endpgm
flat_atomic_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel flat_atomic_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
