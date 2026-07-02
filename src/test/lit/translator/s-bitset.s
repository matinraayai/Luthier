// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:bitset_kern.kd \
// RUN:   -initial-execution-point=0:bitset_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// S_BITSET{0,1}_B{32,64} have a `$sdst_in` operand tied to `$sdst`. Neither
// the MCInst nor the assembly syntax mentions `$sdst_in` — the slot is
// synthesized by CodeDiscoveryPass's general tied-operand fixup.
//
// Semantics (per the AMDGPU ISA, also encoded in SOP1Sem.td):
//   S_BITSET0_B32: sdst = sdst_in & ~(1 << (src0 & 31))
//   S_BITSET1_B32: sdst = sdst_in |  (1 << (src0 & 31))
//
// If the tied operand isn't synthesized correctly, the read of `$sdst_in`
// would return garbage (the wrong physical register), and the and/or
// instructions below would reference a different value than the prior `s0`
// write.

// CHECK: define {{.*}} @bitset_kern

// S_BITSET0_B32: clears one bit — uses xor-with-(-1) to invert + and.
// CHECK-DAG: and i32 {{.*}}, 31
// CHECK-DAG: shl i32 1
// CHECK-DAG: xor i32 {{.*}}, -1
// CHECK-DAG: and i32

// S_BITSET1_B32: sets one bit — uses or.
// CHECK-DAG: or i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  bitset_kern
  .p2align  8
  .type   bitset_kern,@function
bitset_kern:
  // Load non-constant bit-position (s2) and initial sdst (s0) from kernargs;
  // constants would all be folded by InstSimplifyFolder.
  s_load_dwordx2 s[0:1], s[6:7], 0x0
  s_load_dword   s2,    s[6:7], 0x8
  s_waitcnt lgkmcnt(0)
  // S_BITSET1_B32: s0 |= (1 << (s2 & 31))
  s_bitset1_b32 s0, s2
  // S_BITSET0_B32: s0 &= ~(1 << (s2 & 31))
  s_bitset0_b32 s0, s2
  s_endpgm
bitset_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel bitset_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
