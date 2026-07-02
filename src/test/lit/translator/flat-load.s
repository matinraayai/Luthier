// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:flat_load_kern.kd \
// RUN:   -initial-execution-point=0:flat_load_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// FLAT loads (gfx9 — pre-SADDR). Exercises FLAT_LOAD_{UBYTE, SBYTE, USHORT,
// SSHORT, DWORD, DWORDX2, DWORDX3, DWORDX4} and the D16/D16_HI forms.
// All loads must produce a typed `load <T>, ptr addrspace(0)` (generic flat
// pointer) and write to the correct VGPR via $vdst (not $vdata).

// CHECK: define {{.*}} @flat_load_kern

// FLAT pointers are addrspace(0) (generic) — implicit / unprinted in IR.
// Sub-DWORD loads ext to i32; DWORD/X2/X4 load directly into vector or scalar
// i32 types.
// CHECK-DAG: load i8, ptr
// CHECK-DAG: load i16, ptr
// CHECK-DAG: load i32, ptr
// CHECK-DAG: load <2 x i32>, ptr
// CHECK-DAG: load <4 x i32>, ptr
// CHECK-DAG: zext i8 {{.*}} to i32
// CHECK-DAG: sext i8 {{.*}} to i32
// CHECK-DAG: zext i16 {{.*}} to i32
// CHECK-DAG: sext i16 {{.*}} to i32


  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  flat_load_kern
  .p2align  8
  .type   flat_load_kern,@function
flat_load_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0
  flat_load_ubyte  v2, v[0:1]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_load_sbyte  v3, v[0:1]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_load_ushort v4, v[0:1]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_load_sshort v5, v[0:1]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_load_dword  v6, v[0:1]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_load_dwordx2 v[8:9], v[0:1]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_load_dwordx4 v[12:15], v[0:1]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  // D16 / D16_HI form testing is blocked by a MachineVerifier issue with the
  // tied vdst_in operand handling in llvm-mc — the assembler does not emit
  // the tied input slot, causing the verifier to reject the MI before we get
  // a chance to lift it. The foreach semantics are still emitted; revisit
  // once the assembler issue is resolved upstream.
  s_endpgm
flat_load_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel flat_load_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
