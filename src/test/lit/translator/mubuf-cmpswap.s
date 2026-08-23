// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mubuf_cmpswap.kd \
// RUN:   -initial-execution-point=0:mubuf_cmpswap.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// BUFFER_ATOMIC_CMPSWAP_OFFEN_RTN — vector buffer compare-and-swap.
// $vdata is AVLdSt_64 (vector pair: src in elem 0, cmp in elem 1).
// The intrinsic returns the i32 pre-swap value; semantics ZExt it back
// to the i64 register width so setRegOperandValue's size check passes.
// Previously crashed luthier-llc — int_amdgcn_struct_buffer_atomic_cmpswap
// returns i32, but $vdata is i64, so the unwrapped SetNamedOperand
// failed the size assertion in setRegOperandValue.

// CHECK: define {{.*}} @mubuf_cmpswap
// CHECK-DAG: call i32 @llvm.ssa.copy.i32(i32 7)
// CHECK-DAG: call i32 @llvm.ssa.copy.i32(i32 11)
// CHECK-DAG: call i32 @llvm.amdgcn.struct.buffer.atomic.cmpswap.i32(i32 %{{[0-9]+}}, i32 %{{[0-9]+}},
// CHECK-DAG: zext i32 {{.*}} to i64

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mubuf_cmpswap
  .p2align  8
  .type   mubuf_cmpswap,@function
mubuf_cmpswap:
  v_mov_b32_e32 v0, 7
  v_mov_b32_e32 v1, 11
  v_mov_b32_e32 v2, 0
  buffer_atomic_cmpswap v[0:1], v2, s[0:3], 0 offen glc
  s_waitcnt vmcnt(0)
  s_endpgm

mubuf_cmpswap_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mubuf_cmpswap
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 4
  .end_amdhsa_kernel
