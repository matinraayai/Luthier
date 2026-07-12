// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx906 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc --disable-verify -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:vop_cmp.kd \
// RUN:   -initial-execution-point=0:vop_cmp.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// VOPCSem smoke test covering V_CMP / V_CMPX semantic fills:
//
//   1. v_cmpx_eq_f32_e32 — V_CMPX_*_e32 dual-write form (writes VCC + EXEC).
//      Was broken: DefVal $result computed the ballot then dropped it.
//      Now: ballot → setRegOperandValue(VCC), setRegOperandValue(EXEC).
//   2. v_cmp_gt_u32_e32 — pre-existing predicate bug: used FCmpUGT instead of
//      ICMP_UGT for unsigned compare. Now uses ICMP_UGT.
//   3. v_cmp_le_i32_e32 — integer-compare section was an empty `!cond` stub;
//      now emits ICMP_SLE ballot.
//
// Notes:
// - VCC/EXEC writes go through setRegOperandValue (internal register map) and
//   are NOT visible as explicit `store` IR. They only surface when a later
//   instruction reads the register.
// - --disable-verify: SIInstrInfo::verifyInstruction doesn't model some
//   operand kinds we synthesize during code discovery.

// CHECK: define {{.*}} @vop_cmp

// (1) V_CMPX_EQ_F32_e32 → fcmp oeq → ballot
// CHECK-DAG: fcmp oeq float
// CHECK-DAG: call i64 @llvm.amdgcn.ballot.i64

// (2) V_CMP_GT_U32_e32 → icmp ugt (NOT ICMP_SGT, NOT FCmpUGT)
// CHECK-DAG: icmp ugt i32

// (3) V_CMP_LE_I32_e32 → icmp sle
// CHECK-DAG: icmp sle i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx906"
  .globl  vop_cmp
  .p2align  8
  .type   vop_cmp,@function
vop_cmp:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 1
  v_cmpx_eq_f32_e32 vcc, v0, v1
  v_cmp_gt_u32_e32 vcc, v0, v1
  v_cmp_le_i32_e32 vcc, v0, v1
  s_endpgm

vop_cmp_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel vop_cmp
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_queue_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_user_sgpr_flat_scratch_init 0
    .amdhsa_user_sgpr_private_segment_size 0
    .amdhsa_uses_dynamic_stack 0
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
