// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:vop3p_packed.kd \
// RUN:   -initial-execution-point=0:vop3p_packed.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// VOP3P packed-math coverage: ops that were empty Semantic = [] now lift
// to typed packed IR ops. Doesn't model op_sel/neg_lo/hi/clamp modifiers —
// coarse data-flow semantic suitable for analysis.

// CHECK: define {{.*}} @vop3p_packed
// V_PK_ADD_F16: fadd <2 x half>
// CHECK-DAG: fadd <2 x half>
// V_PK_MUL_F16: fmul <2 x half>
// CHECK-DAG: fmul <2 x half>
// V_PK_FMA_F16: int_amdgcn-ish fma intrinsic call
// CHECK-DAG: call <2 x half> @llvm.fma.v2f16
// V_PK_ADD_I16: add <2 x i16>
// CHECK-DAG: add <2 x i16>
// V_PK_MAX_I16: smax intrinsic
// CHECK-DAG: call <2 x i16> @llvm.smax.v2i16
// V_PK_LSHLREV_B16: shl <2 x i16> (operand order reversed: src1 << src0)
// CHECK-DAG: shl <2 x i16>

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  vop3p_packed
  .p2align  8
  .type   vop3p_packed,@function
vop3p_packed:
  v_pk_add_f16 v0, v1, v2
  v_pk_mul_f16 v0, v1, v2
  v_pk_fma_f16 v0, v1, v2, v3
  v_pk_add_i16 v0, v1, v2
  v_pk_max_i16 v0, v1, v2
  v_pk_lshlrev_b16 v0, v1, v2
  s_endpgm

vop3p_packed_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel vop3p_packed
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 4
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
