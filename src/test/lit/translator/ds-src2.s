// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_src2_kern.kd \
// RUN:   -initial-execution-point=0:ds_src2_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// DS_*_SRC2_* family — A-relative second-source addressing (pre-RDNA).
//   B = A + 4 * (offset[15] ? sign_ext(A[31:17]) : sign_ext({offset[14:0]}))
//   DS[A] = OP(DS[A], DS[B])  (DS[A] = DS[B] for WRITE)
//
// Each modeled op contributes loads/store at A and a load at B. We check
// that the B-address arithmetic (ashr 17, shl 17/2, etc.) appears and that
// the per-op result instructions are emitted.

// CHECK: define {{.*}} @ds_src2_kern

// B-address sign-extension: displ_A = A ashr 17. With $offset=8 (bit15=0),
// the select folds to the imm path and 4*8 = 32 byte displacement.
// CHECK-DAG: ashr i32 {{.*}}, 17
// CHECK-DAG: add i32 {{.*}}, 32

// AND_SRC2_B32, OR_SRC2_B32, XOR_SRC2_B32 produce the matching integer ops.
// CHECK-DAG: and i32
// CHECK-DAG: or i32
// CHECK-DAG: xor i32

// MIN_SRC2_I32 → signed-LT select. MAX_SRC2_U32 → unsigned-GT select.
// CHECK-DAG: icmp slt i32
// CHECK-DAG: icmp ugt i32

// MIN_SRC2_F32 → minnum.f32. ADD_SRC2_F32 → fadd float.
// CHECK-DAG: call float @llvm.minnum.f32(
// CHECK-DAG: fadd float

// INC_SRC2_U32 → uge compare.
// CHECK-DAG: icmp uge i32

// All ops store back to A's LDS slot (ptr addrspace(3)).
// CHECK-DAG: store i32 {{.*}} ptr addrspace(3)

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx900"
  .globl  ds_src2_kern
  .p2align  8
  .type   ds_src2_kern,@function
ds_src2_kern:
  v_mov_b32_e32 v0, 0
  // offset:8 selects displ = 8 (positive imm form; offset[15]=0).
  ds_add_src2_u32 v0 offset:8
  s_waitcnt lgkmcnt(0)
  ds_and_src2_b32 v0 offset:8
  s_waitcnt lgkmcnt(0)
  ds_or_src2_b32 v0 offset:8
  s_waitcnt lgkmcnt(0)
  ds_xor_src2_b32 v0 offset:8
  s_waitcnt lgkmcnt(0)
  ds_min_src2_i32 v0 offset:8
  s_waitcnt lgkmcnt(0)
  ds_max_src2_u32 v0 offset:8
  s_waitcnt lgkmcnt(0)
  ds_min_src2_f32 v0 offset:8
  s_waitcnt lgkmcnt(0)
  ds_add_src2_f32 v0 offset:8
  s_waitcnt lgkmcnt(0)
  ds_inc_src2_u32 v0 offset:8
  s_waitcnt lgkmcnt(0)
  ds_write_src2_b32 v0 offset:8
  s_waitcnt lgkmcnt(0)
  s_endpgm
ds_src2_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_src2_kern
    .amdhsa_group_segment_fixed_size 1024
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
