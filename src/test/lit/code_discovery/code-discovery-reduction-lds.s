// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: (luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:    '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print-mir-prepare,function(machine-function(print)))' \
// RUN:    -code-object-paths=%t \
// RUN:    -initial-entrypoint=0:reduce_sum_lds.kd \
// RUN:    -initial-execution-point=0:reduce_sum_lds.kd \
// RUN:    -o /dev/null 2>&1 || true) > %t.out && \
// RUN: FileCheck %s < %t.out

// Regression test for two bugs in TraceFunctionTranslator that were exposed by
// kernels with control-flow on scalar-register/immediate comparisons:
//   * In `getOperandAsValue` the immediate operand defaulted to i64, while
//     register operands defaulted to the register's natural width (e.g. i32
//     for an SGPR). Semantics like S_CMP_LT_U32 — which read both operands
//     without an explicit out-type — would then build an ICmp on
//     mismatched-type operands and crash.
//   * In `invalidateOverlaps` for the "Written subset Stored" case the
//     chunk size was gcd(LeftSize, RightSize); it must also divide
//     WNumHalves, otherwise breakdownToVecTyFromAvailableValues cannot
//     evenly partition the stored value.

// CHECK:     define {{.*}} @reduce_sum_lds
// DSSem lifts ds_write_b32 / ds_read_b32 to typed store / load against
// ptr addrspace(3); VOPSem lifts v_add_f32 to a typed fadd.
// CHECK-DAG: store i32 {{.*}}, ptr addrspace(3)
// CHECK-DAG: load i32, ptr addrspace(3)
// CHECK-DAG: fadd float
// The S_CMP_LT_U32 should lift to a 32-bit icmp ult against an i32
// immediate. If the immediate had defaulted to i64 (as it did before the
// fix), the translator would have aborted on the ICmpInst operand-type
// assertion before reaching this point.
// CHECK-DAG: icmp ult i32 %{{[0-9]+}}, 2
// CHECK-DAG: icmp ult i32 %{{[0-9]+}}, 4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.amdhsa_code_object_version 6
	.text
	.protected	reduce_sum_lds          ; -- Begin function reduce_sum_lds
	.globl	reduce_sum_lds
	.p2align	8
	.type	reduce_sum_lds,@function
reduce_sum_lds:                         ; @reduce_sum_lds
; %bb.0:
	s_load_dword s0, s[4:5], 0x24
	s_load_dword s1, s[4:5], 0x10
	v_mov_b32_e32 v2, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s2, s0, 0xffff
	s_mul_i32 s0, s6, s2
	v_add_u32_e32 v1, s0, v0
	v_cmp_gt_u32_e32 vcc, s1, v1
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx2 s[8:9], s[4:5], 0x0
	v_mov_b32_e32 v2, 0
	v_lshlrev_b64 v[1:2], 2, v[1:2]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s9
	v_add_co_u32_e32 v1, vcc, s8, v1
	v_addc_co_u32_e32 v2, vcc, v3, v2, vcc
	global_load_dword v2, v[1:2], off
.LBB0_2:
	s_or_b64 exec, exec, s[0:1]
	v_lshlrev_b32_e32 v1, 2, v0
	s_cmp_lt_u32 s2, 2
	s_waitcnt vmcnt(0)
	ds_write_b32 v1, v2
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc0 .LBB0_7
.LBB0_3:
	s_mov_b32 s7, 0
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_5
; %bb.4:
	s_load_dwordx2 s[0:1], s[4:5], 0x8
	v_mov_b32_e32 v0, 0
	ds_read_b32 v1, v0
	s_lshl_b64 s[2:3], s[6:7], 2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	global_store_dword v0, v1, s[0:1]
.LBB0_5:
	s_endpgm
.LBB0_6:                                ;   in Loop: Header=BB0_7 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_cmp_lt_u32 s2, 4
	s_mov_b32 s2, s3
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_3
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_lshr_b32 s3, s2, 1
	v_cmp_gt_u32_e32 vcc, s3, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.8:                                ;   in Loop: Header=BB0_7 Depth=1
	v_lshl_add_u32 v2, s3, 2, v1
	ds_read_b32 v2, v2
	ds_read_b32 v3, v1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v3
	ds_write_b32 v1, v2
	s_branch .LBB0_6
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel reduce_sum_lds
		.amdhsa_group_segment_fixed_size 1024
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
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
		.amdhsa_next_free_sgpr 10
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	reduce_sum_lds, .Lfunc_end0-reduce_sum_lds
                                        ; -- End function
	.set reduce_sum_lds.num_vgpr, 4
	.set reduce_sum_lds.num_agpr, 0
	.set reduce_sum_lds.numbered_sgpr, 10
	.set reduce_sum_lds.private_seg_size, 0
	.set reduce_sum_lds.uses_vcc, 1
	.set reduce_sum_lds.uses_flat_scratch, 0
	.set reduce_sum_lds.has_dyn_sized_stack, 0
	.set reduce_sum_lds.has_recursion, 0
	.set reduce_sum_lds.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 280
; TotalNumSgprs: 14
; NumVgprs: 4
; NumAgprs: 0
; TotalNumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1024 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 4
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid,@object ; @__hip_cuid
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid
__hip_cuid:
	.byte	0                               ; 0x0
	.size	__hip_cuid, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           reduce_sum_lds
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         reduce_sum_lds.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
