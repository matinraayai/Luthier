// Verifies the at-opcode mock pass creates a payload for every MI whose
// mnemonic contains the user-supplied substring (passed via
// -luthier-mock-inject-opcode). The kernel has exactly one v_fmac_f32, so
// one payload is expected. This test pins to gfx1100 (RDNA3, wave32) —
// the four payload-injection tests intentionally cover four distinct
// architectures including both wave64 (CDNA) and wave32 (RDNA) lifts.
//
// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 \
// RUN:    -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:    %luthier_mock_injection_plugin \
// RUN:    -passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-apply-instrumentation \
// RUN:    -code-object-paths=%t \
// RUN:    -initial-entrypoint=0:vector_add.kd \
// RUN:    -initial-execution-point=0:vector_add.kd \
// RUN:    -imodule-path=%luthier_test_imodule_dir/TrivialCounter-gfx1100.ll \
// RUN:    -imodule-output=%t.imod.ll \
// RUN:    -imodule-mir-passes= \
// RUN:    -imodule-ir-passes=luthier-mock-inject-at-opcode \
// RUN:    -luthier-mock-inject-opcode=V_FMAC_F32 \
// RUN:    -o /dev/null && \
// RUN: FileCheck %s < %t.imod.ll

// CHECK: define internal void @_Z11bumpCounterv
// CHECK-COUNT-1: define internal void @luthier.payload.vector_add.
// CHECK-NOT:    define internal void @luthier.payload.vector_add.{{.*}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
	.amdhsa_code_object_version 6
	.text
	.protected	vector_add              ; -- Begin function vector_add
	.globl	vector_add
	.p2align	8
	.type	vector_add,@function
vector_add:                             ; @vector_add
; %bb.0:
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x2c
	s_load_b64 s[2:3], s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_and_b32 s4, s4, 0xffff
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_u64_u32 v[1:2], null, s15, s4, v[0:1]
	v_cmp_gt_u32_e32 vcc_lo, s2, v1
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_b128 s[4:7], s[0:1], 0x0
	v_mov_b32_e32 v2, 0
	s_load_b64 s[0:1], s[0:1], 0x10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[0:1], 2, v[1:2]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v2, vcc_lo, s4, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e32 v3, vcc_lo, s5, v1, vcc_lo
	v_add_co_u32 v4, vcc_lo, s6, v0
	v_add_co_ci_u32_e32 v5, vcc_lo, s7, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, s0, v0
	global_load_b32 v2, v[2:3], off
	global_load_b32 v3, v[4:5], off
	v_add_co_ci_u32_e32 v1, vcc_lo, s1, v1, vcc_lo
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v3, s3, v2
	global_store_b32 v[0:1], v3, off
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vector_add
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 15
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
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
	.size	vector_add, .Lfunc_end0-vector_add
                                        ; -- End function
	.set vector_add.num_vgpr, 6
	.set vector_add.num_agpr, 0
	.set vector_add.numbered_sgpr, 16
	.set vector_add.private_seg_size, 0
	.set vector_add.uses_vcc, 1
	.set vector_add.uses_flat_scratch, 0
	.set vector_add.has_dyn_sized_stack, 0
	.set vector_add.has_recursion, 0
	.set vector_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 168
; TotalNumSgprs: 18
; NumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 6
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
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
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           vector_add
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         vector_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
