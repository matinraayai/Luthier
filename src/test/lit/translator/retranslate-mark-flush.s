// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   %luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-mark-retranslate-test,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:vector_add.kd \
// RUN:   -initial-execution-point=0:vector_add.kd \
// RUN:   -o /dev/null 2>&1 | %tee_out FileCheck --check-prefix=MARK %s
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   %luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-mark-retranslate-test,luthier-flush-translation-test,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:vector_add.kd \
// RUN:   -initial-execution-point=0:vector_add.kd \
// RUN:   -o /dev/null 2>&1 | %tee_out FileCheck --check-prefix=FLUSH %s

// Verifies the TranslationState dirty-mark/flush cycle: marking an MBB for
// re-translation persists a needs_retranslation annotation in the
// serialized representation (visible in the module's metadata table), and
// flushing clears the mark and re-lifts the function intact.

// MARK: define {{.*}} @vector_add
// MARK: !"luthier.machine_instr.needs_retranslation"
// MARK: !{i1 true}

// The flush clears every mark (only the cleared `i1 false` payload remains)
// and the re-lifted body matches the original translation.
// FLUSH: define {{.*}} @vector_add
// FLUSH-DAG: call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
// FLUSH-DAG: load i32, ptr addrspace(1)
// FLUSH-DAG: call float @llvm.fma.f32
// FLUSH-DAG: store i32 {{.*}}, ptr addrspace(1)
// FLUSH-NOT: !{i1 true}

	.amdhsa_code_object_version 6
	.text
	.protected	vector_add              ; -- Begin function vector_add
	.globl	vector_add
	.p2align	8
	.type	vector_add,@function
vector_add:                             ; @vector_add
; %bb.0:
	s_load_dword s2, s[4:5], 0x2c
	s_load_dwordx2 s[0:1], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_and_b32 s2, s2, 0xffff
	s_mul_i32 s6, s6, s2
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	s_load_dwordx2 s[2:3], s[4:5], 0x10
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s9
	v_add_co_u32_e32 v2, vcc, s8, v0
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc
	global_load_dword v4, v[2:3], off
	v_mov_b32_e32 v3, s11
	v_add_co_u32_e32 v2, vcc, s10, v0
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc
	global_load_dword v2, v[2:3], off
	v_mov_b32_e32 v3, s3
	v_add_co_u32_e32 v0, vcc, s2, v0
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v2, s1, v4
	global_store_dword v[0:1], v2, off
.LBB0_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel vector_add
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
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
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 12
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
	.size	vector_add, .Lfunc_end0-vector_add
                                        ; -- End function
	.set vector_add.num_vgpr, 5
	.set vector_add.num_agpr, 0
	.set vector_add.numbered_sgpr, 12
	.set vector_add.private_seg_size, 0
	.set vector_add.uses_vcc, 1
	.set vector_add.uses_flat_scratch, 0
	.set vector_add.has_dyn_sized_stack, 0
	.set vector_add.has_recursion, 0
	.set vector_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 152
; TotalNumSgprs: 16
; NumVgprs: 5
; NumAgprs: 0
; TotalNumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 16
; NumVGPRsForWavesPerEU: 5
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
    .sgpr_count:     16
    .sgpr_spill_count: 0
    .symbol:         vector_add.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     5
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
