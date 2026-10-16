// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,luthier-populate-debug-info,target(print-mir-prepare,function(machine-function(print)))' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z5scalePffi.kd \
// RUN:   -initial-execution-point=0:_Z5scalePffi.kd \
// RUN:   -o /dev/null > %t.ir 2> %t.mir
// The IR module is printed to stdout and the MIR bodies to stderr. Merging them
// with 2>&1 can splice one stream into the middle of a line of the other, so
// capture them separately and concatenate in a fixed order instead.
// RUN: cat %t.ir %t.mir | FileCheck %s

// Tests that DebugInfoPass skips a code object that has no DWARF compile
// units, and does not crash. Before the fix, DIBuilder::finalize() asserted
// with "creating type nodes without a CU is not supported". The run-time
// case is a ROCclr blit kernel (for example __amd_rocclr_copyBuffer), which
// ROCm ships without debug info.
//
// llvm-mc runs without -g on purpose: -g makes llvm-mc emit DWARF for the
// assembly file itself, so the code object would get 1 compile unit.
//
// Assembly below was generated from:
//
//   // no-dwarf.hip
//   5: __global__ void scale(float *A, float K, int N) {
//   6:   int I = blockIdx.x * blockDim.x + threadIdx.x;
//   7:   if (I < N)
//   8:     A[I] *= K;
//   9: }
//
// with (no -g):
//   clang -x hip --offload-device-only --no-gpu-bundle-output \
//     --offload-arch=gfx908 -O2 -S no-dwarf.hip
//
// Expected: no DICompileUnit, no DISubprogram, and no !dbg on any function
// or instruction.

// === IR: the kernel is lifted, and no debug info is created ===

// CHECK: define{{.*}}@_Z5scalePffi(
// CHECK-NOT: !dbg
// CHECK-NOT: DICompileUnit
// CHECK-NOT: DISubprogram
// CHECK-NOT: DIFile

// === MIR: no instruction has a debug location ===

// CHECK: name: _Z5scalePffi{{$}}
// CHECK-NOT: debug-location

	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z5scalePffi            ; -- Begin function _Z5scalePffi
	.globl	_Z5scalePffi
	.p2align	8
	.type	_Z5scalePffi,@function
_Z5scalePffi:                           ; @_Z5scalePffi
	.cfi_startproc
; %bb.0:                                ; %entry
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_dword s2, s[4:5], 0x1c
	s_load_dwordx2 s[0:1], s[4:5], 0x8
	s_waitcnt lgkmcnt(0)
	s_and_b32 s2, s2, 0xffff
	s_mul_i32 s6, s6, s2
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_i32_e32 vcc, s1, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:                                ; %if.then
	s_load_dwordx2 s[2:3], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, s3
	v_add_co_u32_e32 v0, vcc, s2, v0
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc
	global_load_dword v2, v[0:1], off
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v2, s0, v2
	global_store_dword v[0:1], v2, off
.LBB0_2:                                ; %if.end
	s_endpgm
.Lfunc_end0:
	.size	_Z5scalePffi, .Lfunc_end0-_Z5scalePffi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z5scalePffi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
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
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 7
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
                                        ; -- End function
	.set .L_Z5scalePffi.num_vgpr, 3
	.set .L_Z5scalePffi.num_agpr, 0
	.set .L_Z5scalePffi.numbered_sgpr, 7
	.set .L_Z5scalePffi.num_named_barrier, 0
	.set .L_Z5scalePffi.private_seg_size, 0
	.set .L_Z5scalePffi.uses_vcc, 1
	.set .L_Z5scalePffi.uses_flat_scratch, 0
	.set .L_Z5scalePffi.has_dyn_sized_stack, 0
	.set .L_Z5scalePffi.has_recursion, 0
	.set .L_Z5scalePffi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 112
; TotalNumSgprs: 11
; NumVgprs: 3
; NumAgprs: 0
; TotalNumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 11
; NumVGPRsForWavesPerEU: 3
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
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_5fee1773853f9822,@object ; @__hip_cuid_5fee1773853f9822
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5fee1773853f9822
__hip_cuid_5fee1773853f9822:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5fee1773853f9822, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5fee1773853f9822
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           A.coerce
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .name:           K
        .offset:         8
        .size:           4
        .value_kind:     by_value
      - .name:           N
        .offset:         12
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z5scalePffi
    .private_segment_fixed_size: 0
    .sgpr_count:     11
    .sgpr_spill_count: 0
    .symbol:         _Z5scalePffi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
