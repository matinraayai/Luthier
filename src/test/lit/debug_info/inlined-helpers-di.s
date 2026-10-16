// RUN: llvm-mc -g --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,luthier-populate-debug-info,target(print-mir-prepare,function(machine-function(print)))' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z13square_kernelPffi.kd \
// RUN:   -initial-execution-point=0:_Z13square_kernelPffi.kd \
// RUN:   -o /dev/null > %t.ir 2> %t.mir
// The IR module is printed to stdout and the MIR bodies to stderr. Merging them
// with 2>&1 can splice one stream into the middle of a line of the other, so
// capture them separately and concatenate in a fixed order instead.
// RUN: cat %t.ir %t.mir | FileCheck %s

// Tests that DebugInfoPass rebuilds inlinedAt chains for code inlined from a
// different file, two levels deep. Assembly below was generated from:
//
//   // helper.h
//   5: __device__ __forceinline__ float square(float X) { return X * X; }
//   7: __device__ __forceinline__ float scaledSquare(float X, float K) {
//   8:   return K * square(X);
//   9: }
//
//   // tiny.hip (#include "helper.h")
//   15: __global__ void square_kernel(float *A, float K, int N) {
//   16:   int I = blockIdx.x * blockDim.x + threadIdx.x;
//   17:   if (I < N)
//   18:     A[I] = scaledSquare(A[I], K);
//   19: }
//
// with:
//   clang --offload-device-only --no-gpu-bundle-output --offload-arch=gfx942 \
//     -O2 -gline-tables-only -fdebug-info-for-profiling -S tiny.hip
//
// -gline-tables-only keeps the DWARF small (line tables + subprogram and
// inlined-subroutine DIEs only). -fdebug-info-for-profiling is needed because
// plain -gline-tables-only omits DW_AT_linkage_name, which DebugInfoPass uses
// to key its subprogram cache.
//
// Expected chains (cross-checked with llvm-symbolizer --inlining):
//   first fmul  (X * X):   square       helper.h:5:61
//                          -> inlined at scaledSquare helper.h:8:14
//                          -> inlined at square_kernel tiny.hip:18:12
//   second fmul (K * ...): scaledSquare helper.h:8:12
//                          -> inlined at square_kernel tiny.hip:18:12
//   store (A[I] = ...):    square_kernel tiny.hip:18:10, not inlined

// === IR: capture the !dbg of the kernel and of the three instructions ===

// CHECK: define{{.*}}@_Z13square_kernelPffi
// CHECK-SAME: !dbg [[KERNEL_SP:![0-9]+]]
// CHECK: fmul float {{.*}}, !dbg [[SQUARE_LOC:![0-9]+]]
// CHECK: fmul float {{.*}}, !dbg [[SCALED_LOC:![0-9]+]]
// CHECK: store i32 {{.*}}, !dbg [[STORE_LOC:![0-9]+]]

// === Metadata: subprograms, files, and the inlinedAt chains ===

// CHECK: [[TINY_FILE:![0-9]+]] = !DIFile(filename: "tiny.hip"
// CHECK: [[KERNEL_SP]] = distinct !DISubprogram(name: "square_kernel", linkageName: "_Z13square_kernelPffi"
// CHECK-SAME: file: [[TINY_FILE]]

// square's code: scoped to square (so its file is helper.h), inlined into
// scaledSquare, which is inlined into the kernel.
// CHECK: [[SQUARE_LOC]] = !DILocation(line: 5, column: 61, scope: [[SQUARE_SP:![0-9]+]], inlinedAt: [[SQUARE_CALL:![0-9]+]])
// CHECK: [[SQUARE_SP]] = distinct !DISubprogram(name: "square", linkageName: "_Z6squaref"
// CHECK-SAME: file: [[HELPER_FILE:![0-9]+]]
// CHECK: [[HELPER_FILE]] = !DIFile(filename: "helper.h"
// CHECK: [[SQUARE_CALL]] = !DILocation(line: 8, column: 14, scope: [[SCALED_SP:![0-9]+]], inlinedAt: [[SCALED_CALL:![0-9]+]])
// CHECK: [[SCALED_SP]] = distinct !DISubprogram(name: "scaledSquare", linkageName: "_Z12scaledSquareff"
// CHECK-SAME: file: [[HELPER_FILE]]
// The chain ends at the kernel's own subprogram.
// CHECK: [[SCALED_CALL]] = !DILocation(line: 18, column: 12, scope: [[KERNEL_SP]])

// scaledSquare's own multiply: one level of inlining, same call site.
// CHECK: [[SCALED_LOC]] = !DILocation(line: 8, column: 12, scope: [[SCALED_SP]], inlinedAt: [[SCALED_CALL]])

// The kernel's own store: scoped directly to the kernel, no inlinedAt.
// CHECK: [[STORE_LOC]] = !DILocation(line: 18, column: 10, scope: [[KERNEL_SP]])

// === MIR: the same locations are attached to the machine instructions ===

// CHECK: name: _Z13square_kernelPffi
// CHECK: V_MUL_F32_e32 {{.*}}debug-location [[SQUARE_LOC]]
// CHECK: V_MUL_F32_e32 {{.*}}debug-location [[SCALED_LOC]]
// CHECK: GLOBAL_STORE_DWORD {{.*}}debug-location [[STORE_LOC]]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z13square_kernelPffi   ; -- Begin function _Z13square_kernelPffi
	.globl	_Z13square_kernelPffi
	.p2align	8
	.type	_Z13square_kernelPffi,@function
_Z13square_kernelPffi:                  ; @_Z13square_kernelPffi
.Lfunc_begin0:
	.file	0 "/home/Luthier/sandbox" "tiny.hip" md5 0x906b49a55fb008d6a6c7be352517e39d
	.cfi_startproc
; %bb.0:                                ; %entry
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.file	1 "/opt/rocm/include/hip/amd_detail" "amd_hip_runtime.h" md5 0xf3ac153f6fbc746611d9adb666a6ab03
	.loc	1 264 58 prologue_end           ; /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:264:58 @[ /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:296:27 @[ tiny.hip:16:24 ] ]
	s_load_dword s3, s[0:1], 0x1c
.Ltmp0:
	.loc	1 259 58                        ; /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:259:58 @[ /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:287:27 @[ tiny.hip:16:11 ] ]
	s_load_dwordx2 s[4:5], s[0:1], 0x8
.Ltmp1:
	.loc	1 264 58                        ; /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:264:58 @[ /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:296:27 @[ tiny.hip:16:24 ] ]
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
.Ltmp2:
	.loc	0 16 22                         ; tiny.hip:16:22
	s_mul_i32 s2, s2, s3
	.loc	0 16 35 is_stmt 0               ; tiny.hip:16:35
	v_add_u32_e32 v0, s2, v0
	.loc	0 17 9 is_stmt 1                ; tiny.hip:17:9
	v_cmp_gt_i32_e32 vcc, s5, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:                                ; %if.then
.Ltmp3:
	.loc	1 259 58                        ; /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:259:58 @[ /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:287:27 @[ tiny.hip:16:11 ] ]
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[0:1]
.Ltmp4:
	.loc	0 18 25                         ; tiny.hip:18:25
	global_load_dword v2, v[0:1], off
.Ltmp5:
	.file	2 "." "helper.h" md5 0xd5b9e467088e3e11c8975ebf330a6c08
	.loc	2 5 61                          ; ./helper.h:5:61 @[ ./helper.h:8:14 @[ tiny.hip:18:12 ] ]
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v2, v2, v2
.Ltmp6:
	.loc	2 8 12                          ; ./helper.h:8:12 @[ tiny.hip:18:12 ]
	v_mul_f32_e32 v2, s4, v2
.Ltmp7:
	.loc	0 18 10                         ; tiny.hip:18:10
	global_store_dword v[0:1], v2, off
.LBB0_2:                                ; %if.end
	.loc	0 19 1                          ; tiny.hip:19:1
	s_endpgm
.Ltmp8:
.Lfunc_end0:
	.size	_Z13square_kernelPffi, .Lfunc_end0-_Z13square_kernelPffi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z13square_kernelPffi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
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
	.set .L_Z13square_kernelPffi.num_vgpr, 3
	.set .L_Z13square_kernelPffi.num_agpr, 0
	.set .L_Z13square_kernelPffi.numbered_sgpr, 6
	.set .L_Z13square_kernelPffi.num_named_barrier, 0
	.set .L_Z13square_kernelPffi.private_seg_size, 0
	.set .L_Z13square_kernelPffi.uses_vcc, 1
	.set .L_Z13square_kernelPffi.uses_flat_scratch, 0
	.set .L_Z13square_kernelPffi.has_dyn_sized_stack, 0
	.set .L_Z13square_kernelPffi.has_recursion, 0
	.set .L_Z13square_kernelPffi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 104
; TotalNumSgprs: 12
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
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 3
; AccumOffset: 4
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 0
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_664c9dfa42ae5906,@object ; @__hip_cuid_664c9dfa42ae5906
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_664c9dfa42ae5906
__hip_cuid_664c9dfa42ae5906:
	.byte	0                               ; 0x0
	.size	__hip_cuid_664c9dfa42ae5906, 1

	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	37                              ; DW_FORM_strx1
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	114                             ; DW_AT_str_offsets_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	37                              ; DW_FORM_strx1
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	115                             ; DW_AT_addr_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	116                             ; DW_AT_rnglists_base
	.byte	23                              ; DW_FORM_sec_offset
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	5                               ; DW_FORM_data2
	.byte	32                              ; DW_AT_inline
	.byte	33                              ; DW_FORM_implicit_const
	.byte	1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	32                              ; DW_AT_inline
	.byte	33                              ; DW_FORM_implicit_const
	.byte	1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	122                             ; DW_AT_call_all_calls
	.byte	25                              ; DW_FORM_flag_present
	.byte	110                             ; DW_AT_linkage_name
	.byte	37                              ; DW_FORM_strx1
	.byte	3                               ; DW_AT_name
	.byte	37                              ; DW_FORM_strx1
	.byte	58                              ; DW_AT_decl_file
	.byte	11                              ; DW_FORM_data1
	.byte	59                              ; DW_AT_decl_line
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.ascii	"\266B"                         ; DW_AT_GNU_discriminator
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	6                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	5                               ; DW_FORM_data2
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	7                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	35                              ; DW_FORM_rnglistx
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	8                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	9                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	27                              ; DW_FORM_addrx
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	1                               ; Abbrev [1] 0xc:0x8d DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	48                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.long	.Lstr_offsets_base0             ; DW_AT_str_offsets_base
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.byte	2                               ; DW_AT_comp_dir
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	.Laddr_table_base0              ; DW_AT_addr_base
	.long	.Lrnglists_table_base0          ; DW_AT_rnglists_base
	.byte	2                               ; Abbrev [2] 0x27:0x6 DW_TAG_subprogram
	.byte	3                               ; DW_AT_linkage_name
	.byte	4                               ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.short	264                             ; DW_AT_decl_line
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x2d:0x6 DW_TAG_subprogram
	.byte	5                               ; DW_AT_linkage_name
	.byte	6                               ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.short	296                             ; DW_AT_decl_line
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x33:0x6 DW_TAG_subprogram
	.byte	7                               ; DW_AT_linkage_name
	.byte	8                               ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.short	259                             ; DW_AT_decl_line
                                        ; DW_AT_inline
	.byte	2                               ; Abbrev [2] 0x39:0x6 DW_TAG_subprogram
	.byte	9                               ; DW_AT_linkage_name
	.byte	6                               ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.short	287                             ; DW_AT_decl_line
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x3f:0x5 DW_TAG_subprogram
	.byte	10                              ; DW_AT_linkage_name
	.byte	11                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.byte	5                               ; DW_AT_decl_line
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x44:0x5 DW_TAG_subprogram
	.byte	12                              ; DW_AT_linkage_name
	.byte	13                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.byte	7                               ; DW_AT_decl_line
                                        ; DW_AT_inline
	.byte	4                               ; Abbrev [4] 0x49:0x4f DW_TAG_subprogram
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	14                              ; DW_AT_linkage_name
	.byte	15                              ; DW_AT_name
	.byte	0                               ; DW_AT_decl_file
	.byte	15                              ; DW_AT_decl_line
	.byte	5                               ; Abbrev [5] 0x53:0x15 DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.byte	16                              ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	2                               ; DW_AT_GNU_discriminator
	.byte	6                               ; Abbrev [6] 0x5d:0xa DW_TAG_inlined_subroutine
	.long	39                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	296                             ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x68:0x14 DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.byte	16                              ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x71:0xa DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	287                             ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	8                               ; Abbrev [8] 0x7c:0x1b DW_TAG_inlined_subroutine
	.long	68                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_low_pc
	.long	.Ltmp7-.Ltmp5                   ; DW_AT_high_pc
	.byte	0                               ; DW_AT_call_file
	.byte	18                              ; DW_AT_call_line
	.byte	12                              ; DW_AT_call_column
	.byte	9                               ; Abbrev [9] 0x89:0xd DW_TAG_inlined_subroutine
	.long	63                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_low_pc
	.long	.Ltmp6-.Ltmp5                   ; DW_AT_high_pc
	.byte	2                               ; DW_AT_call_file
	.byte	8                               ; DW_AT_call_line
	.byte	14                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 ; Length
.Ldebug_list_header_start0:
	.short	5                               ; Version
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
	.long	2                               ; Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_table_base0
	.long	.Ldebug_ranges1-.Lrnglists_table_base0
.Ldebug_ranges0:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    ;   starting offset
	.uleb128 .Ltmp0-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp1-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp2-.Lfunc_begin0           ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges1:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp0-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp1-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp3-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp4-.Lfunc_begin0           ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	68                              ; Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)" ; string offset=0 ; AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)
.Linfo_string1:
	.asciz	"tiny.hip"                      ; string offset=109 ; tiny.hip
.Linfo_string2:
	.asciz	"/home/Luthier/sandbox"         ; string offset=118 ; /home/Luthier/sandbox
.Linfo_string3:
	.asciz	"_ZL21__hip_get_block_dim_xv"   ; string offset=140 ; _ZL21__hip_get_block_dim_xv
.Linfo_string4:
	.asciz	"__hip_get_block_dim_x"         ; string offset=168 ; __hip_get_block_dim_x
.Linfo_string5:
	.asciz	"_ZN24__hip_builtin_blockDim_t7__get_xEv" ; string offset=190 ; _ZN24__hip_builtin_blockDim_t7__get_xEv
.Linfo_string6:
	.asciz	"__get_x"                       ; string offset=230 ; __get_x
.Linfo_string7:
	.asciz	"_ZL21__hip_get_block_idx_xv"   ; string offset=238 ; _ZL21__hip_get_block_idx_xv
.Linfo_string8:
	.asciz	"__hip_get_block_idx_x"         ; string offset=266 ; __hip_get_block_idx_x
.Linfo_string9:
	.asciz	"_ZN24__hip_builtin_blockIdx_t7__get_xEv" ; string offset=288 ; _ZN24__hip_builtin_blockIdx_t7__get_xEv
.Linfo_string10:
	.asciz	"_Z6squaref"                    ; string offset=328 ; _Z6squaref
.Linfo_string11:
	.asciz	"square"                        ; string offset=339 ; square
.Linfo_string12:
	.asciz	"_Z12scaledSquareff"            ; string offset=346 ; _Z12scaledSquareff
.Linfo_string13:
	.asciz	"scaledSquare"                  ; string offset=365 ; scaledSquare
.Linfo_string14:
	.asciz	"_Z13square_kernelPffi"         ; string offset=378 ; _Z13square_kernelPffi
.Linfo_string15:
	.asciz	"square_kernel"                 ; string offset=400 ; square_kernel
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.long	.Linfo_string6
	.long	.Linfo_string7
	.long	.Linfo_string8
	.long	.Linfo_string9
	.long	.Linfo_string10
	.long	.Linfo_string11
	.long	.Linfo_string12
	.long	.Linfo_string13
	.long	.Linfo_string14
	.long	.Linfo_string15
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 ; Length of contribution
.Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp5
.Ldebug_addr_end0:
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_664c9dfa42ae5906
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
    .name:           _Z13square_kernelPffi
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z13square_kernelPffi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     3
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
