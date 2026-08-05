// RUN: llvm-mc -g --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared -o %t %t.o && \
// RUN: luthier-llc --load-pass-plugin=%luthier-opt-plugin \
// RUN:   --passes="luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-debug-info" \
// RUN:   --mtriple=amdgcn-amd-amdhsa --mcpu=gfx942 --disable-verify \
// RUN:   --code-object-paths=%t \
// RUN:   --initial-entrypoint="0:_Z6vecAddPKfS0_Pfi.kd" \
// RUN:   --initial-execution-point="0:_Z6vecAddPKfS0_Pfi.kd" \
// RUN:   --empty-input -o %t.mir && \
// RUN: FileCheck %s < %t.mir

// === IR-level debug-info metadata correctly created and linked ===

// The kernel function carries a !dbg reference to a DISubprogram. Capture
// the subprogram metadata id so later CHECKs can cross-reference it.
// CHECK: define{{.*}}@_Z6vecAddPKfS0_Pfi
// CHECK-SAME: !dbg [[SP:![0-9]+]]

// IR instructions now carry !dbg alongside !pcsections — capture one DILocation
// id here from the function body (before the metadata table in document order)
// so we can later verify it is well-formed and linked to the subprogram.
// CHECK: !dbg [[LOC:![0-9]+]], !pcsections

// Checking for the hard-coded producer string.
// CHECK: [[CU:![0-9]+]] = distinct !DICompileUnit(
// CHECK-SAME: producer: "Luthier Debug Info Pass"

// DIFile captured the source filename from the DWARF.
// CHECK: [[FILE:![0-9]+]] = !DIFile(filename: "vec-add.hip"

// The DISubprogram definition matches the captured SP id and correctly
// references both the DIFile and DICompileUnit — full IR graph is consistent.
// CHECK: [[SP]] = distinct !DISubprogram(
// CHECK-SAME: name: "vecAdd"
// CHECK-SAME: linkageName: "_Z6vecAddPKfS0_Pfi"
// CHECK-SAME: file: [[FILE]]
// CHECK-SAME: unit: [[CU]]

// [[LOC]] was captured from an IR instruction above. Verify the DILocation
// entry in the metadata table is well-formed and scoped to the subprogram.
// CHECK: [[LOC]] = !DILocation(line: {{[0-9]+}}, column: {{[0-9]+}}, scope: [[SP]])

// === Per-MI debug-location attachment verified at both ends of the kernel ===

// The first MI in the entry block references [[LOC]] directly — in LLVM 23 the
// MIR serializer emits a metadata id reference rather than an inline DILocation.
// CHECK: bb.0
// CHECK: debug-location [[LOC]]

// The kernel terminator S_ENDPGM also carries a debug-location, confirming
// attachment reached the last MI and not only early ones.
// CHECK: S_ENDPGM
// CHECK-SAME: debug-location !{{[0-9]+}}
 
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942" 
	.amdhsa_code_object_version 6
	.text
	.protected	_Z6vecAddPKfS0_Pfi      ; -- Begin function _Z6vecAddPKfS0_Pfi
	.globl	_Z6vecAddPKfS0_Pfi
	.p2align	8
	.type	_Z6vecAddPKfS0_Pfi,@function
_Z6vecAddPKfS0_Pfi:                     ; @_Z6vecAddPKfS0_Pfi
.Lfunc_begin0:
	.file	0 "/home/Luthier/vec-add" "vec-add.hip" md5 0x268a20f5a93bf4b6ce28bf82383222b6
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; 
	.cfi_undefined 16
	.file	1 "/opt/rocm-7.1.1/lib/llvm/bin/../../../include/hip/amd_detail" "amd_hip_runtime.h"
	.loc	1 264 116 prologue_end          ; /opt/rocm-7.1.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:264:116
	s_load_dword s3, s[0:1], 0x2c
.Ltmp0:
	.loc	1 259 116                       ; /opt/rocm-7.1.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:259:116
	s_load_dword s4, s[0:1], 0x18
.Ltmp1:
	.loc	1 264 116                       ; /opt/rocm-7.1.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:264:116
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
.Ltmp2:
	.file	2 "vec-add.hip"
	.loc	2 6 22                          ; vec-add.hip:6:22
	s_mul_i32 s2, s2, s3
	.loc	2 6 35 is_stmt 0                ; vec-add.hip:6:35
	v_add_u32_e32 v0, s2, v0
	.loc	2 7 9 is_stmt 1                 ; vec-add.hip:7:9
	v_cmp_gt_i32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
.Ltmp3:
	.loc	1 259 116                       ; /opt/rocm-7.1.1/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:259:116
	s_load_dwordx4 s[4:7], s[0:1], 0x0
	s_load_dwordx2 s[2:3], s[0:1], 0x10
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[4:5], s[4:5], 0, v[0:1]
	v_lshl_add_u64 v[2:3], s[6:7], 0, v[0:1]
.Ltmp4:
	.loc	2 7 21 discriminator 2          ; vec-add.hip:7:21
	global_load_dword v6, v[4:5], off
	.loc	2 7 28 is_stmt 0 discriminator 2 ; vec-add.hip:7:28
	global_load_dword v7, v[2:3], off
	v_lshl_add_u64 v[0:1], s[2:3], 0, v[0:1]
	.loc	2 7 26 discriminator 2          ; vec-add.hip:7:26
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v2, v6, v7
	.loc	2 7 19 discriminator 2          ; vec-add.hip:7:19
	global_store_dword v[0:1], v2, off
.LBB0_2:
	.loc	2 8 1 is_stmt 1                 ; vec-add.hip:8:1
	s_endpgm
.Ltmp5:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z6vecAddPKfS0_Pfi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
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
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_accum_offset 8
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
.Lfunc_end0:
	.size	_Z6vecAddPKfS0_Pfi, .Lfunc_end0-_Z6vecAddPKfS0_Pfi
	.cfi_endproc
                                        ; -- End function
	.set _Z6vecAddPKfS0_Pfi.num_vgpr, 8
	.set _Z6vecAddPKfS0_Pfi.num_agpr, 0
	.set _Z6vecAddPKfS0_Pfi.numbered_sgpr, 8
	.set _Z6vecAddPKfS0_Pfi.private_seg_size, 0
	.set _Z6vecAddPKfS0_Pfi.uses_vcc, 1
	.set _Z6vecAddPKfS0_Pfi.uses_flat_scratch, 0
	.set _Z6vecAddPKfS0_Pfi.has_dyn_sized_stack, 0
	.set _Z6vecAddPKfS0_Pfi.has_recursion, 0
	.set _Z6vecAddPKfS0_Pfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 140
; TotalNumSgprs: 14
; NumVgprs: 8
; NumAgprs: 0
; TotalNumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 8
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_8c31d0e4c505adf3,@object ; @__hip_cuid_8c31d0e4c505adf3
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_8c31d0e4c505adf3
__hip_cuid_8c31d0e4c505adf3:
	.byte	0                               ; 0x0
	.size	__hip_cuid_8c31d0e4c505adf3, 1

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
	.byte	4                               ; Abbreviation Code
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
	.byte	5                               ; Abbreviation Code
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
	.byte	6                               ; Abbreviation Code
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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	1                               ; Abbrev [1] 0xc:0x68 DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	33                              ; DW_AT_language
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
	.byte	3                               ; Abbrev [3] 0x3f:0x34 DW_TAG_subprogram
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	10                              ; DW_AT_linkage_name
	.byte	11                              ; DW_AT_name
	.byte	2                               ; DW_AT_decl_file
	.byte	5                               ; DW_AT_decl_line
	.byte	4                               ; Abbrev [4] 0x49:0x15 DW_TAG_inlined_subroutine
	.long	45                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	6                               ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	2                               ; DW_AT_GNU_discriminator
	.byte	5                               ; Abbrev [5] 0x53:0xa DW_TAG_inlined_subroutine
	.long	39                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	296                             ; DW_AT_call_line
	.byte	160                             ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	6                               ; Abbrev [6] 0x5e:0x14 DW_TAG_inlined_subroutine
	.long	57                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	2                               ; DW_AT_call_file
	.byte	6                               ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x67:0xa DW_TAG_inlined_subroutine
	.long	51                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	287                             ; DW_AT_call_line
	.byte	160                             ; DW_AT_call_column
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
	.long	52                              ; Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)" ; string offset=0
.Linfo_string1:
	.asciz	"vec-add.hip"                   ; string offset=137
.Linfo_string2:
	.asciz	"/home/Luthier/vec-add"         ; string offset=149
.Linfo_string3:
	.asciz	"_ZL21__hip_get_block_dim_xv"   ; string offset=171
.Linfo_string4:
	.asciz	"__hip_get_block_dim_x"         ; string offset=199
.Linfo_string5:
	.asciz	"_ZN24__hip_builtin_blockDim_t7__get_xEv" ; string offset=221
.Linfo_string6:
	.asciz	"__get_x"                       ; string offset=261
.Linfo_string7:
	.asciz	"_ZL21__hip_get_block_idx_xv"   ; string offset=269
.Linfo_string8:
	.asciz	"__hip_get_block_idx_x"         ; string offset=297
.Linfo_string9:
	.asciz	"_ZN24__hip_builtin_blockIdx_t7__get_xEv" ; string offset=319
.Linfo_string10:
	.asciz	"_Z6vecAddPKfS0_Pfi"            ; string offset=359
.Linfo_string11:
	.asciz	"vecAdd"                        ; string offset=378
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
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 ; Length of contribution
.Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.1 25444 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_8c31d0e4c505adf3
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
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
    .name:           _Z6vecAddPKfS0_Pfi
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z6vecAddPKfS0_Pfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
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
