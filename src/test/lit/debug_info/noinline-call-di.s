// RUN: llvm-mc -g --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,luthier-populate-debug-info,target(print-mir-prepare,function(machine-function(print)))' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z11scaleKernelPffi.kd \
// RUN:   -initial-execution-point=0:_Z11scaleKernelPffi.kd \
// RUN:   -o /dev/null > %t.ir 2> %t.mir
// The IR module is printed to stdout and the MIR bodies to stderr. Merging them
// with 2>&1 can splice one stream into the middle of a line of the other, so
// capture them separately and concatenate in a fixed order instead.
// RUN: cat %t.ir %t.mir | FileCheck %s

// Tests that DebugInfoPass gives debug info to every lifted function of a
// code object, not only to the kernel. The kernel calls a device function
// that is not inlined, so code discovery lifts two functions from the same
// code object. The second function reuses the DWARF context and the
// DISubprogram cache that the first function built.
//
// Assembly below was generated from:
//
//   // noinline-call.hip
//    6: __device__ __attribute__((noinline)) float scaleBy(float X, float K) {
//    7:   return X * K;
//    8: }
//   10: __global__ void scaleKernel(float *A, float K, int N) {
//   11:   int I = blockIdx.x * blockDim.x + threadIdx.x;
//   12:   if (I < N)
//   13:     A[I] = scaleBy(A[I], K);
//   14: }
//
// with:
//   clang -x hip --offload-device-only --no-gpu-bundle-output \
//     --offload-arch=gfx908 -O2 -gline-tables-only \
//     -fdebug-info-for-profiling -S noinline-call.hip
//
// Expected: one DICompileUnit. The kernel and the lifted callee
// (_Z7scaleByff) each have a !dbg to their own DISubprogram
// (scaleKernel line 10, scaleBy line 6). The instructions of the callee
// have DILocations at noinline-call.hip:7.

// === IR: each lifted function has its own !dbg ===

// CHECK: define{{.*}}@_Z11scaleKernelPffi(
// CHECK-SAME: !dbg [[KERNEL_SP:![0-9]+]]
// CHECK: define{{.*}}@_Z7scaleByff{{[^(]*}}(
// CHECK-SAME: !dbg [[CALLEE_SP:![0-9]+]]

// === Metadata: exactly one compile unit, and both subprograms point to it ===

// CHECK: !llvm.dbg.cu = !{[[CU:![0-9]+]]}
// CHECK-DAG: [[CU]] = distinct !DICompileUnit({{.*}}file: [[FILE:![0-9]+]]
// CHECK-DAG: [[FILE]] = !DIFile(filename: "noinline-call.hip"
// CHECK-DAG: [[KERNEL_SP]] = distinct !DISubprogram(name: "scaleKernel", linkageName: "_Z11scaleKernelPffi", scope: [[FILE]], file: [[FILE]], line: 10,{{.*}}unit: [[CU]])
// CHECK-DAG: [[CALLEE_SP]] = distinct !DISubprogram(name: "scaleBy", linkageName: "_Z7scaleByff", scope: [[FILE]], file: [[FILE]], line: 6,{{.*}}unit: [[CU]])
// The callee's multiply (X * K) is on line 7.
// CHECK-DAG: !DILocation(line: 7,{{.*}}scope: [[CALLEE_SP]])

// === MIR: the callee's instructions have debug locations ===

// CHECK-LABEL: name: _Z7scaleByff{{[^ ]*$}}
// CHECK: V_MUL_F32_e32 {{.*}}debug-location

	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.amdhsa_code_object_version 6
	.text
	.p2align	6                               ; -- Begin function _Z7scaleByff
	.type	_Z7scaleByff,@function
_Z7scaleByff:                           ; @_Z7scaleByff
.Lfunc_begin0:
	.file	0 "/home/Luthier/sandbox/di-tests" "noinline-call.hip" md5 0xd12f7eb5f5c77f522d12be972d5eb631
	.loc	0 6 0                           ; noinline-call.hip:6:0
	.cfi_startproc
; %bb.0:                                ; %entry
	.cfi_llvm_def_aspace_cfa 64, 0, 6
	.cfi_llvm_register_pair 16, 62, 32, 63, 32
	.cfi_undefined 2560
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b64 s[4:5], exec
	.cfi_llvm_register_pair 17, 36, 32, 37, 32
.Ltmp0:
	.loc	0 7 12 prologue_end             ; noinline-call.hip:7:12
	v_mul_f32_e32 v0, v0, v1
	.loc	0 7 3 is_stmt 0                 ; noinline-call.hip:7:3
	s_setpc_b64 s[30:31]
.Ltmp1:
.Lfunc_end0:
	.size	_Z7scaleByff, .Lfunc_end0-_Z7scaleByff
	.cfi_endproc
                                        ; -- End function
	.set .L_Z7scaleByff.num_vgpr, 2
	.set .L_Z7scaleByff.num_agpr, 0
	.set .L_Z7scaleByff.numbered_sgpr, 32
	.set .L_Z7scaleByff.num_named_barrier, 0
	.set .L_Z7scaleByff.private_seg_size, 0
	.set .L_Z7scaleByff.uses_vcc, 0
	.set .L_Z7scaleByff.uses_flat_scratch, 0
	.set .L_Z7scaleByff.has_dyn_sized_stack, 0
	.set .L_Z7scaleByff.has_recursion, 0
	.set .L_Z7scaleByff.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 16
; TotalNumSgprs: 36
; NumVgprs: 2
; NumAgprs: 0
; TotalNumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
	.text
	.protected	_Z11scaleKernelPffi     ; -- Begin function _Z11scaleKernelPffi
	.globl	_Z11scaleKernelPffi
	.p2align	8
	.type	_Z11scaleKernelPffi,@function
_Z11scaleKernelPffi:                    ; @_Z11scaleKernelPffi
.Lfunc_begin1:
	.loc	0 10 0 is_stmt 1                ; noinline-call.hip:10:0
	.cfi_startproc
; %bb.0:                                ; %entry
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_add_u32 s0, s0, s7
.Ltmp2:
	.file	1 "/opt/rocm/include/hip/amd_detail" "amd_hip_runtime.h" md5 0xf3ac153f6fbc746611d9adb666a6ab03
	.loc	1 264 58 prologue_end           ; /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:264:58 @[ /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:296:27 @[ noinline-call.hip:11:24 ] ]
	s_load_dword s7, s[4:5], 0x1c
.Ltmp3:
	.loc	1 259 58                        ; /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:259:58 @[ /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:287:27 @[ noinline-call.hip:11:11 ] ]
	s_load_dwordx2 s[8:9], s[4:5], 0x8
	s_addc_u32 s1, s1, 0
	s_mov_b32 s32, 0
.Ltmp4:
	.loc	1 264 58                        ; /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:264:58 @[ /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:296:27 @[ noinline-call.hip:11:24 ] ]
	s_waitcnt lgkmcnt(0)
	s_and_b32 s7, s7, 0xffff
.Ltmp5:
	.loc	0 11 22                         ; noinline-call.hip:11:22
	s_mul_i32 s6, s6, s7
	.loc	0 11 35 is_stmt 0               ; noinline-call.hip:11:35
	v_add_u32_e32 v0, s6, v0
	.loc	0 12 9 is_stmt 1                ; noinline-call.hip:12:9
	v_cmp_gt_i32_e32 vcc, s9, v0
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB1_2
; %bb.1:                                ; %if.then
.Ltmp6:
	.loc	1 259 58                        ; /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:259:58 @[ /opt/rocm/include/hip/amd_detail/amd_hip_runtime.h:287:27 @[ noinline-call.hip:11:11 ] ]
	s_load_dwordx2 s[4:5], s[4:5], 0x0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s5
	v_add_co_u32_e32 v2, vcc, s4, v0
	v_addc_co_u32_e32 v3, vcc, v3, v1, vcc
.Ltmp7:
	.loc	0 13 20                         ; noinline-call.hip:13:20
	global_load_dword v0, v[2:3], off
	.loc	0 13 12 is_stmt 0               ; noinline-call.hip:13:12
	s_getpc_b64 s[4:5]
	s_add_u32 s4, s4, _Z7scaleByff@rel32@lo+4
	s_addc_u32 s5, s5, _Z7scaleByff@rel32@hi+12
	v_mov_b32_e32 v1, s8
	s_swappc_b64 s[30:31], s[4:5]
.Ltmp8:
	.loc	0 13 10                         ; noinline-call.hip:13:10
	global_store_dword v[2:3], v0, off
.LBB1_2:                                ; %if.end
	.loc	0 14 1 is_stmt 1                ; noinline-call.hip:14:1
	s_endpgm
.Ltmp9:
.Lfunc_end1:
	.size	_Z11scaleKernelPffi, .Lfunc_end1-_Z11scaleKernelPffi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11scaleKernelPffi
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
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 33
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
	.set .L_Z11scaleKernelPffi.num_vgpr, max(4, .L_Z7scaleByff.num_vgpr)
	.set .L_Z11scaleKernelPffi.num_agpr, max(0, .L_Z7scaleByff.num_agpr)
	.set .L_Z11scaleKernelPffi.numbered_sgpr, max(33, .L_Z7scaleByff.numbered_sgpr)
	.set .L_Z11scaleKernelPffi.num_named_barrier, max(0, .L_Z7scaleByff.num_named_barrier)
	.set .L_Z11scaleKernelPffi.private_seg_size, 0+max(.L_Z7scaleByff.private_seg_size)
	.set .L_Z11scaleKernelPffi.uses_vcc, or(1, .L_Z7scaleByff.uses_vcc)
	.set .L_Z11scaleKernelPffi.uses_flat_scratch, or(0, .L_Z7scaleByff.uses_flat_scratch)
	.set .L_Z11scaleKernelPffi.has_dyn_sized_stack, or(0, .L_Z7scaleByff.has_dyn_sized_stack)
	.set .L_Z11scaleKernelPffi.has_recursion, or(0, .L_Z7scaleByff.has_recursion)
	.set .L_Z11scaleKernelPffi.has_indirect_call, or(0, .L_Z7scaleByff.has_indirect_call)
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 144
; TotalNumSgprs: 37
; NumVgprs: 4
; NumAgprs: 0
; TotalNumVgprs: 4
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 37
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
	.set amdgpu.max_num_vgpr, 2
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_6ae7c3e59783589f,@object ; @__hip_cuid_6ae7c3e59783589f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_6ae7c3e59783589f
__hip_cuid_6ae7c3e59783589f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_6ae7c3e59783589f, 1

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
	.byte	5                               ; DW_FORM_data2
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
	.byte	72                              ; DW_TAG_call_site
	.byte	0                               ; DW_CHILDREN_no
	.ascii	"\204\001"                      ; DW_AT_call_target_clobbered
	.byte	24                              ; DW_FORM_exprloc
	.byte	125                             ; DW_AT_call_return_pc
	.byte	27                              ; DW_FORM_addrx
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
	.byte	1                               ; Abbrev [1] 0xc:0x7d DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	48                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.long	.Lstr_offsets_base0             ; DW_AT_str_offsets_base
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.byte	2                               ; DW_AT_comp_dir
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin0       ; DW_AT_high_pc
	.long	.Laddr_table_base0              ; DW_AT_addr_base
	.long	.Lrnglists_table_base0          ; DW_AT_rnglists_base
	.byte	2                               ; Abbrev [2] 0x27:0xa DW_TAG_subprogram
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	10                              ; DW_AT_linkage_name
	.byte	11                              ; DW_AT_name
	.byte	0                               ; DW_AT_decl_file
	.byte	6                               ; DW_AT_decl_line
	.byte	3                               ; Abbrev [3] 0x31:0x6 DW_TAG_subprogram
	.byte	3                               ; DW_AT_linkage_name
	.byte	4                               ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.short	264                             ; DW_AT_decl_line
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x37:0x6 DW_TAG_subprogram
	.byte	5                               ; DW_AT_linkage_name
	.byte	6                               ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.short	296                             ; DW_AT_decl_line
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x3d:0x6 DW_TAG_subprogram
	.byte	7                               ; DW_AT_linkage_name
	.byte	8                               ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.short	259                             ; DW_AT_decl_line
                                        ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x43:0x6 DW_TAG_subprogram
	.byte	9                               ; DW_AT_linkage_name
	.byte	6                               ; DW_AT_name
	.byte	1                               ; DW_AT_decl_file
	.short	287                             ; DW_AT_decl_line
                                        ; DW_AT_inline
	.byte	4                               ; Abbrev [4] 0x49:0x3f DW_TAG_subprogram
	.byte	1                               ; DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	12                              ; DW_AT_linkage_name
	.byte	13                              ; DW_AT_name
	.byte	0                               ; DW_AT_decl_file
	.byte	10                              ; DW_AT_decl_line
	.byte	5                               ; Abbrev [5] 0x53:0x15 DW_TAG_inlined_subroutine
	.long	55                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.byte	11                              ; DW_AT_call_line
	.byte	24                              ; DW_AT_call_column
	.byte	2                               ; DW_AT_GNU_discriminator
	.byte	6                               ; Abbrev [6] 0x5d:0xa DW_TAG_inlined_subroutine
	.long	49                              ; DW_AT_abstract_origin
	.byte	0                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	296                             ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	7                               ; Abbrev [7] 0x68:0x14 DW_TAG_inlined_subroutine
	.long	67                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	0                               ; DW_AT_call_file
	.byte	11                              ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	6                               ; Abbrev [6] 0x71:0xa DW_TAG_inlined_subroutine
	.long	61                              ; DW_AT_abstract_origin
	.byte	1                               ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.short	287                             ; DW_AT_call_line
	.byte	27                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	8                               ; Abbrev [8] 0x7c:0xb DW_TAG_call_site
	.byte	8                               ; DW_AT_call_target_clobbered
	.byte	144
	.byte	36
	.byte	147
	.byte	4
	.byte	144
	.byte	37
	.byte	147
	.byte	4
	.byte	2                               ; DW_AT_call_return_pc
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
	.uleb128 .Ltmp2-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp3-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp4-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp5-.Lfunc_begin0           ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_ranges1:
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp3-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp4-.Lfunc_begin0           ;   ending offset
	.byte	4                               ; DW_RLE_offset_pair
	.uleb128 .Ltmp6-.Lfunc_begin0           ;   starting offset
	.uleb128 .Ltmp7-.Lfunc_begin0           ;   ending offset
	.byte	0                               ; DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	60                              ; Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)" ; string offset=0 ; AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)
.Linfo_string1:
	.asciz	"noinline-call.hip"             ; string offset=109 ; noinline-call.hip
.Linfo_string2:
	.asciz	"/home/Luthier/sandbox/di-tests" ; string offset=127 ; /home/Luthier/sandbox/di-tests
.Linfo_string3:
	.asciz	"_ZL21__hip_get_block_dim_xv"   ; string offset=158 ; _ZL21__hip_get_block_dim_xv
.Linfo_string4:
	.asciz	"__hip_get_block_dim_x"         ; string offset=186 ; __hip_get_block_dim_x
.Linfo_string5:
	.asciz	"_ZN24__hip_builtin_blockDim_t7__get_xEv" ; string offset=208 ; _ZN24__hip_builtin_blockDim_t7__get_xEv
.Linfo_string6:
	.asciz	"__get_x"                       ; string offset=248 ; __get_x
.Linfo_string7:
	.asciz	"_ZL21__hip_get_block_idx_xv"   ; string offset=256 ; _ZL21__hip_get_block_idx_xv
.Linfo_string8:
	.asciz	"__hip_get_block_idx_x"         ; string offset=284 ; __hip_get_block_idx_x
.Linfo_string9:
	.asciz	"_ZN24__hip_builtin_blockIdx_t7__get_xEv" ; string offset=306 ; _ZN24__hip_builtin_blockIdx_t7__get_xEv
.Linfo_string10:
	.asciz	"_Z7scaleByff"                  ; string offset=346 ; _Z7scaleByff
.Linfo_string11:
	.asciz	"scaleBy"                       ; string offset=359 ; scaleBy
.Linfo_string12:
	.asciz	"_Z11scaleKernelPffi"           ; string offset=367 ; _Z11scaleKernelPffi
.Linfo_string13:
	.asciz	"scaleKernel"                   ; string offset=387 ; scaleKernel
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
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 ; Length of contribution
.Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	.Ltmp8
.Ldebug_addr_end0:
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_6ae7c3e59783589f
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
    .name:           _Z11scaleKernelPffi
    .private_segment_fixed_size: 0
    .sgpr_count:     37
    .sgpr_spill_count: 0
    .symbol:         _Z11scaleKernelPffi.kd
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
	.section	.debug_line,"",@progbits
.Lline_table_start0:
