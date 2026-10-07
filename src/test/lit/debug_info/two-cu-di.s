// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-mc -g --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %t/kernel.s -o %t/kernel.o
// RUN: llvm-mc -g --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %t/callee.s -o %t/callee.o
// RUN: ld.lld -shared -o %t/co %t/kernel.o %t/callee.o
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,luthier-populate-debug-info,target(print-mir-prepare,function(machine-function(print)))' \
// RUN:   -code-object-paths=%t/co \
// RUN:   -initial-entrypoint=0:_Z11scaleKernelPffi.kd \
// RUN:   -initial-execution-point=0:_Z11scaleKernelPffi.kd \
// RUN:   -o /dev/null > %t.ir 2> %t.mir
// The IR module is printed to stdout and the MIR bodies to stderr. Merging them
// with 2>&1 can splice one stream into the middle of a line of the other, so
// capture them separately and concatenate in a fixed order instead.
// RUN: cat %t.ir %t.mir | FileCheck %s

// Tests that DebugInfoPass builds one DICompileUnit for each DWARF compile
// unit in a code object, with a separate DIBuilder for each one, and that each
// DISubprogram points to the unit it came from. The code object links two
// translation units: the kernel and the function that it calls.
//
// One clang run gives one compile unit, so the two files are compiled
// separately and linked by ld.lld. split-file divides this file into
// kernel.s and callee.s at the //--- markers below.
//
// Assembly below was generated from:
//
//   // two-cu-kernel.hip
//    5: __attribute__((device)) float scaleBy(float X, float K);
//    7: __attribute__((global)) void scaleKernel(float *A, float K, int N) {
//    8:   int I = __builtin_amdgcn_workitem_id_x();
//    9:   if (I < N)
//   10:     A[I] = scaleBy(A[I], K);
//   11: }
//
//   // two-cu-callee.hip
//    2: __attribute__((device, noinline)) float scaleBy(float X, float K) {
//    3:   return X * K;
//    4: }
//
// with, for each file:
//   clang -x hip --offload-device-only --no-gpu-bundle-output \
//     --offload-arch=gfx908 -O2 -gline-tables-only \
//     -fdebug-info-for-profiling -nogpulib -nogpuinc -fgpu-rdc \
//     -emit-llvm -c <file>.hip -o <file>.bc
//   llc -O2 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 <file>.bc -o <file>.s
//
// -fgpu-rdc stops clang from deleting scaleBy, which no kernel in its own file
// calls. With -fgpu-rdc, -S emits LLVM IR, so llc lowers the IR to assembly.
// The sources use no HIP headers, because with -fgpu-rdc the headers add
// calls to device-library functions (__ockl_*) that ld.lld cannot resolve.
//
// Expected: two DICompileUnits, one for two-cu-kernel.hip and one for
// two-cu-callee.hip. The scaleKernel DISubprogram has unit: set to the kernel
// unit, and the scaleBy DISubprogram has unit: set to the callee unit.

// === IR: each lifted function has its own !dbg ===

// CHECK: define{{.*}}@_Z11scaleKernelPffi(
// CHECK-SAME: !dbg [[KERNEL_SP:![0-9]+]]
// CHECK: define{{.*}}@_Z7scaleByff{{[^(]*}}(
// CHECK-SAME: !dbg [[CALLEE_SP:![0-9]+]]

// === Metadata: two compile units, one for each file ===

// kernel.o is linked first, so its DWARF unit comes first.
// CHECK: !llvm.dbg.cu = !{[[KERNEL_CU:![0-9]+]], [[CALLEE_CU:![0-9]+]]}
// CHECK-DAG: [[KERNEL_CU]] = distinct !DICompileUnit({{.*}}file: [[KERNEL_FILE:![0-9]+]]
// CHECK-DAG: [[CALLEE_CU]] = distinct !DICompileUnit({{.*}}file: [[CALLEE_FILE:![0-9]+]]
// CHECK-DAG: [[KERNEL_FILE]] = !DIFile(filename: "two-cu-kernel.hip"
// CHECK-DAG: [[CALLEE_FILE]] = !DIFile(filename: "two-cu-callee.hip"

// === Metadata: each subprogram points to the unit it came from ===

// CHECK-DAG: [[KERNEL_SP]] = distinct !DISubprogram(name: "scaleKernel", linkageName: "_Z11scaleKernelPffi", scope: [[KERNEL_FILE]], file: [[KERNEL_FILE]], line: 7,{{.*}}unit: [[KERNEL_CU]])
// CHECK-DAG: [[CALLEE_SP]] = distinct !DISubprogram(name: "scaleBy", linkageName: "_Z7scaleByff", scope: [[CALLEE_FILE]], file: [[CALLEE_FILE]], line: 2,{{.*}}unit: [[CALLEE_CU]])
// The callee's multiply (X * K) is on line 3 of two-cu-callee.hip.
// CHECK-DAG: !DILocation(line: 3,{{.*}}scope: [[CALLEE_SP]])

// === MIR: the callee's instructions have debug locations ===

// CHECK-LABEL: name: _Z7scaleByff{{[^ ]*$}}
// CHECK: V_MUL_F32_e32 {{.*}}debug-location

//--- kernel.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z11scaleKernelPffi     ; -- Begin function _Z11scaleKernelPffi
	.globl	_Z11scaleKernelPffi
	.p2align	8
	.type	_Z11scaleKernelPffi,@function
_Z11scaleKernelPffi:                    ; @_Z11scaleKernelPffi
.Lfunc_begin0:
	.file	0 "/home/Luthier/sandbox/di-tests" "two-cu-kernel.hip" md5 0xd4fcdc76058b48ef1428a0957c81b34b
	.cfi_startproc
; %bb.0:                                ; %entry
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.loc	0 8 11 prologue_end             ; two-cu-kernel.hip:8:11
	s_load_dwordx2 s[18:19], s[8:9], 0x8
	s_add_u32 flat_scratch_lo, s12, s17
	s_addc_u32 flat_scratch_hi, s13, 0
	s_add_u32 s0, s0, s17
	s_addc_u32 s1, s1, 0
	.loc	0 9 9                           ; two-cu-kernel.hip:9:9
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_i32_e32 vcc, s19, v0
	s_mov_b32 s32, 0
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:                                ; %if.then
	.loc	0 8 11                          ; two-cu-kernel.hip:8:11
	s_load_dwordx2 s[34:35], s[8:9], 0x0
	v_lshlrev_b32_e32 v40, 2, v0
	.loc	0 10 12                         ; two-cu-kernel.hip:10:12
	s_add_u32 s8, s8, 16
	v_lshlrev_b32_e32 v2, 20, v2
	v_lshlrev_b32_e32 v1, 10, v1
	.loc	0 10 20 is_stmt 0               ; two-cu-kernel.hip:10:20
	s_waitcnt lgkmcnt(0)
	global_load_dword v3, v40, s[34:35]
	.loc	0 10 12                         ; two-cu-kernel.hip:10:12
	s_addc_u32 s9, s9, 0
	s_mov_b32 s13, s15
	s_getpc_b64 s[20:21]
	s_add_u32 s20, s20, _Z7scaleByff@rel32@lo+4
	s_addc_u32 s21, s21, _Z7scaleByff@rel32@hi+12
	v_or3_b32 v31, v0, v1, v2
	s_mov_b32 s12, s14
	s_mov_b32 s14, s16
	v_mov_b32_e32 v1, s18
                                        ; implicit-def: $sgpr15
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v0, v3
	s_swappc_b64 s[30:31], s[20:21]
.Ltmp0:
	.loc	0 10 10                         ; two-cu-kernel.hip:10:10
	global_store_dword v40, v0, s[34:35]
.LBB0_2:                                ; %if.end
	.loc	0 11 1 is_stmt 1                ; two-cu-kernel.hip:11:1
	s_endpgm
.Ltmp1:
.Lfunc_end0:
	.size	_Z11scaleKernelPffi, .Lfunc_end0-_Z11scaleKernelPffi
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11scaleKernelPffi
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 14
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_flat_scratch_init 1
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 1
		.amdhsa_system_sgpr_private_segment_wavefront_offset 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr max(totalnumvgprs(.L_Z11scaleKernelPffi.num_agpr, .L_Z11scaleKernelPffi.num_vgpr), 1, 0)
		.amdhsa_next_free_sgpr max(.L_Z11scaleKernelPffi.numbered_sgpr+6, 1, 0)-6
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 1
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
	.set .L_Z11scaleKernelPffi.num_vgpr, max(41, amdgpu.max_num_vgpr)
	.set .L_Z11scaleKernelPffi.num_agpr, max(0, amdgpu.max_num_agpr)
	.set .L_Z11scaleKernelPffi.numbered_sgpr, max(36, amdgpu.max_num_sgpr)
	.set .L_Z11scaleKernelPffi.num_named_barrier, max(0, amdgpu.max_num_named_barrier)
	.set .L_Z11scaleKernelPffi.private_seg_size, 0
	.set .L_Z11scaleKernelPffi.uses_vcc, 1
	.set .L_Z11scaleKernelPffi.uses_flat_scratch, 1
	.set .L_Z11scaleKernelPffi.has_dyn_sized_stack, 1
	.set .L_Z11scaleKernelPffi.has_recursion, 1
	.set .L_Z11scaleKernelPffi.has_indirect_call, 1
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 152
; TotalNumSgprs: .L_Z11scaleKernelPffi.numbered_sgpr+6
; NumVgprs: .L_Z11scaleKernelPffi.num_vgpr
; NumAgprs: .L_Z11scaleKernelPffi.num_agpr
; TotalNumVgprs: totalnumvgprs(.L_Z11scaleKernelPffi.num_agpr, .L_Z11scaleKernelPffi.num_vgpr)
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: (alignto(max(max(.L_Z11scaleKernelPffi.numbered_sgpr+extrasgprs(.L_Z11scaleKernelPffi.uses_vcc, .L_Z11scaleKernelPffi.uses_flat_scratch, 1), 1, 0), 1), 8)/8)-1
; VGPRBlocks: (alignto(max(max(totalnumvgprs(.L_Z11scaleKernelPffi.num_agpr, .L_Z11scaleKernelPffi.num_vgpr), 1, 0), 1), 4)/4)-1
; NumSGPRsForWavesPerEU: max(.L_Z11scaleKernelPffi.numbered_sgpr+6, 1, 0)
; NumVGPRsForWavesPerEU: max(totalnumvgprs(.L_Z11scaleKernelPffi.num_agpr, .L_Z11scaleKernelPffi.num_vgpr), 1, 0)
; Occupancy: occupancy(10, 4, 256, 8, 10, max(.L_Z11scaleKernelPffi.numbered_sgpr+extrasgprs(.L_Z11scaleKernelPffi.uses_vcc, .L_Z11scaleKernelPffi.uses_flat_scratch, 1), 1, 0), max(totalnumvgprs(.L_Z11scaleKernelPffi.num_agpr, .L_Z11scaleKernelPffi.num_vgpr), 1, 0))
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_64083bd4f62556ad,@object ; @__hip_cuid_64083bd4f62556ad
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_64083bd4f62556ad
__hip_cuid_64083bd4f62556ad:
	.byte	0                               ; 0x0
	.size	__hip_cuid_64083bd4f62556ad, 1

	.hidden	_Z7scaleByff
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
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
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
	.byte	3                               ; Abbreviation Code
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
	.byte	1                               ; Abbrev [1] 0xc:0x2e DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	48                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.long	.Lstr_offsets_base0             ; DW_AT_str_offsets_base
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.byte	2                               ; DW_AT_comp_dir
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	.Laddr_table_base0              ; DW_AT_addr_base
	.byte	2                               ; Abbrev [2] 0x23:0x16 DW_TAG_subprogram
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	3                               ; DW_AT_linkage_name
	.byte	4                               ; DW_AT_name
	.byte	0                               ; DW_AT_decl_file
	.byte	7                               ; DW_AT_decl_line
	.byte	3                               ; Abbrev [3] 0x2d:0xb DW_TAG_call_site
	.byte	8                               ; DW_AT_call_target_clobbered
	.byte	144
	.byte	52
	.byte	147
	.byte	4
	.byte	144
	.byte	53
	.byte	147
	.byte	4
	.byte	1                               ; DW_AT_call_return_pc
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	24                              ; Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)" ; string offset=0 ; AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)
.Linfo_string1:
	.asciz	"two-cu-kernel.hip"             ; string offset=109 ; two-cu-kernel.hip
.Linfo_string2:
	.asciz	"/home/Luthier/sandbox/di-tests" ; string offset=127 ; /home/Luthier/sandbox/di-tests
.Linfo_string3:
	.asciz	"_Z11scaleKernelPffi"           ; string offset=158 ; _Z11scaleKernelPffi
.Linfo_string4:
	.asciz	"scaleKernel"                   ; string offset=178 ; scaleKernel
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 ; Length of contribution
.Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp0
.Ldebug_addr_end0:
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)"
	.section	".note.GNU-stack","",@progbits
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
      - .offset:         96
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         104
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         112
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         120
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         128
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         216
        .size:           8
        .value_kind:     hidden_queue_ptr
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           _Z11scaleKernelPffi
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         _Z11scaleKernelPffi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: true
    .vgpr_count:     41
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
//--- callee.s
	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.amdhsa_code_object_version 6
	.text
	.hidden	_Z7scaleByff                    ; -- Begin function _Z7scaleByff
	.globl	_Z7scaleByff
	.p2align	6
	.type	_Z7scaleByff,@function
_Z7scaleByff:                           ; @_Z7scaleByff
.Lfunc_begin0:
	.file	0 "/home/Luthier/sandbox/di-tests" "two-cu-callee.hip" md5 0x4c677c5eaba2be37f1d771947ce5dfb3
	.loc	0 2 0                           ; two-cu-callee.hip:2:0
	.cfi_startproc
; %bb.0:                                ; %entry
	.cfi_llvm_def_aspace_cfa 64, 0, 6
	.cfi_llvm_register_pair 16, 62, 32, 63, 32
	.cfi_undefined 2560
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
.Ltmp0:
	.loc	0 3 12 prologue_end             ; two-cu-callee.hip:3:12
	v_mul_f32_e32 v0, v0, v1
	.loc	0 3 3 is_stmt 0                 ; two-cu-callee.hip:3:3
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
; codeLenInByte = 12
; TotalNumSgprs: 36
; NumVgprs: 2
; NumAgprs: 0
; TotalNumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 2
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_37fed19c6221f96f,@object ; @__hip_cuid_37fed19c6221f96f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_37fed19c6221f96f
__hip_cuid_37fed19c6221f96f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_37fed19c6221f96f, 1

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
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	5                               ; DWARF version number
	.byte	1                               ; DWARF Unit Type
	.byte	8                               ; Address Size (in bytes)
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	1                               ; Abbrev [1] 0xc:0x22 DW_TAG_compile_unit
	.byte	0                               ; DW_AT_producer
	.short	48                              ; DW_AT_language
	.byte	1                               ; DW_AT_name
	.long	.Lstr_offsets_base0             ; DW_AT_str_offsets_base
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.byte	2                               ; DW_AT_comp_dir
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	.Laddr_table_base0              ; DW_AT_addr_base
	.byte	2                               ; Abbrev [2] 0x23:0xa DW_TAG_subprogram
	.byte	0                               ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
                                        ; DW_AT_call_all_calls
	.byte	3                               ; DW_AT_linkage_name
	.byte	4                               ; DW_AT_name
	.byte	0                               ; DW_AT_decl_file
	.byte	2                               ; DW_AT_decl_line
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	24                              ; Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)" ; string offset=0 ; AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)
.Linfo_string1:
	.asciz	"two-cu-callee.hip"             ; string offset=109 ; two-cu-callee.hip
.Linfo_string2:
	.asciz	"/home/Luthier/sandbox/di-tests" ; string offset=127 ; /home/Luthier/sandbox/di-tests
.Linfo_string3:
	.asciz	"_Z7scaleByff"                  ; string offset=158 ; _Z7scaleByff
.Linfo_string4:
	.asciz	"scaleBy"                       ; string offset=171 ; scaleBy
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 ; Length of contribution
.Ldebug_addr_start0:
	.short	5                               ; DWARF version number
	.byte	8                               ; Address size
	.byte	0                               ; Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project/ e8ab2aed15d22cd217e9a2c1938e7d2ec9e11893)"
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
