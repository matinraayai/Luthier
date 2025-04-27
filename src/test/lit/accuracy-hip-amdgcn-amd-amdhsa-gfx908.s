// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s | \
// RUN: comgr-link -o %t && llvm-readelf --file-header %t | \
// RUN: FileCheck --check-prefix=SHAREDOBJ %s

// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s | \
// RUN: object-test --triple-test --symbol-lookup-test | \
// RUN: FileCheck --check-prefix=TRIPLE %s

// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t | \
// RUN: lld -flavor gnu -m elf64_amdgpu --no-undefined -shared -plugin-opt=-amdgpu-internalize-symbols \
// RUN: -plugin-opt=mcpu=gfx908 -plugin-opt=O3 --lto-CGO3 --whole-archive -o - %t | \
// RUN: object-test --triple-test --symbol-lookup-test | \
// RUN: FileCheck --check-prefix=TRIPLE %s

// SHAREDOBJ: Class: ELF64
// SHAREDOBJ: Data: 2's complement, little endian
// SHAREDOBJ: Version: 1 (current)
// SHAREDOBJ: OS/ABI: AMDGPU - HSA
// SHAREDOBJ: ABI Version: 4
// SHAREDOBJ: Type: DYN (Shared object file)
// SHAREDOBJ: Machine: EM_AMDGPU
// SHAREDOBJ: Version: 0x1
// SHAREDOBJ: Entry point address: 0x0
// SHAREDOBJ: Flags: 0x530, gfx908, xnack, sramecc

// TRIPLE: Target Triple: amdgcn-amd-amdhsa--gfx908

  .text
        .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
        .protected      _Z15accuracy_kerneliiiPKfPKiPi ; -- Begin function _Z15accuracy_kerneliiiPKfPKiPi
        .globl  _Z15accuracy_kerneliiiPKfPKiPi
        .p2align        8
        .type   _Z15accuracy_kerneliiiPKfPKiPi,@function
_Z15accuracy_kerneliiiPKfPKiPi:         ; @_Z15accuracy_kerneliiiPKfPKiPi
; %bb.0:
        s_load_dwordx8 s[12:19], s[4:5], 0x0
        s_load_dwordx2 s[10:11], s[4:5], 0x20
        v_mov_b32_e32 v1, 0
        s_waitcnt lgkmcnt(0)
        s_cmp_ge_i32 s6, s12
        s_cbranch_scc1 .LBB0_15
; %bb.1:                                ; %.lr.ph43
        s_load_dword s15, s[4:5], 0x28
        v_mbcnt_lo_u32_b32 v2, -1, 0
        v_mbcnt_hi_u32_b32 v4, -1, v2
        s_add_u32 s20, s4, 40
        v_lshlrev_b32_e32 v2, 2, v4
        v_cmp_eq_u32_e64 s[0:1], 0, v4
        v_lshrrev_b32_e32 v4, 4, v0
        v_cmp_gt_i32_e32 vcc, s13, v0
        s_addc_u32 s21, s5, 0
        v_mov_b32_e32 v1, 0
        v_or_b32_e32 v3, 0xfc, v2
        v_and_b32_e32 v4, 60, v4
        v_cmp_gt_u32_e64 s[2:3], 4, v0
        v_or_b32_e32 v5, 12, v2
        s_mul_i32 s30, s6, s13
        s_waitcnt lgkmcnt(0)
        s_mul_i32 s31, s15, s13
        s_branch .LBB0_3
.LBB0_2:                                ; %_ZN6hipcub11BlockReduceIiLi256ELNS_20BlockReduceAlgorithmE0ELi1ELi1ELi1EE3SumEi.exit
                                        ;   in Loop: Header=BB0_3 Depth=1
        s_or_b64 exec, exec, s[4:5]
        s_waitcnt lgkmcnt(0)
        v_cmp_ge_i32_e64 s[4:5], s14, v6
        s_add_i32 s6, s15, s6
        s_add_i32 s30, s30, s31
        v_addc_co_u32_e64 v1, s[4:5], 0, v1, s[4:5]
        s_cmp_ge_i32 s6, s12
        s_barrier
        s_cbranch_scc1 .LBB0_15
.LBB0_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_6 Depth 2
        v_mov_b32_e32 v6, 0
        s_and_saveexec_b64 s[22:23], vcc
        s_cbranch_execz .LBB0_11
; %bb.4:                                ; %.lr.ph
                                        ;   in Loop: Header=BB0_3 Depth=1
        s_ashr_i32 s7, s6, 31
        s_lshl_b64 s[4:5], s[6:7], 2
        s_add_u32 s4, s18, s4
        s_addc_u32 s5, s19, s5
        s_load_dword s7, s[4:5], 0x0
        s_mul_i32 s4, s6, s13
        s_load_dword s8, s[20:21], 0xc
        s_mov_b64 s[24:25], 0
        v_mov_b32_e32 v6, 0
        s_waitcnt lgkmcnt(0)
        s_add_i32 s4, s7, s4
        s_ashr_i32 s5, s4, 31
        s_lshl_b64 s[4:5], s[4:5], 2
        s_add_u32 s4, s16, s4
        s_addc_u32 s5, s17, s5
        s_load_dword s33, s[4:5], 0x0
        s_and_b32 s34, s8, 0xffff
        v_mov_b32_e32 v7, v0
        s_branch .LBB0_6
.LBB0_5:                                ;   in Loop: Header=BB0_6 Depth=2
        s_or_b64 exec, exec, s[4:5]
        v_add_u32_e32 v7, s34, v7
        v_cmp_le_i32_e64 s[4:5], s13, v7
        s_or_b64 s[24:25], s[4:5], s[24:25]
        s_andn2_b64 exec, exec, s[24:25]
        s_cbranch_execz .LBB0_10
.LBB0_6:                                ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
        v_add_u32_e32 v8, s30, v7
        v_ashrrev_i32_e32 v9, 31, v8
        v_lshlrev_b64 v[8:9], 2, v[8:9]
        v_mov_b32_e32 v10, s17
        v_add_co_u32_e64 v8, s[4:5], s16, v8
        v_addc_co_u32_e64 v9, s[4:5], v10, v9, s[4:5]
        global_load_dword v8, v[8:9], off
        s_waitcnt vmcnt(0) lgkmcnt(0)
        v_cmp_lt_f32_e64 s[26:27], s33, v8
        v_cmp_nlt_f32_e64 s[4:5], s33, v8
        s_and_saveexec_b64 s[28:29], s[4:5]
; %bb.7:                                ;   in Loop: Header=BB0_6 Depth=2
        v_cmp_eq_f32_e64 s[4:5], s33, v8
        v_cmp_ge_i32_e64 s[8:9], s7, v7
        s_and_b64 s[4:5], s[8:9], s[4:5]
        s_andn2_b64 s[8:9], s[26:27], exec
        s_and_b64 s[4:5], s[4:5], exec
        s_or_b64 s[26:27], s[8:9], s[4:5]
; %bb.8:                                ; %Flow67
                                        ;   in Loop: Header=BB0_6 Depth=2
        s_or_b64 exec, exec, s[28:29]
        s_and_saveexec_b64 s[4:5], s[26:27]
        s_cbranch_execz .LBB0_5
; %bb.9:                                ;   in Loop: Header=BB0_6 Depth=2
        v_add_u32_e32 v6, 1, v6
        s_branch .LBB0_5
.LBB0_10:                               ; %Flow69
                                        ;   in Loop: Header=BB0_3 Depth=1
        s_or_b64 exec, exec, s[24:25]
.LBB0_11:                               ; %Flow70
                                        ;   in Loop: Header=BB0_3 Depth=1
        s_or_b64 exec, exec, s[22:23]
        v_mov_b32_dpp v7, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
        v_add_u32_e32 v6, v7, v6
        s_nop 1
        v_mov_b32_dpp v7, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
        v_add_u32_e32 v6, v6, v7
        s_nop 1
        v_mov_b32_dpp v7, v6 row_shr:4 row_mask:0xf bank_mask:0xf
        v_add_u32_e32 v6, v6, v7
        s_nop 1
        v_mov_b32_dpp v7, v6 row_shr:8 row_mask:0xf bank_mask:0xf
        v_add_u32_e32 v6, v6, v7
        s_nop 1
        v_mov_b32_dpp v7, v6 row_bcast:15 row_mask:0xf bank_mask:0xf
        v_add_u32_e32 v6, v6, v7
        s_nop 1
        v_mov_b32_dpp v7, v6 row_bcast:31 row_mask:0xf bank_mask:0xf
        v_add_u32_e32 v6, v6, v7
        ds_bpermute_b32 v6, v3, v6
        s_and_saveexec_b64 s[4:5], s[0:1]
        s_cbranch_execz .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_3 Depth=1
        s_waitcnt lgkmcnt(0)
        ds_write_b32 v4, v6
.LBB0_13:                               ;   in Loop: Header=BB0_3 Depth=1
        s_or_b64 exec, exec, s[4:5]
        s_waitcnt lgkmcnt(0)
        s_barrier
        s_and_saveexec_b64 s[4:5], s[2:3]
        s_cbranch_execz .LBB0_2
; %bb.14:                               ;   in Loop: Header=BB0_3 Depth=1
        ds_read_b32 v6, v2
        s_waitcnt lgkmcnt(0)
        s_nop 0
        v_mov_b32_dpp v7, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
        v_add_u32_e32 v6, v7, v6
        s_nop 1
        v_mov_b32_dpp v7, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf
        v_add_u32_e32 v6, v6, v7
        ds_bpermute_b32 v6, v5, v6
        s_branch .LBB0_2
.LBB0_15:                               ; %Flow73
        s_mov_b32 s2, 0
        v_cmp_eq_u32_e32 vcc, 0, v0
        s_and_saveexec_b64 s[0:1], vcc
        s_cbranch_execz .LBB0_20
; %bb.16:
        s_mov_b64 s[0:1], exec
.LBB0_17:                               ; %ComputeLoop
                                        ; =>This Inner Loop Header: Depth=1
        s_ff1_i32_b64 s3, s[0:1]
        v_readlane_b32 s6, v1, s3
        s_lshl_b64 s[4:5], 1, s3
        s_add_i32 s2, s2, s6
        s_andn2_b64 s[0:1], s[0:1], s[4:5]
        s_cmp_lg_u64 s[0:1], 0
        s_cbranch_scc1 .LBB0_17
; %bb.18:                               ; %ComputeEnd
        v_mbcnt_lo_u32_b32 v0, exec_lo, 0
        v_mbcnt_hi_u32_b32 v0, exec_hi, v0
        v_cmp_eq_u32_e32 vcc, 0, v0
        s_and_saveexec_b64 s[0:1], vcc
        s_xor_b64 s[0:1], exec, s[0:1]
        s_cbranch_execz .LBB0_20
; %bb.19:
        v_mov_b32_e32 v0, 0
        v_mov_b32_e32 v1, s2
        global_atomic_add v0, v1, s[10:11]
.LBB0_20:                               ; %Flow66
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel _Z15accuracy_kerneliiiPKfPKiPi
                .amdhsa_group_segment_fixed_size 16
                .amdhsa_private_segment_fixed_size 0
                .amdhsa_kernarg_size 296
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
                .amdhsa_next_free_vgpr 11
                .amdhsa_next_free_sgpr 35
                .amdhsa_reserve_flat_scratch 0
                .amdhsa_reserve_xnack_mask 1
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
        .size   _Z15accuracy_kerneliiiPKfPKiPi, .Lfunc_end0-_Z15accuracy_kerneliiiPKfPKiPi
                                        ; -- End function
        .section        .AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 732
; NumSgprs: 39
; NumVgprs: 11
; NumAgprs: 0
; TotalNumVgprs: 11
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 16 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 39
; NumVGPRsForWavesPerEU: 11
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
        .type   __hip_cuid_c8619780aadae295,@object ; @__hip_cuid_c8619780aadae295
        .section        .bss,"aw",@nobits
        .globl  __hip_cuid_c8619780aadae295
__hip_cuid_c8619780aadae295:
        .byte   0                               ; 0x0
        .size   __hip_cuid_c8619780aadae295, 1

        .ident  "AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
        .addrsig_sym __hip_cuid_c8619780aadae295
        .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 16
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z15accuracy_kerneliiiPKfPKiPi
    .private_segment_fixed_size: 0
    .sgpr_count:     39
    .sgpr_spill_count: 0
    .symbol:         _Z15accuracy_kerneliiiPKfPKiPi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     11
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 2
...

        .end_amdgpu_metadata
