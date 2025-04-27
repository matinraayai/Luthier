// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s | \
// RUN: object-test --triple-test --symbol-lookup-test | \
// RUN: FileCheck --check-prefix=TRIPLE %s

// TRIPLE: Target Triple: amdgcn-amd-amdhsa--gfx942

// SYMLOOKUP: Passed symbol name lookup tests.
        .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
        .p2align        2                               ; -- Begin function _Z5myAddi
        .type   _Z5myAddi,@function
_Z5myAddi:                              ; @_Z5myAddi
; %bb.0:
        s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
        v_add_u32_e32 v0, 5, v0
        s_setpc_b64 s[30:31]
.Lfunc_end0:
        .size   _Z5myAddi, .Lfunc_end0-_Z5myAddi
                                        ; -- End function
        .section        .AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 12
; NumSgprs: 38
; NumVgprs: 1
; NumAgprs: 0
; TotalNumVgprs: 1
; ScratchSize: 0
; MemoryBound: 0
        .text
        .protected      _Z6squarePii            ; -- Begin function _Z6squarePii
        .globl  _Z6squarePii
        .p2align        8
        .type   _Z6squarePii,@function
_Z6squarePii:                           ; @_Z6squarePii
; %bb.0:
        s_load_dword s3, s[0:1], 0x1c
        s_load_dword s4, s[0:1], 0x8
        s_mov_b32 s32, 0
        s_waitcnt lgkmcnt(0)
        s_and_b32 s3, s3, 0xffff
        s_mul_i32 s2, s2, s3
        v_add_u32_e32 v0, s2, v0
        v_cmp_gt_i32_e32 vcc, s4, v0
        s_and_saveexec_b64 s[2:3], vcc
        s_cbranch_execz .LBB1_2
; %bb.1:
        s_load_dwordx2 s[0:1], s[0:1], 0x0
        v_ashrrev_i32_e32 v1, 31, v0
        s_waitcnt lgkmcnt(0)
        v_lshl_add_u64 v[2:3], v[0:1], 2, s[0:1]
        global_load_dword v0, v[2:3], off
        s_getpc_b64 s[0:1]
        s_add_u32 s0, s0, _Z5myAddi@rel32@lo+4
        s_addc_u32 s1, s1, _Z5myAddi@rel32@hi+12
        s_waitcnt vmcnt(0)
        v_mul_lo_u32 v0, v0, v0
        s_swappc_b64 s[30:31], s[0:1]
        s_getpc_b64 s[0:1]
        s_add_u32 s0, s0, MyManagedVar@rel32@lo+4
        s_addc_u32 s1, s1, MyManagedVar@rel32@hi+12
        s_load_dwordx2 s[0:1], s[0:1], 0x0
        s_getpc_b64 s[2:3]
        s_add_u32 s2, s2, MyDeviceVar@rel32@lo+4
        s_addc_u32 s3, s3, MyDeviceVar@rel32@hi+12
        s_load_dword s4, s[2:3], 0x0
        s_waitcnt lgkmcnt(0)
        s_load_dword s5, s[0:1], 0x0
        s_waitcnt lgkmcnt(0)
        s_mul_i32 s5, s5, s4
        v_add_u32_e32 v0, s5, v0
        global_store_dword v[2:3], v0, off
.LBB1_2:
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel _Z6squarePii
                .amdhsa_group_segment_fixed_size 0
                .amdhsa_private_segment_fixed_size 0
                .amdhsa_kernarg_size 272
                .amdhsa_user_sgpr_count 2
                .amdhsa_user_sgpr_dispatch_ptr 0
                .amdhsa_user_sgpr_queue_ptr 0
                .amdhsa_user_sgpr_kernarg_segment_ptr 1
                .amdhsa_user_sgpr_dispatch_id 0
                .amdhsa_user_sgpr_kernarg_preload_length  0
                .amdhsa_user_sgpr_kernarg_preload_offset  0
                .amdhsa_user_sgpr_private_segment_size 0
                .amdhsa_uses_dynamic_stack 0
                .amdhsa_enable_private_segment 0
                .amdhsa_system_sgpr_workgroup_id_x 1
                .amdhsa_system_sgpr_workgroup_id_y 0
                .amdhsa_system_sgpr_workgroup_id_z 0
                .amdhsa_system_sgpr_workgroup_info 0
                .amdhsa_system_vgpr_workitem_id 0
                .amdhsa_next_free_vgpr 4
                .amdhsa_next_free_sgpr 33
                .amdhsa_accum_offset 4
                .amdhsa_reserve_xnack_mask 1
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
.Lfunc_end1:
        .size   _Z6squarePii, .Lfunc_end1-_Z6squarePii
                                        ; -- End function
        .section        .AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 212
; NumSgprs: 39
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
; NumSGPRsForWavesPerEU: 39
; NumVGPRsForWavesPerEU: 4
; AccumOffset: 4
; Occupancy: 8
; WaveLimiterHint : 1
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
        .protected      MyDeviceVar             ; @MyDeviceVar
        .type   MyDeviceVar,@object
        .section        .bss,"aw",@nobits
        .globl  MyDeviceVar
        .p2align        2, 0x0
MyDeviceVar:
        .long   0                               ; 0x0
        .size   MyDeviceVar, 4

        .protected      MyManagedVar.managed    ; @MyManagedVar.managed
        .type   MyManagedVar.managed,@object
        .globl  MyManagedVar.managed
        .p2align        2, 0x0
MyManagedVar.managed:
        .long   0                               ; 0x0
        .size   MyManagedVar.managed, 4

        .protected      MyManagedVar            ; @MyManagedVar
        .type   MyManagedVar,@object
        .globl  MyManagedVar
        .p2align        3, 0x0
MyManagedVar:
        .quad   0
        .size   MyManagedVar, 8

        .type   __hip_cuid_7730498e1315b1c0,@object ; @__hip_cuid_7730498e1315b1c0
        .globl  __hip_cuid_7730498e1315b1c0
__hip_cuid_7730498e1315b1c0:
        .byte   0                               ; 0x0
        .size   __hip_cuid_7730498e1315b1c0, 1

        .ident  "AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)"
        .section        ".note.GNU-stack","",@progbits
        .addrsig
        .addrsig_sym MyDeviceVar
        .addrsig_sym MyManagedVar.managed
        .addrsig_sym MyManagedVar
        .addrsig_sym __hip_cuid_7730498e1315b1c0
        .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z6squarePii
    .private_segment_fixed_size: 0
    .sgpr_count:     39
    .sgpr_spill_count: 0
    .symbol:         _Z6squarePii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     4
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

        .end_amdgpu_metadata