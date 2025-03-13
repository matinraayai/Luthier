	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.protected	_Z14calculateForcePA400_A400_dS1_S1_S1_dddddd ; -- Begin function _Z14calculateForcePA400_A400_dS1_S1_S1_dddddd
	.globl	_Z14calculateForcePA400_A400_dS1_S1_S1_dddddd
	.p2align	8
	.type	_Z14calculateForcePA400_A400_dS1_S1_S1_dddddd,@function
_Z14calculateForcePA400_A400_dS1_S1_S1_dddddd: ; @_Z14calculateForcePA400_A400_dS1_S1_S1_dddddd
; %bb.0:
	s_load_dwordx2 s[0:1], s[4:5], 0x5c
	s_load_dwordx16 s[12:27], s[4:5], 0x0
	v_mov_b32_e32 v4, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_lshr_b32 s2, s0, 16
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	s_mul_i32 s7, s7, s2
	s_mul_i32 s8, s8, s1
	v_add_u32_e32 v3, s6, v0
	v_add_u32_e32 v32, s7, v1
	v_add_u32_e32 v33, s8, v2
	v_max3_u32 v0, v3, v32, v33
	s_movk_i32 s0, 0x18e
	v_cmp_lt_u32_e32 vcc, s0, v0
	v_cmp_eq_u32_e64 s[0:1], 0, v33
	v_cmp_eq_u32_e64 s[2:3], 0, v32
	s_or_b64 s[0:1], s[0:1], s[2:3]
	s_or_b64 s[0:1], s[0:1], vcc
	v_cmp_eq_u32_e32 vcc, 0, v3
	s_mov_b32 s6, 0
	s_or_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_xor_b64 s[0:1], exec, s[2:3]
	s_cbranch_execnz .LBB0_3
; %bb.1:                                ; %Flow142
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execnz .LBB0_4
.LBB0_2:
	s_endpgm
.LBB0_3:
	s_mov_b32 s2, 0x138800
	v_mad_u64_u32 v[0:1], s[2:3], v33, s2, 0
	s_movk_i32 s7, 0xc80
	v_mad_u64_u32 v[5:6], s[2:3], v32, s7, 0
	v_mov_b32_e32 v2, s15
	v_add_co_u32_e32 v7, vcc, s14, v0
	v_addc_co_u32_e32 v2, vcc, v2, v1, vcc
	v_add_co_u32_e32 v7, vcc, v7, v5
	v_addc_co_u32_e32 v8, vcc, v2, v6, vcc
	v_lshlrev_b64 v[2:3], 3, v[3:4]
	s_mov_b32 s7, s6
	v_add_co_u32_e32 v7, vcc, v7, v2
	v_mov_b32_e32 v10, s7
	v_addc_co_u32_e32 v8, vcc, v8, v3, vcc
	v_mov_b32_e32 v9, s6
	global_store_dwordx2 v[7:8], v[9:10], off
	v_mov_b32_e32 v4, s17
	v_add_co_u32_e32 v7, vcc, s16, v0
	v_addc_co_u32_e32 v4, vcc, v4, v1, vcc
	v_add_co_u32_e32 v7, vcc, v7, v5
	v_addc_co_u32_e32 v4, vcc, v4, v6, vcc
	v_add_co_u32_e32 v7, vcc, v7, v2
	v_addc_co_u32_e32 v8, vcc, v4, v3, vcc
	v_mov_b32_e32 v4, s19
	v_add_co_u32_e32 v0, vcc, s18, v0
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc
	v_add_co_u32_e32 v0, vcc, v0, v5
	v_addc_co_u32_e32 v1, vcc, v1, v6, vcc
	v_add_co_u32_e32 v0, vcc, v0, v2
	v_addc_co_u32_e32 v1, vcc, v1, v3, vcc
	global_store_dwordx2 v[7:8], v[9:10], off
	global_store_dwordx2 v[0:1], v[9:10], off
                                        ; implicit-def: $vgpr33
                                        ; implicit-def: $vgpr32
                                        ; implicit-def: $vgpr3
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_2
.LBB0_4:
	v_mov_b32_e32 v0, s12
	v_mov_b32_e32 v1, s13
	s_mov_b32 s0, 0x138800
	v_mad_u64_u32 v[5:6], s[0:1], v33, s0, v[0:1]
	s_movk_i32 s2, 0xc80
	v_mov_b32_e32 v4, 0
	v_mad_u64_u32 v[7:8], s[0:1], v32, s2, v[5:6]
	v_lshlrev_b64 v[0:1], 3, v[3:4]
	s_mov_b32 s0, 0x138000
	v_add_co_u32_e32 v9, vcc, v7, v0
	v_addc_co_u32_e32 v10, vcc, v8, v1, vcc
	v_add_co_u32_e32 v11, vcc, s0, v9
	v_addc_co_u32_e32 v12, vcc, 0, v10, vcc
	s_mov_b32 s0, 0xffec8000
	v_add_u32_e32 v2, -1, v32
	v_add_co_u32_e32 v13, vcc, s0, v9
	v_mad_u64_u32 v[5:6], s[0:1], v2, s2, v[5:6]
	v_addc_co_u32_e32 v14, vcc, -1, v10, vcc
	v_add_co_u32_e32 v5, vcc, v5, v0
	global_load_dwordx2 v[15:16], v[11:12], off offset:2048
	global_load_dwordx2 v[17:18], v[13:14], off offset:-2048
	v_addc_co_u32_e32 v6, vcc, v6, v1, vcc
	global_load_dwordx2 v[11:12], v[5:6], off
	global_load_dwordx2 v[13:14], v[9:10], off offset:3200
	v_add_u32_e32 v3, -1, v3
	v_lshlrev_b64 v[2:3], 3, v[3:4]
	v_add_f64 v[19:20], s[24:25], s[24:25]
	v_add_co_u32_e32 v2, vcc, v7, v2
	v_addc_co_u32_e32 v3, vcc, v8, v3, vcc
	global_load_dwordx2 v[4:5], v[2:3], off
	global_load_dwordx2 v[6:7], v[9:10], off offset:8
	v_add_f64 v[2:3], s[20:21], s[20:21]
	v_add_f64 v[8:9], s[22:23], s[22:23]
	s_waitcnt vmcnt(4)
	v_add_f64 v[15:16], v[15:16], -v[17:18]
	s_waitcnt vmcnt(2)
	v_add_f64 v[10:11], v[13:14], -v[11:12]
	v_div_scale_f64 v[12:13], s[0:1], v[2:3], v[2:3], v[15:16]
	v_div_scale_f64 v[17:18], s[0:1], v[8:9], v[8:9], v[10:11]
	s_waitcnt vmcnt(0)
	v_add_f64 v[6:7], v[6:7], -v[4:5]
	v_div_scale_f64 v[4:5], s[0:1], v[19:20], v[19:20], v[6:7]
	v_div_scale_f64 v[36:37], s[0:1], v[10:11], v[8:9], v[10:11]
	v_rcp_f64_e32 v[21:22], v[12:13]
	v_rcp_f64_e32 v[23:24], v[17:18]
	v_rcp_f64_e32 v[25:26], v[4:5]
	v_fma_f64 v[27:28], -v[12:13], v[21:22], 1.0
	v_fma_f64 v[29:30], -v[17:18], v[23:24], 1.0
	v_fma_f64 v[21:22], v[21:22], v[27:28], v[21:22]
	v_div_scale_f64 v[27:28], vcc, v[15:16], v[2:3], v[15:16]
	v_fma_f64 v[23:24], v[23:24], v[29:30], v[23:24]
	v_fma_f64 v[29:30], -v[4:5], v[25:26], 1.0
	v_fma_f64 v[34:35], -v[12:13], v[21:22], 1.0
	v_fma_f64 v[38:39], -v[17:18], v[23:24], 1.0
	v_fma_f64 v[25:26], v[25:26], v[29:30], v[25:26]
	v_div_scale_f64 v[29:30], s[2:3], v[6:7], v[19:20], v[6:7]
	v_fma_f64 v[21:22], v[21:22], v[34:35], v[21:22]
	v_fma_f64 v[23:24], v[23:24], v[38:39], v[23:24]
	v_fma_f64 v[34:35], -v[4:5], v[25:26], 1.0
	v_mul_f64 v[38:39], v[27:28], v[21:22]
	v_mul_f64 v[40:41], v[36:37], v[23:24]
	v_fma_f64 v[25:26], v[25:26], v[34:35], v[25:26]
	v_fma_f64 v[12:13], -v[12:13], v[38:39], v[27:28]
	v_fma_f64 v[17:18], -v[17:18], v[40:41], v[36:37]
	v_mul_f64 v[27:28], v[29:30], v[25:26]
	v_div_fmas_f64 v[12:13], v[12:13], v[21:22], v[38:39]
	s_mov_b64 vcc, s[0:1]
	v_div_fmas_f64 v[17:18], v[17:18], v[23:24], v[40:41]
	v_fma_f64 v[4:5], -v[4:5], v[27:28], v[29:30]
	s_mov_b64 vcc, s[2:3]
	v_div_fmas_f64 v[21:22], v[4:5], v[25:26], v[27:28]
                                        ; implicit-def: $vgpr28_vgpr29
	v_div_fixup_f64 v[14:15], v[12:13], v[2:3], v[15:16]
	v_div_fixup_f64 v[8:9], v[17:18], v[8:9], v[10:11]
	v_cmp_neq_f64_e64 s[2:3], 0, v[14:15]
	v_mul_f64 v[4:5], v[8:9], v[8:9]
	v_div_fixup_f64 v[2:3], v[21:22], v[19:20], v[6:7]
	v_cmp_neq_f64_e64 s[0:1], 0, v[8:9]
	v_fma_f64 v[6:7], v[14:15], v[14:15], v[4:5]
	v_cmp_neq_f64_e64 s[6:7], 0, v[2:3]
	s_or_b64 s[8:9], s[2:3], s[0:1]
	v_fma_f64 v[18:19], v[2:3], v[2:3], v[6:7]
	s_or_b64 s[8:9], s[8:9], s[6:7]
	s_xor_b64 s[10:11], s[8:9], -1
	s_and_saveexec_b64 s[12:13], s[10:11]
	s_xor_b64 s[10:11], exec, s[12:13]
; %bb.5:
	v_mov_b32_e32 v10, 0xaaaaaaab
	v_mov_b32_e32 v11, 0xbffaaaaa
	v_fma_f64 v[28:29], s[26:27], v[10:11], 1.0
; %bb.6:                                ; %Flow
	s_or_saveexec_b64 s[10:11], s[10:11]
	v_mul_f64 v[16:17], v[14:15], v[14:15]
	v_mul_f64 v[12:13], v[4:5], v[4:5]
	v_mul_f64 v[10:11], v[2:3], v[2:3]
	v_mul_f64 v[20:21], v[18:19], v[18:19]
	v_fma_f64 v[12:13], v[16:17], v[16:17], v[12:13]
	s_xor_b64 exec, exec, s[10:11]
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0xc0080000
	v_fma_f64 v[22:23], s[26:27], v[22:23], 1.0
	v_mul_f64 v[24:25], s[26:27], 4.0
	v_div_scale_f64 v[26:27], s[12:13], v[22:23], v[22:23], v[24:25]
	v_rcp_f64_e32 v[28:29], v[26:27]
	v_fma_f64 v[30:31], -v[26:27], v[28:29], 1.0
	v_fma_f64 v[28:29], v[28:29], v[30:31], v[28:29]
	v_fma_f64 v[30:31], -v[26:27], v[28:29], 1.0
	v_fma_f64 v[28:29], v[28:29], v[30:31], v[28:29]
	v_div_scale_f64 v[30:31], vcc, v[24:25], v[22:23], v[24:25]
	v_mul_f64 v[34:35], v[30:31], v[28:29]
	v_fma_f64 v[26:27], -v[26:27], v[34:35], v[30:31]
	s_nop 1
	v_div_fmas_f64 v[26:27], v[26:27], v[28:29], v[34:35]
	v_div_fixup_f64 v[24:25], v[26:27], v[22:23], v[24:25]
	v_fma_f64 v[26:27], v[10:11], v[10:11], v[12:13]
	v_div_scale_f64 v[28:29], s[12:13], v[20:21], v[20:21], v[26:27]
	v_rcp_f64_e32 v[30:31], v[28:29]
	v_fma_f64 v[34:35], -v[28:29], v[30:31], 1.0
	v_fma_f64 v[30:31], v[30:31], v[34:35], v[30:31]
	v_fma_f64 v[34:35], -v[28:29], v[30:31], 1.0
	v_fma_f64 v[30:31], v[30:31], v[34:35], v[30:31]
	v_div_scale_f64 v[34:35], vcc, v[26:27], v[20:21], v[26:27]
	v_mul_f64 v[36:37], v[34:35], v[30:31]
	v_fma_f64 v[28:29], -v[28:29], v[36:37], v[34:35]
	s_nop 1
	v_div_fmas_f64 v[28:29], v[28:29], v[30:31], v[36:37]
	v_div_fixup_f64 v[26:27], v[28:29], v[20:21], v[26:27]
	v_fma_f64 v[24:25], v[24:25], v[26:27], 1.0
	v_mul_f64 v[28:29], v[22:23], v[24:25]
.LBB0_8:                                ; %_Z2Wnddddd.exit
	s_or_b64 exec, exec, s[10:11]
	v_add_f64 v[22:23], v[4:5], v[10:11]
	v_mul_f64 v[26:27], v[10:11], v[10:11]
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v31, v25
	v_mov_b32_e32 v30, v24
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_fma_f64 v[30:31], v[4:5], v[4:5], v[26:27]
	v_mul_f64 v[34:35], v[14:15], v[16:17]
	v_mul_f64 v[30:31], v[14:15], v[30:31]
	v_fma_f64 v[30:31], v[34:35], v[22:23], -v[30:31]
	v_div_scale_f64 v[34:35], s[8:9], v[20:21], v[20:21], v[30:31]
	v_div_scale_f64 v[40:41], vcc, v[30:31], v[20:21], v[30:31]
	v_rcp_f64_e32 v[36:37], v[34:35]
	v_fma_f64 v[38:39], -v[34:35], v[36:37], 1.0
	v_fma_f64 v[36:37], v[36:37], v[38:39], v[36:37]
	v_fma_f64 v[38:39], -v[34:35], v[36:37], 1.0
	v_fma_f64 v[36:37], v[36:37], v[38:39], v[36:37]
	v_mul_f64 v[38:39], v[40:41], v[36:37]
	v_fma_f64 v[34:35], -v[34:35], v[38:39], v[40:41]
	v_div_fmas_f64 v[34:35], v[34:35], v[36:37], v[38:39]
	v_div_fixup_f64 v[30:31], v[34:35], v[20:21], v[30:31]
.LBB0_10:                               ; %_Z5dFuncddd.exit
	s_or_b64 exec, exec, s[10:11]
	s_load_dwordx2 s[4:5], s[4:5], 0x40
	s_mov_b32 s8, 0x138800
	s_movk_i32 s9, 0xc80
	s_waitcnt lgkmcnt(0)
	v_ldexp_f64 v[20:21], s[4:5], 4
	v_mul_f64 v[28:29], v[28:29], s[4:5]
	v_mul_f64 v[20:21], v[20:21], s[26:27]
	v_mul_f64 v[18:19], v[18:19], v[28:29]
	v_mul_f64 v[20:21], v[20:21], v[18:19]
	v_mul_f64 v[18:19], v[28:29], v[28:29]
	v_mul_f64 v[28:29], v[20:21], v[30:31]
	v_mov_b32_e32 v31, s15
	v_mov_b32_e32 v30, s14
	v_mad_u64_u32 v[30:31], s[4:5], v33, s8, v[30:31]
	v_mad_u64_u32 v[30:31], s[4:5], v32, s9, v[30:31]
	v_fma_f64 v[28:29], v[14:15], v[18:19], v[28:29]
	v_add_f64 v[14:15], v[16:17], v[10:11]
	s_or_b64 s[4:5], s[0:1], s[6:7]
	v_add_co_u32_e32 v30, vcc, v30, v0
	v_addc_co_u32_e32 v31, vcc, v31, v1, vcc
	s_or_b64 s[10:11], s[2:3], s[4:5]
	global_store_dwordx2 v[30:31], v[28:29], off
	s_and_saveexec_b64 s[4:5], s[10:11]
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_fma_f64 v[24:25], v[16:17], v[16:17], v[26:27]
	v_mul_f64 v[26:27], v[8:9], v[4:5]
	v_add_f64 v[16:17], v[16:17], v[22:23]
	v_mul_f64 v[22:23], v[8:9], v[24:25]
	v_mul_f64 v[16:17], v[16:17], v[16:17]
	v_fma_f64 v[22:23], v[26:27], v[14:15], -v[22:23]
	v_div_scale_f64 v[24:25], s[10:11], v[16:17], v[16:17], v[22:23]
	v_div_scale_f64 v[30:31], vcc, v[22:23], v[16:17], v[22:23]
	v_rcp_f64_e32 v[26:27], v[24:25]
	v_fma_f64 v[28:29], -v[24:25], v[26:27], 1.0
	v_fma_f64 v[26:27], v[26:27], v[28:29], v[26:27]
	v_fma_f64 v[28:29], -v[24:25], v[26:27], 1.0
	v_fma_f64 v[26:27], v[26:27], v[28:29], v[26:27]
	v_mul_f64 v[28:29], v[30:31], v[26:27]
	v_fma_f64 v[24:25], -v[24:25], v[28:29], v[30:31]
	v_div_fmas_f64 v[24:25], v[24:25], v[26:27], v[28:29]
	v_div_fixup_f64 v[24:25], v[24:25], v[16:17], v[22:23]
.LBB0_12:                               ; %_Z5dFuncddd.exit109
	s_or_b64 exec, exec, s[4:5]
	v_mul_f64 v[16:17], v[20:21], v[24:25]
	v_mov_b32_e32 v23, s17
	v_mov_b32_e32 v22, s16
	v_mad_u64_u32 v[22:23], s[4:5], v33, s8, v[22:23]
	s_or_b64 s[2:3], s[2:3], s[6:7]
	s_or_b64 s[2:3], s[0:1], s[2:3]
	v_mad_u64_u32 v[22:23], s[4:5], v32, s9, v[22:23]
	v_fma_f64 v[16:17], v[8:9], v[18:19], v[16:17]
	v_mov_b32_e32 v8, 0
	v_add_co_u32_e32 v22, vcc, v22, v0
	v_mov_b32_e32 v9, 0
	v_addc_co_u32_e32 v23, vcc, v23, v1, vcc
	global_store_dwordx2 v[22:23], v[16:17], off
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_14
; %bb.13:
	v_mul_f64 v[8:9], v[2:3], v[10:11]
	v_mul_f64 v[10:11], v[2:3], v[12:13]
	v_add_f64 v[4:5], v[4:5], v[14:15]
	v_fma_f64 v[6:7], v[6:7], v[8:9], -v[10:11]
	v_mul_f64 v[4:5], v[4:5], v[4:5]
	v_div_scale_f64 v[8:9], s[2:3], v[4:5], v[4:5], v[6:7]
	v_div_scale_f64 v[14:15], vcc, v[6:7], v[4:5], v[6:7]
	v_rcp_f64_e32 v[10:11], v[8:9]
	v_fma_f64 v[12:13], -v[8:9], v[10:11], 1.0
	v_fma_f64 v[10:11], v[10:11], v[12:13], v[10:11]
	v_fma_f64 v[12:13], -v[8:9], v[10:11], 1.0
	v_fma_f64 v[10:11], v[10:11], v[12:13], v[10:11]
	v_mul_f64 v[12:13], v[14:15], v[10:11]
	v_fma_f64 v[8:9], -v[8:9], v[12:13], v[14:15]
	v_div_fmas_f64 v[8:9], v[8:9], v[10:11], v[12:13]
	v_div_fixup_f64 v[8:9], v[8:9], v[4:5], v[6:7]
.LBB0_14:                               ; %_Z5dFuncddd.exit113
	s_or_b64 exec, exec, s[0:1]
	v_mul_f64 v[4:5], v[20:21], v[8:9]
	v_mov_b32_e32 v6, s18
	s_mov_b32 s0, 0x138800
	v_mov_b32_e32 v7, s19
	v_mad_u64_u32 v[6:7], s[0:1], v33, s0, v[6:7]
	s_movk_i32 s0, 0xc80
	v_fma_f64 v[2:3], v[2:3], v[18:19], v[4:5]
	v_mad_u64_u32 v[4:5], s[0:1], v32, s0, v[6:7]
	v_add_co_u32_e32 v0, vcc, v4, v0
	v_addc_co_u32_e32 v1, vcc, v5, v1, vcc
	global_store_dwordx2 v[0:1], v[2:3], off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z14calculateForcePA400_A400_dS1_S1_S1_dddddd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 336
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 42
		.amdhsa_next_free_sgpr 28
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
	.size	_Z14calculateForcePA400_A400_dS1_S1_S1_dddddd, .Lfunc_end0-_Z14calculateForcePA400_A400_dS1_S1_S1_dddddd
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1908
; NumSgprs: 32
; NumVgprs: 42
; NumAgprs: 0
; TotalNumVgprs: 42
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 10
; NumSGPRsForWavesPerEU: 32
; NumVGPRsForWavesPerEU: 42
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.text
	.protected	_Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd ; -- Begin function _Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd
	.globl	_Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd
	.p2align	8
	.type	_Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd,@function
_Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd: ; @_Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd
; %bb.0:
	s_load_dwordx2 s[0:1], s[4:5], 0x7c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_lshr_b32 s2, s0, 16
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	s_mul_i32 s7, s7, s2
	s_mul_i32 s8, s8, s1
	v_add_u32_e32 v6, s6, v0
	v_add_u32_e32 v28, s7, v1
	v_add_u32_e32 v29, s8, v2
	v_max3_u32 v0, v6, v28, v29
	s_movk_i32 s0, 0x18f
	v_cmp_gt_u32_e32 vcc, s0, v0
	v_cmp_ne_u32_e64 s[0:1], 0, v29
	v_cmp_ne_u32_e64 s[2:3], 0, v28
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_and_b64 s[0:1], s[0:1], vcc
	v_cmp_ne_u32_e32 vcc, 0, v6
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB1_10
; %bb.1:
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	s_mov_b32 s0, 0x138800
	s_movk_i32 s2, 0xc80
	v_mov_b32_e32 v7, 0
	v_lshlrev_b64 v[4:5], 3, v[6:7]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v0, s10
	v_mov_b32_e32 v1, s11
	v_mad_u64_u32 v[0:1], s[0:1], v29, s0, v[0:1]
	v_add_u32_e32 v30, -1, v28
	v_add_u32_e32 v6, -1, v6
	v_mad_u64_u32 v[8:9], s[0:1], v28, s2, v[0:1]
	s_mov_b32 s0, 0x138000
	s_load_dwordx4 s[24:27], s[4:5], 0x30
	v_add_co_u32_e32 v10, vcc, v8, v4
	v_addc_co_u32_e32 v11, vcc, v9, v5, vcc
	v_add_co_u32_e32 v2, vcc, s0, v10
	v_addc_co_u32_e32 v3, vcc, 0, v11, vcc
	s_mov_b32 s0, 0xffec8000
	v_add_co_u32_e32 v12, vcc, s0, v10
	v_addc_co_u32_e32 v13, vcc, -1, v11, vcc
	global_load_dwordx2 v[14:15], v[2:3], off offset:2048
	global_load_dwordx2 v[16:17], v[12:13], off offset:-2048
	v_mad_u64_u32 v[0:1], s[0:1], v30, s2, v[0:1]
	v_lshlrev_b64 v[12:13], 3, v[6:7]
	v_add_co_u32_e32 v18, vcc, v0, v4
	v_addc_co_u32_e32 v19, vcc, v1, v5, vcc
	v_add_co_u32_e32 v6, vcc, v8, v12
	global_load_dwordx2 v[20:21], v[10:11], off offset:3200
	global_load_dwordx4 v[0:3], v[10:11], off
	v_addc_co_u32_e32 v7, vcc, v9, v13, vcc
	global_load_dwordx2 v[22:23], v[18:19], off
	global_load_dwordx2 v[24:25], v[6:7], off
	s_load_dwordx8 s[16:23], s[4:5], 0x40
	s_load_dwordx4 s[0:3], s[4:5], 0x60
	s_waitcnt lgkmcnt(0)
	v_add_f64 v[6:7], s[22:23], s[22:23]
	v_add_f64 v[8:9], s[0:1], s[0:1]
	v_add_f64 v[10:11], s[2:3], s[2:3]
	s_waitcnt vmcnt(4)
	v_add_f64 v[14:15], v[14:15], -v[16:17]
	v_div_scale_f64 v[18:19], s[0:1], v[6:7], v[6:7], v[14:15]
	s_waitcnt vmcnt(1)
	v_add_f64 v[16:17], v[20:21], -v[22:23]
	s_waitcnt vmcnt(0)
	v_add_f64 v[2:3], v[2:3], -v[24:25]
	v_div_scale_f64 v[20:21], s[0:1], v[8:9], v[8:9], v[16:17]
	v_div_scale_f64 v[22:23], s[0:1], v[10:11], v[10:11], v[2:3]
	v_rcp_f64_e32 v[24:25], v[18:19]
	v_rcp_f64_e32 v[26:27], v[20:21]
	v_rcp_f64_e32 v[31:32], v[22:23]
	v_fma_f64 v[33:34], -v[18:19], v[24:25], 1.0
	v_fma_f64 v[24:25], v[24:25], v[33:34], v[24:25]
	v_fma_f64 v[35:36], -v[20:21], v[26:27], 1.0
	v_fma_f64 v[37:38], -v[22:23], v[31:32], 1.0
	v_fma_f64 v[33:34], -v[18:19], v[24:25], 1.0
	v_fma_f64 v[26:27], v[26:27], v[35:36], v[26:27]
	v_div_scale_f64 v[35:36], vcc, v[14:15], v[6:7], v[14:15]
	v_fma_f64 v[31:32], v[31:32], v[37:38], v[31:32]
	v_fma_f64 v[24:25], v[24:25], v[33:34], v[24:25]
	v_div_scale_f64 v[33:34], s[0:1], v[16:17], v[8:9], v[16:17]
	v_fma_f64 v[37:38], -v[20:21], v[26:27], 1.0
	v_fma_f64 v[39:40], -v[22:23], v[31:32], 1.0
	v_fma_f64 v[26:27], v[26:27], v[37:38], v[26:27]
	v_mul_f64 v[37:38], v[35:36], v[24:25]
	v_fma_f64 v[31:32], v[31:32], v[39:40], v[31:32]
	v_div_scale_f64 v[39:40], s[2:3], v[2:3], v[10:11], v[2:3]
	v_fma_f64 v[18:19], -v[18:19], v[37:38], v[35:36]
	v_mul_f64 v[35:36], v[33:34], v[26:27]
	v_div_fmas_f64 v[18:19], v[18:19], v[24:25], v[37:38]
	v_fma_f64 v[20:21], -v[20:21], v[35:36], v[33:34]
	s_mov_b64 vcc, s[0:1]
	v_mul_f64 v[33:34], v[39:40], v[31:32]
	v_div_fmas_f64 v[20:21], v[20:21], v[26:27], v[35:36]
	s_mov_b64 vcc, s[2:3]
	v_fma_f64 v[22:23], -v[22:23], v[33:34], v[39:40]
	v_div_fmas_f64 v[24:25], v[22:23], v[31:32], v[33:34]
	v_div_fixup_f64 v[22:23], v[18:19], v[6:7], v[14:15]
	v_div_fixup_f64 v[14:15], v[20:21], v[8:9], v[16:17]
	v_cmp_eq_f64_e32 vcc, 0, v[22:23]
	v_cmp_eq_f64_e64 s[0:1], 0, v[14:15]
	v_div_fixup_f64 v[2:3], v[24:25], v[10:11], v[2:3]
                                        ; implicit-def: $vgpr24_vgpr25
	s_and_b64 s[0:1], vcc, s[0:1]
	v_cmp_eq_f64_e64 s[2:3], 0, v[2:3]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_xor_b64 s[2:3], exec, s[2:3]
; %bb.2:
	v_mov_b32_e32 v16, 0xaaaaaaab
	v_mov_b32_e32 v17, 0xbffaaaaa
	v_fma_f64 v[24:25], s[24:25], v[16:17], 1.0
; %bb.3:                                ; %Flow118
	s_or_saveexec_b64 s[2:3], s[2:3]
	v_mul_f64 v[26:27], v[14:15], v[14:15]
	v_mul_f64 v[18:19], v[22:23], v[22:23]
	v_mul_f64 v[16:17], v[2:3], v[2:3]
	v_mul_f64 v[14:15], s[24:25], 4.0
	s_load_dwordx4 s[4:7], s[4:5], 0x20
	v_mul_f64 v[20:21], v[26:27], v[26:27]
	v_fma_f64 v[22:23], v[22:23], v[22:23], v[26:27]
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB1_5
; %bb.4:
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0xc0080000
	v_fma_f64 v[24:25], s[24:25], v[24:25], 1.0
	v_fma_f64 v[35:36], v[18:19], v[18:19], v[20:21]
	v_div_scale_f64 v[26:27], s[10:11], v[24:25], v[24:25], v[14:15]
	v_fma_f64 v[35:36], v[16:17], v[16:17], v[35:36]
	v_div_scale_f64 v[39:40], vcc, v[14:15], v[24:25], v[14:15]
	v_rcp_f64_e32 v[31:32], v[26:27]
	v_fma_f64 v[33:34], -v[26:27], v[31:32], 1.0
	v_fma_f64 v[31:32], v[31:32], v[33:34], v[31:32]
	v_fma_f64 v[33:34], v[2:3], v[2:3], v[22:23]
	v_fma_f64 v[37:38], -v[26:27], v[31:32], 1.0
	v_mul_f64 v[33:34], v[33:34], v[33:34]
	v_fma_f64 v[31:32], v[31:32], v[37:38], v[31:32]
	v_div_scale_f64 v[37:38], s[10:11], v[33:34], v[33:34], v[35:36]
	v_mul_f64 v[41:42], v[39:40], v[31:32]
	v_fma_f64 v[26:27], -v[26:27], v[41:42], v[39:40]
	v_rcp_f64_e32 v[39:40], v[37:38]
	v_div_fmas_f64 v[26:27], v[26:27], v[31:32], v[41:42]
	v_div_scale_f64 v[31:32], vcc, v[35:36], v[33:34], v[35:36]
	v_fma_f64 v[43:44], -v[37:38], v[39:40], 1.0
	v_div_fixup_f64 v[26:27], v[26:27], v[24:25], v[14:15]
	v_fma_f64 v[39:40], v[39:40], v[43:44], v[39:40]
	v_fma_f64 v[41:42], -v[37:38], v[39:40], 1.0
	v_fma_f64 v[39:40], v[39:40], v[41:42], v[39:40]
	v_mul_f64 v[41:42], v[31:32], v[39:40]
	v_fma_f64 v[31:32], -v[37:38], v[41:42], v[31:32]
	v_div_fmas_f64 v[31:32], v[31:32], v[39:40], v[41:42]
	v_div_fixup_f64 v[31:32], v[31:32], v[33:34], v[35:36]
	v_fma_f64 v[26:27], v[26:27], v[31:32], 1.0
	v_mul_f64 v[24:25], v[24:25], v[26:27]
.LBB1_5:                                ; %_Z2Andddd.exit.i
	s_or_b64 exec, exec, s[2:3]
                                        ; implicit-def: $vgpr26_vgpr27
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_xor_b64 s[0:1], exec, s[2:3]
	s_cbranch_execz .LBB1_7
; %bb.6:
	v_mov_b32_e32 v2, 0xaaaaaaab
	v_mov_b32_e32 v3, 0xbffaaaaa
	v_fma_f64 v[26:27], s[24:25], v[2:3], 1.0
                                        ; implicit-def: $vgpr14_vgpr15
                                        ; implicit-def: $vgpr18_vgpr19
                                        ; implicit-def: $vgpr20_vgpr21
                                        ; implicit-def: $vgpr16_vgpr17
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr22_vgpr23
	s_andn2_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execnz .LBB1_8
	s_branch .LBB1_9
.LBB1_7:                                ; %Flow
	s_andn2_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB1_9
.LBB1_8:
	v_fma_f64 v[18:19], v[18:19], v[18:19], v[20:21]
	v_fma_f64 v[2:3], v[2:3], v[2:3], v[22:23]
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0xc0080000
	v_fma_f64 v[20:21], s[24:25], v[20:21], 1.0
	v_fma_f64 v[16:17], v[16:17], v[16:17], v[18:19]
	v_mul_f64 v[2:3], v[2:3], v[2:3]
	v_div_scale_f64 v[18:19], s[0:1], v[20:21], v[20:21], v[14:15]
	v_div_scale_f64 v[22:23], s[0:1], v[2:3], v[2:3], v[16:17]
	v_div_scale_f64 v[39:40], s[0:1], v[16:17], v[2:3], v[16:17]
	v_rcp_f64_e32 v[26:27], v[18:19]
	v_rcp_f64_e32 v[31:32], v[22:23]
	v_fma_f64 v[33:34], -v[18:19], v[26:27], 1.0
	v_fma_f64 v[35:36], -v[22:23], v[31:32], 1.0
	v_fma_f64 v[26:27], v[26:27], v[33:34], v[26:27]
	v_div_scale_f64 v[33:34], vcc, v[14:15], v[20:21], v[14:15]
	v_fma_f64 v[31:32], v[31:32], v[35:36], v[31:32]
	v_fma_f64 v[35:36], -v[18:19], v[26:27], 1.0
	v_fma_f64 v[37:38], -v[22:23], v[31:32], 1.0
	v_fma_f64 v[26:27], v[26:27], v[35:36], v[26:27]
	v_fma_f64 v[31:32], v[31:32], v[37:38], v[31:32]
	v_mul_f64 v[35:36], v[33:34], v[26:27]
	v_mul_f64 v[37:38], v[39:40], v[31:32]
	v_fma_f64 v[18:19], -v[18:19], v[35:36], v[33:34]
	v_fma_f64 v[22:23], -v[22:23], v[37:38], v[39:40]
	v_div_fmas_f64 v[18:19], v[18:19], v[26:27], v[35:36]
	s_mov_b64 vcc, s[0:1]
	v_div_fmas_f64 v[22:23], v[22:23], v[31:32], v[37:38]
	v_div_fixup_f64 v[14:15], v[18:19], v[20:21], v[14:15]
	v_div_fixup_f64 v[2:3], v[22:23], v[2:3], v[16:17]
	v_fma_f64 v[2:3], v[14:15], v[2:3], 1.0
	v_mul_f64 v[26:27], v[20:21], v[2:3]
.LBB1_9:                                ; %_Z4taunddddd.exit
	s_or_b64 exec, exec, s[2:3]
	v_mul_f64 v[2:3], v[24:25], v[26:27]
	s_movk_i32 s2, 0xc80
	v_mov_b32_e32 v26, s15
	s_mov_b32 s3, 0x138000
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v31, s5
	v_mov_b32_e32 v34, s13
	v_mul_f64 v[2:3], v[2:3], s[16:17]
	v_div_scale_f64 v[18:19], s[0:1], v[2:3], v[2:3], s[20:21]
	v_div_scale_f64 v[20:21], vcc, s[20:21], v[2:3], s[20:21]
	s_mov_b32 s0, 0x138800
	v_rcp_f64_e32 v[14:15], v[18:19]
	v_fma_f64 v[16:17], -v[18:19], v[14:15], 1.0
	v_fma_f64 v[14:15], v[14:15], v[16:17], v[14:15]
	v_fma_f64 v[16:17], -v[18:19], v[14:15], 1.0
	v_fma_f64 v[22:23], v[14:15], v[16:17], v[14:15]
	v_mad_u64_u32 v[16:17], s[0:1], v29, s0, 0
	v_mad_u64_u32 v[14:15], s[0:1], v28, s2, 0
	v_add_co_u32_e64 v27, s[0:1], s14, v16
	v_addc_co_u32_e64 v26, s[0:1], v26, v17, s[0:1]
	v_mul_f64 v[24:25], v[20:21], v[22:23]
	v_fma_f64 v[18:19], -v[18:19], v[24:25], v[20:21]
	v_add_co_u32_e64 v20, s[0:1], v27, v14
	v_addc_co_u32_e64 v21, s[0:1], v26, v15, s[0:1]
	v_add_co_u32_e64 v26, s[0:1], v20, v4
	v_addc_co_u32_e64 v27, s[0:1], v21, v5, s[0:1]
	v_div_fmas_f64 v[18:19], v[18:19], v[22:23], v[24:25]
	v_add_co_u32_e32 v20, vcc, s3, v26
	v_addc_co_u32_e32 v21, vcc, 0, v27, vcc
	v_add_co_u32_e32 v22, vcc, 0xffec8000, v26
	v_addc_co_u32_e32 v23, vcc, -1, v27, vcc
	global_load_dwordx2 v[24:25], v[20:21], off offset:2048
	global_load_dwordx2 v[26:27], v[22:23], off offset:-2048
	v_add_co_u32_e32 v20, vcc, s4, v16
	v_addc_co_u32_e32 v21, vcc, v31, v17, vcc
	v_mov_b32_e32 v22, s7
	v_add_co_u32_e32 v28, vcc, s6, v16
	v_addc_co_u32_e32 v29, vcc, v22, v17, vcc
	v_add_co_u32_e32 v22, vcc, v20, v14
	v_addc_co_u32_e32 v23, vcc, v21, v15, vcc
	v_mad_u64_u32 v[20:21], s[0:1], v30, s2, v[20:21]
	v_add_co_u32_e32 v22, vcc, v22, v4
	v_addc_co_u32_e32 v23, vcc, v23, v5, vcc
	v_add_co_u32_e32 v20, vcc, v20, v4
	v_addc_co_u32_e32 v21, vcc, v21, v5, vcc
	global_load_dwordx2 v[22:23], v[22:23], off offset:3200
	s_nop 0
	global_load_dwordx2 v[20:21], v[20:21], off
	v_add_co_u32_e32 v30, vcc, v28, v14
	v_addc_co_u32_e32 v31, vcc, v29, v15, vcc
	v_add_co_u32_e32 v28, vcc, v30, v4
	v_addc_co_u32_e32 v29, vcc, v31, v5, vcc
	v_add_co_u32_e32 v12, vcc, v30, v12
	v_addc_co_u32_e32 v13, vcc, v31, v13, vcc
	global_load_dwordx2 v[28:29], v[28:29], off offset:8
	s_nop 0
	global_load_dwordx2 v[12:13], v[12:13], off
	v_add_co_u32_e32 v38, vcc, s12, v16
	v_addc_co_u32_e32 v39, vcc, v34, v17, vcc
	v_div_fixup_f64 v[2:3], v[18:19], v[2:3], s[20:21]
	s_waitcnt vmcnt(4)
	v_add_f64 v[24:25], v[24:25], -v[26:27]
	v_div_scale_f64 v[26:27], s[0:1], v[6:7], v[6:7], v[24:25]
	v_div_scale_f64 v[34:35], vcc, v[24:25], v[6:7], v[24:25]
	s_waitcnt vmcnt(2)
	v_add_f64 v[20:21], v[22:23], -v[20:21]
	v_rcp_f64_e32 v[30:31], v[26:27]
	v_div_scale_f64 v[22:23], s[0:1], v[8:9], v[8:9], v[20:21]
	s_waitcnt vmcnt(0)
	v_add_f64 v[12:13], v[28:29], -v[12:13]
	v_fma_f64 v[28:29], -v[26:27], v[30:31], 1.0
	v_div_scale_f64 v[32:33], s[0:1], v[10:11], v[10:11], v[12:13]
	v_add_co_u32_e64 v38, s[0:1], v38, v14
	v_fma_f64 v[28:29], v[30:31], v[28:29], v[30:31]
	v_rcp_f64_e32 v[30:31], v[22:23]
	v_addc_co_u32_e64 v39, s[0:1], v39, v15, s[0:1]
	v_add_co_u32_e64 v38, s[0:1], v38, v4
	v_addc_co_u32_e64 v39, s[0:1], v39, v5, s[0:1]
	v_fma_f64 v[36:37], -v[26:27], v[28:29], 1.0
	global_load_dwordx2 v[38:39], v[38:39], off
	v_fma_f64 v[28:29], v[28:29], v[36:37], v[28:29]
	v_fma_f64 v[40:41], -v[22:23], v[30:31], 1.0
	v_rcp_f64_e32 v[36:37], v[32:33]
	v_fma_f64 v[30:31], v[30:31], v[40:41], v[30:31]
	v_mul_f64 v[40:41], v[34:35], v[28:29]
	v_fma_f64 v[26:27], -v[26:27], v[40:41], v[34:35]
	v_fma_f64 v[34:35], -v[32:33], v[36:37], 1.0
	v_div_fmas_f64 v[26:27], v[26:27], v[28:29], v[40:41]
	v_fma_f64 v[34:35], v[36:37], v[34:35], v[36:37]
	v_fma_f64 v[36:37], -v[22:23], v[30:31], 1.0
	v_div_scale_f64 v[28:29], vcc, v[20:21], v[8:9], v[20:21]
	v_fma_f64 v[40:41], -v[32:33], v[34:35], 1.0
	v_fma_f64 v[30:31], v[30:31], v[36:37], v[30:31]
	v_div_scale_f64 v[36:37], s[0:1], v[12:13], v[10:11], v[12:13]
	v_div_fixup_f64 v[6:7], v[26:27], v[6:7], v[24:25]
	v_fma_f64 v[34:35], v[34:35], v[40:41], v[34:35]
	v_mul_f64 v[40:41], v[28:29], v[30:31]
	v_fma_f64 v[22:23], -v[22:23], v[40:41], v[28:29]
	v_mul_f64 v[28:29], v[36:37], v[34:35]
	v_div_fmas_f64 v[22:23], v[22:23], v[30:31], v[40:41]
	v_fma_f64 v[32:33], -v[32:33], v[28:29], v[36:37]
	s_mov_b64 vcc, s[0:1]
	v_fma_f64 v[30:31], -v[0:1], v[0:1], 1.0
	v_div_fmas_f64 v[28:29], v[32:33], v[34:35], v[28:29]
	s_waitcnt vmcnt(0)
	v_mul_f64 v[32:33], v[38:39], s[18:19]
	v_div_fixup_f64 v[8:9], v[22:23], v[8:9], v[20:21]
	v_mul_f64 v[20:21], v[0:1], v[30:31]
	v_mul_f64 v[22:23], v[30:31], v[32:33]
	v_add_f64 v[6:7], v[6:7], v[8:9]
	v_div_fixup_f64 v[10:11], v[28:29], v[10:11], v[12:13]
	v_fma_f64 v[8:9], v[30:31], v[22:23], -v[20:21]
	v_add_f64 v[6:7], v[6:7], v[10:11]
	v_add_f64 v[6:7], v[6:7], -v[8:9]
	v_mov_b32_e32 v8, s9
	v_add_co_u32_e32 v9, vcc, s8, v16
	v_addc_co_u32_e32 v8, vcc, v8, v17, vcc
	v_fma_f64 v[0:1], v[2:3], v[6:7], v[0:1]
	v_add_co_u32_e32 v2, vcc, v9, v14
	v_addc_co_u32_e32 v3, vcc, v8, v15, vcc
	v_add_co_u32_e32 v2, vcc, v2, v4
	v_addc_co_u32_e32 v3, vcc, v3, v5, vcc
	global_store_dwordx2 v[2:3], v[0:1], off
.LBB1_10:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 368
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 45
		.amdhsa_next_free_sgpr 28
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
.Lfunc_end1:
	.size	_Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd, .Lfunc_end1-_Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2124
; NumSgprs: 32
; NumVgprs: 45
; NumAgprs: 0
; TotalNumVgprs: 45
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 11
; NumSGPRsForWavesPerEU: 32
; NumVGPRsForWavesPerEU: 45
; Occupancy: 5
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.text
	.protected	_Z21boundaryConditionsPhiPA400_A400_d ; -- Begin function _Z21boundaryConditionsPhiPA400_A400_d
	.globl	_Z21boundaryConditionsPhiPA400_A400_d
	.p2align	8
	.type	_Z21boundaryConditionsPhiPA400_A400_d,@function
_Z21boundaryConditionsPhiPA400_A400_d:  ; @_Z21boundaryConditionsPhiPA400_A400_d
; %bb.0:
	s_load_dwordx2 s[2:3], s[4:5], 0x14
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	s_lshr_b32 s4, s2, 16
	s_and_b32 s2, s2, 0xffff
	s_mul_i32 s8, s8, s3
	s_mul_i32 s6, s6, s2
	s_mul_i32 s7, s7, s4
	v_add_u32_e32 v4, s8, v2
	s_movk_i32 s2, 0x18e
	v_add_u32_e32 v0, s6, v0
	v_add_u32_e32 v3, s7, v1
	v_cmp_lt_i32_e32 vcc, s2, v4
	s_mov_b64 s[2:3], 0
	s_mov_b64 s[8:9], 0
	s_mov_b64 s[4:5], 0
                                        ; implicit-def: $vgpr1_vgpr2
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execnz .LBB2_6
; %bb.1:                                ; %Flow
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execnz .LBB2_9
.LBB2_2:                                ; %Flow55
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[8:9]
	s_cbranch_execnz .LBB2_10
.LBB2_3:                                ; %Flow56
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execnz .LBB2_16
.LBB2_4:                                ; %Flow66
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execnz .LBB2_17
.LBB2_5:
	s_endpgm
.LBB2_6:                                ; %LeafBlock40
	s_movk_i32 s4, 0x18f
	v_cmp_eq_u32_e32 vcc, s4, v4
	s_mov_b64 s[10:11], -1
	s_mov_b64 s[4:5], 0
                                        ; implicit-def: $vgpr1_vgpr2
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB2_8
; %bb.7:
	s_add_u32 s10, s0, 0x1e70f800
	s_addc_u32 s11, s1, 0
	v_mov_b32_e32 v1, s10
	s_movk_i32 s12, 0xc80
	v_mov_b32_e32 v2, s11
	v_mad_u64_u32 v[5:6], s[10:11], v3, s12, v[1:2]
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[1:2], 3, v[0:1]
	s_mov_b64 s[4:5], exec
	v_add_co_u32_e32 v1, vcc, v5, v1
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	s_xor_b64 s[10:11], exec, -1
.LBB2_8:                                ; %Flow54
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[4:5], s[4:5], exec
	s_and_b64 s[8:9], s[10:11], exec
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execz .LBB2_2
.LBB2_9:                                ; %LeafBlock
	v_cmp_ne_u32_e32 vcc, 0, v4
	s_andn2_b64 s[8:9], s[8:9], exec
	s_and_b64 s[10:11], vcc, exec
	s_mov_b64 s[2:3], exec
	s_or_b64 s[8:9], s[8:9], s[10:11]
                                        ; implicit-def: $vgpr1_vgpr2
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[8:9]
	s_cbranch_execz .LBB2_3
.LBB2_10:                               ; %NodeBlock46
	s_movk_i32 s8, 0x18e
	v_cmp_lt_i32_e32 vcc, s8, v3
	s_mov_b64 s[8:9], 0
	s_mov_b64 s[14:15], 0
	s_mov_b64 s[10:11], s[4:5]
                                        ; implicit-def: $vgpr1_vgpr2
	s_and_saveexec_b64 s[12:13], vcc
	s_xor_b64 s[12:13], exec, s[12:13]
	s_cbranch_execnz .LBB2_18
; %bb.11:                               ; %Flow57
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execnz .LBB2_21
.LBB2_12:                               ; %Flow59
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execnz .LBB2_22
.LBB2_13:                               ; %Flow60
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[12:13], s[8:9]
	s_xor_b64 s[8:9], exec, s[12:13]
.LBB2_14:
	v_mov_b32_e32 v2, s1
	s_mov_b32 s12, 0x138800
	v_mov_b32_e32 v1, s0
	v_mad_u64_u32 v[4:5], s[12:13], v4, s12, v[1:2]
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[1:2], 3, v[0:1]
	s_or_b64 s[10:11], s[10:11], exec
	v_add_co_u32_e32 v1, vcc, v4, v1
	v_addc_co_u32_e32 v2, vcc, v5, v2, vcc
.LBB2_15:                               ; %Flow65
	s_or_b64 exec, exec, s[8:9]
	s_andn2_b64 s[4:5], s[4:5], exec
	s_and_b64 s[8:9], s[10:11], exec
	s_or_b64 s[4:5], s[4:5], s[8:9]
	s_andn2_b64 s[2:3], s[2:3], exec
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[2:3]
	s_cbranch_execz .LBB2_4
.LBB2_16:
	v_mov_b32_e32 v2, s1
	s_movk_i32 s2, 0xc80
	v_mov_b32_e32 v1, s0
	v_mad_u64_u32 v[2:3], s[0:1], v3, s2, v[1:2]
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[4:5], 3, v[0:1]
	s_or_b64 s[4:5], s[4:5], exec
	v_add_co_u32_e32 v1, vcc, v2, v4
	v_addc_co_u32_e32 v2, vcc, v3, v5, vcc
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[0:1], s[4:5]
	s_cbranch_execz .LBB2_5
.LBB2_17:                               ; %.sink.split
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0xbff00000
	global_store_dwordx2 v[1:2], v[3:4], off
	s_endpgm
.LBB2_18:                               ; %LeafBlock44
	s_movk_i32 s10, 0x18f
	v_cmp_eq_u32_e32 vcc, s10, v3
	s_mov_b64 s[14:15], -1
	s_mov_b64 s[16:17], s[4:5]
                                        ; implicit-def: $vgpr1_vgpr2
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB2_20
; %bb.19:
	v_mov_b32_e32 v2, s1
	s_mov_b32 s14, 0x138800
	v_mov_b32_e32 v1, s0
	v_mad_u64_u32 v[5:6], s[14:15], v4, s14, v[1:2]
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[1:2], 3, v[0:1]
	s_or_b64 s[16:17], s[4:5], exec
	v_add_co_u32_e32 v1, vcc, v5, v1
	v_addc_co_u32_e32 v2, vcc, v6, v2, vcc
	v_add_co_u32_e32 v1, vcc, 0x137b80, v1
	v_addc_co_u32_e32 v2, vcc, 0, v2, vcc
	s_xor_b64 s[14:15], exec, -1
.LBB2_20:                               ; %Flow58
	s_or_b64 exec, exec, s[10:11]
	s_andn2_b64 s[10:11], s[4:5], exec
	s_and_b64 s[16:17], s[16:17], exec
	s_or_b64 s[10:11], s[10:11], s[16:17]
	s_and_b64 s[14:15], s[14:15], exec
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB2_12
.LBB2_21:                               ; %LeafBlock42
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[16:17], vcc, exec
	s_mov_b64 s[8:9], exec
	s_or_b64 s[14:15], s[14:15], s[16:17]
                                        ; implicit-def: $vgpr1_vgpr2
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[12:13], s[14:15]
	s_cbranch_execz .LBB2_13
.LBB2_22:                               ; %NodeBlock52
	s_movk_i32 s14, 0x18e
	v_cmp_lt_i32_e32 vcc, s14, v0
	s_mov_b64 s[14:15], s[10:11]
                                        ; implicit-def: $vgpr1_vgpr2
	s_and_saveexec_b64 s[16:17], vcc
	s_xor_b64 s[16:17], exec, s[16:17]
	s_cbranch_execz .LBB2_26
; %bb.23:                               ; %LeafBlock50
	s_movk_i32 s14, 0x18f
	v_cmp_eq_u32_e32 vcc, s14, v0
	s_mov_b64 s[18:19], s[10:11]
                                        ; implicit-def: $vgpr1_vgpr2
	s_and_saveexec_b64 s[14:15], vcc
; %bb.24:
	v_mov_b32_e32 v2, s1
	s_mov_b32 s18, 0x138800
	v_mov_b32_e32 v1, s0
	v_mad_u64_u32 v[1:2], s[18:19], v4, s18, v[1:2]
	s_movk_i32 s18, 0xc80
	v_mad_u64_u32 v[1:2], s[18:19], v3, s18, v[1:2]
	s_or_b64 s[18:19], s[10:11], exec
	v_add_co_u32_e32 v1, vcc, 0xc78, v1
	v_addc_co_u32_e32 v2, vcc, 0, v2, vcc
; %bb.25:                               ; %Flow62
	s_or_b64 exec, exec, s[14:15]
	s_andn2_b64 s[14:15], s[10:11], exec
	s_and_b64 s[18:19], s[18:19], exec
	s_or_b64 s[14:15], s[14:15], s[18:19]
.LBB2_26:                               ; %Flow61
	s_andn2_saveexec_b64 s[16:17], s[16:17]
	s_cbranch_execz .LBB2_30
; %bb.27:                               ; %LeafBlock48
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_mov_b64 s[18:19], s[14:15]
                                        ; implicit-def: $vgpr1_vgpr2
	s_and_saveexec_b64 s[20:21], vcc
	s_xor_b64 s[20:21], exec, s[20:21]
; %bb.28:
	v_mov_b32_e32 v2, s1
	s_mov_b32 s18, 0x138800
	v_mov_b32_e32 v1, s0
	v_mad_u64_u32 v[1:2], s[18:19], v4, s18, v[1:2]
	s_movk_i32 s18, 0xc80
	v_mad_u64_u32 v[1:2], s[18:19], v3, s18, v[1:2]
	s_or_b64 s[18:19], s[14:15], exec
; %bb.29:                               ; %Flow64
	s_or_b64 exec, exec, s[20:21]
	s_andn2_b64 s[14:15], s[14:15], exec
	s_and_b64 s[18:19], s[18:19], exec
	s_or_b64 s[14:15], s[14:15], s[18:19]
.LBB2_30:                               ; %Flow63
	s_or_b64 exec, exec, s[16:17]
	s_andn2_b64 s[10:11], s[10:11], exec
	s_and_b64 s[14:15], s[14:15], exec
	s_or_b64 s[10:11], s[10:11], s[14:15]
	s_andn2_b64 s[8:9], s[8:9], exec
	s_or_b64 exec, exec, s[12:13]
	s_and_saveexec_b64 s[12:13], s[8:9]
	s_xor_b64 s[8:9], exec, s[12:13]
	s_cbranch_execnz .LBB2_14
	s_branch .LBB2_15
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21boundaryConditionsPhiPA400_A400_d
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 264
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 22
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
.Lfunc_end2:
	.size	_Z21boundaryConditionsPhiPA400_A400_d, .Lfunc_end2-_Z21boundaryConditionsPhiPA400_A400_d
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 880
; NumSgprs: 26
; NumVgprs: 7
; NumAgprs: 0
; TotalNumVgprs: 7
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 26
; NumVGPRsForWavesPerEU: 7
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.text
	.protected	_Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd ; -- Begin function _Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd
	.globl	_Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd
	.p2align	8
	.type	_Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd,@function
_Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd: ; @_Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd
; %bb.0:
	s_load_dwordx2 s[0:1], s[4:5], 0x54
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_lshr_b32 s2, s0, 16
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	s_mul_i32 s7, s7, s2
	s_mul_i32 s8, s8, s1
	v_add_u32_e32 v6, s6, v0
	v_add_u32_e32 v12, s7, v1
	v_add_u32_e32 v0, s8, v2
	v_max3_u32 v1, v6, v12, v0
	s_movk_i32 s0, 0x18f
	v_cmp_gt_u32_e32 vcc, s0, v1
	v_cmp_ne_u32_e64 s[0:1], 0, v0
	v_cmp_ne_u32_e64 s[2:3], 0, v12
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_and_b64 s[0:1], s[0:1], vcc
	v_cmp_ne_u32_e32 vcc, 0, v6
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB3_2
; %bb.1:
	s_load_dwordx16 s[8:23], s[4:5], 0x0
	s_mov_b32 s0, 0x138800
	v_mad_u64_u32 v[2:3], s[0:1], v0, s0, 0
	s_movk_i32 s2, 0xc80
	v_mad_u64_u32 v[0:1], s[0:1], v12, s2, 0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, s11
	v_add_co_u32_e32 v10, vcc, s10, v2
	v_mov_b32_e32 v7, 0
	v_addc_co_u32_e32 v11, vcc, v4, v3, vcc
	v_add_co_u32_e32 v13, vcc, v10, v0
	v_lshlrev_b64 v[4:5], 3, v[6:7]
	v_addc_co_u32_e32 v14, vcc, v11, v1, vcc
	v_add_co_u32_e32 v8, vcc, v13, v4
	v_addc_co_u32_e32 v9, vcc, v14, v5, vcc
	s_mov_b32 s0, 0x138000
	v_add_co_u32_e32 v15, vcc, s0, v8
	v_addc_co_u32_e32 v16, vcc, 0, v9, vcc
	s_mov_b32 s0, 0xffec8000
	v_add_co_u32_e32 v17, vcc, s0, v8
	v_addc_co_u32_e32 v18, vcc, -1, v9, vcc
	global_load_dwordx2 v[19:20], v[15:16], off offset:2048
	global_load_dwordx2 v[21:22], v[17:18], off offset:-2048
	v_add_u32_e32 v12, -1, v12
	v_mad_u64_u32 v[10:11], s[0:1], v12, s2, v[10:11]
	v_add_u32_e32 v6, -1, v6
	v_lshlrev_b64 v[6:7], 3, v[6:7]
	v_add_co_u32_e32 v10, vcc, v10, v4
	v_addc_co_u32_e32 v11, vcc, v11, v5, vcc
	global_load_dwordx2 v[23:24], v[8:9], off offset:3200
	global_load_dwordx2 v[25:26], v[10:11], off
	global_load_dwordx4 v[15:18], v[8:9], off
	v_add_co_u32_e32 v6, vcc, v13, v6
	v_addc_co_u32_e32 v7, vcc, v14, v7, vcc
	global_load_dwordx2 v[6:7], v[6:7], off
	v_mul_f64 v[12:13], s[20:21], s[20:21]
	s_load_dwordx2 s[2:3], s[4:5], 0x40
	v_mov_b32_e32 v14, s13
	s_waitcnt vmcnt(4)
	v_add_f64 v[8:9], v[19:20], v[21:22]
	v_mul_f64 v[19:20], s[22:23], s[22:23]
	s_waitcnt vmcnt(2)
	v_add_f64 v[10:11], v[23:24], v[25:26]
	s_waitcnt vmcnt(1)
	v_fma_f64 v[8:9], v[15:16], -2.0, v[8:9]
	s_waitcnt vmcnt(0)
	v_add_f64 v[6:7], v[17:18], v[6:7]
	v_fma_f64 v[10:11], v[15:16], -2.0, v[10:11]
	v_div_scale_f64 v[21:22], s[0:1], v[12:13], v[12:13], v[8:9]
	v_fma_f64 v[6:7], v[15:16], -2.0, v[6:7]
	v_div_scale_f64 v[23:24], s[0:1], v[19:20], v[19:20], v[10:11]
	v_rcp_f64_e32 v[25:26], v[21:22]
	v_rcp_f64_e32 v[27:28], v[23:24]
	v_fma_f64 v[29:30], -v[21:22], v[25:26], 1.0
	v_fma_f64 v[31:32], -v[23:24], v[27:28], 1.0
	v_fma_f64 v[25:26], v[25:26], v[29:30], v[25:26]
	v_fma_f64 v[27:28], v[27:28], v[31:32], v[27:28]
	v_fma_f64 v[29:30], -v[21:22], v[25:26], 1.0
	v_div_scale_f64 v[31:32], vcc, v[8:9], v[12:13], v[8:9]
	v_fma_f64 v[25:26], v[25:26], v[29:30], v[25:26]
	v_fma_f64 v[29:30], -v[23:24], v[27:28], 1.0
	v_fma_f64 v[27:28], v[27:28], v[29:30], v[27:28]
	v_mul_f64 v[29:30], v[31:32], v[25:26]
	v_fma_f64 v[21:22], -v[21:22], v[29:30], v[31:32]
	v_div_scale_f64 v[31:32], s[0:1], v[10:11], v[19:20], v[10:11]
	v_div_fmas_f64 v[17:18], v[21:22], v[25:26], v[29:30]
	s_waitcnt lgkmcnt(0)
	v_mul_f64 v[25:26], s[2:3], s[2:3]
	s_mov_b64 vcc, s[0:1]
	v_mul_f64 v[21:22], v[31:32], v[27:28]
	v_div_scale_f64 v[29:30], s[2:3], v[25:26], v[25:26], v[6:7]
	v_div_fixup_f64 v[8:9], v[17:18], v[12:13], v[8:9]
	v_fma_f64 v[23:24], -v[23:24], v[21:22], v[31:32]
	v_div_fmas_f64 v[21:22], v[23:24], v[27:28], v[21:22]
	v_rcp_f64_e32 v[23:24], v[29:30]
	v_div_fixup_f64 v[10:11], v[21:22], v[19:20], v[10:11]
	v_fma_f64 v[27:28], -v[29:30], v[23:24], 1.0
	v_add_f64 v[8:9], v[8:9], v[10:11]
	v_fma_f64 v[23:24], v[23:24], v[27:28], v[23:24]
	v_mov_b32_e32 v10, s18
	v_mov_b32_e32 v11, s19
	v_mul_f64 v[10:11], s[16:17], v[10:11]
	v_fma_f64 v[27:28], -v[29:30], v[23:24], 1.0
	v_fma_f64 v[23:24], v[23:24], v[27:28], v[23:24]
	v_div_scale_f64 v[27:28], vcc, v[6:7], v[25:26], v[6:7]
	v_mul_f64 v[31:32], v[27:28], v[23:24]
	v_fma_f64 v[27:28], -v[29:30], v[31:32], v[27:28]
	v_mov_b32_e32 v29, s15
	s_nop 0
	v_div_fmas_f64 v[23:24], v[27:28], v[23:24], v[31:32]
	v_add_co_u32_e32 v27, vcc, s12, v2
	v_addc_co_u32_e32 v14, vcc, v14, v3, vcc
	v_add_co_u32_e32 v27, vcc, v27, v0
	v_addc_co_u32_e32 v14, vcc, v14, v1, vcc
	v_add_co_u32_e32 v27, vcc, v27, v4
	v_addc_co_u32_e32 v28, vcc, v14, v5, vcc
	v_add_co_u32_e32 v14, vcc, s14, v2
	v_addc_co_u32_e32 v29, vcc, v29, v3, vcc
	v_add_co_u32_e32 v14, vcc, v14, v0
	v_addc_co_u32_e32 v30, vcc, v29, v1, vcc
	v_add_co_u32_e32 v29, vcc, v14, v4
	v_addc_co_u32_e32 v30, vcc, v30, v5, vcc
	global_load_dwordx2 v[27:28], v[27:28], off
	s_nop 0
	global_load_dwordx2 v[29:30], v[29:30], off
	v_div_fixup_f64 v[6:7], v[23:24], v[25:26], v[6:7]
	v_add_f64 v[6:7], v[8:9], v[6:7]
	v_mov_b32_e32 v8, s9
	v_add_co_u32_e32 v9, vcc, s8, v2
	v_addc_co_u32_e32 v8, vcc, v8, v3, vcc
	v_add_co_u32_e32 v0, vcc, v9, v0
	v_addc_co_u32_e32 v1, vcc, v8, v1, vcc
	v_add_co_u32_e32 v0, vcc, v0, v4
	v_addc_co_u32_e32 v1, vcc, v1, v5, vcc
	s_waitcnt vmcnt(0)
	v_add_f64 v[12:13], v[27:28], -v[29:30]
	v_fma_f64 v[12:13], v[12:13], 0.5, v[15:16]
	v_fma_f64 v[2:3], v[10:11], v[6:7], v[12:13]
	global_store_dwordx2 v[0:1], v[2:3], off
.LBB3_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 328
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 33
		.amdhsa_next_free_sgpr 24
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
.Lfunc_end3:
	.size	_Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd, .Lfunc_end3-_Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 848
; NumSgprs: 28
; NumVgprs: 33
; NumAgprs: 0
; TotalNumVgprs: 33
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 28
; NumVGPRsForWavesPerEU: 33
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.text
	.protected	_Z19boundaryConditionsUPA400_A400_dd ; -- Begin function _Z19boundaryConditionsUPA400_A400_dd
	.globl	_Z19boundaryConditionsUPA400_A400_dd
	.p2align	8
	.type	_Z19boundaryConditionsUPA400_A400_dd,@function
_Z19boundaryConditionsUPA400_A400_dd:   ; @_Z19boundaryConditionsUPA400_A400_dd
; %bb.0:
	s_load_dwordx2 s[10:11], s[4:5], 0x1c
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s4, s11, 0xffff
	s_lshr_b32 s5, s10, 16
	s_and_b32 s9, s10, 0xffff
	s_mul_i32 s8, s8, s4
	s_mul_i32 s6, s6, s9
	s_mul_i32 s7, s7, s5
	v_add_u32_e32 v2, s8, v2
	s_movk_i32 s4, 0x18e
	v_add_u32_e32 v0, s6, v0
	v_add_u32_e32 v3, s7, v1
	v_cmp_lt_i32_e32 vcc, s4, v2
	s_mov_b64 s[4:5], 0
	s_mov_b64 s[8:9], 0
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execnz .LBB4_5
; %bb.1:                                ; %Flow72
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execnz .LBB4_8
.LBB4_2:                                ; %Flow74
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[8:9]
	s_cbranch_execnz .LBB4_9
.LBB4_3:                                ; %Flow75
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execnz .LBB4_15
.LBB4_4:
	s_endpgm
.LBB4_5:                                ; %LeafBlock50
	s_movk_i32 s8, 0x18f
	v_cmp_eq_u32_e32 vcc, s8, v2
	s_mov_b64 s[10:11], -1
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB4_7
; %bb.6:
	s_xor_b32 s10, s3, 0x80000000
	v_mov_b32_e32 v5, s10
	s_add_u32 s10, s0, 0x1e70f800
	s_addc_u32 s11, s1, 0
	v_mov_b32_e32 v6, s10
	s_movk_i32 s12, 0xc80
	v_mov_b32_e32 v7, s11
	v_mad_u64_u32 v[6:7], s[10:11], v3, s12, v[6:7]
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[8:9], 3, v[0:1]
	v_mov_b32_e32 v4, s2
	v_add_co_u32_e32 v6, vcc, v6, v8
	v_addc_co_u32_e32 v7, vcc, v7, v9, vcc
	global_store_dwordx2 v[6:7], v[4:5], off
	s_xor_b64 s[10:11], exec, -1
.LBB4_7:                                ; %Flow73
	s_or_b64 exec, exec, s[8:9]
	s_and_b64 s[8:9], s[10:11], exec
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execz .LBB4_2
.LBB4_8:                                ; %LeafBlock
	v_cmp_ne_u32_e32 vcc, 0, v2
	s_andn2_b64 s[8:9], s[8:9], exec
	s_and_b64 s[10:11], vcc, exec
	s_mov_b64 s[4:5], exec
	s_or_b64 s[8:9], s[8:9], s[10:11]
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[8:9]
	s_cbranch_execz .LBB4_3
.LBB4_9:                                ; %NodeBlock56
	s_movk_i32 s8, 0x18e
	v_cmp_lt_i32_e32 vcc, s8, v3
	s_mov_b64 s[8:9], 0
	s_mov_b64 s[12:13], 0
	s_and_saveexec_b64 s[10:11], vcc
	s_xor_b64 s[10:11], exec, s[10:11]
	s_cbranch_execnz .LBB4_16
; %bb.10:                               ; %Flow67
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execnz .LBB4_19
.LBB4_11:                               ; %Flow69
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[10:11], s[12:13]
	s_cbranch_execnz .LBB4_20
.LBB4_12:                               ; %Flow70
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execz .LBB4_14
.LBB4_13:
	s_xor_b32 s8, s3, 0x80000000
	v_mov_b32_e32 v7, s1
	v_mov_b32_e32 v5, s8
	s_mov_b32 s8, 0x138800
	v_mov_b32_e32 v6, s0
	v_mad_u64_u32 v[6:7], s[8:9], v2, s8, v[6:7]
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[1:2], 3, v[0:1]
	v_mov_b32_e32 v4, s2
	v_add_co_u32_e32 v1, vcc, v6, v1
	v_addc_co_u32_e32 v2, vcc, v7, v2, vcc
	global_store_dwordx2 v[1:2], v[4:5], off
.LBB4_14:                               ; %Flow71
	s_or_b64 exec, exec, s[10:11]
	s_andn2_b64 s[4:5], s[4:5], exec
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB4_4
.LBB4_15:
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v4, s2
	s_movk_i32 s2, 0xc80
	v_mov_b32_e32 v1, s0
	v_mad_u64_u32 v[2:3], s[0:1], v3, s2, v[1:2]
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	s_xor_b32 s3, s3, 0x80000000
	v_add_co_u32_e32 v0, vcc, v2, v0
	v_mov_b32_e32 v5, s3
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc
	global_store_dwordx2 v[0:1], v[4:5], off
	s_endpgm
.LBB4_16:                               ; %LeafBlock54
	s_movk_i32 s12, 0x18f
	v_cmp_eq_u32_e32 vcc, s12, v3
	s_mov_b64 s[14:15], -1
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB4_18
; %bb.17:
	s_xor_b32 s14, s3, 0x80000000
	v_mov_b32_e32 v7, s1
	v_mov_b32_e32 v5, s14
	s_mov_b32 s14, 0x138800
	v_mov_b32_e32 v6, s0
	v_mad_u64_u32 v[6:7], s[14:15], v2, s14, v[6:7]
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[8:9], 3, v[0:1]
	v_mov_b32_e32 v4, s2
	v_add_co_u32_e32 v1, vcc, v6, v8
	v_addc_co_u32_e32 v7, vcc, v7, v9, vcc
	v_add_co_u32_e32 v6, vcc, 0x137000, v1
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	global_store_dwordx2 v[6:7], v[4:5], off offset:2944
	s_xor_b64 s[14:15], exec, -1
.LBB4_18:                               ; %Flow68
	s_or_b64 exec, exec, s[12:13]
	s_and_b64 s[12:13], s[14:15], exec
	s_andn2_saveexec_b64 s[10:11], s[10:11]
	s_cbranch_execz .LBB4_11
.LBB4_19:                               ; %LeafBlock52
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_andn2_b64 s[12:13], s[12:13], exec
	s_and_b64 s[14:15], vcc, exec
	s_mov_b64 s[8:9], exec
	s_or_b64 s[12:13], s[12:13], s[14:15]
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[10:11], s[12:13]
	s_cbranch_execz .LBB4_12
.LBB4_20:                               ; %NodeBlock62
	s_movk_i32 s12, 0x18e
	v_cmp_lt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[12:13], vcc
	s_xor_b64 s[12:13], exec, s[12:13]
	s_cbranch_execz .LBB4_24
; %bb.21:                               ; %LeafBlock60
	s_movk_i32 s14, 0x18f
	v_cmp_eq_u32_e32 vcc, s14, v0
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB4_23
; %bb.22:
	v_mov_b32_e32 v5, s1
	s_mov_b32 s16, 0x138800
	v_mov_b32_e32 v4, s0
	v_mad_u64_u32 v[4:5], s[16:17], v2, s16, v[4:5]
	s_movk_i32 s16, 0xc80
	v_mov_b32_e32 v6, s2
	v_mad_u64_u32 v[4:5], s[16:17], v3, s16, v[4:5]
	s_xor_b32 s16, s3, 0x80000000
	v_mov_b32_e32 v7, s16
	global_store_dwordx2 v[4:5], v[6:7], off offset:3192
.LBB4_23:                               ; %Flow
	s_or_b64 exec, exec, s[14:15]
.LBB4_24:                               ; %Flow65
	s_andn2_saveexec_b64 s[12:13], s[12:13]
	s_cbranch_execz .LBB4_28
; %bb.25:                               ; %LeafBlock58
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB4_27
; %bb.26:
	v_mov_b32_e32 v5, s1
	s_mov_b32 s16, 0x138800
	v_mov_b32_e32 v4, s0
	v_mad_u64_u32 v[4:5], s[16:17], v2, s16, v[4:5]
	s_movk_i32 s16, 0xc80
	v_mov_b32_e32 v6, s2
	v_mad_u64_u32 v[4:5], s[16:17], v3, s16, v[4:5]
	s_xor_b32 s16, s3, 0x80000000
	v_mov_b32_e32 v7, s16
	global_store_dwordx2 v[4:5], v[6:7], off
.LBB4_27:                               ; %Flow64
	s_or_b64 exec, exec, s[14:15]
.LBB4_28:                               ; %Flow66
	s_or_b64 exec, exec, s[12:13]
	s_andn2_b64 s[8:9], s[8:9], exec
	s_or_b64 exec, exec, s[10:11]
	s_and_saveexec_b64 s[10:11], s[8:9]
	s_cbranch_execnz .LBB4_13
	s_branch .LBB4_14
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z19boundaryConditionsUPA400_A400_dd
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 18
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
.Lfunc_end4:
	.size	_Z19boundaryConditionsUPA400_A400_dd, .Lfunc_end4-_Z19boundaryConditionsUPA400_A400_dd
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 852
; NumSgprs: 22
; NumVgprs: 10
; NumAgprs: 0
; TotalNumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 10
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.text
	.protected	_Z8swapGridPA400_A400_dS1_ ; -- Begin function _Z8swapGridPA400_A400_dS1_
	.globl	_Z8swapGridPA400_A400_dS1_
	.p2align	8
	.type	_Z8swapGridPA400_A400_dS1_,@function
_Z8swapGridPA400_A400_dS1_:             ; @_Z8swapGridPA400_A400_dS1_
; %bb.0:
	s_load_dwordx2 s[0:1], s[4:5], 0x1c
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_lshr_b32 s2, s0, 16
	s_and_b32 s0, s0, 0xffff
	s_mul_i32 s6, s6, s0
	s_mul_i32 s7, s7, s2
	s_mul_i32 s8, s8, s1
	v_add_u32_e32 v0, s6, v0
	v_add_u32_e32 v3, s7, v1
	v_add_u32_e32 v2, s8, v2
	v_max3_u32 v1, v0, v3, v2
	s_movk_i32 s0, 0x190
	v_cmp_gt_u32_e32 vcc, s0, v1
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB5_2
; %bb.1:
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[0:1], 3, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v6, s2
	v_mov_b32_e32 v7, s3
	s_mov_b32 s2, 0x138800
	v_mov_b32_e32 v4, s0
	v_mov_b32_e32 v5, s1
	v_mad_u64_u32 v[6:7], s[0:1], v2, s2, v[6:7]
	v_mad_u64_u32 v[4:5], s[0:1], v2, s2, v[4:5]
	s_movk_i32 s3, 0xc80
	v_mad_u64_u32 v[6:7], s[0:1], v3, s3, v[6:7]
	v_mad_u64_u32 v[2:3], s[0:1], v3, s3, v[4:5]
	v_add_co_u32_e32 v4, vcc, v6, v0
	v_addc_co_u32_e32 v5, vcc, v7, v1, vcc
	v_add_co_u32_e32 v0, vcc, v2, v0
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc
	global_load_dwordx2 v[2:3], v[4:5], off
	global_load_dwordx2 v[6:7], v[0:1], off
	s_waitcnt vmcnt(1)
	global_store_dwordx2 v[0:1], v[2:3], off
	s_waitcnt vmcnt(1)
	global_store_dwordx2 v[4:5], v[6:7], off
.LBB5_2:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8swapGridPA400_A400_dS1_
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 9
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
.Lfunc_end5:
	.size	_Z8swapGridPA400_A400_dS1_, .Lfunc_end5-_Z8swapGridPA400_A400_dS1_
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 224
; NumSgprs: 13
; NumVgprs: 8
; NumAgprs: 0
; TotalNumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 13
; NumVGPRsForWavesPerEU: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.type	__hip_cuid_23646d403994057,@object ; @__hip_cuid_23646d403994057
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_23646d403994057
__hip_cuid_23646d403994057:
	.byte	0                               ; 0x0
	.size	__hip_cuid_23646d403994057, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_23646d403994057
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
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           8
        .value_kind:     by_value
      - .offset:         40
        .size:           8
        .value_kind:     by_value
      - .offset:         48
        .size:           8
        .value_kind:     by_value
      - .offset:         56
        .size:           8
        .value_kind:     by_value
      - .offset:         64
        .size:           8
        .value_kind:     by_value
      - .offset:         72
        .size:           8
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         84
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         88
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         92
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         94
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         96
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         98
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         100
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         102
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         136
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         144
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 336
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z14calculateForcePA400_A400_dS1_S1_S1_dddddd
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         _Z14calculateForcePA400_A400_dS1_S1_S1_dddddd.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     42
    .vgpr_spill_count: 0
    .wavefront_size: 64
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
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .offset:         48
        .size:           8
        .value_kind:     by_value
      - .offset:         56
        .size:           8
        .value_kind:     by_value
      - .offset:         64
        .size:           8
        .value_kind:     by_value
      - .offset:         72
        .size:           8
        .value_kind:     by_value
      - .offset:         80
        .size:           8
        .value_kind:     by_value
      - .offset:         88
        .size:           8
        .value_kind:     by_value
      - .offset:         96
        .size:           8
        .value_kind:     by_value
      - .offset:         104
        .size:           8
        .value_kind:     by_value
      - .offset:         112
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         116
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         120
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         124
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         126
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         128
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         130
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         132
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         134
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         152
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         160
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         168
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         176
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 368
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         _Z9allenCahnPA400_A400_dS1_S1_S1_S1_S1_dddddddd.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     45
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z21boundaryConditionsPhiPA400_A400_d
    .private_segment_fixed_size: 0
    .sgpr_count:     26
    .sgpr_spill_count: 0
    .symbol:         _Z21boundaryConditionsPhiPA400_A400_d.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
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
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           8
        .value_kind:     by_value
      - .offset:         40
        .size:           8
        .value_kind:     by_value
      - .offset:         48
        .size:           8
        .value_kind:     by_value
      - .offset:         56
        .size:           8
        .value_kind:     by_value
      - .offset:         64
        .size:           8
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         76
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         80
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         84
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         86
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         88
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         90
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         92
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         94
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         112
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         120
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         136
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 328
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd
    .private_segment_fixed_size: 0
    .sgpr_count:     28
    .sgpr_spill_count: 0
    .symbol:         _Z15thermalEquationPA400_A400_dS1_S1_S1_ddddd.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     33
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           8
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
    .name:           _Z19boundaryConditionsUPA400_A400_dd
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z19boundaryConditionsUPA400_A400_dd.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
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
    .name:           _Z8swapGridPA400_A400_dS1_
    .private_segment_fixed_size: 0
    .sgpr_count:     13
    .sgpr_spill_count: 0
    .symbol:         _Z8swapGridPA400_A400_dS1_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
