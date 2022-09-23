
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <_Z29simpleNonSeparableConvolutionPjPfPi15HIP_vector_typeIjLj2EES3_j>:
	s_load_dwordx4 s[0:3], s[6:7], 0x18                        // 000000001000: C00A0003 00000018
	s_waitcnt lgkmcnt(0)                                       // 000000001008: BF8CC07F
	s_load_dword s3, s[4:5], 0x4                               // 00000000100C: C00200C2 00000004
	v_mov_b32_e32 v4, s8                                       // 000000001014: 7E080208
	v_cvt_f32_u32_e32 v1, s0                                   // 000000001018: 7E020C00
	s_sub_i32 s4, 0, s0                                        // 00000000101C: 81840080
	s_waitcnt lgkmcnt(0)                                       // 000000001020: BF8CC07F
	s_and_b32 s3, s3, 0xffff                                   // 000000001024: 8603FF03 0000FFFF
	v_rcp_iflag_f32_e32 v1, v1                                 // 00000000102C: 7E024701
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001030: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v2, v1                                   // 000000001038: 7E040F01
	v_mov_b32_e32 v1, 0                                        // 00000000103C: 7E020280
	v_mul_lo_u32 v3, s4, v2                                    // 000000001040: D2850003 00020404
	v_mad_u64_u32 v[0:1], s[4:5], s3, v4, v[0:1]               // 000000001048: D1E80400 04020803
	v_mul_hi_u32 v3, v2, v3                                    // 000000001050: D2860003 00020702
	v_add_u32_e32 v1, v2, v3                                   // 000000001058: 68020702
	v_mul_hi_u32 v1, v0, v1                                    // 00000000105C: D2860001 00020300
	v_mul_lo_u32 v2, v1, s0                                    // 000000001064: D2850002 00000101
	v_add_u32_e32 v3, 1, v1                                    // 00000000106C: 68060281
	v_sub_u32_e32 v2, v0, v2                                   // 000000001070: 6A040500
	v_cmp_le_u32_e32 vcc, s0, v2                               // 000000001074: 7D960400
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 000000001078: 00020701
	v_subrev_u32_e32 v3, s0, v2                                // 00000000107C: 6C060400
	v_cndmask_b32_e32 v2, v2, v3, vcc                          // 000000001080: 00040702
	v_add_u32_e32 v3, 1, v1                                    // 000000001084: 68060281
	v_cmp_le_u32_e32 vcc, s0, v2                               // 000000001088: 7D960400
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 00000000108C: 00020701
	v_cmp_gt_u32_e32 vcc, s1, v1                               // 000000001090: 7D980201
	s_and_saveexec_b64 s[4:5], vcc                             // 000000001094: BE84206A
	s_cbranch_execz 76                                         // 000000001098: BF88004C <_Z29simpleNonSeparableConvolutionPjPfPi15HIP_vector_typeIjLj2EES3_j+0x1cc>
	s_load_dword s3, s[6:7], 0x24                              // 00000000109C: C00200C3 00000024
	s_load_dwordx4 s[8:11], s[6:7], 0x0                        // 0000000010A4: C00A0203 00000000
	s_load_dwordx2 s[4:5], s[6:7], 0x10                        // 0000000010AC: C0060103 00000010
	v_mov_b32_e32 v3, 0                                        // 0000000010B4: 7E060280
	s_waitcnt lgkmcnt(0)                                       // 0000000010B8: BF8CC07F
	v_add_u32_e32 v2, s3, v1                                   // 0000000010BC: 68040203
	v_cmp_lt_u32_e32 vcc, v1, v2                               // 0000000010C0: 7D920501
	s_and_saveexec_b64 s[12:13], vcc                           // 0000000010C4: BE8C206A
	s_cbranch_execz 53                                         // 0000000010C8: BF880035 <_Z29simpleNonSeparableConvolutionPjPfPi15HIP_vector_typeIjLj2EES3_j+0x1a0>
	s_load_dword s16, s[6:7], 0x28                             // 0000000010CC: C0020403 00000028
	v_mul_lo_u32 v3, v1, s0                                    // 0000000010D4: D2850003 00000101
	s_mov_b32 s7, 0                                            // 0000000010DC: BE870080
	v_mov_b32_e32 v2, 0                                        // 0000000010E0: 7E040280
	s_mov_b32 s17, 0                                           // 0000000010E4: BE910080
	s_waitcnt lgkmcnt(0)                                       // 0000000010E8: BF8CC07F
	s_sub_i32 s0, s16, s0                                      // 0000000010EC: 81800010
	v_mul_lo_u32 v1, v1, s0                                    // 0000000010F0: D2850001 00000101
	v_sub_u32_e32 v3, v0, v3                                   // 0000000010F8: 6A060700
	v_add_u32_e32 v4, s2, v3                                   // 0000000010FC: 68080602
	v_cmp_lt_u32_e32 vcc, v3, v4                               // 000000001100: 7D920903
	v_add_u32_e32 v4, v0, v1                                   // 000000001104: 68080300
	v_mov_b32_e32 v3, 0                                        // 000000001108: 7E060280
	s_mov_b32 s18, 0                                           // 00000000110C: BE920080
	s_branch 6                                                 // 000000001110: BF820006 <_Z29simpleNonSeparableConvolutionPjPfPi15HIP_vector_typeIjLj2EES3_j+0x12c>
	s_or_b64 exec, exec, s[14:15]                              // 000000001114: 87FE0E7E
	s_add_i32 s18, s18, 1                                      // 000000001118: 81128112
	s_add_i32 s17, s17, s2                                     // 00000000111C: 81110211
	s_cmp_eq_u32 s18, s3                                       // 000000001120: BF060312
	v_add_u32_e32 v4, s16, v4                                  // 000000001124: 68080810
	s_cbranch_scc1 29                                          // 000000001128: BF85001D <_Z29simpleNonSeparableConvolutionPjPfPi15HIP_vector_typeIjLj2EES3_j+0x1a0>
	s_and_saveexec_b64 s[14:15], vcc                           // 00000000112C: BE8E206A
	s_cbranch_execz 65528                                      // 000000001130: BF88FFF8 <_Z29simpleNonSeparableConvolutionPjPfPi15HIP_vector_typeIjLj2EES3_j+0x114>
	s_mov_b32 s19, s2                                          // 000000001134: BE930002
	s_mov_b32 s6, s17                                          // 000000001138: BE860011
	v_mov_b32_e32 v1, v4                                       // 00000000113C: 7E020304
	v_lshlrev_b64 v[5:6], 2, v[1:2]                            // 000000001140: D28F0005 00020282
	v_mov_b32_e32 v7, s9                                       // 000000001148: 7E0E0209
	v_add_co_u32_e64 v5, s[0:1], s8, v5                        // 00000000114C: D1190005 00020A08
	v_addc_co_u32_e64 v6, s[0:1], v7, v6, s[0:1]               // 000000001154: D11C0006 00020D07
	global_load_dword v5, v[5:6], off                          // 00000000115C: DC508000 057F0005
	s_lshl_b64 s[0:1], s[6:7], 2                               // 000000001164: 8E808206
	s_add_u32 s0, s10, s0                                      // 000000001168: 8000000A
	s_addc_u32 s1, s11, s1                                     // 00000000116C: 8201010B
	s_load_dword s0, s[0:1], 0x0                               // 000000001170: C0020000 00000000
	s_add_i32 s6, s6, 1                                        // 000000001178: 81068106
	s_add_i32 s19, s19, -1                                     // 00000000117C: 8113C113
	v_add_u32_e32 v1, 1, v1                                    // 000000001180: 68020281
	s_cmp_eq_u32 s19, 0                                        // 000000001184: BF068013
	s_waitcnt vmcnt(0)                                         // 000000001188: BF8C0F70
	v_cvt_f32_u32_e32 v5, v5                                   // 00000000118C: 7E0A0D05
	s_waitcnt lgkmcnt(0)                                       // 000000001190: BF8CC07F
	v_fmac_f32_e32 v3, s0, v5                                  // 000000001194: 76060A00
	s_cbranch_scc0 65513                                       // 000000001198: BF84FFE9 <_Z29simpleNonSeparableConvolutionPjPfPi15HIP_vector_typeIjLj2EES3_j+0x140>
	s_branch 65501                                             // 00000000119C: BF82FFDD <_Z29simpleNonSeparableConvolutionPjPfPi15HIP_vector_typeIjLj2EES3_j+0x114>
	s_or_b64 exec, exec, s[12:13]                              // 0000000011A0: 87FE0C7E
	v_add_f32_e32 v1, 0.5, v3                                  // 0000000011A4: 020206F0
	v_cvt_i32_f32_e32 v2, v1                                   // 0000000011A8: 7E041101
	v_mov_b32_e32 v1, 0                                        // 0000000011AC: 7E020280
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 0000000011B0: D28F0000 00020082
	v_mov_b32_e32 v3, s5                                       // 0000000011B8: 7E060205
	v_add_co_u32_e32 v0, vcc, s4, v0                           // 0000000011BC: 32000004
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 0000000011C0: 38020303
	global_store_dword v[0:1], v2, off                         // 0000000011C4: DC708000 007F0200
	s_endpgm                                                   // 0000000011CC: BF810000
	s_nop 0                                                    // 0000000011D0: BF800000
	s_nop 0                                                    // 0000000011D4: BF800000
	s_nop 0                                                    // 0000000011D8: BF800000
	s_nop 0                                                    // 0000000011DC: BF800000
	s_nop 0                                                    // 0000000011E0: BF800000
	s_nop 0                                                    // 0000000011E4: BF800000
	s_nop 0                                                    // 0000000011E8: BF800000
	s_nop 0                                                    // 0000000011EC: BF800000
	s_nop 0                                                    // 0000000011F0: BF800000
	s_nop 0                                                    // 0000000011F4: BF800000
	s_nop 0                                                    // 0000000011F8: BF800000
	s_nop 0                                                    // 0000000011FC: BF800000

0000000000001200 <_Z31simpleSeparableConvolutionPass1PjPfS0_15HIP_vector_typeIjLj2EEjS2_>:
	s_load_dwordx4 s[0:3], s[6:7], 0x18                        // 000000001200: C00A0003 00000018
	s_waitcnt lgkmcnt(0)                                       // 000000001208: BF8CC07F
	s_load_dword s3, s[4:5], 0x4                               // 00000000120C: C00200C2 00000004
	v_cvt_f32_u32_e32 v1, s0                                   // 000000001214: 7E020C00
	s_sub_i32 s4, 0, s0                                        // 000000001218: 81840080
	s_waitcnt lgkmcnt(0)                                       // 00000000121C: BF8CC07F
	s_and_b32 s3, s3, 0xffff                                   // 000000001220: 8603FF03 0000FFFF
	s_mul_i32 s8, s8, s3                                       // 000000001228: 92080308
	v_rcp_iflag_f32_e32 v1, v1                                 // 00000000122C: 7E024701
	v_add_u32_e32 v0, s8, v0                                   // 000000001230: 68000008
	s_add_i32 s1, s2, s1                                       // 000000001234: 81010102
	s_add_i32 s1, s1, -1                                       // 000000001238: 8101C101
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 00000000123C: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 000000001244: 7E020F01
	v_mul_lo_u32 v2, s4, v1                                    // 000000001248: D2850002 00020204
	v_mul_hi_u32 v2, v1, v2                                    // 000000001250: D2860002 00020501
	v_add_u32_e32 v1, v1, v2                                   // 000000001258: 68020501
	v_mul_hi_u32 v1, v0, v1                                    // 00000000125C: D2860001 00020300
	v_mul_lo_u32 v2, v1, s0                                    // 000000001264: D2850002 00000101
	v_add_u32_e32 v3, 1, v1                                    // 00000000126C: 68060281
	v_sub_u32_e32 v2, v0, v2                                   // 000000001270: 6A040500
	v_cmp_le_u32_e32 vcc, s0, v2                               // 000000001274: 7D960400
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 000000001278: 00020701
	v_subrev_u32_e32 v3, s0, v2                                // 00000000127C: 6C060400
	v_cndmask_b32_e32 v2, v2, v3, vcc                          // 000000001280: 00040702
	v_add_u32_e32 v3, 1, v1                                    // 000000001284: 68060281
	v_cmp_le_u32_e32 vcc, s0, v2                               // 000000001288: 7D960400
	v_cndmask_b32_e32 v2, v1, v3, vcc                          // 00000000128C: 00040701
	v_cmp_gt_u32_e32 vcc, s1, v2                               // 000000001290: 7D980401
	s_and_saveexec_b64 s[4:5], vcc                             // 000000001294: BE84206A
	s_cbranch_execz 55                                         // 000000001298: BF880037 <_Z31simpleSeparableConvolutionPass1PjPfS0_15HIP_vector_typeIjLj2EEjS2_+0x178>
	v_mul_lo_u32 v1, v2, s0                                    // 00000000129C: D2850001 00000102
	s_load_dwordx4 s[8:11], s[6:7], 0x0                        // 0000000012A4: C00A0203 00000000
	s_load_dwordx2 s[4:5], s[6:7], 0x10                        // 0000000012AC: C0060103 00000010
	v_mov_b32_e32 v4, 0                                        // 0000000012B4: 7E080280
	v_sub_u32_e32 v3, v0, v1                                   // 0000000012B8: 6A060300
	v_add_u32_e32 v1, s2, v3                                   // 0000000012BC: 68020602
	v_cmp_lt_u32_e32 vcc, v3, v1                               // 0000000012C0: 7D920303
	s_and_saveexec_b64 s[12:13], vcc                           // 0000000012C4: BE8C206A
	s_cbranch_execz 28                                         // 0000000012C8: BF88001C <_Z31simpleSeparableConvolutionPass1PjPfS0_15HIP_vector_typeIjLj2EEjS2_+0x13c>
	s_load_dword s1, s[6:7], 0x28                              // 0000000012CC: C0020043 00000028
	v_mov_b32_e32 v1, 0                                        // 0000000012D4: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 0000000012D8: BF8CC07F
	s_sub_i32 s0, s1, s0                                       // 0000000012DC: 81800001
	v_mul_lo_u32 v4, v2, s0                                    // 0000000012E0: D2850004 00000102
	v_add_u32_e32 v0, v0, v4                                   // 0000000012E8: 68000900
	v_mov_b32_e32 v4, v1                                       // 0000000012EC: 7E080301
	v_lshlrev_b64 v[5:6], 2, v[0:1]                            // 0000000012F0: D28F0005 00020082
	v_mov_b32_e32 v7, s9                                       // 0000000012F8: 7E0E0209
	v_add_co_u32_e32 v5, vcc, s8, v5                           // 0000000012FC: 320A0A08
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc                     // 000000001300: 380C0D07
	global_load_dword v5, v[5:6], off                          // 000000001304: DC508000 057F0005
	s_load_dword s0, s[10:11], 0x0                             // 00000000130C: C0020005 00000000
	s_add_i32 s2, s2, -1                                       // 000000001314: 8102C102
	s_add_u32 s10, s10, 4                                      // 000000001318: 800A840A
	s_addc_u32 s11, s11, 0                                     // 00000000131C: 820B800B
	v_add_u32_e32 v0, 1, v0                                    // 000000001320: 68000081
	s_cmp_eq_u32 s2, 0                                         // 000000001324: BF068002
	s_waitcnt vmcnt(0)                                         // 000000001328: BF8C0F70
	v_cvt_f32_u32_e32 v5, v5                                   // 00000000132C: 7E0A0D05
	s_waitcnt lgkmcnt(0)                                       // 000000001330: BF8CC07F
	v_fmac_f32_e32 v4, s0, v5                                  // 000000001334: 76080A00
	s_cbranch_scc0 65517                                       // 000000001338: BF84FFED <_Z31simpleSeparableConvolutionPass1PjPfS0_15HIP_vector_typeIjLj2EEjS2_+0xf0>
	s_or_b64 exec, exec, s[12:13]                              // 00000000133C: 87FE0C7E
	s_load_dword s0, s[6:7], 0x2c                              // 000000001340: C0020003 0000002C
	v_mov_b32_e32 v1, 0                                        // 000000001348: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 00000000134C: BF8CC07F
	v_mul_lo_u32 v0, v3, s0                                    // 000000001350: D2850000 00000103
	v_mov_b32_e32 v3, s5                                       // 000000001358: 7E060205
	v_add_u32_e32 v0, v0, v2                                   // 00000000135C: 68000500
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000001360: D28F0000 00020082
	v_add_co_u32_e32 v0, vcc, s4, v0                           // 000000001368: 32000004
	v_addc_co_u32_e32 v1, vcc, v3, v1, vcc                     // 00000000136C: 38020303
	global_store_dword v[0:1], v4, off                         // 000000001370: DC708000 007F0400
	s_endpgm                                                   // 000000001378: BF810000
	s_nop 0                                                    // 00000000137C: BF800000
	s_nop 0                                                    // 000000001380: BF800000
	s_nop 0                                                    // 000000001384: BF800000
	s_nop 0                                                    // 000000001388: BF800000
	s_nop 0                                                    // 00000000138C: BF800000
	s_nop 0                                                    // 000000001390: BF800000
	s_nop 0                                                    // 000000001394: BF800000
	s_nop 0                                                    // 000000001398: BF800000
	s_nop 0                                                    // 00000000139C: BF800000
	s_nop 0                                                    // 0000000013A0: BF800000
	s_nop 0                                                    // 0000000013A4: BF800000
	s_nop 0                                                    // 0000000013A8: BF800000
	s_nop 0                                                    // 0000000013AC: BF800000
	s_nop 0                                                    // 0000000013B0: BF800000
	s_nop 0                                                    // 0000000013B4: BF800000
	s_nop 0                                                    // 0000000013B8: BF800000
	s_nop 0                                                    // 0000000013BC: BF800000
	s_nop 0                                                    // 0000000013C0: BF800000
	s_nop 0                                                    // 0000000013C4: BF800000
	s_nop 0                                                    // 0000000013C8: BF800000
	s_nop 0                                                    // 0000000013CC: BF800000
	s_nop 0                                                    // 0000000013D0: BF800000
	s_nop 0                                                    // 0000000013D4: BF800000
	s_nop 0                                                    // 0000000013D8: BF800000
	s_nop 0                                                    // 0000000013DC: BF800000
	s_nop 0                                                    // 0000000013E0: BF800000
	s_nop 0                                                    // 0000000013E4: BF800000
	s_nop 0                                                    // 0000000013E8: BF800000
	s_nop 0                                                    // 0000000013EC: BF800000
	s_nop 0                                                    // 0000000013F0: BF800000
	s_nop 0                                                    // 0000000013F4: BF800000
	s_nop 0                                                    // 0000000013F8: BF800000
	s_nop 0                                                    // 0000000013FC: BF800000

0000000000001400 <_Z31simpleSeparableConvolutionPass2PfS_Pi15HIP_vector_typeIjLj2EEjS2_>:
	s_load_dwordx4 s[0:3], s[6:7], 0x18                        // 000000001400: C00A0003 00000018
	s_waitcnt lgkmcnt(0)                                       // 000000001408: BF8CC07F
	s_load_dword s3, s[4:5], 0x4                               // 00000000140C: C00200C2 00000004
	v_cvt_f32_u32_e32 v1, s1                                   // 000000001414: 7E020C01
	s_sub_i32 s4, 0, s1                                        // 000000001418: 81840180
	s_waitcnt lgkmcnt(0)                                       // 00000000141C: BF8CC07F
	s_and_b32 s3, s3, 0xffff                                   // 000000001420: 8603FF03 0000FFFF
	s_mul_i32 s8, s8, s3                                       // 000000001428: 92080308
	v_rcp_iflag_f32_e32 v1, v1                                 // 00000000142C: 7E024701
	v_add_u32_e32 v0, s8, v0                                   // 000000001430: 68000008
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001434: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 00000000143C: 7E020F01
	v_mul_lo_u32 v2, s4, v1                                    // 000000001440: D2850002 00020204
	v_mul_hi_u32 v2, v1, v2                                    // 000000001448: D2860002 00020501
	v_add_u32_e32 v1, v1, v2                                   // 000000001450: 68020501
	v_mul_hi_u32 v1, v0, v1                                    // 000000001454: D2860001 00020300
	v_mul_lo_u32 v2, v1, s1                                    // 00000000145C: D2850002 00000301
	v_add_u32_e32 v3, 1, v1                                    // 000000001464: 68060281
	v_sub_u32_e32 v2, v0, v2                                   // 000000001468: 6A040500
	v_cmp_le_u32_e32 vcc, s1, v2                               // 00000000146C: 7D960401
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 000000001470: 00020701
	v_subrev_u32_e32 v3, s1, v2                                // 000000001474: 6C060401
	v_cndmask_b32_e32 v2, v2, v3, vcc                          // 000000001478: 00040702
	v_add_u32_e32 v3, 1, v1                                    // 00000000147C: 68060281
	v_cmp_le_u32_e32 vcc, s1, v2                               // 000000001480: 7D960401
	v_cndmask_b32_e32 v2, v1, v3, vcc                          // 000000001484: 00040701
	v_cmp_gt_u32_e32 vcc, s0, v2                               // 000000001488: 7D980400
	s_and_saveexec_b64 s[4:5], vcc                             // 00000000148C: BE84206A
	s_cbranch_execz 53                                         // 000000001490: BF880035 <_Z31simpleSeparableConvolutionPass2PfS_Pi15HIP_vector_typeIjLj2EEjS2_+0x168>
	v_mul_lo_u32 v1, v2, s1                                    // 000000001494: D2850001 00000302
	s_load_dwordx4 s[8:11], s[6:7], 0x0                        // 00000000149C: C00A0203 00000000
	s_load_dwordx2 s[4:5], s[6:7], 0x10                        // 0000000014A4: C0060103 00000010
	v_sub_u32_e32 v3, v0, v1                                   // 0000000014AC: 6A060300
	v_add_u32_e32 v1, s2, v3                                   // 0000000014B0: 68020602
	v_cmp_lt_u32_e32 vcc, v3, v1                               // 0000000014B4: 7D920303
	v_mov_b32_e32 v1, 0                                        // 0000000014B8: 7E020280
	s_and_saveexec_b64 s[12:13], vcc                           // 0000000014BC: BE8C206A
	s_cbranch_execz 28                                         // 0000000014C0: BF88001C <_Z31simpleSeparableConvolutionPass2PfS_Pi15HIP_vector_typeIjLj2EEjS2_+0x134>
	s_load_dword s3, s[6:7], 0x2c                              // 0000000014C4: C00200C3 0000002C
	v_mov_b32_e32 v1, 0                                        // 0000000014CC: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 0000000014D0: BF8CC07F
	s_sub_i32 s1, s3, s1                                       // 0000000014D4: 81810103
	v_mul_lo_u32 v4, v2, s1                                    // 0000000014D8: D2850004 00000302
	v_add_u32_e32 v0, v0, v4                                   // 0000000014E0: 68000900
	v_mov_b32_e32 v4, v1                                       // 0000000014E4: 7E080301
	v_lshlrev_b64 v[5:6], 2, v[0:1]                            // 0000000014E8: D28F0005 00020082
	v_mov_b32_e32 v7, s9                                       // 0000000014F0: 7E0E0209
	v_add_co_u32_e32 v5, vcc, s8, v5                           // 0000000014F4: 320A0A08
	v_addc_co_u32_e32 v6, vcc, v7, v6, vcc                     // 0000000014F8: 380C0D07
	global_load_dword v5, v[5:6], off                          // 0000000014FC: DC508000 057F0005
	s_load_dword s1, s[10:11], 0x0                             // 000000001504: C0020045 00000000
	s_add_i32 s2, s2, -1                                       // 00000000150C: 8102C102
	s_add_u32 s10, s10, 4                                      // 000000001510: 800A840A
	s_addc_u32 s11, s11, 0                                     // 000000001514: 820B800B
	v_add_u32_e32 v0, 1, v0                                    // 000000001518: 68000081
	s_cmp_eq_u32 s2, 0                                         // 00000000151C: BF068002
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000001520: BF8C0070
	v_fmac_f32_e32 v4, s1, v5                                  // 000000001524: 76080A01
	s_cbranch_scc0 65519                                       // 000000001528: BF84FFEF <_Z31simpleSeparableConvolutionPass2PfS_Pi15HIP_vector_typeIjLj2EEjS2_+0xe8>
	v_add_f32_e32 v0, 0.5, v4                                  // 00000000152C: 020008F0
	v_cvt_i32_f32_e32 v1, v0                                   // 000000001530: 7E021100
	s_or_b64 exec, exec, s[12:13]                              // 000000001534: 87FE0C7E
	v_mul_lo_u32 v0, v3, s0                                    // 000000001538: D2850000 00000103
	v_mov_b32_e32 v3, 0                                        // 000000001540: 7E060280
	s_waitcnt lgkmcnt(0)                                       // 000000001544: BF8CC07F
	v_mov_b32_e32 v4, s5                                       // 000000001548: 7E080205
	v_add_u32_e32 v2, v0, v2                                   // 00000000154C: 68040500
	v_lshlrev_b64 v[2:3], 2, v[2:3]                            // 000000001550: D28F0002 00020482
	v_add_co_u32_e32 v2, vcc, s4, v2                           // 000000001558: 32040404
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc                     // 00000000155C: 38060704
	global_store_dword v[2:3], v1, off                         // 000000001560: DC708000 007F0102
	s_endpgm                                                   // 000000001568: BF810000
