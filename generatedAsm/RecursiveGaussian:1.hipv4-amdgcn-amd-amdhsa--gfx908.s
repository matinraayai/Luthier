
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <_Z16transpose_kernelP15HIP_vector_typeIhLj4EES1_jjj>:
	s_load_dwordx4 s[0:3], s[6:7], 0x0                         // 000000001000: C00A0003 00000000
	s_load_dwordx4 s[12:15], s[6:7], 0x10                      // 000000001008: C00A0303 00000010
	s_load_dword s10, s[4:5], 0x4                              // 000000001010: C0020282 00000004
	v_mov_b32_e32 v3, 0                                        // 000000001018: 7E060280
	s_waitcnt lgkmcnt(0)                                       // 00000000101C: BF8CC07F
	v_mov_b32_e32 v7, s3                                       // 000000001020: 7E0E0203
	v_mul_lo_u32 v6, v1, s14                                   // 000000001024: D2850006 00001D01
	s_lshr_b32 s4, s10, 16                                     // 00000000102C: 8F04900A
	s_mul_i32 s9, s9, s4                                       // 000000001030: 92090409
	v_add_u32_e32 v4, s9, v1                                   // 000000001034: 68080209
	v_mul_lo_u32 v2, v4, s12                                   // 000000001038: D2850002 00001904
	s_and_b32 s4, s10, 0xffff                                  // 000000001040: 8604FF0A 0000FFFF
	s_mul_i32 s8, s8, s4                                       // 000000001048: 92080408
	v_add_u32_e32 v5, s8, v0                                   // 00000000104C: 680A0008
	v_add_u32_e32 v2, v2, v5                                   // 000000001050: 68040B02
	v_lshlrev_b64 v[1:2], 2, v[2:3]                            // 000000001054: D28F0001 00020482
	v_mul_lo_u32 v5, v5, s13                                   // 00000000105C: D2850005 00001B05
	v_add_co_u32_e32 v1, vcc, s2, v1                           // 000000001064: 32020202
	v_addc_co_u32_e32 v2, vcc, v7, v2, vcc                     // 000000001068: 38040507
	global_load_dword v7, v[1:2], off                          // 00000000106C: DC508000 077F0001
	v_add_lshl_u32 v6, v6, v0, 2                               // 000000001074: D1FE0006 020A0106
	v_add_u32_e32 v2, v5, v4                                   // 00000000107C: 68040905
	v_lshlrev_b64 v[0:1], 2, v[2:3]                            // 000000001080: D28F0000 00020482
	v_mov_b32_e32 v8, s1                                       // 000000001088: 7E100201
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 00000000108C: 32000000
	v_addc_co_u32_e32 v1, vcc, v8, v1, vcc                     // 000000001090: 38020308
	s_waitcnt vmcnt(0)                                         // 000000001094: BF8C0F70
	ds_write_b32 v6, v7                                        // 000000001098: D81A0000 00000706
	s_waitcnt lgkmcnt(0)                                       // 0000000010A0: BF8CC07F
	s_barrier                                                  // 0000000010A4: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 0000000010A8: BF8CC07F
	ds_read_b32 v2, v6                                         // 0000000010AC: D86C0000 02000006
	s_waitcnt lgkmcnt(0)                                       // 0000000010B4: BF8CC07F
	global_store_dword v[0:1], v2, off                         // 0000000010B8: DC708000 007F0200
	s_endpgm                                                   // 0000000010C0: BF810000
	s_nop 0                                                    // 0000000010C4: BF800000
	s_nop 0                                                    // 0000000010C8: BF800000
	s_nop 0                                                    // 0000000010CC: BF800000
	s_nop 0                                                    // 0000000010D0: BF800000
	s_nop 0                                                    // 0000000010D4: BF800000
	s_nop 0                                                    // 0000000010D8: BF800000
	s_nop 0                                                    // 0000000010DC: BF800000
	s_nop 0                                                    // 0000000010E0: BF800000
	s_nop 0                                                    // 0000000010E4: BF800000
	s_nop 0                                                    // 0000000010E8: BF800000
	s_nop 0                                                    // 0000000010EC: BF800000
	s_nop 0                                                    // 0000000010F0: BF800000
	s_nop 0                                                    // 0000000010F4: BF800000
	s_nop 0                                                    // 0000000010F8: BF800000
	s_nop 0                                                    // 0000000010FC: BF800000

0000000000001100 <_Z24RecursiveGaussian_kernelPK15HIP_vector_typeIhLj4EEPS0_iiffffffff>:
	s_load_dwordx8 s[12:19], s[6:7], 0x10                      // 000000001100: C00E0303 00000010
	s_load_dword s0, s[4:5], 0x4                               // 000000001108: C0020002 00000004
	s_waitcnt lgkmcnt(0)                                       // 000000001110: BF8CC07F
	s_and_b32 s0, s0, 0xffff                                   // 000000001114: 8600FF00 0000FFFF
	s_mul_i32 s10, s8, s0                                      // 00000000111C: 920A0008
	v_add_u32_e32 v13, s10, v0                                 // 000000001120: 681A000A
	v_cmp_gt_u32_e32 vcc, s12, v13                             // 000000001124: 7D981A0C
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001128: BE80206A
	s_cbranch_execz BB1_7                                      // 00000000112C: BF8800CB
	s_load_dwordx4 s[0:3], s[6:7], 0x0                         // 000000001130: C00A0003 00000000
	s_cmp_gt_i32 s13, 0                                        // 000000001138: BF02800D
	s_cselect_b64 s[8:9], -1, 0                                // 00000000113C: 858880C1
	s_cmp_lt_i32 s13, 1                                        // 000000001140: BF04810D
	s_mov_b32 s4, 0                                            // 000000001144: BE840080
	s_cbranch_scc1 BB1_4                                       // 000000001148: BF850056
	s_mov_b32 s7, s4                                           // 00000000114C: BE870004
	s_mov_b32 s5, s4                                           // 000000001150: BE850004
	s_mov_b32 s6, s4                                           // 000000001154: BE860004
	v_mov_b32_e32 v8, s7                                       // 000000001158: 7E100207
	v_mov_b32_e32 v5, s4                                       // 00000000115C: 7E0A0204
	v_mov_b32_e32 v7, s6                                       // 000000001160: 7E0E0206
	v_mov_b32_e32 v6, s5                                       // 000000001164: 7E0C0205
	v_mov_b32_e32 v1, v5                                       // 000000001168: 7E020305
	v_mov_b32_e32 v12, v8                                      // 00000000116C: 7E180308
	s_mov_b32 s4, s13                                          // 000000001170: BE84000D
	v_mov_b32_e32 v2, v6                                       // 000000001174: 7E040306
	v_mov_b32_e32 v3, v7                                       // 000000001178: 7E060307
	v_mov_b32_e32 v4, v8                                       // 00000000117C: 7E080308
	v_mov_b32_e32 v11, v7                                      // 000000001180: 7E160307
	v_mov_b32_e32 v10, v6                                      // 000000001184: 7E140306
	v_mov_b32_e32 v9, v5                                       // 000000001188: 7E120305

000000000000118c <BB1_3>:
	v_ashrrev_i32_e32 v14, 31, v13                             // 00000000118C: 221C1A9F
	v_lshlrev_b64 v[22:23], 2, v[13:14]                        // 000000001190: D28F0016 00021A82
	s_waitcnt lgkmcnt(0)                                       // 000000001198: BF8CC07F
	v_mov_b32_e32 v15, s1                                      // 00000000119C: 7E1E0201
	v_add_co_u32_e32 v14, vcc, s0, v22                         // 0000000011A0: 321C2C00
	v_addc_co_u32_e32 v15, vcc, v15, v23, vcc                  // 0000000011A4: 381E2F0F
	global_load_dword v17, v[14:15], off                       // 0000000011A8: DC508000 117F000E
	v_mov_b32_e32 v21, v4                                      // 0000000011B0: 7E2A0304
	v_mov_b32_e32 v20, v3                                      // 0000000011B4: 7E280303
	v_mov_b32_e32 v19, v2                                      // 0000000011B8: 7E260302
	v_mov_b32_e32 v18, v1                                      // 0000000011BC: 7E240301
	v_add_co_u32_e32 v22, vcc, s2, v22                         // 0000000011C0: 322C2C02
	s_add_i32 s4, s4, -1                                       // 0000000011C4: 8104C104
	v_add_u32_e32 v13, s12, v13                                // 0000000011C8: 681A1A0C
	s_cmp_eq_u32 s4, 0                                         // 0000000011CC: BF068004
	s_waitcnt vmcnt(0)                                         // 0000000011D0: BF8C0F70
	v_cvt_f32_ubyte0_e32 v14, v17                              // 0000000011D4: 7E1C2311
	v_cvt_f32_ubyte1_e32 v15, v17                              // 0000000011D8: 7E1E2511
	v_mul_f32_e32 v24, s14, v14                                // 0000000011DC: 0A301C0E
	v_cvt_f32_ubyte2_e32 v16, v17                              // 0000000011E0: 7E202711
	v_fmac_f32_e32 v24, s15, v5                                // 0000000011E4: 76300A0F
	v_mul_f32_e32 v5, s14, v15                                 // 0000000011E8: 0A0A1E0E
	v_cvt_f32_ubyte3_e32 v17, v17                              // 0000000011EC: 7E222911
	v_fmac_f32_e32 v5, s15, v6                                 // 0000000011F0: 760A0C0F
	v_mul_f32_e32 v6, s14, v16                                 // 0000000011F4: 0A0C200E
	v_fmac_f32_e32 v6, s15, v7                                 // 0000000011F8: 760C0E0F
	v_mul_f32_e32 v7, s14, v17                                 // 0000000011FC: 0A0E220E
	v_fmac_f32_e32 v7, s15, v8                                 // 000000001200: 760E100F
	v_fma_f32 v1, -v18, s18, v24                               // 000000001204: D1CB0001 24602512
	v_fma_f32 v2, -v19, s18, v5                                // 00000000120C: D1CB0002 24142513
	v_fma_f32 v3, -v20, s18, v6                                // 000000001214: D1CB0003 24182514
	v_fma_f32 v4, -v21, s18, v7                                // 00000000121C: D1CB0004 241C2515
	v_fma_f32 v1, -v9, s19, v1                                 // 000000001224: D1CB0001 24042709
	v_fma_f32 v2, -v10, s19, v2                                // 00000000122C: D1CB0002 2408270A
	v_fma_f32 v3, -v11, s19, v3                                // 000000001234: D1CB0003 240C270B
	v_fma_f32 v4, -v12, s19, v4                                // 00000000123C: D1CB0004 2410270C
	v_cvt_i32_f32_e32 v24, v1                                  // 000000001244: 7E301101
	v_cvt_i32_f32_sdwa v25, v2 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD// 000000001248: 7E3210F9 00060102
	v_cvt_i32_f32_e32 v26, v3                                  // 000000001250: 7E341103
	v_cvt_i32_f32_sdwa v27, v4 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD// 000000001254: 7E3610F9 00060104
	v_mov_b32_e32 v5, s3                                       // 00000000125C: 7E0A0203
	v_addc_co_u32_e32 v23, vcc, v5, v23, vcc                   // 000000001260: 382E2F05
	v_mov_b32_e32 v5, v14                                      // 000000001264: 7E0A030E
	v_mov_b32_e32 v6, v15                                      // 000000001268: 7E0C030F
	v_mov_b32_e32 v7, v16                                      // 00000000126C: 7E0E0310
	v_mov_b32_e32 v8, v17                                      // 000000001270: 7E100311
	v_mov_b32_e32 v9, v18                                      // 000000001274: 7E120312
	v_or_b32_e32 v14, v24, v25                                 // 000000001278: 281C3318
	v_or_b32_sdwa v15, v26, v27 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD// 00000000127C: 281E36F9 0606051A
	v_mov_b32_e32 v10, v19                                     // 000000001284: 7E140313
	v_mov_b32_e32 v11, v20                                     // 000000001288: 7E160314
	v_mov_b32_e32 v12, v21                                     // 00000000128C: 7E180315
	v_or_b32_sdwa v14, v14, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD// 000000001290: 281C1EF9 0604060E
	global_store_dword v[22:23], v14, off                      // 000000001298: DC708000 007F0E16
	s_cbranch_scc0 BB1_3                                       // 0000000012A0: BF84FFBA

00000000000012a4 <BB1_4>:
	s_andn2_b64 vcc, exec, s[8:9]                              // 0000000012A4: 89EA087E
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 0000000012A8: BF8C0070
	s_barrier                                                  // 0000000012AC: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 0000000012B0: BF8CC07F
	s_cbranch_vccnz BB1_7                                      // 0000000012B4: BF870069
	s_add_i32 s5, s13, -1                                      // 0000000012B8: 8105C10D
	s_mul_i32 s5, s12, s5                                      // 0000000012BC: 9205050C
	s_add_i32 s10, s10, s5                                     // 0000000012C0: 810A050A
	s_mov_b32 s8, 0                                            // 0000000012C4: BE880080
	v_add_u32_e32 v16, s10, v0                                 // 0000000012C8: 6820000A
	s_mov_b32 s9, s8                                           // 0000000012CC: BE890008
	s_mov_b32 s10, s8                                          // 0000000012D0: BE8A0008
	s_mov_b32 s11, s8                                          // 0000000012D4: BE8B0008
	v_mov_b32_e32 v8, s8                                       // 0000000012D8: 7E100208
	v_mov_b32_e32 v11, s11                                     // 0000000012DC: 7E16020B
	v_mov_b32_e32 v9, s9                                       // 0000000012E0: 7E120209
	v_mov_b32_e32 v10, s10                                     // 0000000012E4: 7E14020A
	v_mov_b32_e32 v0, v8                                       // 0000000012E8: 7E000308
	v_mov_b32_e32 v15, v11                                     // 0000000012EC: 7E1E030B
	v_mov_b32_e32 v4, v8                                       // 0000000012F0: 7E080308
	s_add_i32 s4, s13, 1                                       // 0000000012F4: 8104810D
	v_mov_b32_e32 v1, v9                                       // 0000000012F8: 7E020309
	v_mov_b32_e32 v2, v10                                      // 0000000012FC: 7E04030A
	v_mov_b32_e32 v3, v11                                      // 000000001300: 7E06030B
	v_mov_b32_e32 v14, v10                                     // 000000001304: 7E1C030A
	v_mov_b32_e32 v13, v9                                      // 000000001308: 7E1A0309
	v_mov_b32_e32 v12, v8                                      // 00000000130C: 7E180308
	v_mov_b32_e32 v5, v9                                       // 000000001310: 7E0A0309
	v_mov_b32_e32 v6, v10                                      // 000000001314: 7E0C030A
	v_mov_b32_e32 v7, v11                                      // 000000001318: 7E0E030B

000000000000131c <BB1_6>:
	v_mul_f32_e32 v12, s17, v12                                // 00000000131C: 0A181811
	v_mul_f32_e32 v14, s17, v14                                // 000000001320: 0A1C1C11
	v_fmac_f32_e32 v12, s16, v4                                // 000000001324: 76180810
	v_mul_f32_e32 v13, s17, v13                                // 000000001328: 0A1A1A11
	v_mul_f32_e32 v15, s17, v15                                // 00000000132C: 0A1E1E11
	v_fma_f32 v12, -v0, s18, v12                               // 000000001330: D1CB000C 24302500
	v_fmac_f32_e32 v14, s16, v6                                // 000000001338: 761C0C10
	v_fmac_f32_e32 v13, s16, v5                                // 00000000133C: 761A0A10
	v_fma_f32 v18, -v8, s19, v12                               // 000000001340: D1CB0012 24302708
	v_fma_f32 v8, -v2, s18, v14                                // 000000001348: D1CB0008 24382502
	v_fmac_f32_e32 v15, s16, v7                                // 000000001350: 761E0E10
	v_fma_f32 v13, -v1, s18, v13                               // 000000001354: D1CB000D 24342501
	v_ashrrev_i32_e32 v17, 31, v16                             // 00000000135C: 2222209F
	v_fma_f32 v20, -v10, s19, v8                               // 000000001360: D1CB0014 2420270A
	v_fma_f32 v8, -v3, s18, v15                                // 000000001368: D1CB0008 243C2503
	v_fma_f32 v19, -v9, s19, v13                               // 000000001370: D1CB0013 24342709
	v_fma_f32 v21, -v11, s19, v8                               // 000000001378: D1CB0015 2420270B
	v_lshlrev_b64 v[8:9], 2, v[16:17]                          // 000000001380: D28F0008 00022082
	v_mov_b32_e32 v12, s1                                      // 000000001388: 7E180201
	v_add_co_u32_e32 v10, vcc, s0, v8                          // 00000000138C: 32141000
	v_addc_co_u32_e32 v11, vcc, v12, v9, vcc                   // 000000001390: 3816130C
	v_mov_b32_e32 v13, s3                                      // 000000001394: 7E1A0203
	v_add_co_u32_e32 v22, vcc, s2, v8                          // 000000001398: 322C1002
	v_addc_co_u32_e32 v23, vcc, v13, v9, vcc                   // 00000000139C: 382E130D
	global_load_dword v17, v[22:23], off                       // 0000000013A0: DC508000 117F0016
	global_load_dword v24, v[10:11], off                       // 0000000013A8: DC508000 187F000A
	v_mov_b32_e32 v11, v3                                      // 0000000013B0: 7E160303
	v_mov_b32_e32 v10, v2                                      // 0000000013B4: 7E140302
	v_mov_b32_e32 v9, v1                                       // 0000000013B8: 7E120301
	v_mov_b32_e32 v8, v0                                       // 0000000013BC: 7E100300
	v_mov_b32_e32 v15, v7                                      // 0000000013C0: 7E1E0307
	v_mov_b32_e32 v14, v6                                      // 0000000013C4: 7E1C0306
	v_mov_b32_e32 v13, v5                                      // 0000000013C8: 7E1A0305
	v_mov_b32_e32 v12, v4                                      // 0000000013CC: 7E180304
	s_add_i32 s4, s4, -1                                       // 0000000013D0: 8104C104
	v_subrev_u32_e32 v16, s12, v16                             // 0000000013D4: 6C20200C
	s_cmp_gt_u32 s4, 1                                         // 0000000013D8: BF088104
	s_waitcnt vmcnt(1)                                         // 0000000013DC: BF8C0F71
	v_cvt_f32_ubyte0_e32 v0, v17                               // 0000000013E0: 7E002311
	v_cvt_f32_ubyte1_e32 v1, v17                               // 0000000013E4: 7E022511
	v_add_f32_e32 v0, v18, v0                                  // 0000000013E8: 02000112
	v_add_f32_e32 v1, v19, v1                                  // 0000000013EC: 02020313
	v_cvt_i32_f32_e32 v0, v0                                   // 0000000013F0: 7E001100
	v_cvt_i32_f32_sdwa v1, v1 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD// 0000000013F4: 7E0210F9 00060101
	s_waitcnt vmcnt(0)                                         // 0000000013FC: BF8C0F70
	v_cvt_f32_ubyte3_e32 v7, v24                               // 000000001400: 7E0E2918
	v_or_b32_e32 v4, v0, v1                                    // 000000001404: 28080300
	v_cvt_f32_ubyte2_e32 v0, v17                               // 000000001408: 7E002711
	v_cvt_f32_ubyte3_e32 v1, v17                               // 00000000140C: 7E022911
	v_add_f32_e32 v5, v20, v0                                  // 000000001410: 020A0114
	v_add_f32_e32 v6, v21, v1                                  // 000000001414: 020C0315
	v_cvt_i32_f32_e32 v5, v5                                   // 000000001418: 7E0A1105
	v_cvt_i32_f32_sdwa v6, v6 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD// 00000000141C: 7E0C10F9 00060106
	v_mov_b32_e32 v0, v18                                      // 000000001424: 7E000312
	v_mov_b32_e32 v1, v19                                      // 000000001428: 7E020313
	v_mov_b32_e32 v2, v20                                      // 00000000142C: 7E040314
	v_or_b32_sdwa v5, v5, v6 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD// 000000001430: 280A0CF9 06060505
	v_or_b32_sdwa v4, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD// 000000001438: 28080AF9 06040604
	v_mov_b32_e32 v3, v21                                      // 000000001440: 7E060315
	global_store_dword v[22:23], v4, off                       // 000000001444: DC708000 007F0416
	v_cvt_f32_ubyte0_e32 v4, v24                               // 00000000144C: 7E082318
	v_cvt_f32_ubyte1_e32 v5, v24                               // 000000001450: 7E0A2518
	v_cvt_f32_ubyte2_e32 v6, v24                               // 000000001454: 7E0C2718
	s_cbranch_scc1 BB1_6                                       // 000000001458: BF85FFB0

000000000000145c <BB1_7>:
	s_endpgm                                                   // 00000000145C: BF810000
