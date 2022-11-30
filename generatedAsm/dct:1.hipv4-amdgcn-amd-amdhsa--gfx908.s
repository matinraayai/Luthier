
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <_Z9DCTKernelPfS_S_S_jjj>:
	s_load_dwordx4 s[20:23], s[6:7], 0x20                      // 000000001000: C00A0503 00000020
	s_load_dwordx8 s[12:19], s[6:7], 0x0                       // 000000001008: C00E0303 00000000
	s_waitcnt lgkmcnt(0)                                       // 000000001010: BF8CC07F
	v_mul_lo_u32 v6, v1, s21                                   // 000000001014: D2850006 00002B01
	s_cmp_eq_u32 s22, 0                                        // 00000000101C: BF068016
	s_cselect_b64 s[0:1], -1, 0                                // 000000001020: 858080C1
	s_cmp_lg_u32 s21, 0                                        // 000000001024: BF078015
	s_cselect_b64 s[2:3], -1, 0                                // 000000001028: 858280C1
	s_cmp_eq_u32 s21, 0                                        // 00000000102C: BF068015
	s_cbranch_scc1 38                                          // 000000001030: BF850026 <_Z9DCTKernelPfS_S_S_jjj+0xcc>
	s_mul_i32 s6, s9, s20                                      // 000000001034: 92061409
	v_mov_b32_e32 v2, s17                                      // 000000001038: 7E040211
	v_mov_b32_e32 v3, s19                                      // 00000000103C: 7E060213
	s_add_i32 s6, s8, s6                                       // 000000001040: 81060608
	v_cndmask_b32_e64 v8, v2, v3, s[0:1]                       // 000000001044: D1000008 00020702
	v_mov_b32_e32 v2, s16                                      // 00000000104C: 7E040210
	v_mov_b32_e32 v3, s18                                      // 000000001050: 7E060212
	s_mul_i32 s6, s21, s6                                      // 000000001054: 92060615
	v_mov_b32_e32 v5, 0                                        // 000000001058: 7E0A0280
	v_cndmask_b32_e64 v9, v2, v3, s[0:1]                       // 00000000105C: D1000009 00020702
	v_add_u32_e32 v2, s6, v0                                   // 000000001064: 68040006
	v_mov_b32_e32 v4, v6                                       // 000000001068: 7E080306
	s_mov_b32 s6, s21                                          // 00000000106C: BE860015
	v_mov_b32_e32 v7, v5                                       // 000000001070: 7E0E0305
	v_lshlrev_b64 v[10:11], 2, v[4:5]                          // 000000001074: D28F000A 00020882
	v_mov_b32_e32 v3, v5                                       // 00000000107C: 7E060305
	v_lshlrev_b64 v[12:13], 2, v[2:3]                          // 000000001080: D28F000C 00020482
	v_add_co_u32_e32 v10, vcc, v9, v10                         // 000000001088: 32141509
	v_addc_co_u32_e32 v11, vcc, v8, v11, vcc                   // 00000000108C: 38161708
	v_mov_b32_e32 v14, s15                                     // 000000001090: 7E1C020F
	global_load_dword v3, v[10:11], off                        // 000000001094: DC508000 037F000A
	v_add_co_u32_e32 v10, vcc, s14, v12                        // 00000000109C: 3214180E
	v_addc_co_u32_e32 v11, vcc, v14, v13, vcc                  // 0000000010A0: 38161B0E
	global_load_dword v10, v[10:11], off                       // 0000000010A4: DC508000 0A7F000A
	s_add_i32 s6, s6, -1                                       // 0000000010AC: 8106C106
	v_add_u32_e32 v4, 1, v4                                    // 0000000010B0: 68080881
	v_add_u32_e32 v2, s20, v2                                  // 0000000010B4: 68040414
	s_cmp_eq_u32 s6, 0                                         // 0000000010B8: BF068006
	s_waitcnt vmcnt(0)                                         // 0000000010BC: BF8C0F70
	v_fmac_f32_e32 v7, v3, v10                                 // 0000000010C0: 760E1503
	s_cbranch_scc0 65515                                       // 0000000010C4: BF84FFEB <_Z9DCTKernelPfS_S_S_jjj+0x74>
	s_branch 1                                                 // 0000000010C8: BF820001 <_Z9DCTKernelPfS_S_S_jjj+0xd0>
	v_mov_b32_e32 v7, 0                                        // 0000000010CC: 7E0E0280
	s_load_dword s4, s[4:5], 0x4                               // 0000000010D0: C0020102 00000004
	v_add_lshl_u32 v2, v6, v0, 2                               // 0000000010D8: D1FE0002 020A0106
	s_andn2_b64 vcc, exec, s[2:3]                              // 0000000010E0: 89EA027E
	ds_write_b32 v2, v7                                        // 0000000010E4: D81A0000 00000702
	s_waitcnt lgkmcnt(0)                                       // 0000000010EC: BF8CC07F
	s_barrier                                                  // 0000000010F0: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 0000000010F4: BF8CC07F
	s_cbranch_vccnz 116                                        // 0000000010F8: BF870074 <_Z9DCTKernelPfS_S_S_jjj+0x2cc>
	s_add_i32 s2, s21, -1                                      // 0000000010FC: 8102C115
	s_cmp_lt_u32 s2, 7                                         // 000000001100: BF0A8702
	s_cbranch_scc1 115                                         // 000000001104: BF850073 <_Z9DCTKernelPfS_S_S_jjj+0x2d4>
	v_mov_b32_e32 v2, s19                                      // 000000001108: 7E040213
	v_mov_b32_e32 v3, s17                                      // 00000000110C: 7E060211
	v_cndmask_b32_e64 v5, v2, v3, s[0:1]                       // 000000001110: D1000005 00020702
	v_mov_b32_e32 v2, s18                                      // 000000001118: 7E040212
	v_mov_b32_e32 v3, s16                                      // 00000000111C: 7E060210
	v_cndmask_b32_e64 v7, v2, v3, s[0:1]                       // 000000001120: D1000007 00020702
	v_mov_b32_e32 v3, 0                                        // 000000001128: 7E060280
	s_and_b32 s2, s21, -8                                      // 00000000112C: 8602C815
	s_lshl_b32 s3, s21, 3                                      // 000000001130: 8E038315
	s_lshl_b32 s5, s21, 1                                      // 000000001134: 8E058115
	s_mul_i32 s6, s21, 3                                       // 000000001138: 92068315
	s_lshl_b32 s7, s21, 2                                      // 00000000113C: 8E078215
	s_mul_i32 s10, s21, 5                                      // 000000001140: 920A8515
	s_mul_i32 s11, s21, 6                                      // 000000001144: 920B8615
	s_mul_i32 s14, s21, 7                                      // 000000001148: 920E8715
	v_lshlrev_b32_e32 v8, 2, v6                                // 00000000114C: 24100C82
	s_mov_b32 s15, 0                                           // 000000001150: BE8F0080
	v_mov_b32_e32 v2, v0                                       // 000000001154: 7E040300
	v_mov_b32_e32 v4, v3                                       // 000000001158: 7E080303
	v_lshlrev_b64 v[9:10], 2, v[2:3]                           // 00000000115C: D28F0009 00020482
	v_add_u32_e32 v11, s21, v2                                 // 000000001164: 68160415
	v_mov_b32_e32 v12, v3                                      // 000000001168: 7E180303
	v_lshlrev_b64 v[11:12], 2, v[11:12]                        // 00000000116C: D28F000B 00021682
	v_add_co_u32_e32 v9, vcc, v7, v9                           // 000000001174: 32121307
	v_add_u32_e32 v13, s5, v2                                  // 000000001178: 681A0405
	v_mov_b32_e32 v14, v3                                      // 00000000117C: 7E1C0303
	v_addc_co_u32_e32 v10, vcc, v5, v10, vcc                   // 000000001180: 38141505
	v_lshlrev_b64 v[13:14], 2, v[13:14]                        // 000000001184: D28F000D 00021A82
	v_add_co_u32_e32 v11, vcc, v7, v11                         // 00000000118C: 32161707
	v_add_u32_e32 v15, s6, v2                                  // 000000001190: 681E0406
	v_mov_b32_e32 v16, v3                                      // 000000001194: 7E200303
	v_addc_co_u32_e32 v12, vcc, v5, v12, vcc                   // 000000001198: 38181905
	v_lshlrev_b64 v[15:16], 2, v[15:16]                        // 00000000119C: D28F000F 00021E82
	v_add_co_u32_e32 v13, vcc, v7, v13                         // 0000000011A4: 321A1B07
	v_add_u32_e32 v17, s7, v2                                  // 0000000011A8: 68220407
	v_mov_b32_e32 v18, v3                                      // 0000000011AC: 7E240303
	v_addc_co_u32_e32 v14, vcc, v5, v14, vcc                   // 0000000011B0: 381C1D05
	v_lshlrev_b64 v[17:18], 2, v[17:18]                        // 0000000011B4: D28F0011 00022282
	v_add_co_u32_e32 v15, vcc, v7, v15                         // 0000000011BC: 321E1F07
	v_add_u32_e32 v19, s10, v2                                 // 0000000011C0: 6826040A
	v_mov_b32_e32 v20, v3                                      // 0000000011C4: 7E280303
	v_addc_co_u32_e32 v16, vcc, v5, v16, vcc                   // 0000000011C8: 38202105
	v_lshlrev_b64 v[19:20], 2, v[19:20]                        // 0000000011CC: D28F0013 00022682
	v_add_co_u32_e32 v17, vcc, v7, v17                         // 0000000011D4: 32222307
	v_add_u32_e32 v21, s11, v2                                 // 0000000011D8: 682A040B
	v_mov_b32_e32 v22, v3                                      // 0000000011DC: 7E2C0303
	v_addc_co_u32_e32 v18, vcc, v5, v18, vcc                   // 0000000011E0: 38242505
	v_lshlrev_b64 v[21:22], 2, v[21:22]                        // 0000000011E4: D28F0015 00022A82
	v_add_co_u32_e32 v19, vcc, v7, v19                         // 0000000011EC: 32262707
	v_add_u32_e32 v23, s14, v2                                 // 0000000011F0: 682E040E
	v_mov_b32_e32 v24, v3                                      // 0000000011F4: 7E300303
	v_addc_co_u32_e32 v20, vcc, v5, v20, vcc                   // 0000000011F8: 38282905
	v_lshlrev_b64 v[23:24], 2, v[23:24]                        // 0000000011FC: D28F0017 00022E82
	v_add_co_u32_e32 v21, vcc, v7, v21                         // 000000001204: 322A2B07
	v_addc_co_u32_e32 v22, vcc, v5, v22, vcc                   // 000000001208: 382C2D05
	v_add_co_u32_e32 v23, vcc, v7, v23                         // 00000000120C: 322E2F07
	v_addc_co_u32_e32 v24, vcc, v5, v24, vcc                   // 000000001210: 38303105
	global_load_dword v25, v[9:10], off                        // 000000001214: DC508000 197F0009
	global_load_dword v26, v[11:12], off                       // 00000000121C: DC508000 1A7F000B
	global_load_dword v27, v[13:14], off                       // 000000001224: DC508000 1B7F000D
	global_load_dword v28, v[15:16], off                       // 00000000122C: DC508000 1C7F000F
	global_load_dword v29, v[17:18], off                       // 000000001234: DC508000 1D7F0011
	global_load_dword v30, v[19:20], off                       // 00000000123C: DC508000 1E7F0013
	s_nop 0                                                    // 000000001244: BF800000
	global_load_dword v17, v[21:22], off                       // 000000001248: DC508000 117F0015
	global_load_dword v18, v[23:24], off                       // 000000001250: DC508000 127F0017
	ds_read2_b64 v[9:12], v8 offset1:1                         // 000000001258: D8EE0100 09000008
	ds_read2_b64 v[13:16], v8 offset0:2 offset1:3              // 000000001260: D8EE0302 0D000008
	s_add_i32 s15, s15, 8                                      // 000000001268: 810F880F
	v_add_u32_e32 v2, s3, v2                                   // 00000000126C: 68040403
	v_add_u32_e32 v8, 32, v8                                   // 000000001270: 681010A0
	s_cmp_eq_u32 s2, s15                                       // 000000001274: BF060F02
	s_waitcnt vmcnt(7) lgkmcnt(1)                              // 000000001278: BF8C0177
	v_fmac_f32_e32 v4, v9, v25                                 // 00000000127C: 76083309
	s_waitcnt vmcnt(6)                                         // 000000001280: BF8C0F76
	v_fmac_f32_e32 v4, v10, v26                                // 000000001284: 7608350A
	s_waitcnt vmcnt(5)                                         // 000000001288: BF8C0F75
	v_fmac_f32_e32 v4, v11, v27                                // 00000000128C: 7608370B
	s_waitcnt vmcnt(4)                                         // 000000001290: BF8C0F74
	v_fmac_f32_e32 v4, v12, v28                                // 000000001294: 7608390C
	s_waitcnt vmcnt(3) lgkmcnt(0)                              // 000000001298: BF8C0073
	v_fmac_f32_e32 v4, v13, v29                                // 00000000129C: 76083B0D
	s_waitcnt vmcnt(2)                                         // 0000000012A0: BF8C0F72
	v_fmac_f32_e32 v4, v14, v30                                // 0000000012A4: 76083D0E
	s_waitcnt vmcnt(1)                                         // 0000000012A8: BF8C0F71
	v_fmac_f32_e32 v4, v15, v17                                // 0000000012AC: 7608230F
	s_waitcnt vmcnt(0)                                         // 0000000012B0: BF8C0F70
	v_fmac_f32_e32 v4, v16, v18                                // 0000000012B4: 76082510
	s_cbranch_scc0 65448                                       // 0000000012B8: BF84FFA8 <_Z9DCTKernelPfS_S_S_jjj+0x15c>
	s_and_b32 s3, s21, 7                                       // 0000000012BC: 86038715
	s_cmp_eq_u32 s3, 0                                         // 0000000012C0: BF068003
	s_cbranch_scc0 8                                           // 0000000012C4: BF840008 <_Z9DCTKernelPfS_S_S_jjj+0x2e8>
	s_branch 35                                                // 0000000012C8: BF820023 <_Z9DCTKernelPfS_S_S_jjj+0x358>
	v_mov_b32_e32 v4, 0                                        // 0000000012CC: 7E080280
	s_branch 33                                                // 0000000012D0: BF820021 <_Z9DCTKernelPfS_S_S_jjj+0x358>
	s_mov_b32 s2, 0                                            // 0000000012D4: BE820080
	v_mov_b32_e32 v4, 0                                        // 0000000012D8: 7E080280
	s_and_b32 s3, s21, 7                                       // 0000000012DC: 86038715
	s_cmp_eq_u32 s3, 0                                         // 0000000012E0: BF068003
	s_cbranch_scc1 28                                          // 0000000012E4: BF85001C <_Z9DCTKernelPfS_S_S_jjj+0x358>
	v_mov_b32_e32 v2, s19                                      // 0000000012E8: 7E040213
	v_mov_b32_e32 v3, s17                                      // 0000000012EC: 7E060211
	v_cndmask_b32_e64 v5, v2, v3, s[0:1]                       // 0000000012F0: D1000005 00020702
	v_mov_b32_e32 v2, s18                                      // 0000000012F8: 7E040212
	v_mov_b32_e32 v3, s16                                      // 0000000012FC: 7E060210
	v_cndmask_b32_e64 v7, v2, v3, s[0:1]                       // 000000001300: D1000007 00020702
	s_mul_i32 s0, s2, s21                                      // 000000001308: 92001502
	v_add_u32_e32 v2, s0, v0                                   // 00000000130C: 68040000
	v_add_lshl_u32 v6, s2, v6, 2                               // 000000001310: D1FE0006 020A0C02
	v_mov_b32_e32 v3, 0                                        // 000000001318: 7E060280
	v_lshlrev_b64 v[8:9], 2, v[2:3]                            // 00000000131C: D28F0008 00020482
	s_add_i32 s3, s3, -1                                       // 000000001324: 8103C103
	v_add_co_u32_e32 v8, vcc, v7, v8                           // 000000001328: 32101107
	v_addc_co_u32_e32 v9, vcc, v5, v9, vcc                     // 00000000132C: 38121305
	global_load_dword v8, v[8:9], off                          // 000000001330: DC508000 087F0008
	ds_read_b32 v9, v6                                         // 000000001338: D86C0000 09000006
	v_add_u32_e32 v2, s21, v2                                  // 000000001340: 68040415
	v_add_u32_e32 v6, 4, v6                                    // 000000001344: 680C0C84
	s_cmp_lg_u32 s3, 0                                         // 000000001348: BF078003
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 00000000134C: BF8C0070
	v_fmac_f32_e32 v4, v9, v8                                  // 000000001350: 76081109
	s_cbranch_scc1 65521                                       // 000000001354: BF85FFF1 <_Z9DCTKernelPfS_S_S_jjj+0x31c>
	s_lshr_b32 s0, s4, 16                                      // 000000001358: 8F009004
	s_mul_i32 s9, s9, s0                                       // 00000000135C: 92090009
	v_add_u32_e32 v1, s9, v1                                   // 000000001360: 68020209
	v_mul_lo_u32 v2, v1, s20                                   // 000000001364: D2850002 00002901
	s_and_b32 s0, s4, 0xffff                                   // 00000000136C: 8600FF04 0000FFFF
	s_mul_i32 s8, s8, s0                                       // 000000001374: 92080008
	v_mov_b32_e32 v1, 0                                        // 000000001378: 7E020280
	v_add3_u32 v0, s8, v0, v2                                  // 00000000137C: D1FF0000 040A0008
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 000000001384: D28F0000 00020082
	v_mov_b32_e32 v2, s13                                      // 00000000138C: 7E04020D
	v_add_co_u32_e32 v0, vcc, s12, v0                          // 000000001390: 3200000C
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc                     // 000000001394: 38020302
	global_store_dword v[0:1], v4, off                         // 000000001398: DC708000 007F0400
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 0000000013A0: BF8C0070
	s_barrier                                                  // 0000000013A4: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 0000000013A8: BF8CC07F
	s_endpgm                                                   // 0000000013AC: BF810000
