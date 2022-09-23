
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <_Z9dwtHaar1DPfS_S_jjjj>:
	s_load_dword s9, s[4:5], 0x4                               // 000000001000: C0020242 00000004
	s_load_dwordx4 s[0:3], s[6:7], 0x0                         // 000000001008: C00A0003 00000000
	s_load_dwordx2 s[10:11], s[6:7], 0x10                      // 000000001010: C0060283 00000010
	v_mov_b32_e32 v2, 0                                        // 000000001018: 7E040280
	s_waitcnt lgkmcnt(0)                                       // 00000000101C: BF8CC07F
	s_and_b32 s12, s9, 0xffff                                  // 000000001020: 860CFF09 0000FFFF
	s_mul_i32 s4, s8, s12                                      // 000000001028: 92040C08
	v_lshl_add_u32 v1, s4, 1, v0                               // 00000000102C: D1FD0001 04010204
	v_lshlrev_b64 v[3:4], 2, v[1:2]                            // 000000001034: D28F0003 00020282
	v_add_u32_e32 v1, s12, v1                                  // 00000000103C: 6802020C
	v_mov_b32_e32 v5, s1                                       // 000000001040: 7E0A0201
	v_add_co_u32_e32 v3, vcc, s0, v3                           // 000000001044: 32060600
	v_lshlrev_b64 v[1:2], 2, v[1:2]                            // 000000001048: D28F0001 00020282
	v_addc_co_u32_e32 v4, vcc, v5, v4, vcc                     // 000000001050: 38080905
	v_mov_b32_e32 v6, s1                                       // 000000001054: 7E0C0201
	v_add_co_u32_e32 v5, vcc, s0, v1                           // 000000001058: 320A0200
	v_addc_co_u32_e32 v6, vcc, v6, v2, vcc                     // 00000000105C: 380C0506
	global_load_dword v1, v[3:4], off                          // 000000001060: DC508000 017F0003
	global_load_dword v2, v[5:6], off                          // 000000001068: DC508000 027F0005
	s_load_dwordx4 s[4:7], s[6:7], 0x18                        // 000000001070: C00A0103 00000018
	s_mov_b32 s9, 0                                            // 000000001078: BE890080
	s_waitcnt lgkmcnt(0)                                       // 00000000107C: BF8CC07F
	s_cmp_lg_u32 s6, 0                                         // 000000001080: BF078006
	s_cbranch_scc1 16                                          // 000000001084: BF850010 <_Z9dwtHaar1DPfS_S_jjjj+0xc8>
	v_cvt_f32_u32_e32 v3, s5                                   // 000000001088: 7E060C05
	s_cmp_eq_u32 s5, 0                                         // 00000000108C: BF068005
	s_cselect_b64 vcc, -1, 0                                   // 000000001090: 85EA80C1
	v_mov_b32_e32 v5, 0x71800000                               // 000000001094: 7E0A02FF 71800000
	v_cndmask_b32_e32 v5, 1.0, v5, vcc                         // 00000000109C: 000A0AF2
	v_mul_f32_e32 v3, v5, v3                                   // 0000000010A0: 0A060705
	v_rsq_f32_e32 v3, v3                                       // 0000000010A4: 7E064903
	v_mov_b32_e32 v4, 0x58800000                               // 0000000010A8: 7E0802FF 58800000
	v_cndmask_b32_e32 v4, 1.0, v4, vcc                         // 0000000010B0: 000808F2
	v_mul_f32_e32 v3, v4, v3                                   // 0000000010B4: 0A060704
	s_waitcnt vmcnt(1)                                         // 0000000010B8: BF8C0F71
	v_mul_f32_e32 v1, v3, v1                                   // 0000000010BC: 0A020303
	s_waitcnt vmcnt(0)                                         // 0000000010C0: BF8C0F70
	v_mul_f32_e32 v2, v3, v2                                   // 0000000010C4: 0A040503
	v_lshlrev_b32_e32 v5, 2, v0                                // 0000000010C8: 240A0082
	s_min_u32 s4, s4, s7                                       // 0000000010CC: 83840704
	s_waitcnt vmcnt(1)                                         // 0000000010D0: BF8C0F71
	ds_write_b32 v5, v1                                        // 0000000010D4: D81A0000 00000105
	v_lshl_add_u32 v1, s12, 2, v5                              // 0000000010DC: D1FD0001 0415040C
	s_cmp_eq_u32 s4, 0                                         // 0000000010E4: BF068004
	s_waitcnt vmcnt(0)                                         // 0000000010E8: BF8C0F70
	ds_write_b32 v1, v2                                        // 0000000010EC: D81A0000 00000201
	s_waitcnt lgkmcnt(0)                                       // 0000000010F4: BF8CC07F
	s_barrier                                                  // 0000000010F8: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 0000000010FC: BF8CC07F
	s_cbranch_scc1 48                                          // 000000001100: BF850030 <_Z9dwtHaar1DPfS_S_jjjj+0x1c4>
	v_rsq_f32_e32 v6, 2.0                                      // 000000001104: 7E0C48F4
	s_lshl_b32 s1, 1, s4                                       // 000000001108: 8E010481
	s_lshr_b32 s0, s5, 1                                       // 00000000110C: 8F008105
	s_lshr_b32 s5, s1, 31                                      // 000000001110: 8F059F01
	s_add_i32 s1, s1, s5                                       // 000000001114: 81010501
	v_lshl_add_u32 v7, v0, 2, v5                               // 000000001118: D1FD0007 04150500
	s_ashr_i32 s5, s1, 1                                       // 000000001120: 90058101
	v_mov_b32_e32 v8, s0                                       // 000000001124: 7E100200
	v_mov_b32_e32 v2, 0                                        // 000000001128: 7E040280
	s_branch 8                                                 // 00000000112C: BF820008 <_Z9dwtHaar1DPfS_S_jjjj+0x150>
	s_or_b64 exec, exec, s[0:1]                                // 000000001130: 87FE007E
	s_lshr_b32 s5, s5, 1                                       // 000000001134: 8F058105
	s_add_i32 s4, s4, -1                                       // 000000001138: 8104C104
	s_cmp_eq_u32 s4, 0                                         // 00000000113C: BF068004
	s_waitcnt vmcnt(0) lgkmcnt(0)                              // 000000001140: BF8C0070
	s_barrier                                                  // 000000001144: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 000000001148: BF8CC07F
	s_cbranch_scc1 29                                          // 00000000114C: BF85001D <_Z9dwtHaar1DPfS_S_jjjj+0x1c4>
	v_cmp_gt_u32_e32 vcc, s5, v0                               // 000000001150: 7D980005
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001154: BE80206A
	s_cbranch_execz 2                                          // 000000001158: BF880002 <_Z9dwtHaar1DPfS_S_jjjj+0x164>
	ds_read_b64 v[3:4], v7                                     // 00000000115C: D8EC0000 03000007
	s_or_b64 exec, exec, s[0:1]                                // 000000001164: 87FE007E
	s_waitcnt lgkmcnt(0)                                       // 000000001168: BF8CC07F
	s_barrier                                                  // 00000000116C: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 000000001170: BF8CC07F
	s_and_saveexec_b64 s[0:1], vcc                             // 000000001174: BE80206A
	s_cbranch_execz 65517                                      // 000000001178: BF88FFED <_Z9dwtHaar1DPfS_S_jjjj+0x130>
	v_add_f32_e32 v1, v3, v4                                   // 00000000117C: 02020903
	v_mul_f32_e32 v1, v6, v1                                   // 000000001180: 0A020306
	s_mul_i32 s6, s5, s8                                       // 000000001184: 92060805
	ds_write_b32 v5, v1                                        // 000000001188: D81A0000 00000105
	v_add3_u32 v1, s6, v0, v8                                  // 000000001190: D1FF0001 04220006
	v_sub_f32_e32 v9, v3, v4                                   // 000000001198: 04120903
	v_mul_f32_e32 v11, v6, v9                                  // 00000000119C: 0A161306
	v_lshlrev_b64 v[9:10], 2, v[1:2]                           // 0000000011A0: D28F0009 00020282
	v_mov_b32_e32 v1, s3                                       // 0000000011A8: 7E020203
	v_add_co_u32_e32 v9, vcc, s2, v9                           // 0000000011AC: 32121202
	v_addc_co_u32_e32 v10, vcc, v1, v10, vcc                   // 0000000011B0: 38141501
	v_lshrrev_b32_e32 v8, 1, v8                                // 0000000011B4: 20101081
	global_store_dword v[9:10], v11, off                       // 0000000011B8: DC708000 007F0B09
	s_branch 65499                                             // 0000000011C0: BF82FFDB <_Z9dwtHaar1DPfS_S_jjjj+0x130>
	v_cmp_eq_u32_e32 vcc, 0, v0                                // 0000000011C4: 7D940080
	s_and_saveexec_b64 s[0:1], vcc                             // 0000000011C8: BE80206A
	s_cbranch_execz 9                                          // 0000000011CC: BF880009 <_Z9dwtHaar1DPfS_S_jjjj+0x1f4>
	v_mov_b32_e32 v0, 0                                        // 0000000011D0: 7E000280
	ds_read_b32 v1, v0                                         // 0000000011D4: D86C0000 01000000
	s_lshl_b64 s[0:1], s[8:9], 2                               // 0000000011DC: 8E808208
	s_add_u32 s0, s10, s0                                      // 0000000011E0: 8000000A
	s_addc_u32 s1, s11, s1                                     // 0000000011E4: 8201010B
	s_waitcnt lgkmcnt(0)                                       // 0000000011E8: BF8CC07F
	global_store_dword v0, v1, s[0:1]                          // 0000000011EC: DC708000 00000100
	s_endpgm                                                   // 0000000011F4: BF810000
