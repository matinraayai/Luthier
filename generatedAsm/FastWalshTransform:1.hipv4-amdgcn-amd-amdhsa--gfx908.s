
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <_Z18fastWalshTransformPfi>:
	s_load_dword s2, s[6:7], 0x8                               // 000000001000: C0020083 00000008
	s_load_dword s0, s[4:5], 0x4                               // 000000001008: C0020002 00000004
	s_waitcnt lgkmcnt(0)                                       // 000000001010: BF8CC07F
	v_cvt_f32_u32_e32 v1, s2                                   // 000000001014: 7E020C02
	s_sub_i32 s1, 0, s2                                        // 000000001018: 81810280
	s_and_b32 s0, s0, 0xffff                                   // 00000000101C: 8600FF00 0000FFFF
	s_mul_i32 s8, s8, s0                                       // 000000001024: 92080008
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001028: 7E024701
	v_add_u32_e32 v0, s8, v0                                   // 00000000102C: 68000008
	v_mul_f32_e32 v1, 0x4f7ffffe, v1                           // 000000001030: 0A0202FF 4F7FFFFE
	v_cvt_u32_f32_e32 v1, v1                                   // 000000001038: 7E020F01
	v_mul_lo_u32 v2, s1, v1                                    // 00000000103C: D2850002 00020201
	s_load_dwordx2 s[0:1], s[6:7], 0x0                         // 000000001044: C0060003 00000000
	v_mul_hi_u32 v2, v1, v2                                    // 00000000104C: D2860002 00020501
	s_waitcnt lgkmcnt(0)                                       // 000000001054: BF8CC07F
	v_mov_b32_e32 v4, s1                                       // 000000001058: 7E080201
	v_add_u32_e32 v1, v1, v2                                   // 00000000105C: 68020501
	v_mul_hi_u32 v1, v0, v1                                    // 000000001060: D2860001 00020300
	v_mul_lo_u32 v2, v1, s2                                    // 000000001068: D2850002 00000501
	v_add_u32_e32 v3, 1, v1                                    // 000000001070: 68060281
	v_sub_u32_e32 v2, v0, v2                                   // 000000001074: 6A040500
	v_cmp_le_u32_e32 vcc, s2, v2                               // 000000001078: 7D960402
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 00000000107C: 00020701
	v_subrev_u32_e32 v3, s2, v2                                // 000000001080: 6C060402
	v_cndmask_b32_e32 v2, v2, v3, vcc                          // 000000001084: 00040702
	v_add_u32_e32 v3, 1, v1                                    // 000000001088: 68060281
	v_cmp_le_u32_e32 vcc, s2, v2                               // 00000000108C: 7D960402
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 000000001090: 00020701
	v_mul_lo_u32 v2, v1, s2                                    // 000000001094: D2850002 00000501
	v_mov_b32_e32 v1, 0                                        // 00000000109C: 7E020280
	v_mov_b32_e32 v3, v1                                       // 0000000010A0: 7E060301
	v_sub_u32_e32 v0, v0, v2                                   // 0000000010A4: 6A000500
	v_lshl_add_u32 v0, v2, 1, v0                               // 0000000010A8: D1FD0000 04010302
	v_add_u32_e32 v2, s2, v0                                   // 0000000010B0: 68040002
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 0000000010B4: D28F0000 00020082
	v_lshlrev_b64 v[2:3], 2, v[2:3]                            // 0000000010BC: D28F0002 00020482
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 0000000010C4: 32000000
	v_addc_co_u32_e32 v1, vcc, v4, v1, vcc                     // 0000000010C8: 38020304
	v_add_co_u32_e32 v2, vcc, s0, v2                           // 0000000010CC: 32040400
	v_addc_co_u32_e32 v3, vcc, v4, v3, vcc                     // 0000000010D0: 38060704
	global_load_dword v4, v[0:1], off                          // 0000000010D4: DC508000 047F0000
	global_load_dword v5, v[2:3], off                          // 0000000010DC: DC508000 057F0002
	s_waitcnt vmcnt(0)                                         // 0000000010E4: BF8C0F70
	v_add_f32_e32 v6, v4, v5                                   // 0000000010E8: 020C0B04
	v_sub_f32_e32 v4, v4, v5                                   // 0000000010EC: 04080B04
	global_store_dword v[0:1], v6, off                         // 0000000010F0: DC708000 007F0600
	global_store_dword v[2:3], v4, off                         // 0000000010F8: DC708000 007F0402
	s_endpgm                                                   // 000000001100: BF810000
