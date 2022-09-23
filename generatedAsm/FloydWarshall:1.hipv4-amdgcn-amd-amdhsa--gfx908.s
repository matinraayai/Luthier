
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <_Z17floydWarshallPassPjS_jj>:
	s_load_dword s12, s[4:5], 0x4                              // 000000001000: C0020302 00000004
	s_load_dwordx2 s[10:11], s[6:7], 0x10                      // 000000001008: C0060283 00000010
	s_load_dwordx4 s[0:3], s[6:7], 0x0                         // 000000001010: C00A0003 00000000
	v_mov_b32_e32 v3, 0                                        // 000000001018: 7E060280
	s_waitcnt lgkmcnt(0)                                       // 00000000101C: BF8CC07F
	s_lshr_b32 s4, s12, 16                                     // 000000001020: 8F04900C
	s_mul_i32 s9, s9, s4                                       // 000000001024: 92090409
	v_add_u32_e32 v1, s9, v1                                   // 000000001028: 68020209
	v_mul_lo_u32 v4, v1, s10                                   // 00000000102C: D2850004 00001501
	s_and_b32 s4, s12, 0xffff                                  // 000000001034: 8604FF0C 0000FFFF
	s_mul_i32 s8, s8, s4                                       // 00000000103C: 92080408
	v_add_u32_e32 v6, s8, v0                                   // 000000001040: 680C0008
	v_add_u32_e32 v2, v4, v6                                   // 000000001044: 68040D04
	v_lshlrev_b64 v[0:1], 2, v[2:3]                            // 000000001048: D28F0000 00020482
	v_mov_b32_e32 v5, s1                                       // 000000001050: 7E0A0201
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 000000001054: 32000000
	v_addc_co_u32_e32 v1, vcc, v5, v1, vcc                     // 000000001058: 38020305
	v_add_u32_e32 v4, s11, v4                                  // 00000000105C: 6808080B
	v_mov_b32_e32 v5, v3                                       // 000000001060: 7E0A0303
	v_lshlrev_b64 v[4:5], 2, v[4:5]                            // 000000001064: D28F0004 00020882
	v_mov_b32_e32 v7, s1                                       // 00000000106C: 7E0E0201
	v_add_co_u32_e32 v4, vcc, s0, v4                           // 000000001070: 32080800
	s_mul_i32 s4, s11, s10                                     // 000000001074: 92040A0B
	v_addc_co_u32_e32 v5, vcc, v7, v5, vcc                     // 000000001078: 380A0B07
	v_add_u32_e32 v6, s4, v6                                   // 00000000107C: 680C0C04
	v_mov_b32_e32 v7, v3                                       // 000000001080: 7E0E0303
	v_lshlrev_b64 v[6:7], 2, v[6:7]                            // 000000001084: D28F0006 00020C82
	v_mov_b32_e32 v8, s1                                       // 00000000108C: 7E100201
	v_add_co_u32_e32 v6, vcc, s0, v6                           // 000000001090: 320C0C00
	v_addc_co_u32_e32 v7, vcc, v8, v7, vcc                     // 000000001094: 380E0F08
	global_load_dword v8, v[4:5], off                          // 000000001098: DC508000 087F0004
	global_load_dword v9, v[6:7], off                          // 0000000010A0: DC508000 097F0006
	global_load_dword v10, v[0:1], off                         // 0000000010A8: DC508000 0A7F0000
	s_waitcnt vmcnt(1)                                         // 0000000010B0: BF8C0F71
	v_add_u32_e32 v4, v9, v8                                   // 0000000010B4: 68081109
	s_waitcnt vmcnt(0)                                         // 0000000010B8: BF8C0F70
	v_cmp_lt_i32_e32 vcc, v4, v10                              // 0000000010BC: 7D821504
	s_and_saveexec_b64 s[0:1], vcc                             // 0000000010C0: BE80206A
	s_cbranch_execz 10                                         // 0000000010C4: BF88000A <_Z17floydWarshallPassPjS_jj+0xf0>
	v_lshlrev_b64 v[2:3], 2, v[2:3]                            // 0000000010C8: D28F0002 00020482
	v_mov_b32_e32 v5, s3                                       // 0000000010D0: 7E0A0203
	v_add_co_u32_e32 v2, vcc, s2, v2                           // 0000000010D4: 32040402
	v_addc_co_u32_e32 v3, vcc, v5, v3, vcc                     // 0000000010D8: 38060705
	global_store_dword v[0:1], v4, off                         // 0000000010DC: DC708000 007F0400
	v_mov_b32_e32 v0, s11                                      // 0000000010E4: 7E00020B
	global_store_dword v[2:3], v0, off                         // 0000000010E8: DC708000 007F0002
	s_endpgm                                                   // 0000000010F0: BF810000
