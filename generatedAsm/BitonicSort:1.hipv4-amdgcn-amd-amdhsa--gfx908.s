
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <_Z11bitonicSortPjjjj>:
	s_load_dwordx4 s[0:3], s[6:7], 0x8                         // 000000001000: C00A0003 00000008
	s_waitcnt lgkmcnt(0)                                       // 000000001008: BF8CC07F
	s_load_dword s3, s[4:5], 0x4                               // 00000000100C: C00200C2 00000004
	s_sub_i32 s1, s0, s1                                       // 000000001014: 81810100
	s_waitcnt lgkmcnt(0)                                       // 000000001018: BF8CC07F
	s_and_b32 s3, s3, 0xffff                                   // 00000000101C: 8603FF03 0000FFFF
	s_mul_i32 s8, s8, s3                                       // 000000001024: 92080308
	v_add_u32_e32 v7, s8, v0                                   // 000000001028: 680E0008
	s_lshl_b32 s3, 1, s1                                       // 00000000102C: 8E030181
	v_lshrrev_b32_e32 v0, s1, v7                               // 000000001030: 20000E01
	v_mul_lo_u32 v0, s3, v0                                    // 000000001034: D2850000 00020003
	s_load_dwordx2 s[4:5], s[6:7], 0x0                         // 00000000103C: C0060103 00000000
	s_bfm_b32 s1, s1, 0                                        // 000000001044: 91018001
	v_and_b32_e32 v1, s1, v7                                   // 000000001048: 26020E01
	v_lshl_add_u32 v0, v0, 1, v1                               // 00000000104C: D1FD0000 04050300
	v_mov_b32_e32 v1, 0                                        // 000000001054: 7E020280
	v_lshlrev_b64 v[3:4], 2, v[0:1]                            // 000000001058: D28F0003 00020082
	v_add_u32_e32 v2, s3, v0                                   // 000000001060: 68040003
	s_waitcnt lgkmcnt(0)                                       // 000000001064: BF8CC07F
	v_add_co_u32_e32 v5, vcc, s4, v3                           // 000000001068: 320A0604
	v_mov_b32_e32 v3, v1                                       // 00000000106C: 7E060301
	v_mov_b32_e32 v8, s5                                       // 000000001070: 7E100205
	v_lshlrev_b64 v[0:1], 2, v[2:3]                            // 000000001074: D28F0000 00020482
	v_addc_co_u32_e32 v6, vcc, v8, v4, vcc                     // 00000000107C: 380C0908
	v_add_co_u32_e32 v0, vcc, s4, v0                           // 000000001080: 32000004
	v_addc_co_u32_e32 v1, vcc, v8, v1, vcc                     // 000000001084: 38020308
	global_load_dword v2, v[5:6], off                          // 000000001088: DC508000 027F0005
	global_load_dword v3, v[0:1], off                          // 000000001090: DC508000 037F0000
	s_sub_i32 s1, 1, s2                                        // 000000001098: 81810281
	v_bfe_u32 v7, v7, s0, 1                                    // 00000000109C: D1C80007 02040107
	v_mov_b32_e32 v4, s2                                       // 0000000010A4: 7E080202
	v_mov_b32_e32 v8, s1                                       // 0000000010A8: 7E100201
	v_cmp_eq_u32_e32 vcc, 0, v7                                // 0000000010AC: 7D940E80
	v_cndmask_b32_e32 v4, v8, v4, vcc                          // 0000000010B0: 00080908
	v_cmp_eq_u32_e32 vcc, 0, v4                                // 0000000010B4: 7D940880
	s_waitcnt vmcnt(0)                                         // 0000000010B8: BF8C0F70
	v_max_u32_e32 v7, v2, v3                                   // 0000000010BC: 1E0E0702
	v_min_u32_e32 v2, v2, v3                                   // 0000000010C0: 1C040702
	v_cndmask_b32_e32 v3, v2, v7, vcc                          // 0000000010C4: 00060F02
	v_cndmask_b32_e32 v2, v7, v2, vcc                          // 0000000010C8: 00040507
	global_store_dword v[5:6], v3, off                         // 0000000010CC: DC708000 007F0305
	global_store_dword v[0:1], v2, off                         // 0000000010D4: DC708000 007F0200
	s_endpgm                                                   // 0000000010DC: BF810000
