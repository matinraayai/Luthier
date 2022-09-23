
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <_Z10helloworldPcS_>:
	s_load_dword s9, s[4:5], 0x4                               // 000000001000: C0020242 00000004
	s_load_dwordx4 s[0:3], s[6:7], 0x0                         // 000000001008: C00A0003 00000000
	s_waitcnt lgkmcnt(0)                                       // 000000001010: BF8CC07F
	s_and_b32 s4, s9, 0xffff                                   // 000000001014: 8604FF09 0000FFFF
	s_mul_i32 s4, s4, s8                                       // 00000000101C: 92040804
	v_add_u32_e32 v2, s4, v0                                   // 000000001020: 68040004
	v_ashrrev_i32_e32 v3, 31, v2                               // 000000001024: 2206049F
	v_mov_b32_e32 v1, s1                                       // 000000001028: 7E020201
	v_add_co_u32_e32 v0, vcc, s0, v2                           // 00000000102C: 32000400
	v_addc_co_u32_e32 v1, vcc, v1, v3, vcc                     // 000000001030: 38020701
	global_load_ubyte v0, v[0:1], off                          // 000000001034: DC408000 007F0000
	v_mov_b32_e32 v1, s3                                       // 00000000103C: 7E020203
	s_waitcnt vmcnt(0)                                         // 000000001040: BF8C0F70
	v_add_u16_e32 v4, 1, v0                                    // 000000001044: 4C080081
	v_add_co_u32_e32 v0, vcc, s2, v2                           // 000000001048: 32000402
	v_addc_co_u32_e32 v1, vcc, v1, v3, vcc                     // 00000000104C: 38020701
	global_store_byte v[0:1], v4, off                          // 000000001050: DC608000 007F0400
	s_endpgm                                                   // 000000001058: BF810000
