s_getpc_b64 s[0:1]                                         // 000000001000: BE801C00
s_add_u32 s0, s0, 0x2004                                   // 000000001004: 8000FF00 00002004
s_addc_u32 s1, s1, 0                                       // 00000000100C: 8201FF01 00000000
s_load_dwordx2 s[0:1], s[0:1], 0x0                         // 000000001014: C0060000 00000000
v_mov_b32_e32 v0, 1                                        // 00000000101C: 7E000281
v_mov_b32_e32 v1, 0                                        // 000000001020: 7E020280
v_mov_b32_e32 v2, 0                                        // 000000001024: 7E040280
s_waitcnt lgkmcnt(0)                                       // 000000001028: BF8CC07F
global_atomic_add_x2 v2, v[0:1], s[0:1]                    // 00000000102C: DD888000 00000002
s_endpgm                                                   // 000000001034: BF810000