
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001000 <_Z15group_prefixSumPfS_jj>:
	s_load_dword s0, s[4:5], 0x4                               // 000000001000: C0020002 00000004
	s_load_dwordx2 s[10:11], s[6:7], 0x10                      // 000000001008: C0060283 00000010
	v_lshlrev_b32_e32 v5, 3, v0                                // 000000001010: 240A0083
	s_load_dwordx4 s[4:7], s[6:7], 0x0                         // 000000001014: C00A0103 00000000
	s_waitcnt lgkmcnt(0)                                       // 00000000101C: BF8CC07F
	s_and_b32 s0, s0, 0xffff                                   // 000000001020: 8600FF00 0000FFFF
	s_mul_i32 s8, s8, s0                                       // 000000001028: 92080008
	v_add_u32_e32 v1, s8, v0                                   // 00000000102C: 68020008
	v_lshl_or_b32 v1, v1, 1, 1                                 // 000000001030: D2000001 02050301
	v_mul_lo_u32 v1, v1, s11                                   // 000000001038: D2850001 00001701
	v_add_u32_e32 v1, -1, v1                                   // 000000001040: 680202C1
	v_cmp_gt_u32_e64 s[0:1], s10, v1                           // 000000001044: D0CC0000 0002020A
	v_ashrrev_i32_e32 v2, 31, v1                               // 00000000104C: 2204029F
	s_and_saveexec_b64 s[2:3], s[0:1]                          // 000000001050: BE822000
	s_cbranch_execz 10                                         // 000000001054: BF88000A <_Z15group_prefixSumPfS_jj+0x80>
	v_lshlrev_b64 v[3:4], 2, v[1:2]                            // 000000001058: D28F0003 00020282
	v_mov_b32_e32 v6, s7                                       // 000000001060: 7E0C0207
	v_add_co_u32_e32 v3, vcc, s6, v3                           // 000000001064: 32060606
	v_addc_co_u32_e32 v4, vcc, v6, v4, vcc                     // 000000001068: 38080906
	global_load_dword v3, v[3:4], off                          // 00000000106C: DC508000 037F0003
	s_waitcnt vmcnt(0)                                         // 000000001074: BF8C0F70
	ds_write_b32 v5, v3                                        // 000000001078: D81A0000 00000305
	s_or_b64 exec, exec, s[2:3]                                // 000000001080: 87FE027E
	v_add_u32_e32 v3, s11, v1                                  // 000000001084: 6806020B
	v_cmp_gt_u32_e64 s[2:3], s10, v3                           // 000000001088: D0CC0002 0002060A
	v_mov_b32_e32 v4, 0                                        // 000000001090: 7E080280
	s_and_saveexec_b64 s[8:9], s[2:3]                          // 000000001094: BE882002
	s_cbranch_execz 10                                         // 000000001098: BF88000A <_Z15group_prefixSumPfS_jj+0xc4>
	v_lshlrev_b64 v[6:7], 2, v[3:4]                            // 00000000109C: D28F0006 00020682
	v_mov_b32_e32 v8, s7                                       // 0000000010A4: 7E100207
	v_add_co_u32_e32 v6, vcc, s6, v6                           // 0000000010A8: 320C0C06
	v_addc_co_u32_e32 v7, vcc, v8, v7, vcc                     // 0000000010AC: 380E0F08
	global_load_dword v6, v[6:7], off                          // 0000000010B0: DC508000 067F0006
	s_waitcnt vmcnt(0)                                         // 0000000010B8: BF8C0F70
	ds_write_b32 v5, v6 offset:4                               // 0000000010BC: D81A0004 00000605
	s_or_b64 exec, exec, s[8:9]                                // 0000000010C4: 87FE087E
	s_cmp_lt_u32 s10, 2                                        // 0000000010C8: BF0A820A
	s_mov_b32 s8, 1                                            // 0000000010CC: BE880081
	s_cbranch_scc1 31                                          // 0000000010D0: BF85001F <_Z15group_prefixSumPfS_jj+0x150>
	v_lshlrev_b32_e32 v6, 1, v0                                // 0000000010D4: 240C0081
	s_mov_b32 s8, 1                                            // 0000000010D8: BE880081
	v_or_b32_e32 v6, 1, v6                                     // 0000000010DC: 280C0C81
	s_mov_b32 s9, s10                                          // 0000000010E0: BE89000A
	s_lshr_b32 s11, s9, 1                                      // 0000000010E4: 8F0B8109
	v_cmp_gt_u32_e32 vcc, s11, v0                              // 0000000010E8: 7D98000B
	s_waitcnt lgkmcnt(0)                                       // 0000000010EC: BF8CC07F
	s_barrier                                                  // 0000000010F0: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 0000000010F4: BF8CC07F
	s_and_saveexec_b64 s[6:7], vcc                             // 0000000010F8: BE86206A
	s_cbranch_execz 14                                         // 0000000010FC: BF88000E <_Z15group_prefixSumPfS_jj+0x138>
	v_mul_lo_u32 v7, s8, v6                                    // 000000001100: D2850007 00020C08
	v_lshl_add_u32 v7, v7, 2, -4                               // 000000001108: D1FD0007 03110507
	v_lshl_add_u32 v8, s8, 2, v7                               // 000000001110: D1FD0008 041D0408
	ds_read_b32 v7, v7                                         // 000000001118: D86C0000 07000007
	ds_read_b32 v9, v8                                         // 000000001120: D86C0000 09000008
	s_waitcnt lgkmcnt(0)                                       // 000000001128: BF8CC07F
	v_add_f32_e32 v7, v7, v9                                   // 00000000112C: 020E1307
	ds_write_b32 v8, v7                                        // 000000001130: D81A0000 00000708
	s_or_b64 exec, exec, s[6:7]                                // 000000001138: 87FE067E
	s_lshl_b32 s8, s8, 1                                       // 00000000113C: 8E088108
	s_cmp_lt_u32 s9, 4                                         // 000000001140: BF0A8409
	s_cbranch_scc1 2                                           // 000000001144: BF850002 <_Z15group_prefixSumPfS_jj+0x150>
	s_mov_b32 s9, s11                                          // 000000001148: BE89000B
	s_branch 65509                                             // 00000000114C: BF82FFE5 <_Z15group_prefixSumPfS_jj+0xe4>
	s_cmp_gt_u32 s10, 2                                        // 000000001150: BF08820A
	s_cbranch_scc0 42                                          // 000000001154: BF84002A <_Z15group_prefixSumPfS_jj+0x200>
	s_cmp_lt_u32 s8, s10                                       // 000000001158: BF0A0A08
	s_cselect_b64 s[6:7], -1, 0                                // 00000000115C: 858680C1
	v_cndmask_b32_e64 v6, 0, 1, s[6:7]                         // 000000001160: D1000006 00190280
	v_lshlrev_b32_e64 v8, v6, s8                               // 000000001168: D1120008 00001106
	v_cmp_gt_i32_e32 vcc, 2, v8                                // 000000001170: 7D881082
	s_and_b64 vcc, exec, vcc                                   // 000000001174: 86EA6A7E
	s_cbranch_vccnz 33                                         // 000000001178: BF870021 <_Z15group_prefixSumPfS_jj+0x200>
	v_ashrrev_i32_e32 v6, 1, v8                                // 00000000117C: 220C1081
	v_add_u32_e32 v7, 1, v0                                    // 000000001180: 680E0081
	s_mov_b32 s6, 0                                            // 000000001184: BE860080
	s_branch 6                                                 // 000000001188: BF820006 <_Z15group_prefixSumPfS_jj+0x1a4>
	s_or_b64 exec, exec, s[6:7]                                // 00000000118C: 87FE067E
	s_lshl_b32 s6, s8, 1                                       // 000000001190: 8E068108
	v_cmp_lt_i32_e32 vcc, s6, v6                               // 000000001194: 7D820C06
	s_and_b64 vcc, exec, vcc                                   // 000000001198: 86EA6A7E
	v_mov_b32_e32 v8, v9                                       // 00000000119C: 7E100309
	s_cbranch_vccz 23                                          // 0000000011A0: BF860017 <_Z15group_prefixSumPfS_jj+0x200>
	s_or_b32 s8, s6, 1                                         // 0000000011A4: 87088106
	v_ashrrev_i32_e32 v9, 1, v8                                // 0000000011A8: 22121081
	v_cmp_gt_i32_e32 vcc, s8, v0                               // 0000000011AC: 7D880008
	s_waitcnt lgkmcnt(0)                                       // 0000000011B0: BF8CC07F
	s_barrier                                                  // 0000000011B4: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 0000000011B8: BF8CC07F
	s_and_saveexec_b64 s[6:7], vcc                             // 0000000011BC: BE86206A
	s_cbranch_execz 65522                                      // 0000000011C0: BF88FFF2 <_Z15group_prefixSumPfS_jj+0x18c>
	v_mul_lo_u32 v10, v9, v7                                   // 0000000011C4: D285000A 00020F09
	v_and_b32_e32 v8, -4, v8                                   // 0000000011CC: 261010C4
	v_lshl_add_u32 v10, v10, 2, -4                             // 0000000011D0: D1FD000A 0311050A
	v_add_u32_e32 v8, v10, v8                                  // 0000000011D8: 6810110A
	ds_read_b32 v10, v10                                       // 0000000011DC: D86C0000 0A00000A
	ds_read_b32 v11, v8                                        // 0000000011E4: D86C0000 0B000008
	s_waitcnt lgkmcnt(0)                                       // 0000000011EC: BF8CC07F
	v_add_f32_e32 v10, v10, v11                                // 0000000011F0: 0214170A
	ds_write_b32 v8, v10                                       // 0000000011F4: D81A0000 00000A08
	s_branch 65507                                             // 0000000011FC: BF82FFE3 <_Z15group_prefixSumPfS_jj+0x18c>
	s_waitcnt lgkmcnt(0)                                       // 000000001200: BF8CC07F
	s_barrier                                                  // 000000001204: BF8A0000
	s_waitcnt lgkmcnt(0)                                       // 000000001208: BF8CC07F
	s_and_saveexec_b64 s[6:7], s[0:1]                          // 00000000120C: BE862000
	s_cbranch_execnz 4                                         // 000000001210: BF890004 <_Z15group_prefixSumPfS_jj+0x224>
	s_or_b64 exec, exec, s[6:7]                                // 000000001214: 87FE067E
	s_and_saveexec_b64 s[0:1], s[2:3]                          // 000000001218: BE802002
	s_cbranch_execnz 14                                        // 00000000121C: BF89000E <_Z15group_prefixSumPfS_jj+0x258>
	s_endpgm                                                   // 000000001220: BF810000
	ds_read_b32 v6, v5                                         // 000000001224: D86C0000 06000005
	v_lshlrev_b64 v[0:1], 2, v[1:2]                            // 00000000122C: D28F0000 00020282
	v_mov_b32_e32 v2, s5                                       // 000000001234: 7E040205
	v_add_co_u32_e32 v0, vcc, s4, v0                           // 000000001238: 32000004
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc                     // 00000000123C: 38020302
	s_waitcnt lgkmcnt(0)                                       // 000000001240: BF8CC07F
	global_store_dword v[0:1], v6, off                         // 000000001244: DC708000 007F0600
	s_or_b64 exec, exec, s[6:7]                                // 00000000124C: 87FE067E
	s_and_saveexec_b64 s[0:1], s[2:3]                          // 000000001250: BE802002
	s_cbranch_execz 65522                                      // 000000001254: BF88FFF2 <_Z15group_prefixSumPfS_jj+0x220>
	v_lshlrev_b64 v[0:1], 2, v[3:4]                            // 000000001258: D28F0000 00020682
	ds_read_b32 v3, v5 offset:4                                // 000000001260: D86C0004 03000005
	v_mov_b32_e32 v2, s5                                       // 000000001268: 7E040205
	v_add_co_u32_e32 v0, vcc, s4, v0                           // 00000000126C: 32000004
	v_addc_co_u32_e32 v1, vcc, v2, v1, vcc                     // 000000001270: 38020302
	s_waitcnt lgkmcnt(0)                                       // 000000001274: BF8CC07F
	global_store_dword v[0:1], v3, off                         // 000000001278: DC708000 007F0300
	s_endpgm                                                   // 000000001280: BF810000
	s_nop 0                                                    // 000000001284: BF800000
	s_nop 0                                                    // 000000001288: BF800000
	s_nop 0                                                    // 00000000128C: BF800000
	s_nop 0                                                    // 000000001290: BF800000
	s_nop 0                                                    // 000000001294: BF800000
	s_nop 0                                                    // 000000001298: BF800000
	s_nop 0                                                    // 00000000129C: BF800000
	s_nop 0                                                    // 0000000012A0: BF800000
	s_nop 0                                                    // 0000000012A4: BF800000
	s_nop 0                                                    // 0000000012A8: BF800000
	s_nop 0                                                    // 0000000012AC: BF800000
	s_nop 0                                                    // 0000000012B0: BF800000
	s_nop 0                                                    // 0000000012B4: BF800000
	s_nop 0                                                    // 0000000012B8: BF800000
	s_nop 0                                                    // 0000000012BC: BF800000
	s_nop 0                                                    // 0000000012C0: BF800000
	s_nop 0                                                    // 0000000012C4: BF800000
	s_nop 0                                                    // 0000000012C8: BF800000
	s_nop 0                                                    // 0000000012CC: BF800000
	s_nop 0                                                    // 0000000012D0: BF800000
	s_nop 0                                                    // 0000000012D4: BF800000
	s_nop 0                                                    // 0000000012D8: BF800000
	s_nop 0                                                    // 0000000012DC: BF800000
	s_nop 0                                                    // 0000000012E0: BF800000
	s_nop 0                                                    // 0000000012E4: BF800000
	s_nop 0                                                    // 0000000012E8: BF800000
	s_nop 0                                                    // 0000000012EC: BF800000
	s_nop 0                                                    // 0000000012F0: BF800000
	s_nop 0                                                    // 0000000012F4: BF800000
	s_nop 0                                                    // 0000000012F8: BF800000
	s_nop 0                                                    // 0000000012FC: BF800000

0000000000001300 <_Z16global_prefixSumPfjj>:
	s_load_dword s0, s[4:5], 0x4                               // 000000001300: C0020002 00000004
	s_mov_b32 s4, 0x4f7ffffe                                   // 000000001308: BE8400FF 4F7FFFFE
	s_load_dwordx2 s[2:3], s[6:7], 0x8                         // 000000001310: C0060083 00000008
	s_waitcnt lgkmcnt(0)                                       // 000000001318: BF8CC07F
	s_and_b32 s0, s0, 0xffff                                   // 00000000131C: 8600FF00 0000FFFF
	v_cvt_f32_u32_e32 v1, s0                                   // 000000001324: 7E020C00
	s_sub_i32 s1, 0, s0                                        // 000000001328: 81810080
	v_cvt_f32_u32_e32 v5, s2                                   // 00000000132C: 7E0A0C02
	v_rcp_iflag_f32_e32 v1, v1                                 // 000000001330: 7E024701
	v_rcp_iflag_f32_e32 v5, v5                                 // 000000001334: 7E0A4705
	v_mul_f32_e32 v1, s4, v1                                   // 000000001338: 0A020204
	v_cvt_u32_f32_e32 v1, v1                                   // 00000000133C: 7E020F01
	v_mul_lo_u32 v2, s1, v1                                    // 000000001340: D2850002 00020201
	s_lshl_b32 s1, s2, 1                                       // 000000001348: 8E018102
	v_mul_hi_u32 v2, v1, v2                                    // 00000000134C: D2860002 00020501
	v_add_u32_e32 v1, v1, v2                                   // 000000001354: 68020501
	v_mul_hi_u32 v1, s2, v1                                    // 000000001358: D2860001 00020202
	v_mul_lo_u32 v2, v1, s0                                    // 000000001360: D2850002 00000101
	v_add_u32_e32 v3, 1, v1                                    // 000000001368: 68060281
	v_sub_u32_e32 v2, s2, v2                                   // 00000000136C: 6A040402
	v_cmp_le_u32_e32 vcc, s0, v2                               // 000000001370: 7D960400
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 000000001374: 00020701
	v_subrev_u32_e32 v3, s0, v2                                // 000000001378: 6C060400
	v_cndmask_b32_e32 v2, v2, v3, vcc                          // 00000000137C: 00040702
	v_add_u32_e32 v3, 1, v1                                    // 000000001380: 68060281
	v_cmp_le_u32_e32 vcc, s0, v2                               // 000000001384: 7D960400
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 000000001388: 00020701
	v_sub_u32_e32 v2, s1, v1                                   // 00000000138C: 6A040201
	v_cvt_f32_u32_e32 v3, v2                                   // 000000001390: 7E060D02
	v_sub_u32_e32 v4, 0, v2                                    // 000000001394: 6A080480
	s_sub_i32 s1, 0, s2                                        // 000000001398: 81810280
	v_rcp_iflag_f32_e32 v3, v3                                 // 00000000139C: 7E064703
	v_mul_f32_e32 v3, s4, v3                                   // 0000000013A0: 0A060604
	v_cvt_u32_f32_e32 v3, v3                                   // 0000000013A4: 7E060F03
	v_mul_lo_u32 v4, v4, v3                                    // 0000000013A8: D2850004 00020704
	v_mul_hi_u32 v4, v3, v4                                    // 0000000013B0: D2860004 00020903
	v_add_u32_e32 v3, v3, v4                                   // 0000000013B8: 68060903
	v_mul_hi_u32 v3, s8, v3                                    // 0000000013BC: D2860003 00020608
	v_mul_f32_e32 v4, s4, v5                                   // 0000000013C4: 0A080A04
	v_cvt_u32_f32_e32 v4, v4                                   // 0000000013C8: 7E080F04
	v_mul_lo_u32 v5, v3, v2                                    // 0000000013CC: D2850005 00020503
	v_add_u32_e32 v7, 1, v3                                    // 0000000013D4: 680E0681
	v_mul_lo_u32 v6, s1, v4                                    // 0000000013D8: D2850006 00020801
	v_sub_u32_e32 v5, s8, v5                                   // 0000000013E0: 6A0A0A08
	v_cmp_ge_u32_e32 vcc, v5, v2                               // 0000000013E4: 7D9C0505
	v_cndmask_b32_e32 v3, v3, v7, vcc                          // 0000000013E8: 00060F03
	v_sub_u32_e32 v7, v5, v2                                   // 0000000013EC: 6A0E0505
	v_cndmask_b32_e32 v5, v5, v7, vcc                          // 0000000013F0: 000A0F05
	v_add_u32_e32 v7, 1, v3                                    // 0000000013F4: 680E0681
	v_cmp_ge_u32_e32 vcc, v5, v2                               // 0000000013F8: 7D9C0505
	v_cndmask_b32_e32 v2, v3, v7, vcc                          // 0000000013FC: 00040F03
	v_add_u32_e32 v2, 1, v2                                    // 000000001400: 68040481
	v_mul_lo_u32 v1, v2, v1                                    // 000000001404: D2850001 00020302
	v_mul_hi_u32 v2, v4, v6                                    // 00000000140C: D2860002 00020D04
	v_add_u32_e32 v1, s8, v1                                   // 000000001414: 68020208
	v_mul_lo_u32 v3, v1, s0                                    // 000000001418: D2850003 00000101
	v_add_u32_e32 v1, v4, v2                                   // 000000001420: 68020504
	v_add_u32_e32 v0, v3, v0                                   // 000000001424: 68000103
	v_add_u32_e32 v2, 1, v0                                    // 000000001428: 68040081
	v_mul_hi_u32 v3, v2, v1                                    // 00000000142C: D2860003 00020302
	v_cmp_gt_u32_e64 s[0:1], s3, v0                            // 000000001434: D0CC0000 00020003
	v_mul_lo_u32 v3, v3, s2                                    // 00000000143C: D2850003 00000503
	v_sub_u32_e32 v2, v2, v3                                   // 000000001444: 6A040702
	v_subrev_u32_e32 v3, s2, v2                                // 000000001448: 6C060402
	v_cmp_le_u32_e32 vcc, s2, v2                               // 00000000144C: 7D960402
	v_cndmask_b32_e32 v2, v2, v3, vcc                          // 000000001450: 00040702
	v_subrev_u32_e32 v3, s2, v2                                // 000000001454: 6C060402
	v_cmp_le_u32_e32 vcc, s2, v2                               // 000000001458: 7D960402
	v_cndmask_b32_e32 v2, v2, v3, vcc                          // 00000000145C: 00040702
	v_cmp_ne_u32_e32 vcc, 0, v2                                // 000000001460: 7D9A0480
	s_and_b64 s[0:1], vcc, s[0:1]                              // 000000001464: 8680006A
	s_and_saveexec_b64 s[4:5], s[0:1]                          // 000000001468: BE842000
	s_cbranch_execz 35                                         // 00000000146C: BF880023 <_Z16global_prefixSumPfjj+0x1fc>
	v_mul_hi_u32 v1, v0, v1                                    // 000000001470: D2860001 00020300
	s_load_dwordx2 s[0:1], s[6:7], 0x0                         // 000000001478: C0060003 00000000
	v_mov_b32_e32 v2, 0                                        // 000000001480: 7E040280
	v_mul_lo_u32 v1, v1, s2                                    // 000000001484: D2850001 00000501
	s_waitcnt lgkmcnt(0)                                       // 00000000148C: BF8CC07F
	v_mov_b32_e32 v5, s1                                       // 000000001490: 7E0A0201
	v_sub_u32_e32 v1, v0, v1                                   // 000000001494: 6A020300
	v_subrev_u32_e32 v3, s2, v1                                // 000000001498: 6C060202
	v_cmp_le_u32_e32 vcc, s2, v1                               // 00000000149C: 7D960202
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 0000000014A0: 00020701
	v_subrev_u32_e32 v3, s2, v1                                // 0000000014A4: 6C060202
	v_cmp_le_u32_e32 vcc, s2, v1                               // 0000000014A8: 7D960202
	v_cndmask_b32_e32 v1, v1, v3, vcc                          // 0000000014AC: 00020701
	v_xad_u32 v1, v1, -1, v0                                   // 0000000014B0: D1F30001 04018301
	v_lshlrev_b64 v[1:2], 2, v[1:2]                            // 0000000014B8: D28F0001 00020282
	v_add_co_u32_e32 v3, vcc, s0, v1                           // 0000000014C0: 32060200
	v_ashrrev_i32_e32 v1, 31, v0                               // 0000000014C4: 2202009F
	v_lshlrev_b64 v[0:1], 2, v[0:1]                            // 0000000014C8: D28F0000 00020082
	v_addc_co_u32_e32 v4, vcc, v5, v2, vcc                     // 0000000014D0: 38080505
	v_add_co_u32_e32 v0, vcc, s0, v0                           // 0000000014D4: 32000000
	v_addc_co_u32_e32 v1, vcc, v5, v1, vcc                     // 0000000014D8: 38020305
	global_load_dword v2, v[3:4], off                          // 0000000014DC: DC508000 027F0003
	global_load_dword v5, v[0:1], off                          // 0000000014E4: DC508000 057F0000
	s_waitcnt vmcnt(0)                                         // 0000000014EC: BF8C0F70
	v_add_f32_e32 v2, v2, v5                                   // 0000000014F0: 02040B02
	global_store_dword v[0:1], v2, off                         // 0000000014F4: DC708000 007F0200
	s_endpgm                                                   // 0000000014FC: BF810000
