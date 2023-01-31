#include "disassembler.h"
#include <memory>
void Disassembler::initializeDecodeTable() {
  // SOP2 instructions
  addInstType(
      {"s_add_u32", 0, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_sub_u32", 1, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_add_i32", 2, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_sub_i32", 3, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_addc_u32", 4, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_subb_u32", 5, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_min_i32", 6, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_min_u32", 7, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_max_i32", 8, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_max_u32", 9, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_cselect_b32", 10, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cselect_b64", 11, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType(
      {"s_and_b32", 12, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_and_b64", 13, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_or_b32", 14, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_or_b64", 15, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_xor_b32", 16, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_xor_b64", 17, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_andn2_b32", 18, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_andn2_b64", 19, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_orn2_b32", 20, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_orn2_b64", 21, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_nand_b32", 22, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_nand_b64", 23, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType(
      {"s_nor_b32", 24, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_nor_b64", 25, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_xnor_b32", 26, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_xnor_b64", 27, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_lshl_b32", 28, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_lshl_b64", 29, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_lshr_b32", 30, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_lshr_b64", 31, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_ashr_i32", 32, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_ashr_i64", 33, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType(
      {"s_bfm_b32", 34, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_bfm_b64", 35, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_mul_i32", 36, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_bfe_u32", 37, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_bfe_i32", 38, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_bfe_u64", 39, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType(
      {"s_bfe_i64", 40, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_cbranch_g_fork", 41, FormatTable[SOP2], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_absdiss_i32", 42, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_rfe_restore_b64", 43, FormatTable[SOP2], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_mul_hi_u32", 44, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_mul_hi_i32", 45, FormatTable[SOP2], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_lshl1_add_u32", 46, FormatTable[SOP2], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_lshl2_add_u32", 47, FormatTable[SOP2], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_lshl3_add_u32", 48, FormatTable[SOP2], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_lshl4_add_u32", 49, FormatTable[SOP2], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_pack_ll_b32_b16", 50, FormatTable[SOP2], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_pack_lh_b32_b16", 51, FormatTable[SOP2], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_pack_hh_b32_b16", 52, FormatTable[SOP2], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});

  // VOP2 instructions
  addInstType({"v_cndmask_b32", 0, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_add_f32", 1, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sub_f32", 2, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_subrev_f32", 3, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_mul_legacy_f32", 4, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_mul_f32", 5, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_mul_i32_i24", 6, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_mul_hi_i32_i24", 7, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_mul_u32_u24", 8, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_mul_hi_u32_u24", 9, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_min_f32", 10, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_max_f32", 11, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_min_i32", 12, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_max_i32", 13, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_min_u32", 14, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_max_u32", 15, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_lshrrev_b32", 16, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_ashrrev_i32", 17, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_lshlrev_b32", 18, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_and_b32", 19, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_or_b32", 20, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_xor_b32", 21, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_mac_f32", 22, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_madmk_f32", 23, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_madak_f32", 24, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_add_co_u32", 25, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_sub_co_u32", 26, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_subrev_co_u32", 27, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_addc_co_u32", 28, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_subb_co_u32", 29, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_subbrev_co_u32", 30, FormatTable[VOP2], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType(
      {"v_add_f16", 31, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sub_f16", 32, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_subrev_f16", 33, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_mul_f16", 34, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_mac_f16", 35, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_madmk_f16", 36, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_madak_f16", 37, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_add_u16", 38, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sub_u16", 39, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_subrev_u16", 40, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_mul_lo_u16", 41, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_lshlrev_b16", 42, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_lshrrev_b16", 43, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_ashrrev_i16", 44, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_max_f16", 45, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_min_f16", 46, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_max_u16", 47, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_max_i16", 48, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_min_u16", 49, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_min_i16", 50, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_ldexp_f16", 51, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_add_u32", 52, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sub_u32", 53, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_subrev_u32", 54, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_dot2c_f32_f16", 55, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_dot2c_i32_i16", 56, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_dot4c_i32_i8", 57, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_dot8c_i32_i4", 58, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_fmac_f32", 59, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_pk_fmac_f16", 60, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_xnor_b32", 61, FormatTable[VOP2], 0, ExeUnitVALU, 32, 32, 32, 0, 0});

  // VOP1 instructions
  addInstType(
      {"v_nop", 0, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_mov_b32", 1, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_readfirstlane_b32", 2, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_i32_f64", 3, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_f64_i32", 4, FormatTable[VOP1], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cvt_f32_i32", 5, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_f32_u32", 6, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_u32_f32", 7, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_i32_f32", 8, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_f16_f32", 10, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_f32_f16", 11, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_rpi_i32_f32", 12, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_flr_i32_f32", 13, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_off_f32_i4", 14, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_f32_f64", 15, FormatTable[VOP1], 0, ExeUnitVALU, 32, 64,
               32, 0, 0});
  addInstType({"v_cvt_f64_f32", 16, FormatTable[VOP1], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cvt_f32_ubyte0", 17, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_f32_ubyte1", 18, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_f32_ubyte2", 19, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_f32_ubyte3", 20, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_u32_f64", 21, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_f64_u32", 22, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_trunc_f64", 23, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_ceil_f64", 24, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_rndne_f64", 25, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_floor_f64", 26, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_fract_f32", 27, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_trunc_f32", 28, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_ceil_f32", 29, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_rndne_f32", 30, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_floor_f32", 31, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_exp_f32", 32, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_log_f32", 33, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_rcp_f32", 34, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_rcp_iflag_f32", 35, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_rsq_f32", 36, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_rcp_f64", 37, FormatTable[VOP1], 0, ExeUnitVALU, 64, 64, 32, 0, 0});
  addInstType(
      {"v_rsq_f64", 38, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sqrt_f32", 39, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sqrt_f64", 40, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sin_f32", 41, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_cos_f32", 42, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_not_b32", 43, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_bfrev_b32", 44, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_ffbh_u32", 45, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_ffbl_b32", 46, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_ffbh_i32", 47, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_frexp_exp_i32_f64", 48, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_frexp_mant_f64", 49, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType(
      {"v_fract_f64", 50, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_frexp_exp_i32_f32", 51, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_frexp_mant_f32", 52, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType(
      {"v_clrexcp", 53, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_screen_partition_4se_b32", 55, FormatTable[VOP1], 0,
               ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_cvt_f16_u16", 57, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_f16_i16", 58, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_u16_f16", 59, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cvt_i16_f16", 60, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_rcp_f16", 61, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sqrt_f16", 62, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_rsq_f16", 63, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_log_f16", 64, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_exp_f16", 65, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_frexp_mant_f16", 66, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_frexp_exp_i16_f16", 67, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType(
      {"v_floor_f16", 68, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_ceil_f16", 69, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_trunc_f16", 70, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_rndne_f16", 71, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_fract_f16", 72, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sin_f16", 73, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_cos_f16", 74, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_exp_legacy_f32", 75, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_log_legacy_f32", 76, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_norm_i16_f16", 77, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_norm_u16_f16", 78, FormatTable[VOP1], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_sat_pk_u8_i16", 79, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_swap_b32", 81, FormatTable[VOP1], 0, ExeUnitVALU, 32, 32, 32, 0, 0});

  // FLAT Instructions
  addInstType(
      {"_load_ubyte", 16, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType(
      {"_load_sbyte", 17, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType({"_load_ushort", 18, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_load_sshort", 19, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType(
      {"_load_dword", 20, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType({"_load_dwordx2", 21, FormatTable[FLAT], 0, ExeUnitVMem, 64, 64,
               64, 0, 0});
  addInstType({"_load_dwordx3", 22, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_load_dwordx4", 23, FormatTable[FLAT], 0, ExeUnitVMem, 128, 32,
               32, 0, 0});
  addInstType(
      {"_store_byte", 24, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType({"_store_byte_d16_hi", 25, FormatTable[FLAT], 0, ExeUnitVMem, 32,
               32, 32, 0, 0});
  addInstType({"_store_short", 26, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_store_short_d16_hi", 27, FormatTable[FLAT], 0, ExeUnitVMem, 32,
               32, 32, 0, 0});
  addInstType({"_store_dword", 28, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_store_dwordx2", 29, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_store_dwordx3", 30, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_store_dwordx4", 31, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_load_ubyte_d16", 32, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_load_ubyte_d16_hi", 33, FormatTable[FLAT], 0, ExeUnitVMem, 32,
               32, 32, 0, 0});
  addInstType({"_load_sbyte_d16", 34, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_load_sbyte_d16_hi", 35, FormatTable[FLAT], 0, ExeUnitVMem, 32,
               32, 32, 0, 0});
  addInstType({"_load_short_d16", 36, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_load_short_d16_hi", 37, FormatTable[FLAT], 0, ExeUnitVMem, 32,
               32, 32, 0, 0});
  addInstType({"_atomic_swap", 64, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_cmpswap", 65, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType(
      {"_atomic_add", 66, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType(
      {"_atomic_sub", 67, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType({"_atomic_smin", 68, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_umin", 69, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_smax", 70, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_umax", 71, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType(
      {"_atomic_and", 72, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType(
      {"_atomic_or", 73, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType(
      {"_atomic_xor", 74, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType(
      {"_atomic_inc", 75, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType(
      {"_atomic_dec", 76, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32, 32, 0, 0});
  addInstType({"_atomic_swap_x2", 96, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_cmpswap_x2", 97, FormatTable[FLAT], 0, ExeUnitVMem, 32,
               32, 32, 0, 0});
  addInstType({"_atomic_add_x2", 98, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_sub_x2", 99, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_smin_x2", 100, FormatTable[FLAT], 0, ExeUnitVMem, 32,
               32, 32, 0, 0});
  addInstType({"_atomic_umin_x2", 101, FormatTable[FLAT], 0, ExeUnitVMem, 32,
               32, 32, 0, 0});
  addInstType({"_atomic_smax_x2", 102, FormatTable[FLAT], 0, ExeUnitVMem, 32,
               32, 32, 0, 0});
  addInstType({"_atomic_umax_x2", 103, FormatTable[FLAT], 0, ExeUnitVMem, 32,
               32, 32, 0, 0});
  addInstType({"_atomic_and_x2", 104, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_or_x2", 105, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_xor_x2", 106, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_inc_x2", 107, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});
  addInstType({"_atomic_dec_x2", 108, FormatTable[FLAT], 0, ExeUnitVMem, 32, 32,
               32, 0, 0});

  // SMEM instructions
  addInstType({"s_load_dword", 0, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_load_dwordx2", 1, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_load_dwordx4", 2, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_load_dwordx8", 3, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_load_dwordx16", 4, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_scratch_load_dword", 5, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_scratch_load_dwordx2", 6, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_scratch_load_dwordx4", 7, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_load_dword", 8, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_load_dwordx2", 9, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_load_dwordx4", 10, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_load_dwordx8", 11, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_load_dwordx16", 12, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_store_dword", 16, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_store_dwordx2", 17, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_store_dwordx4", 18, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_scratch_store_dword", 21, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_scratch_store_dwordx2", 22, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_scratch_store_dwordx4", 23, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_store_dword", 24, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_store_dwordx2", 25, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_store_dwordx4", 26, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_dcache_inv", 32, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_dcache_wb", 33, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_dcache_inv_vol", 34, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_dcache_wb_vol", 35, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType(
      {"s_memtime", 36, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_memrealtime", 37, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_atc_probe", 38, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_atc_probe_buffer", 39, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_dcache_discard", 40, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_dcache_discard_x2", 41, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atmoic_swap", 64, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_cmpswap", 65, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_add", 66, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_sub", 67, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_smin", 68, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_umin", 69, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_smax", 70, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_umax", 71, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_and", 72, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_or", 73, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_xor", 74, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_inc", 75, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_dec", 76, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_swap_x2", 96, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_cmpswap_x2", 97, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_add_x2", 98, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_sub_x2", 99, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_smin_x2", 100, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_umin_x2", 101, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_smax_x2", 102, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_umax_x2", 103, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_and_x2", 104, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_or_x2", 105, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_xor_x2", 106, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_inc_x2", 107, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_buffer_atomic_dec_x2", 108, FormatTable[SMEM], 0,
               ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_atomic_swap", 128, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_cmpswap", 129, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_add", 130, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_atomic_sub", 131, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_atomic_smin", 132, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_umin", 133, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_smax", 134, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_umax", 135, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_and", 136, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_atomic_or", 137, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_atomic_xor", 138, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_atomic_inc", 139, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_atomic_dec", 140, FormatTable[SMEM], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_atomic_swap_x2", 160, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_cmpswap_x2", 161, FormatTable[SMEM], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_atomic_add_x2", 162, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_sub_x2", 163, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_smin_x2", 164, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_umin_x2", 165, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_smax_x2", 166, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_umax_x2", 167, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_and_x2", 168, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_or_x2", 169, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_xor_x2", 170, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_inc_x2", 171, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_atomic_dec_x2", 172, FormatTable[SMEM], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});

  // SOPP instructions
  addInstType(
      {"s_nop", 0, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32, 32, 0, 0});
  addInstType(
      {"s_endpgm", 1, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32, 32, 0, 0});
  addInstType(
      {"s_branch", 2, FormatTable[SOPP], 0, ExeUnitBranch, 32, 32, 32, 0, 0});
  addInstType(
      {"s_wakeup", 3, FormatTable[SOPP], 0, ExeUnitBranch, 32, 32, 32, 0, 0});
  addInstType({"s_cbranch_scc0", 4, FormatTable[SOPP], 0, ExeUnitBranch, 32, 32,
               32, 0, 0});
  addInstType({"s_cbranch_scc1", 5, FormatTable[SOPP], 0, ExeUnitBranch, 32, 32,
               32, 0, 0});
  addInstType({"s_cbranch_vccz", 6, FormatTable[SOPP], 0, ExeUnitBranch, 32, 32,
               32, 0, 0});
  addInstType({"s_cbranch_vccnz", 7, FormatTable[SOPP], 0, ExeUnitBranch, 32,
               32, 32, 0, 0});
  addInstType({"s_cbranch_execz", 8, FormatTable[SOPP], 0, ExeUnitBranch, 32,
               32, 32, 0, 0});
  addInstType({"s_cbranch_execnz", 9, FormatTable[SOPP], 0, ExeUnitBranch, 32,
               32, 32, 0, 0});
  addInstType({"s_barrier", 10, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32,
               32, 0, 0});
  addInstType({"s_setkill", 11, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32,
               32, 0, 0});
  addInstType({"s_waitcnt", 12, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32,
               32, 0, 0});
  addInstType({"s_sethalt", 13, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32,
               32, 0, 0});
  addInstType(
      {"s_sleep", 14, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32, 32, 0, 0});
  addInstType({"s_setprio", 15, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32,
               32, 0, 0});
  addInstType(
      {"s_sendmsg", 16, FormatTable[SOPP], 0, ExeUnitBranch, 32, 32, 32, 0, 0});
  addInstType({"s_sendmsghalt", 17, FormatTable[SOPP], 0, ExeUnitBranch, 32, 32,
               32, 0, 0});
  addInstType(
      {"s_trap", 18, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32, 32, 0, 0});
  addInstType({"s_icache_inv", 19, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32,
               32, 0, 0});
  addInstType({"s_incperflevel", 20, FormatTable[SOPP], 0, ExeUnitSpecial, 32,
               32, 32, 0, 0});
  addInstType({"s_decperflevel", 21, FormatTable[SOPP], 0, ExeUnitSpecial, 32,
               32, 32, 0, 0});
  addInstType({"s_ttracedata", 22, FormatTable[SOPP], 0, ExeUnitSpecial, 32, 32,
               32, 0, 0});
  addInstType({"s_cbranch_cdbgsys", 23, FormatTable[SOPP], 0, ExeUnitBranch, 32,
               32, 32, 0, 0});
  addInstType({"s_cbranch_cdbguser", 24, FormatTable[SOPP], 0, ExeUnitBranch,
               32, 32, 32, 0, 0});
  addInstType({"s_cbranch_cdbgsys_or_user", 25, FormatTable[SOPP], 0,
               ExeUnitBranch, 32, 32, 32, 0, 0});
  addInstType({"s_cbranch_cdbgsys_and_user", 26, FormatTable[SOPP], 0,
               ExeUnitBranch, 32, 32, 32, 0, 0});
  addInstType({"s_endpgm_saved", 27, FormatTable[SOPP], 0, ExeUnitSpecial, 32,
               32, 32, 0, 0});
  addInstType({"s_set_gpr_idx_off", 28, FormatTable[SOPP], 0, ExeUnitSpecial,
               32, 32, 32, 0, 0});
  addInstType({"s_set_gpr_idx_mode", 29, FormatTable[SOPP], 0, ExeUnitSpecial,
               32, 32, 32, 0, 0});
  addInstType({"s_endpgm_ordered_ps_done", 30, FormatTable[SOPP], 0,
               ExeUnitSpecial, 32, 32, 32, 0, 0});

  // SOPC instructions
  addInstType({"s_cmp_eq_i32", 0, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_lg_i32", 1, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_gt_i32", 2, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_ge_i32", 3, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_lt_i32", 4, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_le_i32", 5, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_eq_u32", 6, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_lg_u32", 7, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_gt_u32", 8, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_ge_u32", 9, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_lt_u32", 10, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_le_u32", 11, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_bitcmp0_b32", 12, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_bitcmp1_b32", 13, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_bitcmp0_b64", 14, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_bitcmp1_b64", 15, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_setvskip", 16, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_set_gpr_idx_on", 17, FormatTable[SOPC], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_cmp_eq_u64", 18, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmp_ne_u64", 19, FormatTable[SOPC], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});

  // SOPK instructions
  addInstType(
      {"s_movk_i32", 0, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32, 32, 0, 0});
  addInstType({"s_cmovk_i32", 1, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_eq_i32", 2, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_lg_i32", 3, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_gt_i32", 4, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_ge_i32", 5, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_lt_i32", 6, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_le_i32", 7, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_eq_u32", 8, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_lg_u32", 9, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_gt_u32", 10, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_ge_u32", 11, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_lt_u32", 12, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cmpk_le_u32", 13, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_addk_i32", 14, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_mulk_i32", 15, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_cbranch_i_fork", 16, FormatTable[SOPK], 0, ExeUnitScalar, 32,
               32, 32, 0, 0});
  addInstType({"s_getreg_b32", 17, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_setreg_b32", 18, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});
  addInstType({"s_setreg_imm32_b32", 20, FormatTable[SOPK], 0, ExeUnitScalar,
               32, 32, 32, 0, 0});
  addInstType({"s_call_b64", 21, FormatTable[SOPK], 0, ExeUnitScalar, 32, 32,
               32, 0, 0});

  // VOPC instruction
  addInstType({"v_cmp_class_f32", 16, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_class_f32", 17, FormatTable[VOPC], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cmp_class_f64", 18, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_class_f64", 19, FormatTable[VOPC], 0, ExeUnitVALU, 32,
               64, 64, 0, 0});
  addInstType({"v_cmp_class_f16", 20, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_class_f16", 21, FormatTable[VOPC], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType(
      {"v_cmp_f_f16", 32, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_cmp_lt_f16", 33, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_eq_f16", 34, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_le_f16", 35, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_gt_f16", 36, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_lg_f16", 37, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ge_f16", 38, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_cmp_o_f16", 39, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_cmp_u_f16", 40, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_cmp_nge_f16", 41, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_nlg_f16", 42, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ngt_f16", 43, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_nle_f16", 44, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_neq_f16", 45, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_nlt_f16", 46, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_tru_f16", 47, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_f_f16", 48, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lt_f16", 49, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_eq_f16", 50, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_le_f16", 51, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_gt_f16", 52, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lg_f16", 53, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ge_f16", 54, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_o_f16", 55, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_u_f16", 56, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_nge_f16", 57, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_nlg_f16", 58, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ngt_f16", 59, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_nle_f16", 60, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_neq_f16", 61, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_nlt_f16", 62, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_tru_f16", 63, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_cmp_f_f32", 64, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_cmp_lt_f32", 65, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_eq_f32", 66, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_le_f32", 67, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_gt_f32", 68, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_lg_f32", 69, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ge_f32", 70, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_cmp_o_f32", 71, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_cmp_u_f32", 72, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_cmp_nge_f32", 73, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_nlg_f32", 74, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ngt_f32", 75, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_nle_f32", 76, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_neq_f32", 77, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_nlt_f32", 78, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_tru_f32", 79, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_f_f32", 80, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lt_f32", 81, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_eq_f32", 82, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_le_f32", 83, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_gt_f32", 84, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lg_f32", 85, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ge_f32", 86, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_o_f32", 87, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_u_f32", 88, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_nge_f32", 89, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_nlg_f32", 90, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ngt_f32", 91, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_nle_f32", 92, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_neq_f32", 93, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_nlt_f32", 94, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_tru_f32", 95, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_cmp_f_f64", 96, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64, 64, 0, 0});
  addInstType({"v_cmp_lt_f64", 97, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_eq_f64", 98, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_le_f64", 99, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_gt_f64", 100, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_lg_f64", 101, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_ge_f64", 102, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_o_f64", 103, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_u_f64", 104, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_nge_f64", 105, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_nlg_f64", 106, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_ngt_f64", 107, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_nle_f64", 108, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_neq_f64", 109, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_nlt_f64", 110, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_tru_f64", 111, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_f_f64", 112, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_lt_f64", 113, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_eq_f64", 114, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_le_f64", 115, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_gt_f64", 116, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_lg_f64", 117, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_ge_f64", 118, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_o_f64", 119, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_u_f64", 120, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_nge_f64", 121, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_nlg_f64", 122, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_ngt_f64", 123, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_nle_f64", 124, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_neq_f64", 125, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_nlt_f64", 126, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_tru_f64", 127, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_f_i16", 160, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_lt_i16", 161, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_eq_i16", 162, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_le_i16", 163, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_gt_i16", 164, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ne_i16", 165, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ge_i16", 166, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_t_i16", 167, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_f_u16", 168, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_lt_u16", 169, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_eq_u16", 170, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_le_u16", 171, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_gt_u16", 172, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ne_u16", 173, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ge_u16", 174, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_t_u16", 175, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_f_i16", 176, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lt_i16", 177, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_eq_i16", 178, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_le_i16", 179, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_gt_i16", 180, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ne_i16", 181, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ge_i16", 182, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_t_i16", 183, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_f_u16", 184, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lt_u16", 185, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_eq_u16", 186, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_le_u16", 187, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_gt_u16", 188, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ne_u16", 189, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ge_u16", 190, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_t_u16", 191, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_f_i32", 192, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_lt_i32", 193, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_eq_i32", 194, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_le_i32", 195, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_gt_i32", 196, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ne_i32", 197, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ge_i32", 198, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_t_i32", 199, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_f_u32", 200, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_lt_u32", 201, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_eq_u32", 202, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_le_u32", 203, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_gt_u32", 204, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ne_u32", 205, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ge_u32", 206, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_t_u32", 207, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_f_i32", 208, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lt_i32", 209, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_eq_i32", 210, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_le_i32", 211, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_gt_i32", 212, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ne_i32", 213, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ge_i32", 214, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_t_i32", 215, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_f_u32", 216, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lt_u32", 217, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_eq_u32", 218, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_le_u32", 219, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_gt_u32", 220, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ne_u32", 221, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_ge_u32", 222, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmpx_t_u32", 223, FormatTable[VOPC], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_f_i64", 224, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_lt_i64", 225, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_eq_i64", 226, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_le_i64", 227, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_gt_i64", 228, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_ne_i64", 229, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_ge_i64", 230, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_t_i64", 231, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_f_u64", 232, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_lt_u64", 233, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_eq_u64", 234, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_le_u64", 235, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_gt_u64", 236, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_ne_u64", 237, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_ge_u64", 238, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmp_t_u64", 239, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_f_i64", 240, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_lt_i64", 241, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_eq_i64", 242, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_le_i64", 243, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_gt_i64", 244, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_ne_i64", 245, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_ge_i64", 246, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_t_i64", 247, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_f_u64", 248, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_lt_u64", 249, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_eq_u64", 250, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_le_u64", 251, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_gt_u64", 252, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_ne_u64", 253, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_ge_u64", 254, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});
  addInstType({"v_cmpx_t_u64", 255, FormatTable[VOPC], 0, ExeUnitVALU, 32, 64,
               64, 0, 0});

  // VOP3a Instructions
  addInstType({"v_cmp_class_f32", 0x10, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_class_f32", 0x11, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_class_f64", 0x12, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_class_f64", 0x13, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmp_class_f16", 0x14, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_class_f16", 0x15, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_f_f16", 0x20, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_lt_f16", 0x21, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_eq_f16", 0x22, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_le_f16", 0x23, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_gt_f16", 0x24, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_lg_f16", 0x25, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_ge_f16", 0x26, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_o_f16", 0x27, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_u_f16", 0x28, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_nge_f16", 0x29, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_nlg_f16", 0x2a, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_ngt_f16", 0x2b, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_nle_f16", 0x2c, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_neq_f16", 0x2d, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_nlt_f16", 0x2e, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_tru_f16", 0x2f, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_f_f16", 0x30, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lt_f16", 0x31, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_eq_f16", 0x32, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_le_f16", 0x33, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_gt_f16", 0x34, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_lg_f16", 0x35, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_ge_f16", 0x36, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_o_f16", 0x37, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmpx_u_f16", 0x38, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmpx_nge_f16", 0x39, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_nlg_f16", 0x3a, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_ngt_f16", 0x3b, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_nle_f16", 0x3c, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_neq_f16", 0x3d, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_nlt_f16", 0x3e, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_tru_f16", 0x3f, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_f_f32_e64", 0x40, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_lt_f32_e64", 0x41, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_eq_f32_e64", 0x42, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_le_f32_e64", 0x43, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_gt_f32_e64", 0x44, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_lg_f32_e64", 0x45, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_ge_f32_e64", 0x46, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_o_f32_e64", 0x47, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_u_f32_e64", 0x48, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_nge_f32_e64", 0x49, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmp_nlg_f32_e64", 0x4a, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmp_ngt_f32_e64", 0x4b, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmp_nle_f32_e64", 0x4c, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmp_neq_f32_e64", 0x4d, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmp_nlt_f32_e64", 0x4e, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmp_tru_f32_e64", 0x4f, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_f_f32_e64", 0x50, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_lt_f32_e64", 0x51, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_eq_f32_e64", 0x52, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_le_f32_e64", 0x53, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_gt_f32_e64", 0x54, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_lg_f32_e64", 0x55, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_ge_f32_e64", 0x56, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_o_f32_e64", 0x57, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_u_f32_e64", 0x58, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_nge_f32_e64", 0x59, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_nlg_f32_e64", 0x5a, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_ngt_f32_e64", 0x5b, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_nle_f32_e64", 0x5c, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_neq_f32_e64", 0x5d, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_nlt_f32_e64", 0x5e, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmpx_tru_f32_e64", 0x5f, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 0});
  addInstType({"v_cmp_f_f64", 0x60, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_lt_f64", 0x61, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_eq_f64", 0x62, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_le_f64", 0x63, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_gt_f64", 0x64, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_lg_f64", 0x65, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_ge_f64", 0x66, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_o_f64", 0x67, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_u_f64", 0x68, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_nge_f64", 0x69, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmp_nlg_f64", 0x6a, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmp_ngt_f64", 0x6b, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmp_nle_f64", 0x6c, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmp_neq_f64", 0x6d, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmp_nlt_f64", 0x6e, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmp_tru_f64", 0x6f, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_f_f64", 0x70, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmpx_lt_f64", 0x71, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_eq_f64", 0x72, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_le_f64", 0x73, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_gt_f64", 0x74, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_lg_f64", 0x75, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_ge_f64", 0x76, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_o_f64", 0x77, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmpx_u_f64", 0x78, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmpx_nge_f64", 0x79, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_nlg_f64", 0x7a, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_ngt_f64", 0x7b, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_nle_f64", 0x7c, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_neq_f64", 0x7d, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_nlt_f64", 0x7e, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_tru_f64", 0x7f, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmp_f_i16", 0xa0, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_lt_i16", 0xa1, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_eq_i16", 0xa2, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_le_i16", 0xa3, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_gt_i16", 0xa4, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_lg_i16", 0xa5, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_ge_i16", 0xa6, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cmp_tru_i16", 0xa7, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_f_u16", 0xa8, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_lt_u16", 0xa9, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_eq_u16", 0xaa, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_le_u16", 0xab, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_gt_u16", 0xac, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_lg_u16", 0xad, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_ge_u16", 0xae, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmp_tru_u16", 0xaf, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_f_i16", 0xb0, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lt_i16", 0xb1, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_eq_i16", 0xb2, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_le_i16", 0xb3, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_gt_i16", 0xb4, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_lg_i16", 0xb5, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_ge_i16", 0xb6, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_tru_i16", 0xb7, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_f_u16", 0xb8, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 0, 0});
  addInstType({"v_cmpx_lt_u16", 0xb9, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_eq_u16", 0xba, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_le_u16", 0xbb, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_gt_u16", 0xbc, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_lg_u16", 0xbd, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_ge_u16", 0xbe, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmpx_tru_u16", 0xbf, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 0});
  addInstType({"v_cmp_f_i32_e64", 0xc0, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_lt_i32_e64", 0xc1, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_eq_i32_e64", 0xc2, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_le_i32_e64", 0xc3, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_gt_i32_e64", 0xc4, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_lg_i32_e64", 0xc5, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_ge_i32_e64", 0xc6, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_tru_i32_e64", 0xc7, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmp_f_u32_e64", 0xc8, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_lt_u32_e64", 0xc9, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_eq_u32_e64", 0xca, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_le_u32_e64", 0xcb, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_gt_u32_e64", 0xcc, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_ne_u32_e64", 0xcd, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_ge_u32_e64", 0xce, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmp_tru_u32_e64", 0xcf, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_f_i32_e64", 0xd0, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmpx_lt_i32_e64", 0xd1, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_eq_i32_e64", 0xd2, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_le_i32_e64", 0xd3, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_gt_i32_e64", 0xd4, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_lg_i32_e64", 0xd5, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_ge_i32_e64", 0xd6, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_tru_i32_e64", 0xd7, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_f_u32_e64", 0xd8, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               32, 32, 0, 64});
  addInstType({"v_cmpx_lt_u32_e64", 0xd9, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_eq_u32_e64", 0xda, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_le_u32_e64", 0xdb, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_gt_u32_e64", 0xdc, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_lg_u32_e64", 0xdd, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_ge_u32_e64", 0xde, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmpx_tru_u32_e64", 0xdf, FormatTable[VOP3a], 0, ExeUnitVALU,
               64, 32, 32, 0, 64});
  addInstType({"v_cmp_f_i64", 0xe0, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_lt_i64", 0xe1, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_eq_i64", 0xe2, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_le_i64", 0xe3, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_gt_i64", 0xe4, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_lg_i64", 0xe5, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_ge_i64", 0xe6, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_tru_i64", 0xe7, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmp_f_u64", 0xe8, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_lt_u64", 0xe9, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_eq_u64", 0xea, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_le_u64", 0xeb, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_gt_u64", 0xec, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_lg_u64", 0xed, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_ge_u64", 0xee, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmp_tru_u64", 0xef, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_f_i64", 0xf0, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmpx_lt_i64", 0xf1, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_eq_i64", 0xf2, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_le_i64", 0xf3, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_gt_i64", 0xf4, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_lg_i64", 0xf5, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_ge_i64", 0xf6, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_tru_i64", 0xf7, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_f_u64", 0xf8, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_cmpx_lt_u64", 0xf9, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_eq_u64", 0xfa, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_le_u64", 0xfb, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_gt_u64", 0xfc, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_lg_u64", 0xfd, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_ge_u64", 0xfe, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cmpx_tru_u64", 0xff, FormatTable[VOP3a], 0, ExeUnitVALU, 64,
               64, 64, 0, 0});
  addInstType({"v_cndmask_b32_e64", 0 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 64, 0});
  addInstType({"v_add_f32", 1 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_sub_f32_e64", 2 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_subrev_f32", 3 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mul_legacy_f32", 4 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_mul_f32_e32", 5 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mul_i32_i24", 6 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mul_hi_i32_i24", 7 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_mul_u32_u24_e32", 8 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_mul_hi_u32_u24", 9 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_min_f32", 10 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_max_f32", 11 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_min_i32", 12 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_max_i32", 13 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_min_u32_e32", 14 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_max_u32_e32", 15 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_lshrrev_b32_e32", 16 + 256, FormatTable[VOP3a], 0,
               ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_ashrrev_i32_e32", 17 + 256, FormatTable[VOP3a], 0,
               ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_lshlrev_b32_e32", 18 + 256, FormatTable[VOP3a], 0,
               ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_and_b32_e32", 19 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_or_b32_e32", 20 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_xor_b32_e32", 21 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_mac_f32_e32", 22 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_madmk_f32", 23 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_madak_f32", 24 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_add_u32_e64", 25 + 256, FormatTable[VOP3b], 0, ExeUnitVALU,
               32, 32, 32, 64, 64});
  addInstType({"v_sub_u32_e64", 26 + 256, FormatTable[VOP3b], 0, ExeUnitVALU,
               32, 32, 32, 0, 64});
  addInstType({"v_subrev_u32_e64", 27 + 256, FormatTable[VOP3b], 0, ExeUnitVALU,
               32, 32, 0, 0, 64});
  addInstType({"v_addc_u32_e64", 28 + 256, FormatTable[VOP3b], 0, ExeUnitVALU,
               32, 32, 32, 64, 64});
  addInstType({"v_subb_u32_e64", 29 + 256, FormatTable[VOP3b], 0, ExeUnitVALU,
               32, 32, 32, 0, 64});
  addInstType({"v_subbrev_u32_e64", 30 + 256, FormatTable[VOP3b], 0,
               ExeUnitVALU, 32, 32, 32, 64, 64});
  addInstType({"v_add_f16", 31 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_sub_f16", 32 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_subrev_f16", 33 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mul_f16", 34 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mac_f16", 35 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_madmk_f16", 36 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_madak_f16", 37 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_add_u16", 38 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_sub_u16", 39 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_subrev_u16", 40 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mul_lo_u16", 41 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_lshlrev_b16", 42 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_lshrrev_b16", 43 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_ashrrev_i16", 44 + 256, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_max_f16", 45 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_min_f16", 46 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_max_u16", 47 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_max_i16", 48 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_min_u16", 49 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_min_i16", 50 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_ldexp_f16", 51 + 256, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});

  addInstType({"v_mad_legacy_F32", 448, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType(
      {"v_mad_f32", 449, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_mad_i32_i24", 450, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_mad_u32_u24", 451, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 32, 0});
  addInstType({"v_cubeid_f32", 452, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cubesc_f32", 453, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cubetc_f32", 454, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_cubema_f32", 455, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_bfe_u32", 456, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_bfe_i32", 457, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_bfi_b32", 458, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_fma_f32", 459, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32,
               32, 0});
  addInstType(
      {"v_fma_f64", 460, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_lerp_u8", 461, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_alignbit_b32", 462, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_alignbyte_b32", 463, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_min3_f32", 464, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_min3_i32", 465, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_min3_u32", 466, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_max3_f32", 467, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_max3_i32", 468, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_max3_u32", 469, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_med3_f32", 470, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_med3_i32", 471, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_med3_u32", 472, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_sad_u8", 473, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_sad_hi_u8", 474, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_sad_u16", 475, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sad_u32", 476, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_cvt_pk_u8_f32", 477, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_div_fixup_f32", 478, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_div_fixup_f64", 479, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_div_scale_f32", 480, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_div_scale_f64", 481, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_div_fmas_f32", 482, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_div_fmas_f64", 483, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType(
      {"v_msad_u8", 484, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_qsad_pk_u16_u8", 485, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mqsad_pk_u16_u8", 486, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mqsad_u32_u8", 487, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mad_u64_u32", 488, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 64, 0});
  addInstType({"v_mad_i64_i32", 489, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 32,
               32, 64, 0});
  addInstType({"v_mad_legacy_f16", 490, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mad_legacy_u16", 491, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mad_legacy_i16", 492, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_perm_b32", 493, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_fma_legacy_f16", 494, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_div_fixup_legacy_f16", 495, FormatTable[VOP3a], 0,
               ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_cvt_pkaccum_u8_f32", 496, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_mad_u32_u16", 497, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_mad_i32_i16", 498, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_xad_u32", 499, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_min3_f16", 500, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_min3_i16", 501, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_min3_u16", 502, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_max3_f16", 503, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_max3_i16", 504, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_max3_u16", 505, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_med3_f16", 506, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_med3_i16", 507, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_med3_u16", 508, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_lshl_add_u32", 509, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 32, 0});
  addInstType({"v_add_lshl_u32", 510, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_add3_u32", 511, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 32, 0});
  addInstType({"v_lshl_or_b32", 512, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_and_or_b32", 513, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType(
      {"v_or3_b32", 514, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_mad_f16", 515, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_mad_u16", 516, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_mad_i16", 517, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_fma_f16", 518, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_div_fixup_f16", 519, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_interp_p1ll_f16", 628, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_interp_p1lv_f16", 629, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_interp_p2_legacy_f16", 630, FormatTable[VOP3a], 0,
               ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_interp_p2_f16", 631, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType(
      {"v_add_f64", 640, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_mul_f64", 641, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_min_f64", 642, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_max_f64", 643, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_ldexp_f64", 644, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_mul_lo_u32", 645, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_mul_hi_u32", 646, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_mul_hi_i32", 647, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_ldexp_f32", 648, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_readlane_b32", 649, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_writelane_b32", 650, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_bcnt_u32_b32", 651, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mbcnt_lo_u32_b32", 652, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_mbcnt_hi_u32_b32", 653, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_lshlrev_b64", 655, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_lshrrev_b64", 656, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32,
               32, 0, 0});
  addInstType({"v_ashrrev_i64", 657, FormatTable[VOP3a], 0, ExeUnitVALU, 64, 64,
               64, 0, 0});
  addInstType({"v_trig_preop_f64", 658, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType(
      {"v_bfm_b32", 659, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_cvt_pknorm_i16_f32", 660, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_cvt_pknorm_u16_f32", 661, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_cvt_pkrtz_f16_f32", 662, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_cvt_pk_u16_u32", 663, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_pk_i16_i32", 664, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_cvt_pknorm_i16_f16", 665, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType({"v_cvt_pknorm_u16_f16", 666, FormatTable[VOP3a], 0, ExeUnitVALU,
               32, 32, 32, 0, 0});
  addInstType(
      {"v_add_i32", 668, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sub_i32", 669, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_add_i16", 670, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType(
      {"v_sub_i16", 671, FormatTable[VOP3a], 0, ExeUnitVALU, 32, 32, 32, 0, 0});
  addInstType({"v_pack_b32_f16", 672, FormatTable[VOP3a], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});

  // VOP3b Instructions
  addInstType({"v_add_co_u32_e64", 281, FormatTable[VOP3b], 0, ExeUnitVALU, 32,
               32, 32, 0, 64});
  addInstType({"v_sub_co_u32", 282, FormatTable[VOP3b], 0, ExeUnitVALU, 32, 32,
               32, 0, 64});
  addInstType({"v_addc_co_u32_e64", 284, FormatTable[VOP3b], 0, ExeUnitVALU, 32,
               32, 32, 64, 64});

  addInstType({"v_div_scale_f32", 480, FormatTable[VOP3b], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_div_scale_f64", 481, FormatTable[VOP3b], 0, ExeUnitVALU, 32,
               32, 32, 0, 0});
  addInstType({"v_mad_u64_u32", 488, FormatTable[VOP3b], 0, ExeUnitVALU, 64, 32,
               32, 64, 64});
  addInstType({"v_mad_i64_i32", 489, FormatTable[VOP3b], 0, ExeUnitVALU, 64, 32,
               32, 64, 64});

  // SOP1 Instructions
  addInstType(
      {"s_mov_b32", 0, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32, 0, 0, 0});
  addInstType(
      {"s_mov_b64", 1, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64, 0, 0, 0});
  addInstType(
      {"s_cmov_b32", 2, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32, 0, 0, 0});
  addInstType(
      {"s_cmov_b64", 3, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64, 0, 0, 0});
  addInstType(
      {"s_not_b32", 4, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32, 0, 0, 0});
  addInstType(
      {"s_not_b64", 5, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64, 0, 0, 0});
  addInstType(
      {"s_wqm_b32", 6, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32, 0, 0, 0});
  addInstType(
      {"s_wqm_b64 ", 7, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64, 0, 0, 0});
  addInstType(
      {"s_brev_b32", 8, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32, 0, 0, 0});
  addInstType(
      {"s_brev_b64", 9, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64, 0, 0, 0});
  addInstType({"s_bcnt0_i32_b32", 10, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_bcnt0_i32_b64", 11, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_bcnt1_i32_b32", 12, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_bcnt1_i32_b64", 13, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_ff0_i32_b32", 14, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32,
               0, 0, 0});
  addInstType({"s_ff0_i32_b64", 15, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32,
               0, 0, 0});
  addInstType({"s_ff1_i32_b32", 16, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32,
               0, 0, 0});
  addInstType({"s_ff1_i32_b64", 17, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32,
               0, 0, 0});
  addInstType({"s_flbit_i32_b32", 18, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_flbit_i32_b64", 19, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_flbit_i32", 20, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32,
               0, 0, 0});
  addInstType({"s_flbit_i32_i64", 21, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_sext_i32_i8", 22, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32,
               0, 0, 0});
  addInstType({"s_sext_i32_i16", 23, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_bitset0_b32", 24, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32,
               0, 0, 0});
  addInstType({"s_bitset0_b64", 25, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64,
               0, 0, 0});
  addInstType({"s_bitset1_b32", 26, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32,
               0, 0, 0});
  addInstType({"s_bitset1_b64", 27, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64,
               0, 0, 0});
  addInstType({"s_getpc_b64", 28, FormatTable[SOP1], 0, ExeUnitScalar, 64, 0,
               0, 0, 0});
  addInstType({"s_setpc_b64", 29, FormatTable[SOP1], 0, ExeUnitScalar, 0, 64,
               0, 0, 0});
  addInstType({"s_swappc_b64", 30, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64,
               0, 0, 0});
  addInstType(
      {"s_rfe_b64", 31, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64, 0, 0, 0});
  addInstType({"s_and_saveexec_b64", 32, FormatTable[SOP1], 0, ExeUnitScalar,
               64, 64, 0, 0, 0});
  addInstType({"s_or_saveexec_b64", 33, FormatTable[SOP1], 0, ExeUnitScalar, 64,
               64, 0, 0, 0});
  addInstType({"s_xor_saveexec_b64", 34, FormatTable[SOP1], 0, ExeUnitScalar,
               64, 64, 0, 0, 0});
  addInstType({"s_andn2_saveexec_b64", 35, FormatTable[SOP1], 0, ExeUnitScalar,
               64, 64, 0, 0, 0});
  addInstType({"s_orn2_saveexec_b64", 36, FormatTable[SOP1], 0, ExeUnitScalar,
               64, 64, 0, 0, 0});
  addInstType({"s_nand_saveexec_b64", 37, FormatTable[SOP1], 0, ExeUnitScalar,
               64, 64, 0, 0, 0});
  addInstType({"s_nor_saveexec_b64", 38, FormatTable[SOP1], 0, ExeUnitScalar,
               64, 64, 0, 0, 0});
  addInstType({"s_xnor_saveexec_b64", 39, FormatTable[SOP1], 0, ExeUnitScalar,
               64, 64, 0, 0, 0});
  addInstType({"s_quadmask_b32", 40, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_quadmask_b64", 41, FormatTable[SOP1], 0, ExeUnitScalar, 64,
               64, 0, 0, 0});
  addInstType({"s_movrels_b32", 42, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32,
               0, 0, 0});
  addInstType({"s_movrels_b64", 43, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64,
               0, 0, 0});
  addInstType({"s_movreld_b32", 44, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32,
               0, 0, 0});
  addInstType({"s_movreld_b64", 45, FormatTable[SOP1], 0, ExeUnitScalar, 64, 64,
               0, 0, 0});
  addInstType({"s_cbranch_join", 46, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType(
      {"s_abs_i32", 48, FormatTable[SOP1], 0, ExeUnitScalar, 32, 32, 0, 0, 0});
  addInstType({"s_set_gpr_idx_idx", 50, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_and1_saveexec_b64", 51, FormatTable[SOP1], 0, ExeUnitScalar,
               32, 32, 0, 0, 0});
  addInstType({"s_orn1_saceexec_b64", 52, FormatTable[SOP1], 0, ExeUnitScalar,
               32, 32, 0, 0, 0});
  addInstType({"s_and1_wrexec_b64", 53, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_and2_wrexec_b64", 54, FormatTable[SOP1], 0, ExeUnitScalar, 32,
               32, 0, 0, 0});
  addInstType({"s_bitreplicate_b64_b32", 55, FormatTable[SOP1], 0,
               ExeUnitScalar, 32, 32, 0, 0, 0});

  // DS Instructions
  addInstType({"ds_add_u32", 0, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_sub_u32", 1, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_rsub_u32", 2, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_inc_u32", 3, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_dec_u32", 4, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_min_i32", 5, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_max_i32", 6, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_min_u32", 7, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_max_u32", 8, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_and_b32", 9, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_or_b32", 10, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_xor_b32", 11, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_mskor_b32", 12, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_write_b32", 13, FormatTable[DS], 0, ExeUnitLDS, 0, 32, 0, 0, 0});
  addInstType(
      {"ds_write2_b32", 14, FormatTable[DS], 0, ExeUnitLDS, 0, 32, 32, 0, 0});
  addInstType(
      {"ds_write2st64_b32", 15, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_cmpst_b32", 16, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_cmpst_f32", 17, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_f32", 18, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_f32", 19, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_nop", 20, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_add_f32", 21, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_write_addtid_b32", 29, FormatTable[DS], 0, ExeUnitLDS, 0, 0,
               0, 0, 0});
  addInstType(
      {"ds_write_b8", 30, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_write_b16", 31, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_add_rtn_u32", 32, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_sub_rtn_u32", 33, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_rsub_rtn_u32", 34, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_inc_rtn_u32", 35, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_dec_rtn_u32", 36, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_rtn_i32", 37, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_rtn_i32", 38, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_rtn_u32", 39, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_rtn_u32", 40, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_and_rtn_b32", 41, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_or_rtn_b32", 42, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_xor_rtn_b32", 43, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_mskor_rtn_b32", 44, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_wrxchg_rtn_b32", 45, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_wrxchg2_rtn_b32", 46, FormatTable[DS], 0, ExeUnitLDS, 0, 0,
               0, 0, 0});
  addInstType({"ds_wrxchg2st64_rtn_b32", 47, FormatTable[DS], 0, ExeUnitLDS, 0,
               0, 0, 0, 0});
  addInstType(
      {"ds_cmpst_rtn_b32", 48, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_cmpst_rtn_f32", 49, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_rtn_f32", 50, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_rtn_f32", 51, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_wrap_rtn_b32", 52, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_add_rtn_f32", 53, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_b32", 54, FormatTable[DS], 0, ExeUnitLDS, 32, 0, 0, 0, 0});
  addInstType(
      {"ds_read2_b32", 55, FormatTable[DS], 0, ExeUnitLDS, 64, 0, 0, 0, 0});
  addInstType(
      {"ds_read2st64_b32", 56, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_i8", 57, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_u8", 58, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_i16", 59, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_u16", 60, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_swizzle_b32", 61, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_permute_b32", 62, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_bpermute_b32", 63, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_add_u64", 64, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_sub_u64", 65, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_rsub_u64", 66, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_inc_u64", 67, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_dec_u64", 68, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_i64", 69, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_i64", 70, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_u64", 71, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_u64", 72, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_and_b64", 73, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_or_b64", 74, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_xor_b64", 75, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_mskor_b64", 76, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_write_b64", 77, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_write2_b64", 78, FormatTable[DS], 0, ExeUnitLDS, 0, 64, 64, 0, 0});
  addInstType(
      {"ds_write2st64_b64", 79, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_cmpst_b64", 80, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_cmpst_f64", 81, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_f64", 82, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_f64", 83, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_write_b8_d16_hi", 84, FormatTable[DS], 0, ExeUnitLDS, 0, 0,
               0, 0, 0});
  addInstType({"ds_write_b16_d16_hi", 85, FormatTable[DS], 0, ExeUnitLDS, 0, 0,
               0, 0, 0});
  addInstType(
      {"ds_read_u8_d16", 86, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_u8_d16_hi", 87, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_i8_d16", 88, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_i8_d16_hi", 89, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_u16_d16", 90, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_read_u16_d16_hi", 91, FormatTable[DS], 0, ExeUnitLDS, 0, 0,
               0, 0, 0});
  addInstType(
      {"ds_add_rtn_u64", 96, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_sub_rtn_u64", 97, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_rsub_rtn_u64", 98, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_inc_rtn_u64", 99, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_dec_rtn_u64", 100, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_rtn_i64", 101, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_rtn_i64", 102, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_rtn_u64", 103, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_rtn_u64", 104, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_and_rtn_b64", 105, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_or_rtn_b64", 106, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_xor_rtn_b64", 107, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_mskor_rtn_b64", 108, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_wrxchg_rtn_b64", 109, FormatTable[DS], 0, ExeUnitLDS, 0, 0,
               0, 0, 0});
  addInstType({"ds_wrxchg2_rtn_b64", 110, FormatTable[DS], 0, ExeUnitLDS, 0, 0,
               0, 0, 0});
  addInstType({"ds_wrxchg2st64_rtn_b64", 111, FormatTable[DS], 0, ExeUnitLDS, 0,
               0, 0, 0, 0});
  addInstType(
      {"ds_cmpst_rtn_b64", 112, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_cmpst_rtn_f64", 113, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_rtn_f64", 114, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_rtn_f64", 115, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_b64", 118, FormatTable[DS], 0, ExeUnitLDS, 64, 0, 0, 0, 0});
  addInstType(
      {"ds_read2_b64", 119, FormatTable[DS], 0, ExeUnitLDS, 128, 0, 0, 0, 0});
  addInstType(
      {"ds_read2st64_b64", 120, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_condxchg32_rtn_b64", 126, FormatTable[DS], 0, ExeUnitLDS, 0,
               0, 0, 0, 0});
  addInstType(
      {"ds_add_src2_u32", 128, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_sub_src2_u32", 129, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_rsub_src2_u32", 130, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_inc_src2_u32", 131, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_dec_src2_u32", 132, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_src2_i32", 133, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_src2_i32", 134, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_src2_u32", 135, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_src2_u32", 136, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_and_src2_b32", 137, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_or_src2_b32", 138, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_xor_src2_b32", 139, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_write_src2_b32", 141, FormatTable[DS], 0, ExeUnitLDS, 0, 0,
               0, 0, 0});
  addInstType(
      {"ds_min_src2_f32", 146, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_src2_f32", 147, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_add_src2_f32", 149, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_gws_sema_release_all", 152, FormatTable[DS], 0, ExeUnitLDS,
               0, 0, 0, 0, 0});
  addInstType(
      {"ds_gws_init", 153, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_gws_sema_v", 154, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_gws_sema_br", 155, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_gws_sema_p", 156, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_gws_barrier", 157, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_read_addtid_b32", 182, FormatTable[DS], 0, ExeUnitLDS, 0, 0,
               0, 0, 0});
  addInstType(
      {"ds_consume", 189, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_append", 190, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_ordered_count", 191, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_add_src2_u64", 192, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_sub_src2_u64", 193, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_rsub_src2_u64", 194, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_inc_src2_u64", 195, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_dec_src2_u64", 196, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_src2_i64", 197, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_src2_i64", 198, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_min_src2_u64", 199, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_src2_u64", 200, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_and_src2_b64", 201, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_or_src2_b64", 202, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_xor_src2_b64", 203, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType({"ds_write_src2_b64", 205, FormatTable[DS], 0, ExeUnitLDS, 0, 0,
               0, 0, 0});
  addInstType(
      {"ds_min_src2_f64", 210, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_max_src2_f64", 211, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_write_b96", 222, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_write_b128", 223, FormatTable[DS], 0, ExeUnitLDS, 0, 128, 0, 0, 0});
  addInstType(
      {"ds_read_b96", 254, FormatTable[DS], 0, ExeUnitLDS, 0, 0, 0, 0, 0});
  addInstType(
      {"ds_read_b128", 255, FormatTable[DS], 0, ExeUnitLDS, 128, 0, 0, 0, 0});

  // VOP3P Instructions
  addInstType(
      {"v_pk_mad_i16", 0, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType({"v_pk_mul_lo_u16", 1, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType(
      {"v_pk_add_i16", 2, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_sub_i16", 3, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType({"v_pk_lshlrev_b16", 4, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType({"v_pk_lshrrev_b16", 5, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType({"v_pk_ashrrev_i16", 6, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType(
      {"v_pk_max_i16", 7, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_min_i16", 8, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_mad_u16", 9, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_add_u16", 10, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_sub_u16", 11, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_max_u16", 12, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_min_u16", 13, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_fma_f16", 14, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_add_f16", 15, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_mul_f16", 16, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_min_f16", 17, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_pk_max_f16", 18, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_mad_mix_f32", 32, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType({"v_mad_mixlo_f16", 33, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType({"v_mad_mixhi_f16", 34, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType({"v_dot2_f32_f16", 35, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType({"v_dot2_i32_i16", 38, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType({"v_dot2_u32_u16", 39, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType(
      {"v_dot4_i32_i8", 40, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_dot4_u32_u8", 41, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_dot8_i32_i4", 42, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType(
      {"v_dot8_u32_u4", 43, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_32x32x1f32", 64, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_16x16x1f32", 65, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_4x4x1f32", 66, FormatTable[VOP3P], 0, ExeUnitVALU, 0,
               0, 0, 0, 0});
  addInstType({"v_mfma_f32_32x32x2f32", 68, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_16x16x4f32", 69, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_32x32x4f16", 72, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"f_mfma_f32_16x16x4f16", 73, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_4x4x4f16", 74, FormatTable[VOP3P], 0, ExeUnitVALU, 0,
               0, 0, 0, 0});
  addInstType({"v_mfma_f32_32x32x8f16", 76, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_16x16x16f16", 77, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_i32_32x32x4i8", 80, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_i32_16x16x4i8", 81, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_i32_4x4x4i8", 82, FormatTable[VOP3P], 0, ExeUnitVALU, 0,
               0, 0, 0, 0});
  addInstType({"v_mfma_i32_32x32x8i8", 84, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_i32_16x16x16i8", 85, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_accvgpr_read", 88, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType({"v_accvgpr_write", 89, FormatTable[VOP3P], 0, ExeUnitVALU, 0, 0,
               0, 0, 0});
  addInstType({"v_mfma_f32_32x32x2bf16", 104, FormatTable[VOP3P], 0,
               ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_16x16x2bf16", 105, FormatTable[VOP3P], 0,
               ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_4x4x2bf16", 107, FormatTable[VOP3P], 0, ExeUnitVALU,
               0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_32x32x4bf16", 108, FormatTable[VOP3P], 0,
               ExeUnitVALU, 0, 0, 0, 0, 0});
  addInstType({"v_mfma_f32_16x16x8bf16", 109, FormatTable[VOP3P], 0,
               ExeUnitVALU, 0, 0, 0, 0, 0});
}
