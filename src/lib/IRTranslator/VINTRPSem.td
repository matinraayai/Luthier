//===- VINTRPSem.td ------------------------------------------*- tblgen -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
// Describes the semantics of VINTRP (legacy parameter interpolation),
// VOP3 Interpolation (F16), and LDSDIR/DSDIR (LDS direct/param load)
// instructions for the AMDGPU backend.
//
// Legacy VINTRP instructions (V_INTERP_P1_F32, V_INTERP_P2_F32,
// V_INTERP_MOV_F32) exist from GCN3 through RDNA2 (GFX6–GFX10).
// In RDNA3 (GFX11), they are replaced by LDS_PARAM_LOAD + VINTERP
// (V_INTERP_P2_F32 in new VOP3 encoding).
// In RDNA4 (GFX12), LDS_PARAM_LOAD is replaced by DS_PARAM_LOAD.
//
// VOP3 Interpolation F16 instructions (V_INTERP_P1LL_F16, V_INTERP_P1LV_F16,
// V_INTERP_P2_F16) exist from GCN3 through RDNA2. In RDNA3+, V_INTERP_P2_F16
// is available in new VINTERP encoding; P1LL/P1LV are removed.
//===----------------------------------------------------------------------===//
include "SISemOps.td"

//==============================================================================
// VINTRP Legacy Interpolation Instructions (GCN3–RDNA2)
//==============================================================================
//
// These use the VINTRP encoding format (32-bit). Operands:
//   $vsrc: VGPR source (barycentric coordinate for P1/P2, slot for MOV)
//   $attr: interpolation attribute number (0–32)
//   $attrchan: attribute channel (x=0, y=1, z=2, w=3)
// The LDS parameter memory is accessed implicitly via M0.
//
// All three instructions use intrinsics for their semantics since
// the interpolation behavior is hardware-specific (LDS parameter
// memory access + interpolation math).

//------------------------------------------------------------------------------
// V_INTERP_P1_F32 — Interpolation Step 1 (partial result)
//------------------------------------------------------------------------------
// D.f = P10 * S0.f + P0, where P0 and P10 are read from the
// parameter cache (LDS) at the attribute/channel specified.
// Result is a partial interpolation stored in VGPR.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 264.
//   vega-shader-instruction-set-architecture.pdf, page 163.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 165.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 157.
//   rdna-shader-instruction-set-architecture.pdf, page 176.
//   rdna2-shader-instruction-set-architecture.pdf, page 174.
def V_INTERP_P1_F32_sem : InstSIIntrinsic<V_INTERP_P1_F32,
                                            int_amdgcn_interp_p1>;

//------------------------------------------------------------------------------
// V_INTERP_P2_F32 — Interpolation Step 2 (final result)
//------------------------------------------------------------------------------
// D.f = P20 * S0.f + D.f (where D.f is the result from P1).
// Completes the barycentric interpolation.
// Sources (GCN3–RDNA2, legacy VINTRP encoding):
//   gcn3-instruction-set-architecture.pdf, page 265.
//   vega-shader-instruction-set-architecture.pdf, page 163.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 165.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 157.
//   rdna-shader-instruction-set-architecture.pdf, page 177.
//   rdna2-shader-instruction-set-architecture.pdf, page 174.
// Sources (RDNA3+, new VINTERP VOP3 encoding):
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 507.
//   rdna4-instruction-set-architecture.pdf, page 580.
def V_INTERP_P2_F32_sem : InstSIIntrinsic<V_INTERP_P2_F32,
                                            int_amdgcn_interp_p2>;

//------------------------------------------------------------------------------
// V_INTERP_MOV_F32 — Interpolation Move (flat/center/centroid)
//------------------------------------------------------------------------------
// D.f = Parameter value at the specified slot (P0, P10, or P20)
// for the given attribute/channel. Used for flat shading.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 264.
//   vega-shader-instruction-set-architecture.pdf, page 163.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 165.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 157.
//   rdna-shader-instruction-set-architecture.pdf, page 177.
//   rdna2-shader-instruction-set-architecture.pdf, page 174.
def V_INTERP_MOV_F32_sem : InstSIIntrinsic<V_INTERP_MOV_F32,
                                             int_amdgcn_interp_mov>;

//------------------------------------------------------------------------------
// V_INTERP_P1_F32_16bank — P1 variant for 16-bank LDS (LLVM pseudo)
//------------------------------------------------------------------------------
// Same semantics as V_INTERP_P1_F32 but scheduled differently for
// different LDS bank configurations. LLVM-only pseudo.
def V_INTERP_P1_F32_16bank_sem : InstSIIntrinsic<V_INTERP_P1_F32_16bank,
                                                   int_amdgcn_interp_p1>;

//==============================================================================
// VOP3 Interpolation F16 Instructions (GCN3–RDNA2)
//==============================================================================
//
// These use VOP3 encoding (64-bit) with source modifiers. They perform
// interpolation in half-precision (F16) and are used for 16-bit pixel
// shader inputs. Available from GCN3 (GFX8) through RDNA2. In RDNA3+,
// V_INTERP_P1LL_F16 and V_INTERP_P1LV_F16 are removed; V_INTERP_P2_F16
// uses the new VINTERP encoding.

//------------------------------------------------------------------------------
// V_INTERP_P1LL_F16 — F16 Interpolation P1, Low-Low
//------------------------------------------------------------------------------
// D.f32 = LDS[attr][chan].lo * S0.f + LDS[attr][chan].p0
// First step of F16 interpolation using the low half of the parameter.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 265.
//   vega-shader-instruction-set-architecture.pdf, page 177.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 179.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 171.
//   rdna-shader-instruction-set-architecture.pdf, page 192.
//   rdna2-shader-instruction-set-architecture.pdf, page 188.
def V_INTERP_P1LL_F16_sem : InstSIIntrinsic<V_INTERP_P1LL_F16,
                                              int_amdgcn_interp_p1_f16>;

//------------------------------------------------------------------------------
// V_INTERP_P1LV_F16 — F16 Interpolation P1, Low-High
//------------------------------------------------------------------------------
// D.f32 = LDS[attr][chan].hi * S0.f + LDS[attr][chan].p0 + S2.f
// First step of F16 interpolation using the high half of the parameter,
// combined with partial result from a previous P1LL step in src2.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 266.
//   vega-shader-instruction-set-architecture.pdf, page 178.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 180.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 171.
//   rdna-shader-instruction-set-architecture.pdf, page 193.
//   rdna2-shader-instruction-set-architecture.pdf, page 188.
def V_INTERP_P1LV_F16_sem : InstSISemantic<V_INTERP_P1LV_F16> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// V_INTERP_P2_F16 — F16 Interpolation P2 (final result)
//------------------------------------------------------------------------------
// D.f16 = S0.f * P20 + S2.f (completes F16 interpolation).
// Takes the partial result from P1 in src2 and the barycentric
// coordinate in src0.
// Sources (GCN3–RDNA2, VOP3 encoding):
//   gcn3-instruction-set-architecture.pdf, page 266.
//   vega-shader-instruction-set-architecture.pdf, page 178.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 180.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 171.
//   rdna-shader-instruction-set-architecture.pdf, page 195.
//   rdna2-shader-instruction-set-architecture.pdf, page 190.
// Sources (RDNA3+, new VINTERP encoding):
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 508.
//   rdna4-instruction-set-architecture.pdf, page 581.
def V_INTERP_P2_F16_sem : InstSIIntrinsic<V_INTERP_P2_F16,
                                            int_amdgcn_interp_p2_f16>;

//==============================================================================
// LDSDIR Instructions — LDS Direct/Param Load (RDNA3 / GFX11)
//==============================================================================
//
// RDNA3 replaces the legacy VINTRP encoding with LDSDIR instructions
// for loading interpolation parameters and direct LDS values. These
// use a new 32-bit LDSDIR encoding format.
//
// Encoding format sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 182.

//------------------------------------------------------------------------------
// LDS_PARAM_LOAD — Load Interpolation Parameter from LDS
//------------------------------------------------------------------------------
// D.u = LDS[attr][attrchan]. Loads the interpolation parameter value
// from LDS parameter memory. Replaces V_INTERP_MOV_F32 for loading
// parameters in RDNA3. Used with V_INTERP_P2_F32/F16 for interpolation.
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 510.
def LDS_PARAM_LOAD_sem : InstSIIntrinsic<LDS_PARAM_LOAD,
                                           int_amdgcn_lds_param_load>;

//------------------------------------------------------------------------------
// LDS_DIRECT_LOAD — Direct Load from LDS
//------------------------------------------------------------------------------
// D.u = LDS[M0.u]. Loads a value directly from LDS using the address
// in M0. Used for direct LDS access without interpolation.
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 510.
def LDS_DIRECT_LOAD_sem : InstSIIntrinsic<LDS_DIRECT_LOAD,
                                            int_amdgcn_lds_direct_load>;

//==============================================================================
// DSDIR Instructions — DS Direct/Param Load (RDNA4 / GFX12)
//==============================================================================
//
// RDNA4 replaces the LDSDIR instructions with DSDIR (renamed VDSDIR).
// Same functionality, new encoding name.
//
// Encoding format sources:
//   rdna4-instruction-set-architecture.pdf, page 202.

//------------------------------------------------------------------------------
// DS_PARAM_LOAD — Load Interpolation Parameter from LDS (GFX12)
//------------------------------------------------------------------------------
// D.u = LDS[attr][attrchan]. Same as LDS_PARAM_LOAD but with GFX12
// encoding and additional wait counter operands.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 584.
def DS_PARAM_LOAD_sem : InstSIIntrinsic<DS_PARAM_LOAD,
                                         int_amdgcn_lds_param_load>;

//------------------------------------------------------------------------------
// DS_DIRECT_LOAD — Direct Load from LDS (GFX12)
//------------------------------------------------------------------------------
// D.u = LDS[M0.u]. Same as LDS_DIRECT_LOAD but with GFX12 encoding
// and additional wait counter operands.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 584.
def DS_DIRECT_LOAD_sem : InstSIIntrinsic<DS_DIRECT_LOAD,
                                          int_amdgcn_lds_direct_load>;
