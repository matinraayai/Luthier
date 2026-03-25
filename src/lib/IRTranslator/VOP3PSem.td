//===- VOP3PSem.td -----------------------------------------------*- tblgen -*-===//
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

//==============================================================================
// VOP3P Packed Arithmetic Instructions
//==============================================================================
// Packed operations with op_sel/neg_lo/neg_hi modifiers are too complex
// to express in the current semantic DSL; all use empty semantics.

//------------------------------------------------------------------------------
// V_PK_* Two-Source Packed Arithmetic (f16/i16/u16/f32)
//------------------------------------------------------------------------------

// D.v2f16 = S0.v2f16 + S1.v2f16 (packed f16 add).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 266.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 280.
//   rdna-shader-instruction-set-architecture.pdf, page 175.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 363.
//   rdna4-instruction-set-architecture.pdf, page 402.
def V_PK_ADD_F16_sem : InstSISemantic<V_PK_ADD_F16> {
  let Semantic = [];
}

// D.v2f32 = S0.v2f32 + S1.v2f32 (packed f32 add, GFX10+).
// Sources:
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 156.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 271.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 287.
def V_PK_ADD_F32_sem : InstSISemantic<V_PK_ADD_F32> {
  let Semantic = [];
}

// D.v2i16 = S0.v2i16 + S1.v2i16 (packed i16 add).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 154.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 154.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 263.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 277.
//   rdna-shader-instruction-set-architecture.pdf, page 174.
//   rdna2-shader-instruction-set-architecture.pdf, page 171.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 360.
//   rdna4-instruction-set-architecture.pdf, page 399.
def V_PK_ADD_I16_sem : InstSISemantic<V_PK_ADD_I16> {
  let Semantic = [];
}

// D.v2u16 = S0.v2u16 + S1.v2u16 (packed u16 add).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 265.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 279.
//   rdna-shader-instruction-set-architecture.pdf, page 175.
//   rdna2-shader-instruction-set-architecture.pdf, page 171.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 362.
//   rdna4-instruction-set-architecture.pdf, page 401.
def V_PK_ADD_U16_sem : InstSISemantic<V_PK_ADD_U16> {
  let Semantic = [];
}

// D.v2i16 = S0.v2i16 - S1.v2i16.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 154.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 154.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 263.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 277.
//   rdna-shader-instruction-set-architecture.pdf, page 174.
//   rdna2-shader-instruction-set-architecture.pdf, page 171.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 360.
//   rdna4-instruction-set-architecture.pdf, page 399.
def V_PK_SUB_I16_sem : InstSISemantic<V_PK_SUB_I16> {
  let Semantic = [];
}

// D.v2u16 = S0.v2u16 - S1.v2u16.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 265.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 279.
//   rdna-shader-instruction-set-architecture.pdf, page 175.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 362.
//   rdna4-instruction-set-architecture.pdf, page 401.
def V_PK_SUB_U16_sem : InstSISemantic<V_PK_SUB_U16> {
  let Semantic = [];
}

// D.v2f16 = S0.v2f16 * S1.v2f16.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 266.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 280.
//   rdna-shader-instruction-set-architecture.pdf, page 175.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 363.
//   rdna4-instruction-set-architecture.pdf, page 403.
def V_PK_MUL_F16_sem : InstSISemantic<V_PK_MUL_F16> {
  let Semantic = [];
}

// D.v2f32 = S0.v2f32 * S1.v2f32 (GFX10+).
// Sources:
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 156.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 271.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 287.
def V_PK_MUL_F32_sem : InstSISemantic<V_PK_MUL_F32> {
  let Semantic = [];
}

// D.v2u16 = S0.v2u16 * S1.v2u16 (low 16 bits of each lane).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 154.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 154.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 263.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 277.
//   rdna-shader-instruction-set-architecture.pdf, page 174.
//   rdna2-shader-instruction-set-architecture.pdf, page 171.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 360.
//   rdna4-instruction-set-architecture.pdf, page 399.
def V_PK_MUL_LO_U16_sem : InstSISemantic<V_PK_MUL_LO_U16> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// V_PK_* Three-Source Packed FMA/MAD
//------------------------------------------------------------------------------

// D.v2f16 = FMA(S0.v2f16, S1.v2f16, S2.v2f16).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 266.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 280.
//   rdna-shader-instruction-set-architecture.pdf, page 175.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 362.
//   rdna4-instruction-set-architecture.pdf, page 402.
def V_PK_FMA_F16_sem : InstSISemantic<V_PK_FMA_F16> {
  let Semantic = [];
}

// D.v2f32 = FMA(S0.v2f32, S1.v2f32, S2.v2f32) (GFX10+).
// Sources:
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 156.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 270.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 287.
def V_PK_FMA_F32_sem : InstSISemantic<V_PK_FMA_F32> {
  let Semantic = [];
}

// D.v2i16 = S0.v2i16 * S1.v2i16 + S2.v2i16.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 162.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 154.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 154.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 262.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 276.
//   rdna-shader-instruction-set-architecture.pdf, page 174.
//   rdna2-shader-instruction-set-architecture.pdf, page 170.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 360.
//   rdna4-instruction-set-architecture.pdf, page 399.
def V_PK_MAD_I16_sem : InstSISemantic<V_PK_MAD_I16> {
  let Semantic = [];
}

// D.v2u16 = S0.v2u16 * S1.v2u16 + S2.v2u16.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 264.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 278.
//   rdna-shader-instruction-set-architecture.pdf, page 175.
//   rdna2-shader-instruction-set-architecture.pdf, page 171.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 361.
//   rdna4-instruction-set-architecture.pdf, page 401.
def V_PK_MAD_U16_sem : InstSISemantic<V_PK_MAD_U16> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// V_PK_* Packed Min/Max
//------------------------------------------------------------------------------

// D.v2f16 = max(S0.v2f16, S1.v2f16).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 164.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 267.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 281.
//   rdna-shader-instruction-set-architecture.pdf, page 176.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 363.
def V_PK_MAX_F16_sem : InstSISemantic<V_PK_MAX_F16> {
  let Semantic = [];
}

// D.v2f16 = min(S0.v2f16, S1.v2f16).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 266.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 280.
//   rdna-shader-instruction-set-architecture.pdf, page 175.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 363.
def V_PK_MIN_F16_sem : InstSISemantic<V_PK_MIN_F16> {
  let Semantic = [];
}

// D.v2i16 = max(S0.v2i16, S1.v2i16).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 264.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 278.
//   rdna-shader-instruction-set-architecture.pdf, page 174.
//   rdna2-shader-instruction-set-architecture.pdf, page 171.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 361.
//   rdna4-instruction-set-architecture.pdf, page 400.
def V_PK_MAX_I16_sem : InstSISemantic<V_PK_MAX_I16> {
  let Semantic = [];
}

// D.v2i16 = min(S0.v2i16, S1.v2i16).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 264.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 278.
//   rdna-shader-instruction-set-architecture.pdf, page 175.
//   rdna2-shader-instruction-set-architecture.pdf, page 171.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 361.
//   rdna4-instruction-set-architecture.pdf, page 401.
def V_PK_MIN_I16_sem : InstSISemantic<V_PK_MIN_I16> {
  let Semantic = [];
}

// D.v2u16 = max(S0.v2u16, S1.v2u16).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 265.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 279.
//   rdna-shader-instruction-set-architecture.pdf, page 175.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 362.
//   rdna4-instruction-set-architecture.pdf, page 402.
def V_PK_MAX_U16_sem : InstSISemantic<V_PK_MAX_U16> {
  let Semantic = [];
}

// D.v2u16 = min(S0.v2u16, S1.v2u16).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 265.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 279.
//   rdna-shader-instruction-set-architecture.pdf, page 175.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 362.
//   rdna4-instruction-set-architecture.pdf, page 402.
def V_PK_MIN_U16_sem : InstSISemantic<V_PK_MIN_U16> {
  let Semantic = [];
}

// D.v2f16 = maximum(S0.v2f16, S1.v2f16) (IEEE 754-2019).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 406.
def V_PK_MAXIMUM_F16_sem : InstSISemantic<V_PK_MAXIMUM_F16> {
  let Semantic = [];
}

// D.v2f16 = minimum(S0.v2f16, S1.v2f16) (IEEE 754-2019).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 406.
def V_PK_MINIMUM_F16_sem : InstSISemantic<V_PK_MINIMUM_F16> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// V_PK_* Packed Shift
//------------------------------------------------------------------------------

// D.v2u16 = S1.v2u16 << S0.v2u16.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 154.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 154.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 263.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 277.
//   rdna-shader-instruction-set-architecture.pdf, page 174.
//   rdna2-shader-instruction-set-architecture.pdf, page 171.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 360.
//   rdna4-instruction-set-architecture.pdf, page 400.
def V_PK_LSHLREV_B16_sem : InstSISemantic<V_PK_LSHLREV_B16> {
  let Semantic = [];
}

// D.v2u16 = S1.v2u16 >> S0.v2u16 (logical).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 154.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 154.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 263.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 277.
//   rdna-shader-instruction-set-architecture.pdf, page 174.
//   rdna2-shader-instruction-set-architecture.pdf, page 171.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 361.
//   rdna4-instruction-set-architecture.pdf, page 400.
def V_PK_LSHRREV_B16_sem : InstSISemantic<V_PK_LSHRREV_B16> {
  let Semantic = [];
}

// D.v2i16 = S1.v2i16 >> S0.v2u16 (arithmetic).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 163.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 154.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 154.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 264.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 278.
//   rdna-shader-instruction-set-architecture.pdf, page 174.
//   rdna2-shader-instruction-set-architecture.pdf, page 171.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 361.
//   rdna4-instruction-set-architecture.pdf, page 400.
def V_PK_ASHRREV_I16_sem : InstSISemantic<V_PK_ASHRREV_I16> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// V_PK_MOV
//------------------------------------------------------------------------------

// D.u = S0.u (move with packed semantics).
// Sources:
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 156.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 271.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 288.
def V_PK_MOV_B32_sem : InstSISemantic<V_PK_MOV_B32> {
  let Semantic = [];
}

//==============================================================================
// VOP3P Dot Product Instructions
//==============================================================================

//------------------------------------------------------------------------------
// V_DOT2 (two-element dot product)
//------------------------------------------------------------------------------

// D.f32 = dot2(S0.v2f16, S1.v2f16) + S2.f32.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 164.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 268.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 283.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 363.
//   rdna4-instruction-set-architecture.pdf, page 403.
def V_DOT2_F32_F16_sem : InstSISemantic<V_DOT2_F32_F16> {
  let Semantic = [];
}

// D.f32 = dot2(S0.v2bf16, S1.v2bf16) + S2.f32.
// Sources:
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 281.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 365.
//   rdna4-instruction-set-architecture.pdf, page 405.
def V_DOT2_F32_BF16_sem : InstSISemantic<V_DOT2_F32_BF16> {
  let Semantic = [];
}

// D.i32 = dot2(S0.v2i16, S1.v2i16) + S2.i32.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 164.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 269.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 284.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
def V_DOT2_I32_I16_sem : InstSISemantic<V_DOT2_I32_I16> {
  let Semantic = [];
}

// D.u32 = dot2(S0.v2u16, S1.v2u16) + S2.u32.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 164.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 155.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 155.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 269.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 284.
//   rdna2-shader-instruction-set-architecture.pdf, page 172.
def V_DOT2_U32_U16_sem : InstSISemantic<V_DOT2_U32_U16> {
  let Semantic = [];
}

// D.v2bf16 = dot2 with packed bf16 result.
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 437.
//   rdna4-instruction-set-architecture.pdf, page 499.
def V_DOT2_BF16_BF16_sem : InstSISemantic<V_DOT2_BF16_BF16> {
  let Semantic = [];
}

// D.v2f16 = dot2 with packed f16 result.
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 436.
//   rdna4-instruction-set-architecture.pdf, page 499.
def V_DOT2_F16_F16_sem : InstSISemantic<V_DOT2_F16_F16> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// V_DOT4 (four-element dot product)
//------------------------------------------------------------------------------

// D.i32 = dot4(S0.v4i8, S1.v4i8) + S2.i32.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 164.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 269.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 284.
//   rdna2-shader-instruction-set-architecture.pdf, page 173.
def V_DOT4_I32_I8_sem : InstSISemantic<V_DOT4_I32_I8> {
  let Semantic = [];
}

// D.i32 = dot4(S0.v4iu8, S1.v4iu8) + S2.i32 (mixed sign).
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 364.
//   rdna4-instruction-set-architecture.pdf, page 403.
def V_DOT4_I32_IU8_sem : InstSISemantic<V_DOT4_I32_IU8> {
  let Semantic = [];
}

// D.u32 = dot4(S0.v4u8, S1.v4u8) + S2.u32.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 164.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 156.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 156.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 269.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 285.
//   rdna2-shader-instruction-set-architecture.pdf, page 173.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 364.
//   rdna4-instruction-set-architecture.pdf, page 404.
def V_DOT4_U32_U8_sem : InstSISemantic<V_DOT4_U32_U8> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// V_DOT4 FP8/BF8 (four-element dot product, fp8/bf8)
//------------------------------------------------------------------------------

// D.f32 = dot4(S0.v4bf8, S1.v4bf8) + S2.f32.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 409.
def V_DOT4_F32_BF8_BF8_sem : InstSISemantic<V_DOT4_F32_BF8_BF8> {
  let Semantic = [];
}

// D.f32 = dot4(S0.v4bf8, S1.v4fp8) + S2.f32.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 408.
def V_DOT4_F32_BF8_FP8_sem : InstSISemantic<V_DOT4_F32_BF8_FP8> {
  let Semantic = [];
}

// D.f32 = dot4(S0.v4fp8, S1.v4bf8) + S2.f32.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 408.
def V_DOT4_F32_FP8_BF8_sem : InstSISemantic<V_DOT4_F32_FP8_BF8> {
  let Semantic = [];
}

// D.f32 = dot4(S0.v4fp8, S1.v4fp8) + S2.f32.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 409.
def V_DOT4_F32_FP8_FP8_sem : InstSISemantic<V_DOT4_F32_FP8_FP8> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// V_DOT8 (eight-element dot product)
//------------------------------------------------------------------------------

// D.i32 = dot8(S0.v8i4, S1.v8i4) + S2.i32.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 164.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 270.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 285.
//   rdna2-shader-instruction-set-architecture.pdf, page 173.
def V_DOT8_I32_I4_sem : InstSISemantic<V_DOT8_I32_I4> {
  let Semantic = [];
}

// D.i32 = dot8(S0.v8iu4, S1.v8iu4) + S2.i32 (mixed sign).
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 365.
//   rdna4-instruction-set-architecture.pdf, page 404.
def V_DOT8_I32_IU4_sem : InstSISemantic<V_DOT8_I32_IU4> {
  let Semantic = [];
}

// D.u32 = dot8(S0.v8u4, S1.v8u4) + S2.u32.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 164.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 156.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 156.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 270.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 285.
//   rdna2-shader-instruction-set-architecture.pdf, page 173.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 365.
//   rdna4-instruction-set-architecture.pdf, page 405.
def V_DOT8_U32_U4_sem : InstSISemantic<V_DOT8_U32_U4> {
  let Semantic = [];
}
