//===- SOPCSem.td -------------------------------------------------*- tblgen -*-===//
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

//------------------------------------------------------------------------------
// SOPC Integer Comparison Instructions
// SCC = S0 <cmp> S1.
//------------------------------------------------------------------------------

// Sources (present in all architectures):
//   gcn3-instruction-set-architecture.pdf, pages 138-142.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 117.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 133-135.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 145-147.
//   rdna-shader-instruction-set-architecture.pdf, pages 131-133.
//   rdna2-shader-instruction-set-architecture.pdf, page 131.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 238-240.
//   rdna4-instruction-set-architecture.pdf, pages 267-269.

foreach i = 0-5 in {
  defvar op = [S_CMP_EQ_I32, S_CMP_LG_I32, S_CMP_GT_I32,
               S_CMP_GE_I32, S_CMP_LT_I32, S_CMP_LE_I32][i];
  defvar cmpPred = [LLVMICmpEQ, LLVMICmpNE, LLVMICmpSGT, LLVMICmpSGE, LLVMICmpSLT, LLVMICmpSLE][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      SetSCCImplicit<(LLVMCmp cmpPred,
        (GetNamedOperand $src0), (GetNamedOperand $src1))>.ret
    ];
  }
}

foreach i = 0-5 in {
  defvar op = [S_CMP_EQ_U32, S_CMP_LG_U32, S_CMP_GT_U32,
               S_CMP_GE_U32, S_CMP_LT_U32, S_CMP_LE_U32][i];
  defvar cmpPred = [LLVMICmpEQ, LLVMICmpNE, LLVMICmpUGT, LLVMICmpUGE, LLVMICmpULT, LLVMICmpULE][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      SetSCCImplicit<(LLVMCmp cmpPred,
        (GetNamedOperand $src0, si_llvm_i32_ty), (GetNamedOperand $src1, si_llvm_i32_ty))>.ret
    ];
  }
}

//------------------------------------------------------------------------------
// SOPC 64-bit Comparison Instructions
//------------------------------------------------------------------------------

// SCC = (S0 == S1) (64-bit unsigned).
// Sources (introduced in GFX8/Vega):
//   vega-7nm-shader-instruction-set-architecture.pdf, page 118.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 107.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 137.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 149.
//   rdna-shader-instruction-set-architecture.pdf, page 133.
//   rdna2-shader-instruction-set-architecture.pdf, page 131.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 241.
//   rdna4-instruction-set-architecture.pdf, page 270.
def S_CMP_EQ_U64_sem : InstSISemantic<S_CMP_EQ_U64> {
  let Semantic = [
    SetSCCImplicit<(LLVMCmp LLVMICmpEQ,
      (GetNamedOperand $src0), (GetNamedOperand $src1))>.ret
  ];
}

// SCC = (S0 != S1) (64-bit unsigned).
// Sources: same as S_CMP_EQ_U64.
def S_CMP_LG_U64_sem : InstSISemantic<S_CMP_LG_U64> {
  let Semantic = [
    SetSCCImplicit<(LLVMCmp LLVMICmpNE,
      (GetNamedOperand $src0), (GetNamedOperand $src1))>.ret
  ];
}

//------------------------------------------------------------------------------
// SOPC Bit Compare Instructions
//------------------------------------------------------------------------------

// SCC = (S0.u[S1.u[4:0]] == 0). Test if bit is zero.
// Sources (present in all architectures):
//   gcn3-instruction-set-architecture.pdf, page 138.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 117.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 135.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 147.
//   rdna-shader-instruction-set-architecture.pdf, page 133.
//   rdna2-shader-instruction-set-architecture.pdf, page 133.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 240.
//   rdna4-instruction-set-architecture.pdf, page 269.
def S_BITCMP0_B32_sem : InstSISemantic<S_BITCMP0_B32> {
  let Semantic = [
    SetSCCImplicit<(LLVMCmp LLVMICmpEQ,
      (LLVMAnd (LLVMLShr (GetNamedOperand $src0),
                          (LLVMAnd (GetNamedOperand $src1),
                                   (ConstantInt si_llvm_i32_ty, 31))),
               (ConstantInt si_llvm_i32_ty, 1)),
      (ConstantInt si_llvm_i32_ty, 0))>.ret
  ];
}

// SCC = (S0.u[S1.u[5:0]] == 0). Test if bit is zero (64-bit).
// Sources: same as S_BITCMP0_B32.
def S_BITCMP0_B64_sem : InstSISemantic<S_BITCMP0_B64> {
  let Semantic = [
    SetSCCImplicit<(LLVMCmp LLVMICmpEQ,
      (LLVMAnd (LLVMLShr (GetNamedOperand $src0),
                          (LLVMZExt (LLVMAnd (GetNamedOperand $src1),
                                             (ConstantInt si_llvm_i32_ty, 63)),
                                    si_llvm_i64_ty)),
               (ConstantInt si_llvm_i64_ty, 1)),
      (ConstantInt si_llvm_i64_ty, 0))>.ret
  ];
}

// SCC = (S0.u[S1.u[4:0]] == 1). Test if bit is one.
// Sources: same as S_BITCMP0_B32.
def S_BITCMP1_B32_sem : InstSISemantic<S_BITCMP1_B32> {
  let Semantic = [
    SetSCCImplicit<(LLVMCmp LLVMICmpEQ,
      (LLVMAnd (LLVMLShr (GetNamedOperand $src0),
                          (LLVMAnd (GetNamedOperand $src1),
                                   (ConstantInt si_llvm_i32_ty, 31))),
               (ConstantInt si_llvm_i32_ty, 1)),
      (ConstantInt si_llvm_i32_ty, 1))>.ret
  ];
}

// SCC = (S0.u[S1.u[5:0]] == 1). Test if bit is one (64-bit).
// Sources: same as S_BITCMP0_B32.
def S_BITCMP1_B64_sem : InstSISemantic<S_BITCMP1_B64> {
  let Semantic = [
    SetSCCImplicit<(LLVMCmp LLVMICmpEQ,
      (LLVMAnd (LLVMLShr (GetNamedOperand $src0),
                          (LLVMZExt (LLVMAnd (GetNamedOperand $src1),
                                             (ConstantInt si_llvm_i32_ty, 63)),
                                    si_llvm_i64_ty)),
               (ConstantInt si_llvm_i64_ty, 1)),
      (ConstantInt si_llvm_i64_ty, 1))>.ret
  ];
}

//------------------------------------------------------------------------------
// SOPC Scalar Float Comparison Instructions (HasSALUFloatInsts, GFX11+)
//------------------------------------------------------------------------------

// Sources (introduced in GFX11/RDNA3, HasSALUFloatInsts):
//   rdna4-instruction-set-architecture.pdf, pages 270-275.
//   (Not present in GCN3, Vega, CDNA, RDNA1, RDNA2, or RDNA3 SOPC sections.)

// Ordered float comparisons: SCC = S0 <cmp> S1 (f32).
foreach i = 0-5 in {
  defvar op = [S_CMP_EQ_F32, S_CMP_LG_F32, S_CMP_GT_F32,
               S_CMP_GE_F32, S_CMP_LT_F32, S_CMP_LE_F32][i];
  defvar cmpPred = [LLVMFCmpOEQ, LLVMFCmpONE, LLVMFCmpOGT, LLVMFCmpOGE, LLVMFCmpOLT, LLVMFCmpOLE][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      SetSCCImplicit<(LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty))>.ret,
    ];
  }
}

// Ordered float comparisons: SCC = S0 <cmp> S1 (f16).
foreach i = 0-5 in {
  defvar op = [S_CMP_EQ_F16, S_CMP_LG_F16, S_CMP_GT_F16,
               S_CMP_GE_F16, S_CMP_LT_F16, S_CMP_LE_F16][i];
  defvar cmpPred = [LLVMFCmpOEQ, LLVMFCmpONE, LLVMFCmpOGT, LLVMFCmpOGE, LLVMFCmpOLT, LLVMFCmpOLE][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      SetSCCImplicit<(LLVMFCmp cmpPred,
        (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty), si_llvm_half_ty),
        (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty), si_llvm_half_ty))>.ret
    ];
  }
}

// Unordered float comparisons: SCC = !(S0 <cmp> S1) (f32).
// NEQ = unordered not equal, NGE = unordered less than, etc.
foreach i = 0-5 in {
  defvar op = [S_CMP_NEQ_F32, S_CMP_NLG_F32, S_CMP_NGT_F32,
               S_CMP_NGE_F32, S_CMP_NLT_F32, S_CMP_NLE_F32][i];
  defvar cmpPred = [LLVMFCmpUNE, LLVMFCmpUEQ, LLVMFCmpULE, LLVMFCmpULT, LLVMFCmpUGE, LLVMFCmpUGT][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      SetSCCImplicit<(LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty))>.ret
    ];
  }
}

// Unordered float comparisons: SCC = !(S0 <cmp> S1) (f16).
foreach i = 0-5 in {
  defvar op = [S_CMP_NEQ_F16, S_CMP_NLG_F16, S_CMP_NGT_F16,
               S_CMP_NGE_F16, S_CMP_NLT_F16, S_CMP_NLE_F16][i];
  defvar cmpPred = [LLVMFCmpUNE, LLVMFCmpUEQ, LLVMFCmpULE, LLVMFCmpULT, LLVMFCmpUGE, LLVMFCmpUGT][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      SetSCCImplicit<(LLVMFCmp cmpPred,
        (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty), si_llvm_half_ty),
        (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty), si_llvm_half_ty))>.ret
    ];
  }
}

// SCC = !isNaN(S0) && !isNaN(S1). Ordered (both not NaN).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 272 (F32), page 273 (F16).
def S_CMP_O_F32_sem : InstSISemantic<S_CMP_O_F32> {
  let Semantic = [
    SetSCCImplicit<(LLVMFCmp LLVMFCmpOLE,
      (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
      (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty))>.ret
  ];
}

def S_CMP_O_F16_sem : InstSISemantic<S_CMP_O_F16> {
  let Semantic = [
    SetSCCImplicit<(LLVMFCmp LLVMFCmpOLE,
      (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty), si_llvm_half_ty),
      (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty), si_llvm_half_ty))>.ret
  ];
}

// SCC = isNaN(S0) || isNaN(S1). Unordered (either is NaN).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 273 (F32), page 273 (F16).
def S_CMP_U_F32_sem : InstSISemantic<S_CMP_U_F32> {
  let Semantic = [
    SetSCCImplicit<(LLVMFCmp LLVMFCmpUNO,
      (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
      (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty))>.ret
  ];
}

def S_CMP_U_F16_sem : InstSISemantic<S_CMP_U_F16> {
  let Semantic = [
    SetSCCImplicit<(LLVMFCmp LLVMFCmpUNO,
      (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty), si_llvm_half_ty),
      (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty), si_llvm_half_ty))>.ret
  ];
}

//------------------------------------------------------------------------------
// S_SET_GPR_IDX_ON $src0, $src1 (GFX9)
//
// Enables VGPR indirect indexing for following VALU instructions:
//   M0[7:0]   := $src0 & 0xFF       // the index
//   M0[14:11] := $src1 & 0xF        // per-operand enable mask
//   MODE.GPR_IDX_EN := 1            // bit 27
// Other M0/MODE bits are preserved. The companion read-side wrapper in
// `TraceFunctionTranslator::emitIndexedVGPRSrc` consults MODE.GPR_IDX_EN at
// every VALU operand read; pairs with `S_SET_GPR_IDX_OFF` to bracket
// the indexed region.
//------------------------------------------------------------------------------
def S_SET_GPR_IDX_ON_sem : InstSISemantic<S_SET_GPR_IDX_ON> {
  let Semantic = [
    // Clear M0[7:0] and M0[14:11] in one mask (= 0xFFFF8700).
    (DefVal $m0_cleared,
      (LLVMAnd (ImplicitUse M0),
               (ConstantInt si_llvm_i32_ty, 0xFFFF8700))),
    (DefVal $idx_low,
      (LLVMAnd (GetNamedOperand $src0),
               (ConstantInt si_llvm_i32_ty, 0xFF))),
    (DefVal $mask_shifted,
      (LLVMShl (LLVMAnd (GetNamedOperand $src1),
                        (ConstantInt si_llvm_i32_ty, 0xF)),
               (ConstantInt si_llvm_i32_ty, 11))),
    (ImplicitDef M0,
      (LLVMOr (LLVMOr (GetVal $m0_cleared), (GetVal $idx_low)),
              (GetVal $mask_shifted))),
    // MODE.GPR_IDX_EN := 1 (bit 27 = 0x08000000).
    (ImplicitDef MODE,
      (LLVMOr (ImplicitUse MODE),
              (ConstantInt si_llvm_i32_ty, 0x08000000)))
  ];
}
