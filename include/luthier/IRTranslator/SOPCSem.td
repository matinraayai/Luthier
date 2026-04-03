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
  defvar cmpPred = [SETEQ, SETNE, SETGT, SETGE, SETLT, SETLE][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      (ImplicitDef SCC, (LLVMCmp cmpPred,
        (GetNamedOperand $src0), (GetNamedOperand $src1)))
    ];
  }
}

foreach i = 0-5 in {
  defvar op = [S_CMP_EQ_U32, S_CMP_LG_U32, S_CMP_GT_U32,
               S_CMP_GE_U32, S_CMP_LT_U32, S_CMP_LE_U32][i];
  defvar cmpPred = [SETEQ, SETNE, SETUGT, SETUGE, SETULT, SETULE][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      (ImplicitDef SCC, (LLVMCmp cmpPred,
        (GetNamedOperand $src0), (GetNamedOperand $src1)))
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
    (ImplicitDef SCC, (LLVMCmp LLVMICmpEQ,
      (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// SCC = (S0 != S1) (64-bit unsigned).
// Sources: same as S_CMP_EQ_U64.
def S_CMP_LG_U64_sem : InstSISemantic<S_CMP_LG_U64> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE,
      (GetNamedOperand $src0), (GetNamedOperand $src1)))
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
    (ImplicitDef SCC, (LLVMCmp LLVMICmpEQ,
      (LLVMAnd (LLVMLShr (GetNamedOperand $src0),
                          (LLVMAnd (GetNamedOperand $src1),
                                   (ConstantInt llvm_i32_ty, 31))),
               (ConstantInt llvm_i32_ty, 1)),
      (ConstantInt llvm_i32_ty, 0)))
  ];
}

// SCC = (S0.u[S1.u[5:0]] == 0). Test if bit is zero (64-bit).
// Sources: same as S_BITCMP0_B32.
def S_BITCMP0_B64_sem : InstSISemantic<S_BITCMP0_B64> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp LLVMICmpEQ,
      (LLVMAnd (LLVMLShr (GetNamedOperand $src0),
                          (LLVMZExt (LLVMAnd (GetNamedOperand $src1),
                                             (ConstantInt llvm_i32_ty, 63)),
                                    llvm_i64_ty)),
               (ConstantInt llvm_i64_ty, 1)),
      (ConstantInt llvm_i64_ty, 0)))
  ];
}

// SCC = (S0.u[S1.u[4:0]] == 1). Test if bit is one.
// Sources: same as S_BITCMP0_B32.
def S_BITCMP1_B32_sem : InstSISemantic<S_BITCMP1_B32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp LLVMICmpEQ,
      (LLVMAnd (LLVMLShr (GetNamedOperand $src0),
                          (LLVMAnd (GetNamedOperand $src1),
                                   (ConstantInt llvm_i32_ty, 31))),
               (ConstantInt llvm_i32_ty, 1)),
      (ConstantInt llvm_i32_ty, 1)))
  ];
}

// SCC = (S0.u[S1.u[5:0]] == 1). Test if bit is one (64-bit).
// Sources: same as S_BITCMP0_B32.
def S_BITCMP1_B64_sem : InstSISemantic<S_BITCMP1_B64> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp LLVMICmpEQ,
      (LLVMAnd (LLVMLShr (GetNamedOperand $src0),
                          (LLVMZExt (LLVMAnd (GetNamedOperand $src1),
                                             (ConstantInt llvm_i32_ty, 63)),
                                    llvm_i64_ty)),
               (ConstantInt llvm_i64_ty, 1)),
      (ConstantInt llvm_i64_ty, 1)))
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
  defvar cmpPred = [SETOEQ, SETONE, SETOGT, SETOGE, SETOLT, SETOLE][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      (ImplicitDef SCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }
}

// Ordered float comparisons: SCC = S0 <cmp> S1 (f16).
foreach i = 0-5 in {
  defvar op = [S_CMP_EQ_F16, S_CMP_LG_F16, S_CMP_GT_F16,
               S_CMP_GE_F16, S_CMP_LT_F16, S_CMP_LE_F16][i];
  defvar cmpPred = [SETOEQ, SETONE, SETOGT, SETOGE, SETOLT, SETOLE][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      (ImplicitDef SCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
        (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty)))
    ];
  }
}

// Unordered float comparisons: SCC = !(S0 <cmp> S1) (f32).
// NEQ = unordered not equal, NGE = unordered less than, etc.
foreach i = 0-5 in {
  defvar op = [S_CMP_NEQ_F32, S_CMP_NLG_F32, S_CMP_NGT_F32,
               S_CMP_NGE_F32, S_CMP_NLT_F32, S_CMP_NLE_F32][i];
  defvar cmpPred = [SETUNE, SETUEQ, SETULE, SETULT, SETUGE, SETUGT][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      (ImplicitDef SCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }
}

// Unordered float comparisons: SCC = !(S0 <cmp> S1) (f16).
foreach i = 0-5 in {
  defvar op = [S_CMP_NEQ_F16, S_CMP_NLG_F16, S_CMP_NGT_F16,
               S_CMP_NGE_F16, S_CMP_NLT_F16, S_CMP_NLE_F16][i];
  defvar cmpPred = [SETUNE, SETUEQ, SETULE, SETULT, SETUGE, SETUGT][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      (ImplicitDef SCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
        (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty)))
    ];
  }
}

// SCC = !isNaN(S0) && !isNaN(S1). Ordered (both not NaN).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 272 (F32), page 273 (F16).
def S_CMP_O_F32_sem : InstSISemantic<S_CMP_O_F32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMFCmp LLVMFCmpOLE,
      (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
      (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
  ];
}

def S_CMP_O_F16_sem : InstSISemantic<S_CMP_O_F16> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMFCmp LLVMFCmpOLE,
      (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
      (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty)))
  ];
}

// SCC = isNaN(S0) || isNaN(S1). Unordered (either is NaN).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 273 (F32), page 273 (F16).
def S_CMP_U_F32_sem : InstSISemantic<S_CMP_U_F32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMFCmp LLVMFCmpUNO,
      (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
      (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
  ];
}

def S_CMP_U_F16_sem : InstSISemantic<S_CMP_U_F16> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMFCmp LLVMFCmpUNO,
      (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
      (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty)))
  ];
}
