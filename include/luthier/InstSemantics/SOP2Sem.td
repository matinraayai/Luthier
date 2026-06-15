//===- SOP2Sem.td ---------------------------------------------*- tblgen -*-===//
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
// SOP2 Instructions
//------------------------------------------------------------------------------

// (S_ADD_CO_U32, S_SUB_CO_U32 etc. are GFX12+ renamed mnemonics for
// S_ADD_U32, S_SUB_U32 etc.; no separate pseudo records exist.
// The semantics are already defined under the original names below.)

// D.i = |S0.i - S1.i|; SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 107.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 106.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 118.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 113.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 198.
//   rdna4-instruction-set-architecture.pdf, page 218.
def S_ABSDIFF_I32_sem : InstSISemantic<S_ABSDIFF_I32> {
  let Semantic = [
    (DefVal $diff, (LLVMSub (GetNamedOperand $src0), (GetNamedOperand $src1))),
    (DefVal $result,
      (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $diff), (ConstantInt si_llvm_i32_ty, 0)),
                  (LLVMSub (ConstantInt si_llvm_i32_ty, 0), (GetVal $diff)),
                  (GetVal $diff))),
    SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $result),
                                  (ConstantInt si_llvm_i32_ty, 0))>.ret,
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.u = S0.u + S1.u; SCC = unsigned carry out.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 108.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 100.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 108.
//   rdna-shader-instruction-set-architecture.pdf, page 110.
//   rdna2-shader-instruction-set-architecture.pdf, page 110.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 197 (as S_ADD_CO_U32).
//   rdna4-instruction-set-architecture.pdf, page 217 (as S_ADD_CO_U32).
def S_ADD_U32_sem : InstSISemantic<S_ADD_U32> {
   let Semantic = [
     (DefVal $tmp,
       (LLVMAdd (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
     SetSCCImplicit<(LLVMCmp LLVMICmpUGE, (GetVal $tmp), (ConstantInt si_llvm_i64_ty, 0x100000000))>.ret,
     (SetNamedOperand $sdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
   ];
}

// D.u = S0.u - S1.u; SCC = unsigned borrow.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 117.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 100.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 109.
//   rdna-shader-instruction-set-architecture.pdf, page 110.
//   rdna2-shader-instruction-set-architecture.pdf, page 110.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 197 (as S_SUB_CO_U32).
//   rdna4-instruction-set-architecture.pdf, page 217 (as S_SUB_CO_U32).
def S_SUB_U32_sem : InstSISemantic<S_SUB_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMSub (GetNamedOperand $src0), (GetNamedOperand $src1))),
    SetSCCImplicit<(LLVMCmp LLVMICmpUGT, (GetNamedOperand $src1),
                                       (GetNamedOperand $src0))>.ret,
    (SetNamedOperand $sdst, (GetVal $tmp))
  ];
}

// D.i = S0.i + S1.i; SCC = signed overflow.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 108.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 100.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 109.
//   rdna-shader-instruction-set-architecture.pdf, page 110.
//   rdna2-shader-instruction-set-architecture.pdf, page 110.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 197 (as S_ADD_CO_I32).
//   rdna4-instruction-set-architecture.pdf, page 217 (as S_ADD_CO_I32).
def S_ADD_I32_sem : InstSISemantic<S_ADD_I32> {
  let Semantic = [
    (DefVal $tmp, (LLVMAdd (GetNamedOperand $src0), (GetNamedOperand $src1))),
    (DefVal $src0_31, LLVMExtractValueConstIdx<(GetNamedOperand $src0), si_llvm_v32i1_ty, 31>.Op),
    (DefVal $src1_31, LLVMExtractValueConstIdx<(GetNamedOperand $src1), si_llvm_v32i1_ty, 31>.Op),
    (DefVal $sdst_31, LLVMExtractValueConstIdx<(GetNamedOperand $sdst), si_llvm_v32i1_ty, 31>.Op),
    SetSCCImplicit<(LLVMAnd (LLVMCmp LLVMICmpEQ, (GetVal $src0_31), (GetVal $src1_31)),
                               (LLVMCmp LLVMICmpNE, (GetVal $src0_31), (GetVal $sdst_31)))>.ret,
    (SetNamedOperand $sdst, (GetVal $tmp))
  ];
}

// D.i = S0.i - S1.i; SCC = signed overflow.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 117.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 100.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 109.
//   rdna-shader-instruction-set-architecture.pdf, page 110.
//   rdna2-shader-instruction-set-architecture.pdf, page 110.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 197 (as S_SUB_CO_I32).
//   rdna4-instruction-set-architecture.pdf, page 217 (as S_SUB_CO_I32).
def S_SUB_I32_sem : InstSISemantic<S_SUB_I32> {
  let Semantic = [
    (DefVal $tmp, (LLVMSub (GetNamedOperand $src0), (GetNamedOperand $src1))),
    (DefVal $src0_31, LLVMExtractValueConstIdx<(GetNamedOperand $src0), si_llvm_v32i1_ty, 31>.Op),
    (DefVal $src1_31, LLVMExtractValueConstIdx<(GetNamedOperand $src1), si_llvm_v32i1_ty, 31>.Op),
    (DefVal $sdst_31, LLVMExtractValueConstIdx<(GetNamedOperand $sdst), si_llvm_v32i1_ty, 31>.Op),
    SetSCCImplicit<(LLVMAnd (LLVMCmp LLVMICmpNE, (GetVal $src0_31), (GetVal $src1_31)),
                               (LLVMCmp LLVMICmpNE, (GetVal $src0_31), (GetVal $sdst_31)))>.ret,
    (SetNamedOperand $sdst, (GetVal $tmp))
  ];
}

// D.u = S0.u + S1.u + SCC; SCC = unsigned carry out.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 108.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 100.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 110.
//   rdna-shader-instruction-set-architecture.pdf, page 110.
//   rdna2-shader-instruction-set-architecture.pdf, page 110.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 198 (as S_ADD_CO_CI_U32).
//   rdna4-instruction-set-architecture.pdf, page 217 (as S_ADD_CO_CI_U32).
def S_ADDC_U32_sem : InstSISemantic<S_ADDC_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMAdd (LLVMAdd (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                                    (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty)),
                                    (LLVMZExt GetSCCImplicit<>.ret, si_llvm_i64_ty))),
    SetSCCImplicit<(LLVMCmp LLVMICmpUGE, (GetVal $tmp),
                                       (ConstantInt si_llvm_i64_ty, 0x100000000))>.ret,
    (SetNamedOperand $sdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

// D.u = S0.u - S1.u - SCC; SCC = unsigned borrow.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 117.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 100.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 110.
//   rdna-shader-instruction-set-architecture.pdf, page 110.
//   rdna2-shader-instruction-set-architecture.pdf, page 110.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 198 (as S_SUB_CO_CI_U32).
//   rdna4-instruction-set-architecture.pdf, page 218 (as S_SUB_CO_CI_U32).
def S_SUBB_U32_sem : InstSISemantic<S_SUBB_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMSub (LLVMSub (GetNamedOperand $src0),
                                    (GetNamedOperand $src1)),
                                    (LLVMZExt GetSCCImplicit<>.ret, si_llvm_i32_ty))),
    SetSCCImplicit<(LLVMCmp LLVMICmpUGT, (LLVMAdd (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty),
                                                (LLVMZExt GetSCCImplicit<>.ret, si_llvm_i64_ty)),
                                       (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty))>.ret,
    (SetNamedOperand $sdst, (GetVal $tmp))
  ];
}

// D.u = S0.u << S1.u[4:0]; SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 113.
//   vega-shader-instruction-set-architecture.pdf, page 101.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 115.
//   rdna-shader-instruction-set-architecture.pdf, page 112.
//   rdna2-shader-instruction-set-architecture.pdf, page 112.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 199.
//   rdna4-instruction-set-architecture.pdf, page 219.
def S_LSHL_B32_sem : InstSISemantic<S_LSHL_B32> {
  let Semantic = [
    (DefVal $shifted, (LLVMShl (GetNamedOperand $src0),
                                (LLVMAnd (GetNamedOperand $src1),
                                         (ConstantInt si_llvm_i32_ty, 31)))),
    SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $shifted),
                                      (ConstantInt si_llvm_i32_ty, 0))>.ret,
    (SetNamedOperand $sdst, (GetVal $shifted))
  ];
}

// D.u = S0.u << S1.u[5:0]; SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 113.
//   vega-shader-instruction-set-architecture.pdf, page 101.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 115.
//   rdna-shader-instruction-set-architecture.pdf, page 112.
//   rdna2-shader-instruction-set-architecture.pdf, page 112.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 199.
//   rdna4-instruction-set-architecture.pdf, page 219.
def S_LSHL_B64_sem : InstSISemantic<S_LSHL_B64> {
  let Semantic = [
    (DefVal $shifted, (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                                (LLVMAnd (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty),
                                         (ConstantInt si_llvm_i64_ty, 63)))),
    SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $shifted),
                                      (ConstantInt si_llvm_i64_ty, 0))>.ret,
    (SetNamedOperand $sdst, (GetVal $shifted))
  ];
}

// D.u = S0.u >> S1.u[4:0] (logical); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 113.
//   vega-shader-instruction-set-architecture.pdf, page 101.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 115.
//   rdna-shader-instruction-set-architecture.pdf, page 112.
//   rdna2-shader-instruction-set-architecture.pdf, page 112.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 199.
//   rdna4-instruction-set-architecture.pdf, page 219.
def S_LSHR_B32_sem : InstSISemantic<S_LSHR_B32> {
  let Semantic = [
    (DefVal $shifted, (LLVMLShr (GetNamedOperand $src0, si_llvm_i32_ty),
                                 (LLVMAnd (GetNamedOperand $src1, si_llvm_i32_ty),
                                          (ConstantInt si_llvm_i32_ty, 31)))),
    SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $shifted),
                                      (ConstantInt si_llvm_i32_ty, 0))>.ret,
    (SetNamedOperand $sdst, (GetVal $shifted))
  ];
}

// D.u = S0.u >> S1.u[5:0] (logical); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 113.
//   vega-shader-instruction-set-architecture.pdf, page 101.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 115.
//   rdna-shader-instruction-set-architecture.pdf, page 112.
//   rdna2-shader-instruction-set-architecture.pdf, page 112.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 199.
//   rdna4-instruction-set-architecture.pdf, page 219.
def S_LSHR_B64_sem : InstSISemantic<S_LSHR_B64> {
  let Semantic = [
    // src0 is a 64-bit pair (i64 by default), src1 is a 32-bit count.
    // \c LLVMAnd requires same-width operands, so request src1 as i64
    // explicitly — matches S_ASHR_I64 / S_LSHL_B64's pattern (the
    // sibling S_LSHL_B64 uses \c LLVMZExt; \c GetNamedOperand with an
    // explicit OutType achieves the same widening via the cached
    // zext-or-trunc inside getOperandAsValue).
    (DefVal $shifted, (LLVMLShr (GetNamedOperand $src0),
                                 (LLVMAnd (GetNamedOperand $src1, si_llvm_i64_ty),
                                          (ConstantInt si_llvm_i64_ty, 63)))),
    SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $shifted),
                                      (ConstantInt si_llvm_i64_ty, 0))>.ret,
    (SetNamedOperand $sdst, (GetVal $shifted))
  ];
}

// D.i = S0.i >> S1.u[4:0] (arithmetic); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 109.
//   vega-shader-instruction-set-architecture.pdf, page 102.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 116.
//   rdna-shader-instruction-set-architecture.pdf, page 113.
//   rdna2-shader-instruction-set-architecture.pdf, page 112.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 200.
//   rdna4-instruction-set-architecture.pdf, page 220.
def S_ASHR_I32_sem : InstSISemantic<S_ASHR_I32> {
  let Semantic = [
    (DefVal $shifted, (LLVMAShr (GetNamedOperand $src0),
                                 (LLVMAnd (GetNamedOperand $src1, si_llvm_i32_ty),
                                          (ConstantInt si_llvm_i32_ty, 31)))),
    SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $shifted),
                                      (ConstantInt si_llvm_i32_ty, 0))>.ret,
    (SetNamedOperand $sdst, (GetVal $shifted))
  ];
}

// D.i = S0.i >> S1.u[5:0] (arithmetic); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 110.
//   vega-shader-instruction-set-architecture.pdf, page 102.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 116.
//   rdna-shader-instruction-set-architecture.pdf, page 113.
//   rdna2-shader-instruction-set-architecture.pdf, page 112.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 200.
//   rdna4-instruction-set-architecture.pdf, page 220.
def S_ASHR_I64_sem : InstSISemantic<S_ASHR_I64> {
  let Semantic = [
    (DefVal $shifted, (LLVMAShr (GetNamedOperand $src0),
                                (LLVMAnd (GetNamedOperand $src1, si_llvm_i64_ty),
                                         (ConstantInt si_llvm_i64_ty, 63)))),
    SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $shifted),
                                      (ConstantInt si_llvm_i64_ty, 0))>.ret,
    (SetNamedOperand $sdst, (GetVal $shifted))
  ];
}

// D = (S0 << 1) + S1; SCC = unsigned carry out.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 107.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 119.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 113.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 200.
//   rdna4-instruction-set-architecture.pdf, page 220.
def S_LSHL1_ADD_U32_sem : InstSISemantic<S_LSHL1_ADD_U32> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMAdd
        (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                 (ConstantInt si_llvm_i64_ty, 1)),
        (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    SetSCCImplicit<(LLVMCmp LLVMICmpUGE, (GetVal $tmp),
                                  (ConstantInt si_llvm_i64_ty, 0x100000000))>.ret,
    (SetNamedOperand $sdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

// D = (S0 << 2) + S1; SCC = unsigned carry out.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 107.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 119.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 113.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 200.
//   rdna4-instruction-set-architecture.pdf, page 220.
def S_LSHL2_ADD_U32_sem : InstSISemantic<S_LSHL2_ADD_U32> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMAdd
        (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                 (ConstantInt si_llvm_i64_ty, 2)),
        (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    SetSCCImplicit<(LLVMCmp LLVMICmpUGE, (GetVal $tmp),
                                  (ConstantInt si_llvm_i64_ty, 0x100000000))>.ret,
    (SetNamedOperand $sdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

// D = (S0 << 3) + S1; SCC = unsigned carry out.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 107.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 119.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 201.
//   rdna4-instruction-set-architecture.pdf, page 221.
def S_LSHL3_ADD_U32_sem : InstSISemantic<S_LSHL3_ADD_U32> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMAdd
        (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                 (ConstantInt si_llvm_i64_ty, 3)),
        (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    SetSCCImplicit<(LLVMCmp LLVMICmpUGE, (GetVal $tmp),
                                  (ConstantInt si_llvm_i64_ty, 0x100000000))>.ret,
    (SetNamedOperand $sdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

// D = (S0 << 4) + S1; SCC = unsigned carry out.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 108.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 120.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 201.
//   rdna4-instruction-set-architecture.pdf, page 221.
def S_LSHL4_ADD_U32_sem : InstSISemantic<S_LSHL4_ADD_U32> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMAdd
        (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                 (ConstantInt si_llvm_i64_ty, 4)),
        (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    SetSCCImplicit<(LLVMCmp LLVMICmpUGE, (GetVal $tmp),
                                  (ConstantInt si_llvm_i64_ty, 0x100000000))>.ret,
    (SetNamedOperand $sdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

// D = min/max(S0, S1); SCC = (S0 < S1) or (S0 >= S1) respectively.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 114.
//   rdna4-instruction-set-architecture.pdf, page 221 (MIN), page 221-222 (MAX).
foreach i = 0-3 in {
  defvar op = [S_MIN_I32, S_MIN_U32, S_MAX_I32, S_MAX_U32][i];
  // Integer min/max — the original predicate list used \c LLVMFCmpO*/U*
  // (float-compare) variants on integer operands, which emits an FCmp
  // on i32 and fails IR builder type checks. Use the matching ICmp
  // signed-vs-unsigned predicates.
  defvar cmpSgn = [LLVMICmpSLT, LLVMICmpULT, LLVMICmpSGE, LLVMICmpUGE][i];

  def op#_sem : InstSISemantic<op> {
    let Semantic = [
      SetSCCImplicit<(LLVMCmp cmpSgn, (GetNamedOperand $src0), (GetNamedOperand $src1))>.ret,
      (SetNamedOperand $sdst, (LLVMSelect GetSCCImplicit<>.ret, (GetNamedOperand $src0), (GetNamedOperand $src1)))
    ];
  }
}

// D = S0 op S1; SCC = (D != 0). Bitwise AND/OR/XOR.
// Sources:
//   gcn3-instruction-set-architecture.pdf, pages 108-109 (AND), 116 (OR), 118 (XOR).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 101-102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 100-101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 111-112.
//   rdna-shader-instruction-set-architecture.pdf, pages 111-112.
//   rdna2-shader-instruction-set-architecture.pdf, page 111.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 202-203.
//   rdna4-instruction-set-architecture.pdf, pages 222-223.
foreach op = ["And", "Or", "Xor"] in {
  foreach size = ["32", "64"] in {
    def S_#!toupper(op)#_B#size#_sem : InstSISemantic<!cast<InstSI>("S_"#!toupper(op)#"_B"#size)> {
      let Semantic =
        [(SetNamedOperand $sdst, (!cast<LLVMOp>("LLVM"#op)
          (GetNamedOperand $src0, !cast<SILLVMType>("si_llvm_i"#size#"_ty")),
          (GetNamedOperand $src1, !cast<SILLVMType>("si_llvm_i"#size#"_ty")))),
         SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetNamedOperand $sdst), (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 0))>.ret
        ];
    }
  }
}

// D = S0 op ~S1; SCC = (D != 0). Bitwise ANDN2/ORN2.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 109 (ANDN2), page 116-117 (ORN2).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 113.
//   rdna-shader-instruction-set-architecture.pdf, pages 111-112.
//   rdna2-shader-instruction-set-architecture.pdf, page 111.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 204-205 (as AND_NOT1/OR_NOT1).
//   rdna4-instruction-set-architecture.pdf, pages 224-225 (as AND_NOT1/OR_NOT1).
foreach op = ["And", "Or"] in {
  foreach size = [32, 64] in {
    def S_#!toupper(op)#N2_B#size#_sem : InstSISemantic<!cast<InstSI>("S_"#!toupper(op)#"N2_B"#size)> {
      let Semantic = [
        (SetNamedOperand $sdst, (!cast<LLVMOp>("LLVM"#op) (GetNamedOperand $src0), LLVMNot<(GetNamedOperand $src1), !cast<SILLVMType>("si_llvm_i"#size#"_ty")>.ret)),
        SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetNamedOperand $sdst), (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 0))>.ret
      ];
    }
  }
}

// D = ~(S0 & S1); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 115.
//   rdna4-instruction-set-architecture.pdf, page 223.
foreach size = [32, 64] in {
  def S_NAND_B#size#_sem : InstSISemantic<!cast<InstSI>("S_NAND_B"#size)> {
    let Semantic = [
      (DefVal $result, (LLVMXor (LLVMAnd (GetNamedOperand $src0), (GetNamedOperand $src1)),
                                 (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), -1))),
      SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 0))>.ret,
      (SetNamedOperand $sdst, (GetVal $result))
    ];
  }
}

// D = ~(S0 | S1); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 115 (B32), page 116 (B64).
//   rdna4-instruction-set-architecture.pdf, page 224.
foreach size = [32, 64] in {
  def S_NOR_B#size#_sem : InstSISemantic<!cast<InstSI>("S_NOR_B"#size)> {
    let Semantic = [
      (DefVal $result, (LLVMXor (LLVMOr (GetNamedOperand $src0), (GetNamedOperand $src1)),
                                 (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), -1))),
      SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 0))>.ret,
      (SetNamedOperand $sdst, (GetVal $result))
    ];
  }
}

// D = ~(S0 ^ S1); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 118.
//   rdna4-instruction-set-architecture.pdf, page 224.
foreach size = [32, 64] in {
  def S_XNOR_B#size#_sem : InstSISemantic<!cast<InstSI>("S_XNOR_B"#size)> {
    let Semantic = [
      (DefVal $result, (LLVMXor (LLVMXor (GetNamedOperand $src0), (GetNamedOperand $src1)),
                                 (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), -1))),
      SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 0))>.ret,
      (SetNamedOperand $sdst, (GetVal $result))
    ];
  }
}

// (S_AND_NOT1/S_OR_NOT1 are GFX11+ renamed mnemonics for S_ANDN2/S_ORN2;
// no separate pseudo records exist.)

// D = ((1 << S0[4:0]) - 1) << S1[4:0]. Bitfield mask.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 111 (B32), page 112 (B64).
//   rdna4-instruction-set-architecture.pdf, page 226.
foreach size = [32, 64] in {
  defvar maskBits = !if(!eq(size, 32), 31, 63);
  def S_BFM_B#size#_sem : InstSISemantic<!cast<InstSI>("S_BFM_B"#size)> {
    let Semantic = [
      (DefVal $count, (LLVMAnd (GetNamedOperand $src0), (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), maskBits))),
      (DefVal $shift, (LLVMAnd (GetNamedOperand $src1), (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), maskBits))),
      (DefVal $ones, (LLVMSub (LLVMShl (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 1), (GetVal $count)),
                               (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 1))),
      (SetNamedOperand $sdst, (LLVMShl (GetVal $ones), (GetVal $shift)))
    ];
  }
}

// D.i = S0.i * S1.i (low 32 bits).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 115.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 116.
//   rdna-shader-instruction-set-architecture.pdf, page 113.
//   rdna2-shader-instruction-set-architecture.pdf, page 113.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 207.
//   rdna4-instruction-set-architecture.pdf, page 227.
def S_MUL_I32_sem : InstSISemantic<S_MUL_I32> {
  let Semantic = [
    (SetNamedOperand $sdst,
      (LLVMMul (LLVMZExt (GetNamedOperand $src0), si_llvm_i32_ty),
               (LLVMZExt (GetNamedOperand $src1), si_llvm_i32_ty)))
  ];
}

// D.u = (S0.u * S1.u) >> 32 (high 32 bits of unsigned multiply).
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 103.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 106.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 118.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 207.
//   rdna4-instruction-set-architecture.pdf, page 227.
def S_MUL_HI_U32_sem : InstSISemantic<S_MUL_HI_U32> {
  let Semantic = [
    (DefVal $tmp, (LLVMMul (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                        (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (SetNamedOperand $sdst, (LLVMTrunc (LLVMLShr (GetVal $tmp), (ConstantInt si_llvm_i64_ty, 32)), si_llvm_i32_ty))
  ];
}

// D.i = (S0.i * S1.i) >> 32 (high 32 bits of signed multiply).
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 103.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 107.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 119.
//   rdna-shader-instruction-set-architecture.pdf, page 115.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 207.
//   rdna4-instruction-set-architecture.pdf, page 227.
def S_MUL_HI_I32_sem : InstSISemantic<S_MUL_HI_I32> {
  let Semantic = [
    (DefVal $tmp, (LLVMMul (LLVMSExt (GetNamedOperand $src0), si_llvm_i64_ty),
                        (LLVMSExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (SetNamedOperand $sdst, (LLVMTrunc (LLVMLShr (GetVal $tmp), (ConstantInt si_llvm_i64_ty, 32)), si_llvm_i32_ty))
  ];
}

// D = SCC ? S0 : S1. Conditional select.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 112.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 100.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 111.
//   rdna-shader-instruction-set-architecture.pdf, page 111.
//   rdna2-shader-instruction-set-architecture.pdf, page 111.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 207.
//   rdna4-instruction-set-architecture.pdf, page 227.
foreach size = [32, 64] in {
  def S_CSELECT_B#size#_sem : InstSISemantic<!cast<InstSI>("S_CSELECT_B"#size)> {
    let Semantic = [(SetNamedOperand $sdst,
                    (LLVMSelect GetSCCImplicit<>.ret, (GetNamedOperand $src0), (GetNamedOperand $src1)))];
  }
}

// D = {S0[15:0], S1[15:0]}. Pack low halves.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 108.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 120.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 207.
//   rdna4-instruction-set-architecture.pdf, page 228.
def S_PACK_LL_B32_B16_sem : InstSISemantic<S_PACK_LL_B32_B16> {
  let Semantic = [
    (DefVal $src0_lo, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $src1_lo, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $packed, (LLVMOr (LLVMZExt (GetVal $src1_lo), si_llvm_i32_ty),
                             (LLVMShl (LLVMZExt (GetVal $src0_lo), si_llvm_i32_ty),
                                      (ConstantInt si_llvm_i32_ty, 16)))),
    (SetNamedOperand $sdst, (GetVal $packed))
  ];
}

// D = {S0[15:0], S1[31:16]}. Pack low of S0, high of S1.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 108.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 120.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 208.
//   rdna4-instruction-set-architecture.pdf, page 228.
def S_PACK_LH_B32_B16_sem : InstSISemantic<S_PACK_LH_B32_B16> {
  let Semantic = [
    (DefVal $src0_lo, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $src1_hi, (LLVMTrunc (LLVMLShr (GetNamedOperand $src1), (ConstantInt si_llvm_i32_ty, 16)), si_llvm_i16_ty)),
    (DefVal $packed, (LLVMOr (LLVMZExt (GetVal $src1_hi), si_llvm_i32_ty),
                             (LLVMShl (LLVMZExt (GetVal $src0_lo), si_llvm_i32_ty),
                                      (ConstantInt si_llvm_i32_ty, 16)))),
    (SetNamedOperand $sdst, (GetVal $packed))
  ];
}

// D = {S0[31:16], S1[31:16]}. Pack high halves.
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 108.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 120.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 208.
//   rdna4-instruction-set-architecture.pdf, page 228.
def S_PACK_HH_B32_B16_sem : InstSISemantic<S_PACK_HH_B32_B16> {
  let Semantic = [
    (DefVal $src0_hi, (LLVMTrunc (LLVMLShr (GetNamedOperand $src0), (ConstantInt si_llvm_i32_ty, 16)), si_llvm_i16_ty)),
    (DefVal $src1_hi, (LLVMTrunc (LLVMLShr (GetNamedOperand $src1), (ConstantInt si_llvm_i32_ty, 16)), si_llvm_i16_ty)),
    (DefVal $packed, (LLVMOr (LLVMZExt (GetVal $src1_hi), si_llvm_i32_ty),
                             (LLVMShl (LLVMZExt (GetVal $src0_hi), si_llvm_i32_ty),
                                      (ConstantInt si_llvm_i32_ty, 16)))),
    (SetNamedOperand $sdst, (GetVal $packed))
  ];
}

// D = {S0[31:16], S1[15:0]}. Pack high of S0, low of S1.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 228.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 208.
//   rdna4-instruction-set-architecture.pdf, page 228.
def S_PACK_HL_B32_B16_sem : InstSISemantic<S_PACK_HL_B32_B16> {
  let Semantic = [
    (DefVal $src0_hi, (LLVMTrunc (LLVMLShr (GetNamedOperand $src0), (ConstantInt si_llvm_i32_ty, 16)), si_llvm_i16_ty)),
    (DefVal $src1_lo, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $packed, (LLVMOr (LLVMZExt (GetVal $src1_lo), si_llvm_i32_ty),
                             (LLVMShl (LLVMZExt (GetVal $src0_hi), si_llvm_i32_ty),
                                      (ConstantInt si_llvm_i32_ty, 16)))),
    (SetNamedOperand $sdst, (GetVal $packed))
  ];
}

// D = (S0 >> S1[4:0]) & ((1 << S1[22:16]) - 1). Unsigned bitfield extract.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 111.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 102.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 117.
//   rdna-shader-instruction-set-architecture.pdf, page 113.
//   rdna2-shader-instruction-set-architecture.pdf, page 113.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 205-206.
//   rdna4-instruction-set-architecture.pdf, pages 225-226.
foreach size = [32, 64] in {
  def S_BFE_U#size#_sem : InstSISemantic<!cast<InstSI>("S_BFE_U"#size)> {
    let Semantic = [
      (DefVal $shift_amt,
        LLVMExtractBitsIntStartIntNumBits<(GetNamedOperand $src1), !cast<SILLVMType>("si_llvm_i"#size#"_ty"), false, 0, !if(!eq(size, 64), 6, 5)>.Op),
      (DefVal $bit_width,
        LLVMExtractBitsIntStartIntNumBits<(GetNamedOperand $src1), !cast<SILLVMType>("si_llvm_i"#size#"_ty"), false, 16, 7>.Op),
      (DefVal $shifted, (LLVMLShr (GetNamedOperand $src0), (GetVal $shift_amt))),
      (DefVal $mask, (LLVMSub (LLVMShl (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 1), (GetVal $bit_width)),
                               (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 1))),
      (DefVal $result, (LLVMAnd (GetVal $shifted), (GetVal $mask))),
      SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 0))>.ret,
      (SetNamedOperand $sdst, (GetVal $result))
    ];
  }
}

// D = signext((S0 >> S1[4:0]) & ((1 << S1[22:16]) - 1)). Signed bitfield extract.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 110.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 102.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 96.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 117.
//   rdna-shader-instruction-set-architecture.pdf, page 113.
//   rdna2-shader-instruction-set-architecture.pdf, page 113.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 206.
//   rdna4-instruction-set-architecture.pdf, page 226.
foreach size = [32, 64] in {
  def S_BFE_I#size#_sem : InstSISemantic<!cast<InstSI>("S_BFE_I"#size)> {
    let Semantic = [
      (DefVal $shift_amt,
        LLVMExtractBitsIntStartIntNumBits<(GetNamedOperand $src1), !cast<SILLVMType>("si_llvm_i"#size#"_ty"), false, 0, !if(!eq(size, 64), 6, 5)>.Op),
      (DefVal $bit_width,
        LLVMExtractBitsIntStartIntNumBits<(GetNamedOperand $src1), !cast<SILLVMType>("si_llvm_i"#size#"_ty"), false, 16, 7>.Op),
      (DefVal $shifted, (LLVMLShr (GetNamedOperand $src0), (GetVal $shift_amt))),
      (DefVal $mask, (LLVMSub (LLVMShl (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 1), (GetVal $bit_width)),
                               (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 1))),
      (DefVal $masked, (LLVMAnd (GetVal $shifted), (GetVal $mask))),
      (DefVal $sign_bit, (LLVMTrunc (LLVMLShr (GetVal $masked), (LLVMSub (GetVal $bit_width), (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 1))), si_llvm_i1_ty)),
      (DefVal $sign_ext, (LLVMSExt (GetVal $sign_bit), !cast<SILLVMType>("si_llvm_i"#size#"_ty"))),
      (DefVal $result, (LLVMOr (LLVMAnd (GetVal $masked), (GetVal $mask)),
                              (LLVMAnd (GetVal $sign_ext), LLVMNot<(GetVal $mask), !cast<SILLVMType>("si_llvm_i"#size#"_ty")>.ret))),
      SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt !cast<SILLVMType>("si_llvm_i"#size#"_ty"), 0))>.ret,
      (SetNamedOperand $sdst, (GetVal $result))
    ];
  }
}

// D.f = S0.f + S1.f.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 228.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 208.
//   rdna4-instruction-set-architecture.pdf, page 228.
def S_ADD_F32_sem : InstSISemantic<S_ADD_F32> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMFAdd (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.f = S0.f - S1.f.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 228.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 208.
//   rdna4-instruction-set-architecture.pdf, page 228.
def S_SUB_F32_sem : InstSISemantic<S_SUB_F32> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMFSub (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.f = S0.f * S1.f.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 230.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 210.
//   rdna4-instruction-set-architecture.pdf, page 230.
def S_MUL_F32_sem : InstSISemantic<S_MUL_F32> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMFMul (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D = IEEE minNum(S0, S1). If one is NaN, return the other.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 229 (as S_MIN_NUM_F32).
def S_MIN_F32_sem : InstSISemantic<S_MIN_F32> {
  let Semantic = [
    (DefVal $is_nan0, (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src0), (GetNamedOperand $src0))),
    (DefVal $is_nan1, (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src1), (GetNamedOperand $src1))),
    (DefVal $is_less,
      (LLVMOr (LLVMFCmp LLVMFCmpOLT, (GetNamedOperand $src0), (GetNamedOperand $src1)),
              (LLVMAnd (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src0), (GetNamedOperand $src1)),
                       (LLVMXor (LLVMFCmp LLVMFCmpOLT, (LLVMFAdd (GetNamedOperand $src0), (ConstantFP si_llvm_float_ty, "-0.0")),
                                                             (ConstantFP si_llvm_float_ty, "-0.0")),
                               (LLVMFCmp LLVMFCmpOLT, (LLVMFAdd (GetNamedOperand $src1), (ConstantFP si_llvm_float_ty, "-0.0")),
                                                             (ConstantFP si_llvm_float_ty, "-0.0")))))),
    (SetNamedOperand $sdst,
      (LLVMSelect (LLVMAnd (GetVal $is_less), (LLVMOr (GetVal $is_nan0), (GetVal $is_nan1))),
                  (GetNamedOperand $src0),
                  (GetNamedOperand $src1)))
  ];
}

// D = IEEE maxNum(S0, S1). If one is NaN, return the other.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 229 (as S_MAX_NUM_F32).
def S_MAX_F32_sem : InstSISemantic<S_MAX_F32> {
  let Semantic = [
    (DefVal $is_nan0, (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src0), (GetNamedOperand $src0))),
    (DefVal $is_nan1, (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src1), (GetNamedOperand $src1))),
    (DefVal $is_greater,
      (LLVMOr (LLVMFCmp LLVMFCmpOGT, (GetNamedOperand $src0), (GetNamedOperand $src1)),
              (LLVMAnd (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src0), (GetNamedOperand $src1)),
                       (LLVMXor (LLVMFCmp LLVMFCmpOLT, (LLVMFAdd (GetNamedOperand $src0), (ConstantFP si_llvm_float_ty, "-0.0")),
                                                             (ConstantFP si_llvm_float_ty, "-0.0")),
                               (LLVMFCmp LLVMFCmpOLT, (LLVMFAdd (GetNamedOperand $src1), (ConstantFP si_llvm_float_ty, "-0.0")),
                                                             (ConstantFP si_llvm_float_ty, "-0.0")))))),
    (SetNamedOperand $sdst,
      (LLVMSelect (LLVMAnd (GetVal $is_greater), (LLVMOr (GetVal $is_nan0), (GetVal $is_nan1))),
                  (GetNamedOperand $src0),
                  (GetNamedOperand $src1)))
  ];
}

// D.f = IEEE 754 minimum(S0.f, S1.f). Propagates NaN.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 232.
//   rdna4-instruction-set-architecture.pdf, page 232.
def S_MINIMUM_F32_sem : InstSISemantic<S_MINIMUM_F32> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMSelect (LLVMFCmp LLVMFCmpOLT, (GetNamedOperand $src0), (GetNamedOperand $src1)),
                                          (GetNamedOperand $src0),
                                          (GetNamedOperand $src1)))
  ];
}

// D.f = IEEE 754 maximum(S0.f, S1.f). Propagates NaN.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 233.
//   rdna4-instruction-set-architecture.pdf, page 233.
def S_MAXIMUM_F32_sem : InstSISemantic<S_MAXIMUM_F32> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMSelect (LLVMFCmp LLVMFCmpOGT, (GetNamedOperand $src0), (GetNamedOperand $src1)),
                                          (GetNamedOperand $src0),
                                          (GetNamedOperand $src1)))
  ];
}

// D.f16 = S0.f16 + S1.f16.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 231.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 211.
//   rdna4-instruction-set-architecture.pdf, page 231.
def S_ADD_F16_sem : InstSISemantic<S_ADD_F16> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMFAdd (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.f16 = S0.f16 - S1.f16.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 231.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 211.
//   rdna4-instruction-set-architecture.pdf, page 231.
def S_SUB_F16_sem : InstSISemantic<S_SUB_F16> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMFSub (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.f16 = S0.f16 * S1.f16.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 232.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 212.
//   rdna4-instruction-set-architecture.pdf, page 232.
def S_MUL_F16_sem : InstSISemantic<S_MUL_F16> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMFMul (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D = IEEE minNum(S0, S1). If one is NaN, return the other. (half)
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 231 (as S_MIN_NUM_F16).
def S_MIN_F16_sem : InstSISemantic<S_MIN_F16> {
  let Semantic = [
    (DefVal $is_nan0, (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src0), (GetNamedOperand $src0))),
    (DefVal $is_nan1, (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src1), (GetNamedOperand $src1))),
    (DefVal $is_less,
      (LLVMOr (LLVMFCmp LLVMFCmpOLT, (GetNamedOperand $src0), (GetNamedOperand $src1)),
              (LLVMAnd (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src0), (GetNamedOperand $src1)),
                       (LLVMXor (LLVMFCmp LLVMFCmpOLT, (LLVMFAdd (GetNamedOperand $src0), (ConstantFP si_llvm_half_ty, "-0.0")),
                                                             (ConstantFP si_llvm_half_ty, "-0.0")),
                               (LLVMFCmp LLVMFCmpOLT, (LLVMFAdd (GetNamedOperand $src1), (ConstantFP si_llvm_half_ty, "-0.0")),
                                                             (ConstantFP si_llvm_half_ty, "-0.0")))))),
    (SetNamedOperand $sdst,
      (LLVMSelect (LLVMAnd (GetVal $is_less), (LLVMOr (GetVal $is_nan0), (GetVal $is_nan1))),
                  (GetNamedOperand $src0),
                  (GetNamedOperand $src1)))
  ];
}

// D = IEEE maxNum(S0, S1). If one is NaN, return the other. (half)
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 231 (as S_MAX_NUM_F16).
def S_MAX_F16_sem : InstSISemantic<S_MAX_F16> {
  let Semantic = [
    (DefVal $is_nan0, (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src0), (GetNamedOperand $src0))),
    (DefVal $is_nan1, (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src1), (GetNamedOperand $src1))),
    (DefVal $is_greater,
      (LLVMOr (LLVMFCmp LLVMFCmpOGT, (GetNamedOperand $src0), (GetNamedOperand $src1)),
              (LLVMAnd (LLVMFCmp LLVMFCmpOEQ, (GetNamedOperand $src0), (GetNamedOperand $src1)),
                       (LLVMXor (LLVMFCmp LLVMFCmpOLT, (LLVMFAdd (GetNamedOperand $src0), (ConstantFP si_llvm_half_ty, "-0.0")),
                                                             (ConstantFP si_llvm_half_ty, "-0.0")),
                               (LLVMFCmp LLVMFCmpOLT, (LLVMFAdd (GetNamedOperand $src1), (ConstantFP si_llvm_half_ty, "-0.0")),
                                                             (ConstantFP si_llvm_half_ty, "-0.0")))))),
    (SetNamedOperand $sdst,
      (LLVMSelect (LLVMAnd (GetVal $is_greater), (LLVMOr (GetVal $is_nan0), (GetVal $is_nan1))),
                  (GetNamedOperand $src0),
                  (GetNamedOperand $src1)))
  ];
}

// D.f16 = IEEE 754 minimum(S0.f16, S1.f16). Propagates NaN.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 233.
//   rdna4-instruction-set-architecture.pdf, page 233.
def S_MINIMUM_F16_sem : InstSISemantic<S_MINIMUM_F16> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMSelect (LLVMFCmp LLVMFCmpOLT, (GetNamedOperand $src0), (GetNamedOperand $src1)),
                                          (GetNamedOperand $src0),
                                          (GetNamedOperand $src1)))
  ];
}

// D.f16 = IEEE 754 maximum(S0.f16, S1.f16). Propagates NaN.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 234.
//   rdna4-instruction-set-architecture.pdf, page 234.
def S_MAXIMUM_F16_sem : InstSISemantic<S_MAXIMUM_F16> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMSelect (LLVMFCmp LLVMFCmpOGT, (GetNamedOperand $src0), (GetNamedOperand $src1)),
                                          (GetNamedOperand $src0),
                                          (GetNamedOperand $src1)))
  ];
}

// D.f = S0.f * S1.f + K (FMA with inline constant addend).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 230.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 210.
//   rdna4-instruction-set-architecture.pdf, page 230.
def S_FMAAK_F32_sem : InstSISemantic<S_FMAAK_F32> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMFMA (GetNamedOperand $src0),
                                     (GetNamedOperand $src1),
                                     (GetNamedOperand $imm)))
  ];
}

// D.f = S0.f * K + S1.f (FMA with inline constant multiplier).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 230.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 210.
//   rdna4-instruction-set-architecture.pdf, page 230.
def S_FMAMK_F32_sem : InstSISemantic<S_FMAMK_F32> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMFMA (GetNamedOperand $src0),
                                     (GetNamedOperand $imm),
                                     (GetNamedOperand $src1)))
  ];
}

// D.f = S0.f * S1.f + D.f (FMA accumulate).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 230.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 230.
//   rdna4-instruction-set-architecture.pdf, page 230.
def S_FMAC_F32_sem : InstSISemantic<S_FMAC_F32> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMFMA (GetNamedOperand $src0),
                                     (GetNamedOperand $src1),
                                     (GetNamedOperand $sdst)))
  ];
}

// D.f16 = S0.f16 * S1.f16 + D.f16 (FMA accumulate, half).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 232.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 212.
//   rdna4-instruction-set-architecture.pdf, page 232.
def S_FMAC_F16_sem : InstSISemantic<S_FMAC_F16> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMFMA (GetNamedOperand $src0),
                                     (GetNamedOperand $src1),
                                     (GetNamedOperand $sdst)))
  ];
}

// D = {f32_to_f16(S1), f32_to_f16(S0)}. Pack two f32 to f16 with round-toward-zero.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 230.
def S_CVT_PK_RTZ_F16_F32_sem : InstSISemantic<S_CVT_PK_RTZ_F16_F32> {
  let Semantic = [
    (DefVal $low_f16, (LLVMFPTrunc (GetNamedOperand $src0), si_llvm_half_ty)),
    (DefVal $high_f16, (LLVMFPTrunc (GetNamedOperand $src1), si_llvm_half_ty)),
    (DefVal $low_i16, (LLVMBitCast (GetVal $low_f16), si_llvm_i16_ty)),
    (DefVal $high_i16, (LLVMBitCast (GetVal $high_f16), si_llvm_i16_ty)),
    (SetNamedOperand $sdst, (LLVMOr (LLVMZExt (GetVal $low_i16), si_llvm_i32_ty),
                                    (LLVMShl (LLVMZExt (GetVal $high_i16), si_llvm_i32_ty),
                                             (ConstantInt si_llvm_i32_ty, 16))))
  ];
}

// D = S0 + S1; no carry out.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 234 (as S_ADD_NC_U64).
def S_ADD_U64_sem : InstSISemantic<S_ADD_U64> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMAdd (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D = S0 - S1; no borrow out.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 235 (as S_SUB_NC_U64).
def S_SUB_U64_sem : InstSISemantic<S_SUB_U64> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMSub (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D = S0 * S1 (64-bit unsigned multiply).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 235.
//   rdna4-instruction-set-architecture.pdf, page 235.
def S_MUL_U64_sem : InstSISemantic<S_MUL_U64> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMMul (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

//------------------------------------------------------------------------------
// S_CBRANCH_G_FORK / S_CBRANCH_I_FORK — fork wave control flow.
//
// ISA pseudocode (GFX6-9, gcn3 ISA p.112):
//   mask_pass = exec & src0;      // src0 = comparison mask
//   mask_fail = exec & ~src0;
//   if      (mask_pass == exec)               { PC = src1; }
//   else if (mask_fail == exec)               { PC += 4;   }
//   else if (popcount(mask_fail) < popcount(mask_pass)) {
//     // execute the smaller (fail) side now, push the larger (pass) for JOIN
//     SGPR[CSP*4..CSP*4+3] = { src1, mask_pass };
//     CSP++;  exec = mask_fail;  PC += 4;
//   } else {
//     SGPR[CSP*4..CSP*4+3] = { PC+4, mask_fail };
//     CSP++;  exec = mask_pass;  PC = src1;
//   }
//
// CSP (Control Stack Pointer) lives in MODE[31:29] and indexes one of 6 SGPR
// quads: s[0:3], s[4:7], s[8:11], s[12:15], s[16:19], s[20:23]. We model
// the indexed write by emitting a `select(csp == i, entry, old_quad_i)`
// per slot so liveness sees every potentially-defined quad. The actual
// PC redirection is NOT modeled — fork is a hardware-stack operation that
// has no direct LLVM-IR representation. Downstream analyses get the
// register-definition set; control-flow modeling for wave divergence is
// left to higher-level passes.
//
// `$entry` is `{ saved_pc, saved_exec }` packed in an i128 with saved_pc
// in the low 64 bits and saved_exec in the high 64 bits (mirroring the
// hardware layout described in the ISA manual).
//------------------------------------------------------------------------------

// Per-slot conditional write to one of the 6 SGPR quads in the CSP stack.
// Emits a single (ImplicitDef quad, select(should_push & csp==slot, entry,
// old_quad)) statement. `slot` is the slot index 0..5, `regs` is the list
// of 4 SGPR Register records that make up the quad. Inlined rather than
// split into DefVals because tablegen can't synthesize per-slot dag arg
// names — the emitter binds DefVal targets by literal argname, not by value.
class CSPStackSlot<int slot, list<Register> regs> {
  dag ret = (ImplicitDef SuperRegFactory<regs>,
    (LLVMSelect
      (LLVMAnd (GetVal $should_push),
               (LLVMCmp LLVMICmpEQ, (GetVal $csp),
                                    (ConstantInt si_llvm_i32_ty, slot))),
      (GetVal $entry),
      (LLVMBitCast (ImplicitUse SuperRegFactory<regs>), si_llvm_i128_ty)));
}

def S_CBRANCH_G_FORK_sem : InstSISemantic<S_CBRANCH_G_FORK> {
  let Semantic = !listconcat(
    [
      // Read inputs.
      (DefVal $mask_in, (GetNamedOperand $src0)),
      (DefVal $target,  (GetNamedOperand $src1)),
      (DefVal $cur_pc,  (LLVMIntrinsic si_llvm_i64_ty,
                                       int_amdgcn_s_getpc, []<dag>)),
      (DefVal $exec_mask, (ImplicitUse EXEC)),

      // mask_pass = exec & src0; mask_fail = exec & ~src0.
      // ~src0 expressed as XOR with all-ones at wave-mask width.
      (DefVal $mask_pass, (LLVMAnd (GetVal $exec_mask), (GetVal $mask_in))),
      (DefVal $all_ones, (LLVMSExt (ConstantTrue si_llvm_i1_ty),
                                   (LLVMIntNTy (GetWavefrontSize)))),
      (DefVal $not_mask_in, (LLVMXor (GetVal $mask_in), (GetVal $all_ones))),
      (DefVal $mask_fail, (LLVMAnd (GetVal $exec_mask),
                                   (GetVal $not_mask_in))),

      // Decide which half executes first.
      (DefVal $bc_pass, (LLVMUnaryIntrinsic int_ctpop, (GetVal $mask_pass))),
      (DefVal $bc_fail, (LLVMUnaryIntrinsic int_ctpop, (GetVal $mask_fail))),
      (DefVal $fail_smaller,
              (LLVMCmp LLVMICmpULT, (GetVal $bc_fail), (GetVal $bc_pass))),

      // No-fork: all-lanes one way → no stack push, no EXEC change.
      (DefVal $all_pass,
              (LLVMCmp LLVMICmpEQ, (GetVal $mask_pass), (GetVal $exec_mask))),
      (DefVal $all_fail,
              (LLVMCmp LLVMICmpEQ, (GetVal $mask_fail), (GetVal $exec_mask))),
      (DefVal $no_fork, (LLVMOr (GetVal $all_pass), (GetVal $all_fail))),
      (DefVal $should_push, (LLVMXor (GetVal $no_fork),
                                     (ConstantTrue si_llvm_i1_ty))),

      // Saved entry. When fail-side is smaller (we execute fail first), the
      // PASS side gets saved → saved_pc=src1, saved_exec=mask_pass. Otherwise
      // we execute pass first and save FAIL → saved_pc=pc+4, saved_exec=mask_fail.
      (DefVal $pc_plus_4, (LLVMAdd (GetVal $cur_pc),
                                   (ConstantInt si_llvm_i64_ty, 4))),
      (DefVal $saved_pc, (LLVMSelect (GetVal $fail_smaller),
                                     (GetVal $target), (GetVal $pc_plus_4))),
      (DefVal $saved_exec, (LLVMSelect (GetVal $fail_smaller),
                                       (GetVal $mask_pass),
                                       (GetVal $mask_fail))),
      // Pack into i128: low 64 bits = saved_pc, high 64 bits = saved_exec.
      (DefVal $saved_pc_128, (LLVMZExt (GetVal $saved_pc), si_llvm_i128_ty)),
      (DefVal $saved_exec_64, (LLVMZExt (GetVal $saved_exec), si_llvm_i64_ty)),
      (DefVal $saved_exec_128, (LLVMZExt (GetVal $saved_exec_64),
                                         si_llvm_i128_ty)),
      (DefVal $saved_exec_shifted, (LLVMShl (GetVal $saved_exec_128),
                                            (ConstantInt si_llvm_i128_ty, 64))),
      (DefVal $entry, (LLVMOr (GetVal $saved_pc_128),
                              (GetVal $saved_exec_shifted))),

      // Read CSP from MODE[31:29] (hwreg 5953 = MODE id 1 | (29 << 6) | ((3-1) << 11)).
      (DefVal $csp, (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_s_getreg,
                                   [(ConstantInt si_llvm_i32_ty, 5953)]<dag>))
    ],
    // Per-slot CSP stack writes (s[0:3] through s[20:23]).
    [CSPStackSlot<0, [SGPR0,  SGPR1,  SGPR2,  SGPR3 ]>.ret,
     CSPStackSlot<1, [SGPR4,  SGPR5,  SGPR6,  SGPR7 ]>.ret,
     CSPStackSlot<2, [SGPR8,  SGPR9,  SGPR10, SGPR11]>.ret,
     CSPStackSlot<3, [SGPR12, SGPR13, SGPR14, SGPR15]>.ret,
     CSPStackSlot<4, [SGPR16, SGPR17, SGPR18, SGPR19]>.ret,
     CSPStackSlot<5, [SGPR20, SGPR21, SGPR22, SGPR23]>.ret],
    [
      // EXEC update: when forking, set EXEC to the smaller half. When not
      // forking, EXEC is unchanged (the wave just falls through or branches
      // as a whole — but EXEC stays as it was).
      (DefVal $forked_exec, (LLVMSelect (GetVal $fail_smaller),
                                        (GetVal $mask_fail),
                                        (GetVal $mask_pass))),
      (DefVal $new_exec, (LLVMSelect (GetVal $no_fork),
                                     (GetVal $exec_mask),
                                     (GetVal $forked_exec))),
      (ImplicitDef EXEC, (GetVal $new_exec)),

      // CSP increment (only when we actually pushed).
      (DefVal $csp_inc, (LLVMAdd (GetVal $csp),
                                 (ConstantInt si_llvm_i32_ty, 1))),
      (DefVal $new_csp, (LLVMSelect (GetVal $should_push),
                                    (GetVal $csp_inc), (GetVal $csp))),
      (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_setreg,
                     [(ConstantInt si_llvm_i32_ty, 5953),
                      (GetVal $new_csp)]<dag>)
    ]);
}

// --- old draft kept below for reference, commented out ---
//def S_CBRANCH_G_FORK_sem_draft : InstSISemantic<S_CBRANCH_G_FORK> {
//  let Semantic = [
//    // Get current PC for branch stack entry
//    (DefVal $current_pc, (LLVMIntrinsic si_llvm_i64_ty, int_amdgcn_s_getpc, []<dag>)),
//    // Compute pass and fail masks
//    (DefVal $mask_pass, (LLVMAnd (ImplicitUse EXEC), (GetNamedOperand $src0))),
//    (DefVal $mask_fail, (LLVMAnd (ImplicitUse EXEC),
//                                 (LLVMNot<(GetNamedOperand $src0), si_llvm_i64_ty>.ret -1))),
//    // Get CSP from MODE register (hwmode = 5953, returns i32)
//    (DefVal $csp, (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_s_getreg,
//      [(ConstantInt si_llvm_i16_ty, 5953)]<dag>)),
//    // Read SGPR quads and bitcast to 128-bit vectors
//    (DefVal $sgpr0_3, (ImplicitUse SuperRegFactory<[SGPR0, SGPR1, SGPR2, SGPR3]>)),
//    (DefVal $sgpr4_7, (ImplicitUse SuperRegFactory<[SGPR4, SGPR5, SGPR6, SGPR7]>)),
//    (DefVal $sgpr8_11, (ImplicitUse SuperRegFactory<[SGPR8, SGPR9, SGPR10, SGPR11]>)),
//    (DefVal $sgpr12_15, (ImplicitUse SuperRegFactory<[SGPR12, SGPR13, SGPR14, SGPR15]>)),
//    (DefVal $sgpr16_19, (ImplicitUse SuperRegFactory<[SGPR16, SGPR17, SGPR18, SGPR19]>)),
//    (DefVal $sgpr20_23, (ImplicitUse SuperRegFactory<[SGPR20, SGPR21, SGPR22, SGPR23]>)),
//    // Create 6x128-bit vector of SGPR values
//    (DefVal $sgpr_vec, (LLVMInsertElement
//      (LLVMInsertElement
//        (LLVMInsertElement
//          (LLVMInsertElement
//            (LLVMInsertElement
//              (LLVMInsertElement (PoisionVal si_llvm_v6i128_ty),
//                                 (GetVal $sgpr0_3), (ConstantInt si_llvm_i32_ty, 0)),
//                                 (GetVal $sgpr4_7), (ConstantInt si_llvm_i32_ty, 1)),
//                                 (GetVal $sgpr8_11), (ConstantInt si_llvm_i32_ty, 2)),
//                                 (GetVal $sgpr12_15), (ConstantInt si_llvm_i32_ty, 3)),
//                                 (GetVal $sgpr16_19), (ConstantInt si_llvm_i32_ty, 4)),
//                                 (GetVal $sgpr20_23), (ConstantInt si_llvm_i32_ty, 5))),
//    (DefVal $mask_pass_exec_eq, (LLVMICmpEQ (ImplicitUse EXEC), (GetVal $mask_pass)),
//    (DefVal $mask_fail_exec_eq, (LLVMICmpNE (ImplicitUse EXEC), (GetVal $mask_fail)),
//    (DefVal $mask_pass_bc_greater,(LLVMICmpUGT
//                                              (LLVMUnaryIntrinsic int_ctpop, (GetVal $mask_pass)),
//                                              (LLVMUnaryIntrinsic int_ctpop, (GetVal $mask_fail)))),
//    // Extract element at CSP index
//    (DefVal $csp_val, (LLVMExtractElement (GetVal $sgpr_vec), (GetVal $csp))),
//    // Construct 128-bit branch stack entry: { mask_pass, target[47:2] }
//    (DefVal $target_shifted, (LLVMLShr (GetNamedOperand $src1), (ConstantInt si_llvm_i64_ty, 2))),
//    (DefVal $entry_vec, (LLVMInsertElement
//                          (LLVMInsertElement (ConstantPoison si_llvm_v2i64_ty),
//                                             (GetVal $target_shifted), (ConstantInt si_llvm_i32_ty, 0)),
//                                             (GetVal $pass_mask), (ConstantInt si_llvm_i32_ty, 1))),
//    (DefVal $branch_stack_entry, (LLVMBitCast (GetVal $entry_vec), si_llvm_i128_ty)),
//    // Indirect branch to target address
//    (LLVMIndirectBranch (GetVal $csp_val))
//  ];
//}

// S_RFE_RESTORE_B64 (gfx8/9 only) — Return From Exception with restore.
// Pops the exception state (src1 holds the restored TTMP value) and jumps
// to PC in src0. From an analysis perspective the jump is structurally
// the same as S_RFE_B64 / S_SETPC_B64; the state restore is HW-internal
// with no IR analog.
def S_RFE_RESTORE_B64_sem : InstSISemantic<S_RFE_RESTORE_B64> {
  let Semantic = [
    (IndirectTailCall (GetNamedOperand $src0))
  ];
}
