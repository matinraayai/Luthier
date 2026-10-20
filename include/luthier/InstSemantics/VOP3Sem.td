//===- VOP3Sem.td ------------------------------------------------*- tblgen -*-===//
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
// Describes the semantics of VOP3 (e64) instructions for the AMDGPU backend.
// These are the 64-bit encoded variants of VOP1/VOP2 instructions, plus
// VOP3-only 3-operand instructions.
//===----------------------------------------------------------------------===//

//==============================================================================
// VOP2 e64 Variants
//==============================================================================

//------------------------------------------------------------------------------
// VOP2 Float Arithmetic (F32) e64
//------------------------------------------------------------------------------

// D.f = S0.f + S1.f.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 164.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 126.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 169.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 181.
//   rdna-shader-instruction-set-architecture.pdf, page 140.
//   rdna2-shader-instruction-set-architecture.pdf, page 139.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 259.
//   rdna4-instruction-set-architecture.pdf, page 299.
def V_ADD_F32_e64_sem : InstSISemantic<V_ADD_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = S0.f - S1.f.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 183.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 126.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 169.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 181.
//   rdna-shader-instruction-set-architecture.pdf, page 140.
//   rdna2-shader-instruction-set-architecture.pdf, page 139.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 259.
//   rdna4-instruction-set-architecture.pdf, page 300.
def V_SUB_F32_e64_sem : InstSISemantic<V_SUB_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFSub (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = S1.f - S0.f.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 186.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 126.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 170.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 182.
//   rdna-shader-instruction-set-architecture.pdf, page 140.
//   rdna2-shader-instruction-set-architecture.pdf, page 139.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 260.
//   rdna4-instruction-set-architecture.pdf, page 300.
def V_SUBREV_F32_e64_sem : InstSISemantic<V_SUBREV_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFSub (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = S0.f * S1.f.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 179.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 170.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 182.
//   rdna-shader-instruction-set-architecture.pdf, page 140.
//   rdna2-shader-instruction-set-architecture.pdf, page 139.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 261.
//   rdna4-instruction-set-architecture.pdf, page 301.
def V_MUL_F32_e64_sem : InstSISemantic<V_MUL_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = S0.f * S1.f (DX9 legacy: 0.0 * anything = 0.0).
// Sources (GFX6-GFX10 only, removed in RDNA3+):
//   gcn3-instruction-set-architecture.pdf, page 181.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 126.
//   rdna-shader-instruction-set-architecture.pdf, page 140.
//   rdna2-shader-instruction-set-architecture.pdf, page 139.
def V_MUL_LEGACY_F32_e64_sem : InstSISemantic<V_MUL_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Min/Max Float (F32) e64
//------------------------------------------------------------------------------

// D.f = max(S0.f, S1.f) (IEEE maxNum).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 172.
//   vega-shader-instruction-set-architecture.pdf, page 130.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 172.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 184.
//   rdna-shader-instruction-set-architecture.pdf, page 142.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 263.
//   rdna4-instruction-set-architecture.pdf, page 305 (as V_MAX_NUM_F32).
def V_MAX_F32_e64_sem : InstSISemantic<V_MAX_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMaxNum (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = min(S0.f, S1.f) (IEEE minNum).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 176.
//   vega-shader-instruction-set-architecture.pdf, page 130.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 171.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 183.
//   rdna-shader-instruction-set-architecture.pdf, page 141.
//   rdna2-shader-instruction-set-architecture.pdf, page 140.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 262.
//   rdna4-instruction-set-architecture.pdf, page 304 (as V_MIN_NUM_F32).
def V_MIN_F32_e64_sem : InstSISemantic<V_MIN_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMinNum (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = max(S0.f, S1.f) (DX10 legacy).
def V_MAX_LEGACY_F32_e64_sem : InstSISemantic<V_MAX_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMSelect (LLVMFCmp LLVMFCmpOGE,
                      (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                      (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
                    (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = min(S0.f, S1.f) (DX10 legacy).
def V_MIN_LEGACY_F32_e64_sem : InstSISemantic<V_MIN_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMSelect (LLVMFCmp LLVMFCmpOLT,
                      (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                      (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
                    (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Integer Arithmetic e64
//------------------------------------------------------------------------------

// D.u = S0.u + S1.u (no carry out).
// Sources (GFX9+ only, HasAddNoCarryInsts):
//   vega-shader-instruction-set-architecture.pdf, page 134.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 183.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 195.
def V_ADD_U32_e64_sem : InstSISemantic<V_ADD_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.u = S0.u - S1.u (no carry out).
def V_SUB_U32_e64_sem : InstSISemantic<V_SUB_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSub (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.u = S1.u - S0.u (no carry out, reversed operands).
// Sources (GFX9+ only, HasAddNoCarryInsts):
//   vega-shader-instruction-set-architecture.pdf, page 134.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 183.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 196.
def V_SUBREV_U32_e64_sem : InstSISemantic<V_SUBREV_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSub (GetNamedOperand $src1), (GetNamedOperand $src0)))
  ];
}

//------------------------------------------------------------------------------
// VOP2 24-bit Multiply e64
//------------------------------------------------------------------------------

// D.i = S0.i[23:0] * S1.i[23:0] (signed 24-bit multiply, low 32 bits).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 180.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 171.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 183.
//   rdna-shader-instruction-set-architecture.pdf, page 141.
//   rdna2-shader-instruction-set-architecture.pdf, page 140.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 261.
//   rdna4-instruction-set-architecture.pdf, page 301.
def V_MUL_I32_I24_e64_sem : InstSISemantic<V_MUL_I32_I24_e64> {
  let Semantic = [
    (DefVal $a, (LLVMAShr (LLVMShl (GetNamedOperand $src0),
                                    (ConstantInt si_llvm_i32_ty, 8)),
                           (ConstantInt si_llvm_i32_ty, 8))),
    (DefVal $b, (LLVMAShr (LLVMShl (GetNamedOperand $src1),
                                    (ConstantInt si_llvm_i32_ty, 8)),
                           (ConstantInt si_llvm_i32_ty, 8))),
    (SetNamedOperand $vdst, (LLVMMul (GetVal $a), (GetVal $b)))
  ];
}

// D.i = (S0.i[23:0] * S1.i[23:0]) >> 32 (signed 24-bit multiply, high 32 bits).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 179.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 170.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 182.
//   rdna-shader-instruction-set-architecture.pdf, page 141.
//   rdna2-shader-instruction-set-architecture.pdf, page 140.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 261.
//   rdna4-instruction-set-architecture.pdf, page 301.
def V_MUL_HI_I32_I24_e64_sem : InstSISemantic<V_MUL_HI_I32_I24_e64> {
  let Semantic = [
    (DefVal $a, (LLVMSExt (LLVMAShr (LLVMShl (GetNamedOperand $src0),
                                              (ConstantInt si_llvm_i32_ty, 8)),
                                     (ConstantInt si_llvm_i32_ty, 8)),
                           si_llvm_i64_ty)),
    (DefVal $b, (LLVMSExt (LLVMAShr (LLVMShl (GetNamedOperand $src1),
                                              (ConstantInt si_llvm_i32_ty, 8)),
                                     (ConstantInt si_llvm_i32_ty, 8)),
                           si_llvm_i64_ty)),
    (SetNamedOperand $vdst,
      (LLVMTrunc (LLVMAShr (LLVMMul (GetVal $a), (GetVal $b)),
                            (ConstantInt si_llvm_i64_ty, 32)),
                 si_llvm_i32_ty))
  ];
}

// D.u = S0.u[23:0] * S1.u[23:0] (unsigned 24-bit multiply, low 32 bits).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 182.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 171.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 183.
//   rdna-shader-instruction-set-architecture.pdf, page 141.
//   rdna2-shader-instruction-set-architecture.pdf, page 140.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 262.
//   rdna4-instruction-set-architecture.pdf, page 302.
def V_MUL_U32_U24_e64_sem : InstSISemantic<V_MUL_U32_U24_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMMul (LLVMAnd (GetNamedOperand $src0),
                         (ConstantInt si_llvm_i32_ty, 16777215)),
               (LLVMAnd (GetNamedOperand $src1),
                         (ConstantInt si_llvm_i32_ty, 16777215))))
  ];
}

// D.u = (S0.u[23:0] * S1.u[23:0]) >> 32 (unsigned 24-bit multiply, high 32 bits).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 180.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 171.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 183.
//   rdna-shader-instruction-set-architecture.pdf, page 141.
//   rdna2-shader-instruction-set-architecture.pdf, page 140.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 262.
//   rdna4-instruction-set-architecture.pdf, page 302.
def V_MUL_HI_U32_U24_e64_sem : InstSISemantic<V_MUL_HI_U32_U24_e64> {
  let Semantic = [
    (DefVal $a, (LLVMZExt (LLVMAnd (GetNamedOperand $src0),
                                    (ConstantInt si_llvm_i32_ty, 16777215)),
                           si_llvm_i64_ty)),
    (DefVal $b, (LLVMZExt (LLVMAnd (GetNamedOperand $src1),
                                    (ConstantInt si_llvm_i32_ty, 16777215)),
                           si_llvm_i64_ty)),
    (SetNamedOperand $vdst,
      (LLVMTrunc (LLVMLShr (LLVMMul (GetVal $a), (GetVal $b)),
                            (ConstantInt si_llvm_i64_ty, 32)),
                 si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Min/Max Integer e64
//------------------------------------------------------------------------------

// D.i = max(S0.i, S1.i) (signed).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 173.
//   vega-shader-instruction-set-architecture.pdf, page 130.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 172.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 184.
//   rdna-shader-instruction-set-architecture.pdf, page 142.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 264.
//   rdna4-instruction-set-architecture.pdf, page 303.
def V_MAX_I32_e64_sem : InstSISemantic<V_MAX_I32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetNamedOperand $src0),
                                   (GetNamedOperand $src1)),
                  (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.i = min(S0.i, S1.i) (signed).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 177.
//   vega-shader-instruction-set-architecture.pdf, page 130.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 172.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 184.
//   rdna-shader-instruction-set-architecture.pdf, page 142.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 264.
//   rdna4-instruction-set-architecture.pdf, page 303.
def V_MIN_I32_e64_sem : InstSISemantic<V_MIN_I32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetNamedOperand $src0),
                                   (GetNamedOperand $src1)),
                  (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.u = max(S0.u, S1.u) (unsigned).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 174.
//   vega-shader-instruction-set-architecture.pdf, page 130.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 173.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 185.
//   rdna-shader-instruction-set-architecture.pdf, page 142.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 264.
//   rdna4-instruction-set-architecture.pdf, page 304.
def V_MAX_U32_e64_sem : InstSISemantic<V_MAX_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetNamedOperand $src0),
                                    (GetNamedOperand $src1)),
                  (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.u = min(S0.u, S1.u) (unsigned).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 178.
//   vega-shader-instruction-set-architecture.pdf, page 130.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 173.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 185.
//   rdna-shader-instruction-set-architecture.pdf, page 142.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 264.
//   rdna4-instruction-set-architecture.pdf, page 304.
def V_MIN_U32_e64_sem : InstSISemantic<V_MIN_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpULT, (GetNamedOperand $src0),
                                    (GetNamedOperand $src1)),
                  (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Add/Sub with Carry Out e64 (explicit $sdst output)
//------------------------------------------------------------------------------

// D.u = S0.u + S1.u; sdst = unsigned carry out.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 165 (as V_ADD_I32).
//   vega-shader-instruction-set-architecture.pdf, page 131 (as V_ADD_CO_U32).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 175.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 187.
//   rdna-shader-instruction-set-architecture.pdf, page 178.
//   rdna2-shader-instruction-set-architecture.pdf, page 175.
def V_ADD_CO_U32_e64_sem : InstSISemantic<V_ADD_CO_U32_e64> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMAdd (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
               (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

// D.u = S0.u - S1.u; sdst = unsigned borrow out.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 184 (as V_SUB_I32).
//   vega-shader-instruction-set-architecture.pdf, page 131 (as V_SUB_CO_U32).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 175.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 188.
//   rdna-shader-instruction-set-architecture.pdf, page 178.
//   rdna2-shader-instruction-set-architecture.pdf, page 175.
def V_SUB_CO_U32_e64_sem : InstSISemantic<V_SUB_CO_U32_e64> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMSub (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
               (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

// D.u = S1.u - S0.u; sdst = unsigned borrow out (reversed operands).
// Sources: same as V_SUB_CO_U32 above.
//   gcn3-instruction-set-architecture.pdf, page 187 (as V_SUBREV_I32).
//   vega-shader-instruction-set-architecture.pdf, page 131 (as V_SUBREV_CO_U32).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 176.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 188.
//   rdna-shader-instruction-set-architecture.pdf, page 178.
//   rdna2-shader-instruction-set-architecture.pdf, page 175.
def V_SUBREV_CO_U32_e64_sem : InstSISemantic<V_SUBREV_CO_U32_e64> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMSub (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty),
               (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Add/Sub with Carry In/Out e64 (src2 = carry-in, sdst = carry-out)
//------------------------------------------------------------------------------

// D.u = S0.u + S1.u + src2(carry); sdst = carry out.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 165 (as V_ADDC_U32).
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 175.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 187.
def V_ADDC_U32_e64_sem : InstSISemantic<V_ADDC_U32_e64> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMAdd (LLVMAdd (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                         (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty)),
               (LLVMZExt (GetNamedOperand $src2), si_llvm_i64_ty))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

// D.u = S0.u - S1.u - src2(borrow); sdst = borrow out.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 184 (as V_SUBB_U32).
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 175.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 188.
def V_SUBB_U32_e64_sem : InstSISemantic<V_SUBB_U32_e64> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMSub (LLVMSub (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                         (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty)),
               (LLVMZExt (GetNamedOperand $src2), si_llvm_i64_ty))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

// D.u = S1.u - S0.u - src2(borrow); sdst = borrow out (reversed operands).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 185 (as V_SUBBREV_U32).
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 176.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 188.
def V_SUBBREV_U32_e64_sem : InstSISemantic<V_SUBBREV_U32_e64> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMSub (LLVMSub (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty),
                         (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty)),
               (LLVMZExt (GetNamedOperand $src2), si_llvm_i64_ty))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Logical Operations e64
//------------------------------------------------------------------------------

// D.u = S0.u & S1.u.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 166.
//   vega-shader-instruction-set-architecture.pdf, page 130.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 174.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 186.
//   rdna-shader-instruction-set-architecture.pdf, page 142.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 265.
//   rdna4-instruction-set-architecture.pdf, page 306.
def V_AND_B32_e64_sem : InstSISemantic<V_AND_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAnd (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.u = S0.u | S1.u.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 182.
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 174.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 186.
//   rdna-shader-instruction-set-architecture.pdf, page 142.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 265.
//   rdna4-instruction-set-architecture.pdf, page 306.
def V_OR_B32_e64_sem : InstSISemantic<V_OR_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMOr (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.u = S0.u ^ S1.u.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 188.
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 174.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 186.
//   rdna-shader-instruction-set-architecture.pdf, page 143.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 265.
//   rdna4-instruction-set-architecture.pdf, page 307.
def V_XOR_B32_e64_sem : InstSISemantic<V_XOR_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMXor (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.u = ~(S0.u ^ S1.u) (XNOR).
// Sources (GFX9+ only, HasDLInsts):
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 185.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 197.
//   rdna-shader-instruction-set-architecture.pdf, page 143.
//   rdna2-shader-instruction-set-architecture.pdf, page 142.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 266.
//   rdna4-instruction-set-architecture.pdf, page 307.
def V_XNOR_B32_e64_sem : InstSISemantic<V_XNOR_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      LLVMNot<(LLVMXor (GetNamedOperand $src0),
                        (GetNamedOperand $src1)), si_llvm_i32_ty>.ret)
  ];
}

//------------------------------------------------------------------------------
// VOP2 Shift Operations e64
//------------------------------------------------------------------------------

// D.u = S1.u << S0.u[4:0] (reversed operand order).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 168.
//   vega-shader-instruction-set-architecture.pdf, page 130.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 173.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 185.
//   rdna-shader-instruction-set-architecture.pdf, page 142.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 264.
//   rdna4-instruction-set-architecture.pdf, page 305.
def V_LSHLREV_B32_e64_sem : InstSISemantic<V_LSHLREV_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMShl (GetNamedOperand $src1),
               (LLVMAnd (GetNamedOperand $src0),
                         (ConstantInt si_llvm_i32_ty, 31))))
  ];
}

// D.u = S1.u >> S0.u[4:0] (logical shift right, reversed operand order).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 169.
//   vega-shader-instruction-set-architecture.pdf, page 130.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 173.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 185.
//   rdna-shader-instruction-set-architecture.pdf, page 142.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 265.
//   rdna4-instruction-set-architecture.pdf, page 306.
def V_LSHRREV_B32_e64_sem : InstSISemantic<V_LSHRREV_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMLShr (GetNamedOperand $src1),
                (LLVMAnd (GetNamedOperand $src0),
                          (ConstantInt si_llvm_i32_ty, 31))))
  ];
}

// D.i = S1.i >> S0.u[4:0] (arithmetic shift right, reversed operand order).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 167.
//   vega-shader-instruction-set-architecture.pdf, page 130.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 127.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 126.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 173.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 185.
//   rdna-shader-instruction-set-architecture.pdf, page 142.
//   rdna2-shader-instruction-set-architecture.pdf, page 141.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 265.
//   rdna4-instruction-set-architecture.pdf, page 306.
def V_ASHRREV_I32_e64_sem : InstSISemantic<V_ASHRREV_I32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAShr (GetNamedOperand $src1),
                (LLVMAnd (GetNamedOperand $src0),
                          (ConstantInt si_llvm_i32_ty, 31))))
  ];
}

// D.u = S0.u << S1.u[4:0] (non-reversed, GFX6/GFX7).
def V_LSHL_B32_e64_sem : InstSISemantic<V_LSHL_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMShl (GetNamedOperand $src0),
               (LLVMAnd (GetNamedOperand $src1),
                         (ConstantInt si_llvm_i32_ty, 31))))
  ];
}

// D.u = S0.u >> S1.u[4:0] (logical shift right, non-reversed, GFX6/GFX7).
def V_LSHR_B32_e64_sem : InstSISemantic<V_LSHR_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMLShr (GetNamedOperand $src0),
                (LLVMAnd (GetNamedOperand $src1),
                          (ConstantInt si_llvm_i32_ty, 31))))
  ];
}

// D.i = S0.i >> S1.u[4:0] (arithmetic shift right, non-reversed, GFX6/GFX7).
def V_ASHR_I32_e64_sem : InstSISemantic<V_ASHR_I32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAShr (GetNamedOperand $src0),
                (LLVMAnd (GetNamedOperand $src1),
                          (ConstantInt si_llvm_i32_ty, 31))))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Conditional Move e64
//------------------------------------------------------------------------------

// D.u = src2[lane] ? S1.u : S0.u. Condition from explicit $src2.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 167.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 126.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 169.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 181.
//   rdna-shader-instruction-set-architecture.pdf, page 140.
//   rdna2-shader-instruction-set-architecture.pdf, page 139.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 259.
//   rdna4-instruction-set-architecture.pdf, page 299.
def V_CNDMASK_B32_e64_sem : InstSISemantic<V_CNDMASK_B32_e64> {
  let Semantic = [
    // $src2 is a wave-sized SGPR lane mask (i32 / i64). LLVMSelect needs
    // an i1 condition, so extract the current lane's bit from the mask
    // before selecting between src0/src1.
    (SetNamedOperand $vdst,
      (LLVMSelect LaneBitFromWaveMask<(GetNamedOperand $src2)>.ret,
                  (GetNamedOperand $src1), (GetNamedOperand $src0)))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Bit Operations e64
//------------------------------------------------------------------------------

// D.u = ((1 << S0.u[4:0]) - 1) << S1.u[4:0]. Bitfield mask.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 234.
//   vega-shader-instruction-set-architecture.pdf, page 181.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 183.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 174.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 172.
//   rdna-shader-instruction-set-architecture.pdf, page 197.
//   rdna2-shader-instruction-set-architecture.pdf, page 192.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 443.
//   rdna4-instruction-set-architecture.pdf, page 516.
def V_BFM_B32_e64_sem : InstSISemantic<V_BFM_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMShl
        (LLVMSub (LLVMShl (ConstantInt si_llvm_i32_ty, 1),
                           (LLVMAnd (GetNamedOperand $src0),
                                     (ConstantInt si_llvm_i32_ty, 31))),
                 (ConstantInt si_llvm_i32_ty, 1)),
        (LLVMAnd (GetNamedOperand $src1), (ConstantInt si_llvm_i32_ty, 31))))
  ];
}

// D.u = CountOneBits(S0.u) + S1.u.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 232.
//   vega-shader-instruction-set-architecture.pdf, page 180.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 173.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 171.
//   rdna-shader-instruction-set-architecture.pdf, page 197.
//   rdna2-shader-instruction-set-architecture.pdf, page 192.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 444.
//   rdna4-instruction-set-architecture.pdf, page 516.
def V_BCNT_U32_B32_e64_sem : InstSISemantic<V_BCNT_U32_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMUnaryIntrinsic int_ctpop, (GetNamedOperand $src0)),
               (GetNamedOperand $src1)))
  ];
}

// D.u = MaskedBitCount(S0.u, EXEC[31:0]) + S1.u (intrinsic).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 175.
//   vega-shader-instruction-set-architecture.pdf, page 180.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 173.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 171.
//   rdna-shader-instruction-set-architecture.pdf, page 197.
//   rdna2-shader-instruction-set-architecture.pdf, page 192.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 444.
//   rdna4-instruction-set-architecture.pdf, page 517.
def V_MBCNT_LO_U32_B32_e64_sem : InstSISemantic<V_MBCNT_LO_U32_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_mbcnt_lo, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))
  ];
}

// D.u = MaskedBitCount(S0.u, EXEC[63:32]) + S1.u (intrinsic).
def V_MBCNT_HI_U32_B32_e64_sem : InstSISemantic<V_MBCNT_HI_U32_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_mbcnt_hi, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))
  ];
}

//------------------------------------------------------------------------------
// VOP2 LDEXP e64
//------------------------------------------------------------------------------

// D.f = S0.f * 2^S1.i (ldexp).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 244.
//   vega-shader-instruction-set-architecture.pdf, page 179.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 181.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 172.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 170.
//   rdna-shader-instruction-set-architecture.pdf, page 197.
//   rdna2-shader-instruction-set-architecture.pdf, page 192.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 443.
//   rdna4-instruction-set-architecture.pdf, page 515.
def V_LDEXP_F32_e64_sem : InstSISemantic<V_LDEXP_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMLdexp (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                   (GetNamedOperand $src1)),
        si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 F16 Arithmetic e64
//------------------------------------------------------------------------------

// D.f16 = S0.f16 + S1.f16.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 164.
//   vega-shader-instruction-set-architecture.pdf, page 132.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 177.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 189.
//   rdna-shader-instruction-set-architecture.pdf, page 144.
//   rdna2-shader-instruction-set-architecture.pdf, page 143.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 269.
//   rdna4-instruction-set-architecture.pdf, page 312.
def V_ADD_F16_e64_sem : InstSISemantic<V_ADD_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFAdd
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f16 = S0.f16 - S1.f16.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 183.
//   vega-shader-instruction-set-architecture.pdf, page 132.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 177.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 190.
//   rdna-shader-instruction-set-architecture.pdf, page 144.
//   rdna2-shader-instruction-set-architecture.pdf, page 143.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 269.
//   rdna4-instruction-set-architecture.pdf, page 312.
def V_SUB_F16_e64_sem : InstSISemantic<V_SUB_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFSub
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f16 = S1.f16 - S0.f16 (reversed operands).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 183.
//   vega-shader-instruction-set-architecture.pdf, page 132.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 177.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 190.
//   rdna-shader-instruction-set-architecture.pdf, page 144.
//   rdna2-shader-instruction-set-architecture.pdf, page 143.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 269.
//   rdna4-instruction-set-architecture.pdf, page 312.
def V_SUBREV_F16_e64_sem : InstSISemantic<V_SUBREV_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFSub
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f16 = S0.f16 * S1.f16.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 178.
//   vega-shader-instruction-set-architecture.pdf, page 132.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 178.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 190.
//   rdna-shader-instruction-set-architecture.pdf, page 145.
//   rdna2-shader-instruction-set-architecture.pdf, page 143.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 270.
//   rdna4-instruction-set-architecture.pdf, page 313.
def V_MUL_F16_e64_sem : InstSISemantic<V_MUL_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFMul
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f16 = max(S0.f16, S1.f16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 172.
//   vega-shader-instruction-set-architecture.pdf, page 133.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 181.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 194.
//   rdna-shader-instruction-set-architecture.pdf, page 146.
//   rdna2-shader-instruction-set-architecture.pdf, page 144.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 271.
//   rdna4-instruction-set-architecture.pdf, page 311 (as V_MAX_NUM_F16).
def V_MAX_F16_e64_sem : InstSISemantic<V_MAX_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMaxNum
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f16 = min(S0.f16, S1.f16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 175.
//   vega-shader-instruction-set-architecture.pdf, page 133.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 181.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 194.
//   rdna-shader-instruction-set-architecture.pdf, page 146.
//   rdna2-shader-instruction-set-architecture.pdf, page 144.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 272.
//   rdna4-instruction-set-architecture.pdf, page 311 (as V_MIN_NUM_F16).
def V_MIN_F16_e64_sem : InstSISemantic<V_MIN_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMinNum
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f16 = S0.f16 * 2^S1.i (ldexp, half precision).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 168.
//   vega-shader-instruction-set-architecture.pdf, page 134.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 182.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 195.
//   rdna-shader-instruction-set-architecture.pdf, page 146.
//   rdna2-shader-instruction-set-architecture.pdf, page 144.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 272.
//   rdna4-instruction-set-architecture.pdf, page 314.
def V_LDEXP_F16_e64_sem : InstSISemantic<V_LDEXP_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMLdexp
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 I16 Arithmetic e64
//------------------------------------------------------------------------------

// D.u16 = S0.u16 + S1.u16.
// Sources (GFX8/GFX9 only, isGFX8GFX9):
//   gcn3-instruction-set-architecture.pdf, page 164.
//   vega-shader-instruction-set-architecture.pdf, page 132.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 179.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 191.
foreach inst = [
    V_ADD_U16_e64,
    V_ADD_NC_U16_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMAdd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                   (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
          si_llvm_i32_ty))
    ];
  }
}

// D.u16 = S0.u16 - S1.u16.
// Sources (GFX8/GFX9 only):
//   gcn3-instruction-set-architecture.pdf, page 183.
//   vega-shader-instruction-set-architecture.pdf, page 132.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 179.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 192.
foreach inst = [
    V_SUB_U16_e64,
    V_SUB_NC_U16_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMSub (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                   (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
          si_llvm_i32_ty))
    ];
  }
}

// D.u16 = S1.u16 - S0.u16 (reversed operands).
// Sources (GFX8/GFX9 only):
//   gcn3-instruction-set-architecture.pdf, page 186.
//   vega-shader-instruction-set-architecture.pdf, page 132.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 180.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 192.
def V_SUBREV_U16_e64_sem : InstSISemantic<V_SUBREV_U16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSub (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                 (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// D.u16 = S0.u16 * S1.u16 (low 16 bits).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 181.
//   vega-shader-instruction-set-architecture.pdf, page 133.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 180.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 192.
def V_MUL_LO_U16_e64_sem : InstSISemantic<V_MUL_LO_U16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                 (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// D.i16 = max(S0.i16, S1.i16) (signed).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 173.
//   vega-shader-instruction-set-architecture.pdf, page 134.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 182.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 194.
def V_MAX_I16_e64_sem : InstSISemantic<V_MAX_I16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// D.i16 = min(S0.i16, S1.i16) (signed).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 176.
//   vega-shader-instruction-set-architecture.pdf, page 134.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 182.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 195.
def V_MIN_I16_e64_sem : InstSISemantic<V_MIN_I16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// D.u16 = max(S0.u16, S1.u16) (unsigned).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 173.
//   vega-shader-instruction-set-architecture.pdf, page 133.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 182.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 194.
def V_MAX_U16_e64_sem : InstSISemantic<V_MAX_U16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// D.u16 = min(S0.u16, S1.u16) (unsigned).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 177.
//   vega-shader-instruction-set-architecture.pdf, page 134.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 182.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 194.
def V_MIN_U16_e64_sem : InstSISemantic<V_MIN_U16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 I16 Shifts e64
//------------------------------------------------------------------------------

// D.u16 = S1.u16 << S0.u16 (reversed operand order, 16-bit shift left).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 168.
//   vega-shader-instruction-set-architecture.pdf, page 133.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 180.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 192.
def V_LSHLREV_B16_e64_sem : InstSISemantic<V_LSHLREV_B16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMShl (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                 (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// D.u16 = S1.u16 >> S0.u16 (reversed operand order, 16-bit logical shift right).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 169.
//   vega-shader-instruction-set-architecture.pdf, page 133.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 180.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 193.
def V_LSHRREV_B16_e64_sem : InstSISemantic<V_LSHRREV_B16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMLShr (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                  (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                            (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// D.i16 = S1.i16 >> S0.u16 (reversed operand order, 16-bit arithmetic shift right).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 166.
//   vega-shader-instruction-set-architecture.pdf, page 133.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 180.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 193.
def V_ASHRREV_I16_e64_sem : InstSISemantic<V_ASHRREV_I16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAShr (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                  (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                            (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 MAC (Multiply-Accumulate, non-fused) e64
//------------------------------------------------------------------------------

// D.f += S0.f * S1.f (non-fused multiply-accumulate).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 170.
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 143.
def V_MAC_F32_e64_sem : InstSISemantic<V_MAC_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd
          (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f16 += S0.f16 * S1.f16 (non-fused multiply-accumulate, f16).
// Sources (Has16BitInsts):
//   gcn3-instruction-set-architecture.pdf, page 169.
//   vega-shader-instruction-set-architecture.pdf, page 132.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 178.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 190.
def V_MAC_F16_e64_sem : InstSISemantic<V_MAC_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFAdd
            (LLVMFMul
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f += S0.f * S1.f (DX9 legacy non-fused multiply-accumulate).
// Sources (isGFX6GFX7GFX10):
//   rdna-shader-instruction-set-architecture.pdf, page 140.
def V_MAC_LEGACY_F32_e64_sem : InstSISemantic<V_MAC_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd
          (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 FMAC (Fused Multiply-Accumulate) e64
//------------------------------------------------------------------------------

// D.f = FMA(S0.f, S1.f, D.f) (fused multiply-accumulate).
// Sources (HasDLInsts / GFX9+):
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 185.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 197.
//   rdna-shader-instruction-set-architecture.pdf, page 144.
//   rdna2-shader-instruction-set-architecture.pdf, page 142.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 268.
//   rdna4-instruction-set-architecture.pdf, page 309.
def V_FMAC_F32_e64_sem : InstSISemantic<V_FMAC_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f16 = FMA(S0.f16, S1.f16, D.f16) (fused multiply-accumulate, f16).
// Sources (isGFX10Plus):
//   rdna-shader-instruction-set-architecture.pdf, page 145.
//   rdna2-shader-instruction-set-architecture.pdf, page 143.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 270.
//   rdna4-instruction-set-architecture.pdf, page 313.
def V_FMAC_F16_e64_sem : InstSISemantic<V_FMAC_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFMA
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.d = FMA(S0.d, S1.d, D.d) (fused multiply-accumulate, f64).
// Sources (HasFmacF64Inst):
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 170.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 182.
def V_FMAC_F64_e64_sem : InstSISemantic<V_FMAC_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                 (LLVMBitCast (GetNamedOperand $src1), si_llvm_double_ty),
                 (LLVMBitCast (GetNamedOperand $src2), si_llvm_double_ty)),
        si_llvm_i64_ty))
  ];
}

// D.f = FMA(S0.f, S1.f, D.f) (DX9 legacy fused multiply-accumulate).
// Sources (HasFmaLegacy32):
//   rdna2-shader-instruction-set-architecture.pdf, page 257.
def V_FMAC_LEGACY_F32_e64_sem : InstSISemantic<V_FMAC_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

//==============================================================================
// VOP1 e64 Variants
//==============================================================================

//------------------------------------------------------------------------------
// VOP1 Move e64
//------------------------------------------------------------------------------

// D.u = S0.u.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 214.
//   vega-shader-instruction-set-architecture.pdf, page 134.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 131.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 186.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 199.
//   rdna-shader-instruction-set-architecture.pdf, page 147.
//   rdna2-shader-instruction-set-architecture.pdf, page 145.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 274.
//   rdna4-instruction-set-architecture.pdf, page 315.
def V_MOV_B32_e64_sem : InstSISemantic<V_MOV_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Float-to-Int / Int-to-Float Conversion e64
//------------------------------------------------------------------------------

// D.f = (float)S0.i.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 194.
//   vega-shader-instruction-set-architecture.pdf, page 135.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 131.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 188.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 201.
//   rdna-shader-instruction-set-architecture.pdf, page 147.
//   rdna2-shader-instruction-set-architecture.pdf, page 146.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 275.
//   rdna4-instruction-set-architecture.pdf, page 317.
def V_CVT_F32_I32_e64_sem : InstSISemantic<V_CVT_F32_I32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMSIToFP (GetNamedOperand $src0), si_llvm_float_ty),
                   si_llvm_i32_ty))
  ];
}

// D.f = (float)S0.u.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 195.
//   vega-shader-instruction-set-architecture.pdf, page 135.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 131.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 188.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 201.
//   rdna-shader-instruction-set-architecture.pdf, page 147.
//   rdna2-shader-instruction-set-architecture.pdf, page 146.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 276.
//   rdna4-instruction-set-architecture.pdf, page 317.
def V_CVT_F32_U32_e64_sem : InstSISemantic<V_CVT_F32_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUIToFP (GetNamedOperand $src0), si_llvm_float_ty),
                   si_llvm_i32_ty))
  ];
}

// D.i = (int)S0.f (truncate toward zero).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 200.
//   vega-shader-instruction-set-architecture.pdf, page 135.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 131.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 189.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 202.
//   rdna-shader-instruction-set-architecture.pdf, page 148.
//   rdna2-shader-instruction-set-architecture.pdf, page 146.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 276.
//   rdna4-instruction-set-architecture.pdf, page 317.
def V_CVT_I32_F32_e64_sem : InstSISemantic<V_CVT_I32_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToSI (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                   si_llvm_i32_ty))
  ];
}

// D.u = (uint)S0.f (truncate toward zero).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 203.
//   vega-shader-instruction-set-architecture.pdf, page 135.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 131.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 188.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 201.
//   rdna-shader-instruction-set-architecture.pdf, page 148.
//   rdna2-shader-instruction-set-architecture.pdf, page 146.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 276.
//   rdna4-instruction-set-architecture.pdf, page 317.
def V_CVT_U32_F32_e64_sem : InstSISemantic<V_CVT_U32_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToUI (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                   si_llvm_i32_ty))
  ];
}

// D.f = (float)S0.d. F64 to F32 conversion.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 194.
//   vega-shader-instruction-set-architecture.pdf, page 136.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 137.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 132.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 191.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 204.
//   rdna-shader-instruction-set-architecture.pdf, page 149.
//   rdna2-shader-instruction-set-architecture.pdf, page 147.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 278.
//   rdna4-instruction-set-architecture.pdf, page 319.
def V_CVT_F32_F64_e64_sem : InstSISemantic<V_CVT_F32_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFPTrunc (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                     si_llvm_float_ty),
        si_llvm_i32_ty))
  ];
}

// D.d = (double)S0.f. F32 to F64 conversion.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 197.
//   vega-shader-instruction-set-architecture.pdf, page 136.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 137.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 132.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 191.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 204.
//   rdna-shader-instruction-set-architecture.pdf, page 149.
//   rdna2-shader-instruction-set-architecture.pdf, page 147.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 279.
//   rdna4-instruction-set-architecture.pdf, page 320.
def V_CVT_F64_F32_e64_sem : InstSISemantic<V_CVT_F64_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFPExt (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                   si_llvm_double_ty),
        si_llvm_i64_ty))
  ];
}

// D.d = (double)S0.i. I32 to F64 conversion.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 198.
//   vega-shader-instruction-set-architecture.pdf, page 135.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 131.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 187.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 200.
//   rdna-shader-instruction-set-architecture.pdf, page 147.
//   rdna2-shader-instruction-set-architecture.pdf, page 146.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 275.
//   rdna4-instruction-set-architecture.pdf, page 316.
def V_CVT_F64_I32_e64_sem : InstSISemantic<V_CVT_F64_I32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMSIToFP (GetNamedOperand $src0), si_llvm_double_ty),
                   si_llvm_i64_ty))
  ];
}

// D.d = (double)S0.u. U32 to F64 conversion.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 198.
//   vega-shader-instruction-set-architecture.pdf, page 137.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 132.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 192.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 205.
//   rdna-shader-instruction-set-architecture.pdf, page 150.
//   rdna2-shader-instruction-set-architecture.pdf, page 147.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 280.
//   rdna4-instruction-set-architecture.pdf, page 321.
def V_CVT_F64_U32_e64_sem : InstSISemantic<V_CVT_F64_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUIToFP (GetNamedOperand $src0), si_llvm_double_ty),
                   si_llvm_i64_ty))
  ];
}

// D.i = (int)S0.d (truncate toward zero). F64 to I32.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 200.
//   vega-shader-instruction-set-architecture.pdf, page 135.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 131.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 187.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 200.
//   rdna-shader-instruction-set-architecture.pdf, page 147.
//   rdna2-shader-instruction-set-architecture.pdf, page 145.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 275.
//   rdna4-instruction-set-architecture.pdf, page 316.
def V_CVT_I32_F64_e64_sem : InstSISemantic<V_CVT_I32_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToSI (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                   si_llvm_i32_ty))
  ];
}

// D.u = (uint)S0.d (truncate toward zero). F64 to U32.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 203.
//   vega-shader-instruction-set-architecture.pdf, page 136.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 132.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 192.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 205.
//   rdna-shader-instruction-set-architecture.pdf, page 149.
//   rdna2-shader-instruction-set-architecture.pdf, page 147.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 280.
//   rdna4-instruction-set-architecture.pdf, page 321.
def V_CVT_U32_F64_e64_sem : InstSISemantic<V_CVT_U32_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToUI (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                   si_llvm_i32_ty))
  ];
}

// D.f16 = float_to_half(S0.f). F32 to F16 conversion.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 192.
//   vega-shader-instruction-set-architecture.pdf, page 135.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 137.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 131.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 189.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 202.
//   rdna-shader-instruction-set-architecture.pdf, page 148.
//   rdna2-shader-instruction-set-architecture.pdf, page 146.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 277.
//   rdna4-instruction-set-architecture.pdf, page 318.
def V_CVT_F16_F32_e64_sem : InstSISemantic<V_CVT_F16_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFPTrunc (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f = half_to_float(S0.f16). F16 to F32 conversion.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 193.
//   vega-shader-instruction-set-architecture.pdf, page 135.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 137.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 131.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 189.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 202.
//   rdna-shader-instruction-set-architecture.pdf, page 148.
//   rdna2-shader-instruction-set-architecture.pdf, page 146.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 277.
//   rdna4-instruction-set-architecture.pdf, page 318.
def V_CVT_F32_F16_e64_sem : InstSISemantic<V_CVT_F32_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFPExt (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                            si_llvm_i16_ty),
                                 si_llvm_half_ty),
                   si_llvm_float_ty),
        si_llvm_i32_ty))
  ];
}

// D.i = (int)floor(S0.f). Float to int with floor.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 199.
//   vega-shader-instruction-set-architecture.pdf, page 136.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 137.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 132.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 190.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 203.
//   rdna-shader-instruction-set-architecture.pdf, page 148.
//   rdna2-shader-instruction-set-architecture.pdf, page 146.
def V_CVT_FLR_I32_F32_e64_sem : InstSISemantic<V_CVT_FLR_I32_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToSI
        (LLVMUnaryIntrinsic int_floor, (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.i = (int)floor(S0.f + 0.5). Round to positive infinity.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 202.
//   vega-shader-instruction-set-architecture.pdf, page 136.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 137.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 132.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 189.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 202.
//   rdna-shader-instruction-set-architecture.pdf, page 148.
//   rdna2-shader-instruction-set-architecture.pdf, page 146.
def V_CVT_RPI_I32_F32_e64_sem : InstSISemantic<V_CVT_RPI_I32_F32_e64> {
  let Semantic = [
    // Constant 0.5 cannot be expressed as a tablegen float literal; semantic
    // omitted until a fp-bitpattern primitive is added to SISemOps.td.
  ];
}

// D.f = 4-bit signed int to float.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 201.
//   vega-shader-instruction-set-architecture.pdf, page 136.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 137.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 132.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 190.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 203.
//   rdna-shader-instruction-set-architecture.pdf, page 149.
//   rdna2-shader-instruction-set-architecture.pdf, page 147.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 278.
//   rdna4-instruction-set-architecture.pdf, page 319.
def V_CVT_OFF_F32_I4_e64_sem : InstSISemantic<V_CVT_OFF_F32_I4_e64> {
  let Semantic = [
    (DefVal $sext, (LLVMAShr (LLVMShl (GetNamedOperand $src0),
                                       (ConstantInt si_llvm_i32_ty, 28)),
                              (ConstantInt si_llvm_i32_ty, 28))),
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMSIToFP (GetVal $sext), si_llvm_float_ty),
                   si_llvm_i32_ty))
  ];
}

// D.f = (float)S0.u[byte_n]. Convert unsigned byte N to float.
foreach i = 0-3 in {
  defvar inst = ["V_CVT_F32_UBYTE0_e64", "V_CVT_F32_UBYTE1_e64",
                 "V_CVT_F32_UBYTE2_e64", "V_CVT_F32_UBYTE3_e64"][i];
  defvar shift = [0, 8, 16, 24][i];

  def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMBitCast
          (LLVMUIToFP
            (LLVMAnd (LLVMLShr (GetNamedOperand $src0),
                               (ConstantInt si_llvm_i32_ty, shift)),
                     (ConstantInt si_llvm_i32_ty, 255)),
            si_llvm_float_ty),
          si_llvm_i32_ty))
    ];
  }
}

//------------------------------------------------------------------------------
// VOP1 F16 Conversion e64
//------------------------------------------------------------------------------

// D.f16 = (half)S0.i16. Signed I16 to F16.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 193.
//   vega-shader-instruction-set-architecture.pdf, page 145.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 146.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 140.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 140.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 204.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 217.
//   rdna-shader-instruction-set-architecture.pdf, page 156.
//   rdna2-shader-instruction-set-architecture.pdf, page 154.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 293.
//   rdna4-instruction-set-architecture.pdf, page 335.
def V_CVT_F16_I16_e64_sem : InstSISemantic<V_CVT_F16_I16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMSIToFP (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f16 = (half)S0.u16. Unsigned U16 to F16.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 193.
//   vega-shader-instruction-set-architecture.pdf, page 145.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 146.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 140.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 140.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 204.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 217.
//   rdna-shader-instruction-set-architecture.pdf, page 156.
//   rdna2-shader-instruction-set-architecture.pdf, page 154.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 293.
//   rdna4-instruction-set-architecture.pdf, page 334.
def V_CVT_F16_U16_e64_sem : InstSISemantic<V_CVT_F16_U16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUIToFP (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.i16 = (int16)S0.f16. F16 to signed I16.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 199.
//   vega-shader-instruction-set-architecture.pdf, page 145.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 146.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 140.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 140.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 205.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 218.
//   rdna-shader-instruction-set-architecture.pdf, page 157.
//   rdna2-shader-instruction-set-architecture.pdf, page 154.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 294.
//   rdna4-instruction-set-architecture.pdf, page 335.
def V_CVT_I16_F16_e64_sem : InstSISemantic<V_CVT_I16_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMFPToSI
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.u16 = (uint16)S0.f16. F16 to unsigned U16.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 202.
//   vega-shader-instruction-set-architecture.pdf, page 145.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 146.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 140.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 140.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 204.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 217.
//   rdna-shader-instruction-set-architecture.pdf, page 157.
//   rdna2-shader-instruction-set-architecture.pdf, page 154.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 294.
//   rdna4-instruction-set-architecture.pdf, page 335.
def V_CVT_U16_F16_e64_sem : InstSISemantic<V_CVT_U16_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMFPToUI
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.i = sign_extend(S0.i16). I16 to I32 sign extension.
def V_CVT_I32_I16_e64_sem : InstSISemantic<V_CVT_I32_I16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSExt (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                si_llvm_i32_ty))
  ];
}

// D.u = zero_extend(S0.u16). U16 to U32 zero extension.
def V_CVT_U32_U16_e64_sem : InstSISemantic<V_CVT_U32_U16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                si_llvm_i32_ty))
  ];
}

// Normalized F16 to I16/U16 conversions (intrinsics).
// Sources:
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 210.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 223.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 300.
//   rdna4-instruction-set-architecture.pdf, page 341.
def V_CVT_NORM_I16_F16_e64_sem : InstSISemantic<V_CVT_NORM_I16_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_pknorm_i16, [(GetNamedOperand $src0)]))
  ];
}
def V_CVT_NORM_U16_F16_e64_sem : InstSISemantic<V_CVT_NORM_U16_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_pknorm_u16, [(GetNamedOperand $src0)]))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Math e64 (hardware transcendental approximations)
//------------------------------------------------------------------------------

// Square root.
def V_SQRT_F32_e64_sem : InstSISemantic<V_SQRT_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_sqrt, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 224.
//   vega-shader-instruction-set-architecture.pdf, page 140.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 141.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 135.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 198.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 211.
//   rdna-shader-instruction-set-architecture.pdf, page 152.
//   rdna2-shader-instruction-set-architecture.pdf, page 150.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 286.
//   rdna4-instruction-set-architecture.pdf, page 327.
def V_SQRT_F64_e64_sem : InstSISemantic<V_SQRT_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_double_ty, int_amdgcn_sqrt, [(GetNamedOperand $src0, si_llvm_double_ty)]))
  ];
}
def V_SQRT_F16_e64_sem : InstSISemantic<V_SQRT_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_sqrt, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// Reciprocal square root.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 221.
//   vega-shader-instruction-set-architecture.pdf, page 139.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 140.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 135.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 197.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 210.
//   rdna-shader-instruction-set-architecture.pdf, page 152.
//   rdna2-shader-instruction-set-architecture.pdf, page 150.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 285.
//   rdna4-instruction-set-architecture.pdf, page 326.
def V_RSQ_F32_e64_sem : InstSISemantic<V_RSQ_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_rsq, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_RSQ_F64_e64_sem : InstSISemantic<V_RSQ_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_double_ty, int_amdgcn_rsq, [(GetNamedOperand $src0, si_llvm_double_ty)]))
  ];
}
def V_RSQ_F16_e64_sem : InstSISemantic<V_RSQ_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_rsq, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// Reciprocal.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 218.
//   vega-shader-instruction-set-architecture.pdf, page 138.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 139.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 134.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 196.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 209.
//   rdna-shader-instruction-set-architecture.pdf, page 151.
//   rdna2-shader-instruction-set-architecture.pdf, page 149.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 284.
//   rdna4-instruction-set-architecture.pdf, page 325.
def V_RCP_F32_e64_sem : InstSISemantic<V_RCP_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_rcp, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_RCP_F64_e64_sem : InstSISemantic<V_RCP_F64_e64> {
  let Semantic = [
    // See V_RCP_F64_e32_sem: `llvm.amdgcn.rcp` requires its argument
    // type to match the return type, so request the f64 view of src0.
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_double_ty, int_amdgcn_rcp,
        [(GetNamedOperand $src0, si_llvm_double_ty)]))
  ];
}
def V_RCP_F16_e64_sem : InstSISemantic<V_RCP_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_rcp, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// Reciprocal with integer division-by-zero flag.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 219.
//   vega-shader-instruction-set-architecture.pdf, page 139.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 140.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 134.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 196.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 209.
//   rdna-shader-instruction-set-architecture.pdf, page 152.
//   rdna2-shader-instruction-set-architecture.pdf, page 149.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 284.
//   rdna4-instruction-set-architecture.pdf, page 325.
def V_RCP_IFLAG_F32_e64_sem : InstSISemantic<V_RCP_IFLAG_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_rcp, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}

// Clamped reciprocal (GFX6/GFX7).
def V_RCP_CLAMP_F32_e64_sem : InstSISemantic<V_RCP_CLAMP_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_rcp, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_RCP_CLAMP_F64_e64_sem : InstSISemantic<V_RCP_CLAMP_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_double_ty, int_amdgcn_rcp, [(GetNamedOperand $src0, si_llvm_double_ty)]))
  ];
}

// Clamped RSQ (GFX6/GFX7).
def V_RSQ_CLAMP_F32_e64_sem : InstSISemantic<V_RSQ_CLAMP_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_rsq_clamp, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_RSQ_CLAMP_F64_e64_sem : InstSISemantic<V_RSQ_CLAMP_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_double_ty, int_amdgcn_rsq_clamp, [(GetNamedOperand $src0, si_llvm_double_ty)]))
  ];
}

// Legacy RSQ (GFX6/GFX7).
def V_RSQ_LEGACY_F32_e64_sem : InstSISemantic<V_RSQ_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_rsq_legacy, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}

// Legacy reciprocal (GFX6/GFX7).
def V_RCP_LEGACY_F32_e64_sem : InstSISemantic<V_RCP_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_rcp_legacy, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Trig e64
//------------------------------------------------------------------------------

// D.f = sin(S0.f * 2 * PI) (hardware sine).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 223.
//   vega-shader-instruction-set-architecture.pdf, page 140.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 141.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 135.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 198.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 211.
//   rdna-shader-instruction-set-architecture.pdf, page 153.
//   rdna2-shader-instruction-set-architecture.pdf, page 150.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 286.
//   rdna4-instruction-set-architecture.pdf, page 328.
def V_SIN_F32_e64_sem : InstSISemantic<V_SIN_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_sin, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_SIN_F16_e64_sem : InstSISemantic<V_SIN_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_sin, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// D.f = cos(S0.f * 2 * PI) (hardware cosine).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 192.
//   vega-shader-instruction-set-architecture.pdf, page 140.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 141.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 135.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 199.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 212.
//   rdna-shader-instruction-set-architecture.pdf, page 153.
//   rdna2-shader-instruction-set-architecture.pdf, page 151.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 287.
//   rdna4-instruction-set-architecture.pdf, page 328.
def V_COS_F32_e64_sem : InstSISemantic<V_COS_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_cos, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_COS_F16_e64_sem : InstSISemantic<V_COS_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_cos, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Exp/Log e64
//------------------------------------------------------------------------------

// D.f = 2^S0.f (hardware base-2 exponential).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 204.
//   vega-shader-instruction-set-architecture.pdf, page 138.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 139.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 133.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 195.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 208.
//   rdna-shader-instruction-set-architecture.pdf, page 151.
//   rdna2-shader-instruction-set-architecture.pdf, page 149.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 283.
//   rdna4-instruction-set-architecture.pdf, page 324.
def V_EXP_F32_e64_sem : InstSISemantic<V_EXP_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_exp2, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_EXP_F16_e64_sem : InstSISemantic<V_EXP_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_exp2, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}
def V_EXP_LEGACY_F32_e64_sem : InstSISemantic<V_EXP_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_exp2, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}

// D.f = log2(S0.f) (hardware base-2 logarithm).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 213.
//   vega-shader-instruction-set-architecture.pdf, page 138.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 139.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 134.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 195.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 208.
//   rdna-shader-instruction-set-architecture.pdf, page 151.
//   rdna2-shader-instruction-set-architecture.pdf, page 149.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 283.
//   rdna4-instruction-set-architecture.pdf, page 324.
def V_LOG_F32_e64_sem : InstSISemantic<V_LOG_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_log, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_LOG_F16_e64_sem : InstSISemantic<V_LOG_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_log, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}
def V_LOG_LEGACY_F32_e64_sem : InstSISemantic<V_LOG_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_log, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_LOG_CLAMP_F32_e64_sem : InstSISemantic<V_LOG_CLAMP_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_log_clamp, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Rounding e64
//------------------------------------------------------------------------------

// D.f = ceil(S0.f).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 190.
//   vega-shader-instruction-set-architecture.pdf, page 137.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 133.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 194.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 207.
//   rdna-shader-instruction-set-architecture.pdf, page 150.
//   rdna2-shader-instruction-set-architecture.pdf, page 148.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 282.
//   rdna4-instruction-set-architecture.pdf, page 323.
def V_CEIL_F32_e64_sem : InstSISemantic<V_CEIL_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUnaryIntrinsic int_ceil,
                      (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty)),
                   si_llvm_i32_ty))
  ];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 190.
//   vega-shader-instruction-set-architecture.pdf, page 137.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 133.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 194.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 207.
//   rdna-shader-instruction-set-architecture.pdf, page 150.
//   rdna2-shader-instruction-set-architecture.pdf, page 148.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 282.
//   rdna4-instruction-set-architecture.pdf, page 323.
def V_CEIL_F64_e64_sem : InstSISemantic<V_CEIL_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUnaryIntrinsic int_ceil, (LLVMBitCast (GetNamedOperand $src0),
                                            si_llvm_double_ty)),
                   si_llvm_i64_ty))
  ];
}
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 139.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 140.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 135.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 197.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 210.
//   rdna-shader-instruction-set-architecture.pdf, page 152.
//   rdna2-shader-instruction-set-architecture.pdf, page 150.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 285.
//   rdna4-instruction-set-architecture.pdf, page 327.
def V_CEIL_F16_e64_sem : InstSISemantic<V_CEIL_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_ceil, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                              si_llvm_i16_ty),
                                  si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f = floor(S0.f).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 207.
//   vega-shader-instruction-set-architecture.pdf, page 138.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 139.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 133.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 195.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 208.
//   rdna-shader-instruction-set-architecture.pdf, page 151.
//   rdna2-shader-instruction-set-architecture.pdf, page 148.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 283.
//   rdna4-instruction-set-architecture.pdf, page 324.
def V_FLOOR_F32_e64_sem : InstSISemantic<V_FLOOR_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUnaryIntrinsic int_floor, (LLVMBitCast (GetNamedOperand $src0),
                                             si_llvm_float_ty)),
                   si_llvm_i32_ty))
  ];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 207.
//   vega-shader-instruction-set-architecture.pdf, page 138.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 139.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 133.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 195.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 208.
//   rdna-shader-instruction-set-architecture.pdf, page 151.
//   rdna2-shader-instruction-set-architecture.pdf, page 148.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 283.
//   rdna4-instruction-set-architecture.pdf, page 324.
def V_FLOOR_F64_e64_sem : InstSISemantic<V_FLOOR_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUnaryIntrinsic int_floor, (LLVMBitCast (GetNamedOperand $src0),
                                             si_llvm_double_ty)),
                   si_llvm_i64_ty))
  ];
}
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 139.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 140.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 135.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 197.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 210.
//   rdna-shader-instruction-set-architecture.pdf, page 152.
//   rdna2-shader-instruction-set-architecture.pdf, page 150.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 285.
//   rdna4-instruction-set-architecture.pdf, page 327.
def V_FLOOR_F16_e64_sem : InstSISemantic<V_FLOOR_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_floor, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                               si_llvm_i16_ty),
                                   si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f = trunc(S0.f) (toward zero).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 225.
//   vega-shader-instruction-set-architecture.pdf, page 137.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 133.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 194.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 207.
//   rdna-shader-instruction-set-architecture.pdf, page 150.
//   rdna2-shader-instruction-set-architecture.pdf, page 148.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 282.
//   rdna4-instruction-set-architecture.pdf, page 323.
def V_TRUNC_F32_e64_sem : InstSISemantic<V_TRUNC_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUnaryIntrinsic int_trunc, (LLVMBitCast (GetNamedOperand $src0),
                                             si_llvm_float_ty)),
                   si_llvm_i32_ty))
  ];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 225.
//   vega-shader-instruction-set-architecture.pdf, page 137.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 133.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 194.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 207.
//   rdna-shader-instruction-set-architecture.pdf, page 150.
//   rdna2-shader-instruction-set-architecture.pdf, page 148.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 282.
//   rdna4-instruction-set-architecture.pdf, page 323.
def V_TRUNC_F64_e64_sem : InstSISemantic<V_TRUNC_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUnaryIntrinsic int_trunc, (LLVMBitCast (GetNamedOperand $src0),
                                             si_llvm_double_ty)),
                   si_llvm_i64_ty))
  ];
}
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 139.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 140.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 135.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 197.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 210.
//   rdna-shader-instruction-set-architecture.pdf, page 152.
//   rdna2-shader-instruction-set-architecture.pdf, page 150.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 285.
//   rdna4-instruction-set-architecture.pdf, page 327.
def V_TRUNC_F16_e64_sem : InstSISemantic<V_TRUNC_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_trunc, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                                si_llvm_i16_ty),
                                    si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f = roundeven(S0.f) (round to nearest even).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 220.
//   vega-shader-instruction-set-architecture.pdf, page 137.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 139.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 133.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 194.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 207.
//   rdna-shader-instruction-set-architecture.pdf, page 150.
//   rdna2-shader-instruction-set-architecture.pdf, page 148.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 282.
//   rdna4-instruction-set-architecture.pdf, page 324.
def V_RNDNE_F32_e64_sem : InstSISemantic<V_RNDNE_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUnaryIntrinsic int_roundeven, (LLVMBitCast (GetNamedOperand $src0),
                                                si_llvm_float_ty)),
                   si_llvm_i32_ty))
  ];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 220.
//   vega-shader-instruction-set-architecture.pdf, page 137.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 139.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 133.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 194.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 207.
//   rdna-shader-instruction-set-architecture.pdf, page 150.
//   rdna2-shader-instruction-set-architecture.pdf, page 148.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 282.
//   rdna4-instruction-set-architecture.pdf, page 324.
def V_RNDNE_F64_e64_sem : InstSISemantic<V_RNDNE_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUnaryIntrinsic int_roundeven, (LLVMBitCast (GetNamedOperand $src0),
                                                si_llvm_double_ty)),
                   si_llvm_i64_ty))
  ];
}
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 139.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 140.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 135.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 197.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 210.
//   rdna-shader-instruction-set-architecture.pdf, page 152.
//   rdna2-shader-instruction-set-architecture.pdf, page 150.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 285.
//   rdna4-instruction-set-architecture.pdf, page 327.
def V_RNDNE_F16_e64_sem : InstSISemantic<V_RNDNE_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_roundeven, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                                   si_llvm_i16_ty),
                                       si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Fractional Part e64
//------------------------------------------------------------------------------

// D.f = fract(S0.f) = S0.f - floor(S0.f) (intrinsic).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 209.
//   vega-shader-instruction-set-architecture.pdf, page 137.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 133.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 193.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 206.
//   rdna-shader-instruction-set-architecture.pdf, page 150.
//   rdna2-shader-instruction-set-architecture.pdf, page 148.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 282.
//   rdna4-instruction-set-architecture.pdf, page 323.
def V_FRACT_F32_e64_sem : InstSISemantic<V_FRACT_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_fract, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_FRACT_F64_e64_sem : InstSISemantic<V_FRACT_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_double_ty, int_amdgcn_fract, [(GetNamedOperand $src0, si_llvm_double_ty)]))
  ];
}
def V_FRACT_F16_e64_sem : InstSISemantic<V_FRACT_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_fract, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Frexp e64
//------------------------------------------------------------------------------

// D.f = frexp_mant(S0.f). Extract mantissa (intrinsic).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 212.
//   vega-shader-instruction-set-architecture.pdf, page 143.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 144.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 138.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 203.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 216.
//   rdna-shader-instruction-set-architecture.pdf, page 155.
//   rdna2-shader-instruction-set-architecture.pdf, page 153.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 291.
//   rdna4-instruction-set-architecture.pdf, page 332.
def V_FREXP_MANT_F32_e64_sem : InstSISemantic<V_FREXP_MANT_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_frexp_mant, [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_FREXP_MANT_F64_e64_sem : InstSISemantic<V_FREXP_MANT_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_double_ty, int_amdgcn_frexp_mant, [(GetNamedOperand $src0, si_llvm_double_ty)]))
  ];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 212.
//   vega-shader-instruction-set-architecture.pdf, page 143.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 144.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 138.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 203.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 216.
//   rdna-shader-instruction-set-architecture.pdf, page 155.
//   rdna2-shader-instruction-set-architecture.pdf, page 153.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 291.
//   rdna4-instruction-set-architecture.pdf, page 332.
def V_FREXP_MANT_F16_e64_sem : InstSISemantic<V_FREXP_MANT_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_frexp_mant, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// D.i = frexp_exp(S0.f). Extract exponent (intrinsic).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 210.
//   vega-shader-instruction-set-architecture.pdf, page 143.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 144.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 138.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 203.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 216.
//   rdna-shader-instruction-set-architecture.pdf, page 155.
//   rdna2-shader-instruction-set-architecture.pdf, page 153.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 291.
//   rdna4-instruction-set-architecture.pdf, page 332.
def V_FREXP_EXP_I32_F32_e64_sem : InstSISemantic<V_FREXP_EXP_I32_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_frexp_exp,
        [(GetNamedOperand $src0, si_llvm_float_ty)]))
  ];
}
def V_FREXP_EXP_I32_F64_e64_sem : InstSISemantic<V_FREXP_EXP_I32_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_frexp_exp,
        [(GetNamedOperand $src0, si_llvm_double_ty)]))
  ];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 210.
//   vega-shader-instruction-set-architecture.pdf, page 143.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 144.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 138.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 203.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 216.
//   rdna-shader-instruction-set-architecture.pdf, page 155.
//   rdna2-shader-instruction-set-architecture.pdf, page 153.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 291.
//   rdna4-instruction-set-architecture.pdf, page 332.
def V_FREXP_EXP_I16_F16_e64_sem : InstSISemantic<V_FREXP_EXP_I16_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_frexp_exp,
        [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Find First Bit e64
//------------------------------------------------------------------------------

// D.u = position of first 1 bit in S0.u from MSB. Returns 0xFFFFFFFF if S0 == 0.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 206.
//   vega-shader-instruction-set-architecture.pdf, page 141.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 142.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 137.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 136.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 200.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 213.
//   rdna-shader-instruction-set-architecture.pdf, page 153.
//   rdna2-shader-instruction-set-architecture.pdf, page 151.
def V_FFBH_U32_e64_sem : InstSISemantic<V_FFBH_U32_e64> {
  let Semantic = [
    (DefVal $clz, (LLVMBinaryIntrinsic int_ctlz, (GetNamedOperand $src0),
                             (ConstantFalse si_llvm_i1_ty))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetNamedOperand $src0),
                                   (ConstantInt si_llvm_i32_ty, 0)),
                  (ConstantSignedInt si_llvm_i32_ty, -1),
                  (GetVal $clz)))
  ];
}

// D.u = position of first 1 bit in S0.u from LSB. Returns 0xFFFFFFFF if S0 == 0.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 206.
//   vega-shader-instruction-set-architecture.pdf, page 141.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 142.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 137.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 136.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 200.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 213.
//   rdna-shader-instruction-set-architecture.pdf, page 154.
//   rdna2-shader-instruction-set-architecture.pdf, page 151.
def V_FFBL_B32_e64_sem : InstSISemantic<V_FFBL_B32_e64> {
  let Semantic = [
    (DefVal $ctz, (LLVMBinaryIntrinsic int_cttz, (GetNamedOperand $src0),
                             (ConstantFalse si_llvm_i1_ty))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetNamedOperand $src0),
                                   (ConstantInt si_llvm_i32_ty, 0)),
                  (ConstantSignedInt si_llvm_i32_ty, -1),
                  (GetVal $ctz)))
  ];
}

// D.u = position of first bit different from sign bit. Returns 0xFFFFFFFF if S0 == 0 or -1.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 205.
//   vega-shader-instruction-set-architecture.pdf, page 142.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 143.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 137.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 137.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 201.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 214.
//   rdna-shader-instruction-set-architecture.pdf, page 154.
//   rdna2-shader-instruction-set-architecture.pdf, page 152.
def V_FFBH_I32_e64_sem : InstSISemantic<V_FFBH_I32_e64> {
  let Semantic = [
    (DefVal $xored, (LLVMXor (GetNamedOperand $src0),
                              (LLVMAShr (GetNamedOperand $src0),
                                        (ConstantInt si_llvm_i32_ty, 31)))),
    (DefVal $clz, (LLVMBinaryIntrinsic int_ctlz, (GetVal $xored), (ConstantFalse si_llvm_i1_ty))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpEQ, (GetVal $xored),
                                   (ConstantInt si_llvm_i32_ty, 0)),
                  (ConstantSignedInt si_llvm_i32_ty, -1),
                  (GetVal $clz)))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Bitrev e64
//------------------------------------------------------------------------------

// D.u = bitreverse(S0.u).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 189.
//   vega-shader-instruction-set-architecture.pdf, page 140.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 141.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 136.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 199.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 212.
//   rdna-shader-instruction-set-architecture.pdf, page 153.
//   rdna2-shader-instruction-set-architecture.pdf, page 151.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 288.
//   rdna4-instruction-set-architecture.pdf, page 329.
def V_BFREV_B32_e64_sem : InstSISemantic<V_BFREV_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMUnaryIntrinsic int_bitreverse, (GetNamedOperand $src0)))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Bit Operations e64
//------------------------------------------------------------------------------

// D.u = ~S0.u (32-bit bitwise NOT).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 217.
//   vega-shader-instruction-set-architecture.pdf, page 140.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 141.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 136.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 136.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 199.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 212.
//   rdna-shader-instruction-set-architecture.pdf, page 153.
//   rdna2-shader-instruction-set-architecture.pdf, page 151.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 287.
//   rdna4-instruction-set-architecture.pdf, page 329.
def V_NOT_B32_e64_sem : InstSISemantic<V_NOT_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      LLVMNot<(GetNamedOperand $src0), si_llvm_i32_ty>.ret)
  ];
}

// D.u16 = ~S0.u16 (16-bit bitwise NOT).
def V_NOT_B16_e64_sem : InstSISemantic<V_NOT_B16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMXor (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                 (LLVMSExt (ConstantTrue si_llvm_i1_ty), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// D.i = count leading sign bits (same as sign bit) in S0.i.
// D.u = count leading zeros in S0.u.
//------------------------------------------------------------------------------
// VOP1 Pipeline/Misc e64
//------------------------------------------------------------------------------

// V_NOP: No operation.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 216.
//   vega-shader-instruction-set-architecture.pdf, page 134.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 131.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 186.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 199.
//   rdna-shader-instruction-set-architecture.pdf, page 147.
//   rdna2-shader-instruction-set-architecture.pdf, page 145.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 274.
//   rdna4-instruction-set-architecture.pdf, page 315.
def V_NOP_e64_sem : InstSISemantic<V_NOP_e64> {
  let Semantic = [];
}

// Clear float exception state.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 191.
//   vega-shader-instruction-set-architecture.pdf, page 143.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 144.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 138.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 138.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 203.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 216.
//   rdna-shader-instruction-set-architecture.pdf, page 155.
//   rdna2-shader-instruction-set-architecture.pdf, page 153.
def V_CLREXCP_e64_sem : InstSISemantic<V_CLREXCP_e64> {
  let Semantic = [];
}

// Flush VALU pipeline.
def V_PIPEFLUSH_e64_sem : InstSISemantic<V_PIPEFLUSH_e64> {
  let Semantic = [];
}

// Screen partition (GFX9 only).
def V_SCREEN_PARTITION_4SE_B32_e64_sem : InstSISemantic<V_SCREEN_PARTITION_4SE_B32_e64> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// VOP1 ACCVGPR e64
//------------------------------------------------------------------------------

// D.u = ACCVGPR[S0]. Read from accumulator VGPR.
// Sources:
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 211.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 224.
def V_ACCVGPR_READ_B32_e64_sem : InstSISemantic<V_ACCVGPR_READ_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
}

// ACCVGPR[D] = S0.u. Write to accumulator VGPR.
def V_ACCVGPR_WRITE_B32_e64_sem : InstSISemantic<V_ACCVGPR_WRITE_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
}

//==============================================================================
// VOP3-Only Instructions (no e32 equivalent)
//==============================================================================

//------------------------------------------------------------------------------
// VOP3 F64 Arithmetic
//------------------------------------------------------------------------------

// D.d = S0.d + S1.d.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 230.
//   vega-shader-instruction-set-architecture.pdf, page 178.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 180.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 171.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 169.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 359.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 405.
//   rdna-shader-instruction-set-architecture.pdf, page 184.
//   rdna2-shader-instruction-set-architecture.pdf, page 181.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 446.
//   rdna4-instruction-set-architecture.pdf, page 299.
def V_ADD_F64_e64_sem : InstSISemantic<V_ADD_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $src1), si_llvm_double_ty)),
        si_llvm_i64_ty))
  ];
}

// D.d = S0.d * S1.d.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 256.
//   vega-shader-instruction-set-architecture.pdf, page 178.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 180.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 171.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 170.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 347.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 373.
//   rdna-shader-instruction-set-architecture.pdf, page 184.
//   rdna2-shader-instruction-set-architecture.pdf, page 181.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 446.
//   rdna4-instruction-set-architecture.pdf, page 300.
def V_MUL_F64_e64_sem : InstSISemantic<V_MUL_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $src1), si_llvm_double_ty)),
        si_llvm_i64_ty))
  ];
}

// D.d = max(S0.d, S1.d) (IEEE maxNum).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 250.
//   vega-shader-instruction-set-architecture.pdf, page 179.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 181.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 172.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 170.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 360.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 406.
//   rdna-shader-instruction-set-architecture.pdf, page 185.
//   rdna2-shader-instruction-set-architecture.pdf, page 182.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 447.
def V_MAX_F64_e64_sem : InstSISemantic<V_MAX_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMaxNum (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_double_ty)),
        si_llvm_i64_ty))
  ];
}

// D.d = min(S0.d, S1.d) (IEEE minNum).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 317.
//   vega-shader-instruction-set-architecture.pdf, page 179.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 181.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 172.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 170.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 360.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 406.
//   rdna-shader-instruction-set-architecture.pdf, page 185.
//   rdna2-shader-instruction-set-architecture.pdf, page 182.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 447.
def V_MIN_F64_e64_sem : InstSISemantic<V_MIN_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMinNum (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_double_ty)),
        si_llvm_i64_ty))
  ];
}

// D.d = max(S0.d, S1.d) (IEEE maxNum, GFX12+ renamed).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 303.
def V_MAX_NUM_F64_e64_sem : InstSISemantic<V_MAX_NUM_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMaxNum (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_double_ty)),
        si_llvm_i64_ty))
  ];
}

// D.d = S0.d * 2^S1.i (ldexp, f64).
def V_LDEXP_F64_e64_sem : InstSISemantic<V_LDEXP_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMLdexp (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                   (GetNamedOperand $src1)),
        si_llvm_i64_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP3 64-bit Shift Operations
//------------------------------------------------------------------------------

// D.i64 = S0.i64 >> S1.u[5:0] (arithmetic shift right).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 232.
//   vega-shader-instruction-set-architecture.pdf, page 180.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 173.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 171.
def V_ASHR_I64_e64_sem : InstSISemantic<V_ASHR_I64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAShr (GetNamedOperand $src0),
                (LLVMZExt (LLVMAnd (GetNamedOperand $src1),
                                   (ConstantInt si_llvm_i32_ty, 63)),
                          si_llvm_i64_ty)))
  ];
}

// D.i64 = S1.i64 >> S0.u[5:0] (arithmetic shift right, reversed operands).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 232.
//   vega-shader-instruction-set-architecture.pdf, page 180.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 173.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 171.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 365.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 411.
//   rdna-shader-instruction-set-architecture.pdf, page 190.
//   rdna2-shader-instruction-set-architecture.pdf, page 186.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 451.
//   rdna4-instruction-set-architecture.pdf, page 522.
def V_ASHRREV_I64_e64_sem : InstSISemantic<V_ASHRREV_I64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAShr (GetNamedOperand $src1),
                (LLVMZExt (LLVMAnd (GetNamedOperand $src0),
                                   (ConstantInt si_llvm_i32_ty, 63)),
                          si_llvm_i64_ty)))
  ];
}

// D.u64 = S0.u64 << S1.u[5:0].
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 246.
//   vega-shader-instruction-set-architecture.pdf, page 180.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 173.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 171.
def V_LSHL_B64_e64_sem : InstSISemantic<V_LSHL_B64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMShl (GetNamedOperand $src0),
               (LLVMZExt (LLVMAnd (GetNamedOperand $src1),
                                   (ConstantInt si_llvm_i32_ty, 63)),
                          si_llvm_i64_ty)))
  ];
}

// D.u64 = S1.u64 << S0.u[5:0] (reversed operands).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 246.
//   vega-shader-instruction-set-architecture.pdf, page 180.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 173.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 171.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 364.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 410.
//   rdna-shader-instruction-set-architecture.pdf, page 190.
//   rdna2-shader-instruction-set-architecture.pdf, page 186.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 450.
//   rdna4-instruction-set-architecture.pdf, page 307.
def V_LSHLREV_B64_e64_sem : InstSISemantic<V_LSHLREV_B64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMShl (GetNamedOperand $src1),
               (LLVMZExt (LLVMAnd (GetNamedOperand $src0),
                                   (ConstantInt si_llvm_i32_ty, 63)),
                          si_llvm_i64_ty)))
  ];
}

// D.u64 = S0.u64 >> S1.u[5:0] (logical shift right).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 318.
//   vega-shader-instruction-set-architecture.pdf, page 180.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 173.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 171.
def V_LSHR_B64_e64_sem : InstSISemantic<V_LSHR_B64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMLShr (GetNamedOperand $src0),
                (LLVMZExt (LLVMAnd (GetNamedOperand $src1),
                                   (ConstantInt si_llvm_i32_ty, 63)),
                          si_llvm_i64_ty)))
  ];
}

// D.u64 = S1.u64 >> S0.u[5:0] (logical shift right, reversed operands).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 318.
//   vega-shader-instruction-set-architecture.pdf, page 180.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 173.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 171.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 365.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 410.
//   rdna-shader-instruction-set-architecture.pdf, page 190.
//   rdna2-shader-instruction-set-architecture.pdf, page 186.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 451.
//   rdna4-instruction-set-architecture.pdf, page 521.
def V_LSHRREV_B64_e64_sem : InstSISemantic<V_LSHRREV_B64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMLShr (GetNamedOperand $src1),
                (LLVMZExt (LLVMAnd (GetNamedOperand $src0),
                                   (ConstantInt si_llvm_i32_ty, 63)),
                          si_llvm_i64_ty)))
  ];
}

//------------------------------------------------------------------------------
// VOP3 32-bit Full Multiply
//------------------------------------------------------------------------------

// D.u = S0.u * S1.u (low 32 bits of 32x32 multiply).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 257.
//   vega-shader-instruction-set-architecture.pdf, page 179.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 181.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 172.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 170.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 361.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 407.
//   rdna-shader-instruction-set-architecture.pdf, page 185.
//   rdna2-shader-instruction-set-architecture.pdf, page 182.
//   rdna4-instruction-set-architecture.pdf, page 519.
def V_MUL_LO_U32_e64_sem : InstSISemantic<V_MUL_LO_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMMul (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.i = S0.i * S1.i (low 32 bits, same as unsigned for low bits).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 257.
//   vega-shader-instruction-set-architecture.pdf, page 179.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 181.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 172.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 170.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 361.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 407.
//   rdna-shader-instruction-set-architecture.pdf, page 185.
//   rdna2-shader-instruction-set-architecture.pdf, page 182.
//   rdna4-instruction-set-architecture.pdf, page 519.
def V_MUL_LO_I32_e64_sem : InstSISemantic<V_MUL_LO_I32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMMul (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.u = (S0.u * S1.u) >> 32 (high 32 bits of unsigned 32x32 multiply).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 256.
//   vega-shader-instruction-set-architecture.pdf, page 179.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 181.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 172.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 170.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 362.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 408.
//   rdna-shader-instruction-set-architecture.pdf, page 185.
//   rdna2-shader-instruction-set-architecture.pdf, page 182.
//   rdna4-instruction-set-architecture.pdf, page 520.
def V_MUL_HI_U32_e64_sem : InstSISemantic<V_MUL_HI_U32_e64> {
  let Semantic = [
    (DefVal $a, (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty)),
    (DefVal $b, (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty)),
    (SetNamedOperand $vdst,
      (LLVMTrunc (LLVMLShr (LLVMMul (GetVal $a), (GetVal $b)),
                            (ConstantInt si_llvm_i64_ty, 32)),
                 si_llvm_i32_ty))
  ];
}

// D.i = (S0.i * S1.i) >> 32 (high 32 bits of signed 32x32 multiply).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 255.
//   vega-shader-instruction-set-architecture.pdf, page 179.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 181.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 172.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 170.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 361.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 407.
//   rdna-shader-instruction-set-architecture.pdf, page 185.
//   rdna2-shader-instruction-set-architecture.pdf, page 182.
//   rdna4-instruction-set-architecture.pdf, page 519.
def V_MUL_HI_I32_e64_sem : InstSISemantic<V_MUL_HI_I32_e64> {
  let Semantic = [
    (DefVal $a, (LLVMSExt (GetNamedOperand $src0), si_llvm_i64_ty)),
    (DefVal $b, (LLVMSExt (GetNamedOperand $src1), si_llvm_i64_ty)),
    (SetNamedOperand $vdst,
      (LLVMTrunc (LLVMAShr (LLVMMul (GetVal $a), (GetVal $b)),
                            (ConstantInt si_llvm_i64_ty, 32)),
                 si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP3 MAD/FMA 3-operand (F32)
//------------------------------------------------------------------------------

// D.f = S0.f * S1.f + S2.f (non-fused MAD).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 247.
//   vega-shader-instruction-set-architecture.pdf, page 164.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 166.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 158.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 158.
//   rdna-shader-instruction-set-architecture.pdf, page 178.
def V_MAD_F32_e64_sem : InstSISemantic<V_MAD_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd
          (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = S0.f * S1.f + S2.f (non-fused MAD, DX9 legacy).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 248.
def V_MAD_LEGACY_F32_e64_sem : InstSISemantic<V_MAD_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd
          (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f16 = S0.f16 * S1.f16 + S2.f16 (non-fused MAD, F16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 171.
//   vega-shader-instruction-set-architecture.pdf, page 173.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 175.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 166.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 165.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 350.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 377.
def V_MAD_F16_e64_sem : InstSISemantic<V_MAD_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFAdd
            (LLVMFMul
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f = FMA(S0.f, S1.f, S2.f) (fused multiply-add).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 243.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 338.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 365.
//   rdna-shader-instruction-set-architecture.pdf, page 180.
//   rdna2-shader-instruction-set-architecture.pdf, page 177.
//   rdna4-instruction-set-architecture.pdf, page 479.
def V_FMA_F32_e64_sem : InstSISemantic<V_FMA_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.d = FMA(S0.d, S1.d, S2.d) (fused multiply-add, F64).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 314.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 159.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 339.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 365.
//   rdna-shader-instruction-set-architecture.pdf, page 181.
//   rdna2-shader-instruction-set-architecture.pdf, page 177.
//   rdna4-instruction-set-architecture.pdf, page 479.
def V_FMA_F64_e64_sem : InstSISemantic<V_FMA_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
                 (LLVMBitCast (GetNamedOperand $src1), si_llvm_double_ty),
                 (LLVMBitCast (GetNamedOperand $src2), si_llvm_double_ty)),
        si_llvm_i64_ty))
  ];
}

// D.f16 = FMA(S0.f16, S1.f16, S2.f16) (fused multiply-add, F16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 243.
//   vega-shader-instruction-set-architecture.pdf, page 176.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 178.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 170.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 169.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 358.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 385.
//   rdna-shader-instruction-set-architecture.pdf, page 193.
//   rdna2-shader-instruction-set-architecture.pdf, page 189.
//   rdna4-instruction-set-architecture.pdf, page 492.
def V_FMA_F16_e64_sem : InstSISemantic<V_FMA_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFMA
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f = FMA(S0.f, S1.f, S2.f) (fused multiply-add, legacy).
// Sources:
//   rdna2-shader-instruction-set-architecture.pdf, page 175.
def V_FMA_LEGACY_F32_e64_sem : InstSISemantic<V_FMA_LEGACY_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP3 MAD Integer 24-bit
//------------------------------------------------------------------------------

// D.i = S0.i[23:0] * S1.i[23:0] + S2.i (signed 24-bit MAD).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 248.
//   vega-shader-instruction-set-architecture.pdf, page 164.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 166.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 158.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 158.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 362.
//   rdna-shader-instruction-set-architecture.pdf, page 178.
//   rdna2-shader-instruction-set-architecture.pdf, page 175.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 414.
//   rdna4-instruction-set-architecture.pdf, page 475.
def V_MAD_I32_I24_e64_sem : InstSISemantic<V_MAD_I32_I24_e64> {
  let Semantic = [
    (DefVal $a, (LLVMAShr (LLVMShl (GetNamedOperand $src0),
                                    (ConstantInt si_llvm_i32_ty, 8)),
                           (ConstantInt si_llvm_i32_ty, 8))),
    (DefVal $b, (LLVMAShr (LLVMShl (GetNamedOperand $src1),
                                    (ConstantInt si_llvm_i32_ty, 8)),
                           (ConstantInt si_llvm_i32_ty, 8))),
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (GetVal $a), (GetVal $b)),
               (GetNamedOperand $src2)))
  ];
}

// D.u = S0.u[23:0] * S1.u[23:0] + S2.u (unsigned 24-bit MAD).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 249.
//   vega-shader-instruction-set-architecture.pdf, page 165.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 166.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 158.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 158.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 336.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 362.
//   rdna-shader-instruction-set-architecture.pdf, page 178.
//   rdna2-shader-instruction-set-architecture.pdf, page 175.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 415.
//   rdna4-instruction-set-architecture.pdf, page 476.
def V_MAD_U32_U24_e64_sem : InstSISemantic<V_MAD_U32_U24_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMAnd (GetNamedOperand $src0),
                                   (ConstantInt si_llvm_i32_ty, 16777215)),
                         (LLVMAnd (GetNamedOperand $src1),
                                   (ConstantInt si_llvm_i32_ty, 16777215))),
               (GetNamedOperand $src2)))
  ];
}

// D.i64 = S0.i32 * S1.i32 + S2.i64 (signed 32-to-64 MAD).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 248.
//   vega-shader-instruction-set-architecture.pdf, page 164.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 166.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 158.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 158.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 349.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 376.
//   rdna-shader-instruction-set-architecture.pdf, page 190.
//   rdna2-shader-instruction-set-architecture.pdf, page 186.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 439.
def V_MAD_I64_I32_e64_sem : InstSISemantic<V_MAD_I64_I32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMSExt (GetNamedOperand $src0), si_llvm_i64_ty),
                         (LLVMSExt (GetNamedOperand $src1), si_llvm_i64_ty)),
               (GetNamedOperand $src2)))
  ];
}

// D.u64 = S0.u32 * S1.u32 + S2.u64 (unsigned 32-to-64 MAD).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 249.
//   vega-shader-instruction-set-architecture.pdf, page 164.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 166.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 158.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 158.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 349.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 376.
//   rdna-shader-instruction-set-architecture.pdf, page 190.
//   rdna2-shader-instruction-set-architecture.pdf, page 186.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 439.
def V_MAD_U64_U32_e64_sem : InstSISemantic<V_MAD_U64_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                         (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty)),
               (GetNamedOperand $src2)))
  ];
}

//------------------------------------------------------------------------------
// VOP3 MAD I16/U16
//------------------------------------------------------------------------------

// D.f16 = S0.f16 * S1.f16 + S2.f16 (16-bit MAD, same as MAD_F16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 247.
//   vega-shader-instruction-set-architecture.pdf, page 173.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 175.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 167.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 166.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 351.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 378.
//   rdna-shader-instruction-set-architecture.pdf, page 195.
//   rdna2-shader-instruction-set-architecture.pdf, page 191.
//   rdna4-instruction-set-architecture.pdf, page 493.
def V_MAD_I16_e64_sem : InstSISemantic<V_MAD_I16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAdd (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
                 (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// Sources:
//   gcn3-instruction-set-architecture.pdf, page 249.
//   vega-shader-instruction-set-architecture.pdf, page 173.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 175.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 167.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 166.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 350.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 377.
//   rdna-shader-instruction-set-architecture.pdf, page 192.
//   rdna2-shader-instruction-set-architecture.pdf, page 188.
//   rdna4-instruction-set-architecture.pdf, page 490.
def V_MAD_U16_e64_sem : InstSISemantic<V_MAD_U16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAdd (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
                 (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// D.i = S0.i16 * S1.i16 + S2.i (sext 16 to 32, mul, add).
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 167.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 353.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 380.
//   rdna-shader-instruction-set-architecture.pdf, page 198.
//   rdna2-shader-instruction-set-architecture.pdf, page 193.
//   rdna4-instruction-set-architecture.pdf, page 495.
def V_MAD_I32_I16_e64_sem : InstSISemantic<V_MAD_I32_I16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMSExt (LLVMTrunc (GetNamedOperand $src0),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty),
                         (LLVMSExt (LLVMTrunc (GetNamedOperand $src1),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty)),
               (GetNamedOperand $src2)))
  ];
}

// D.u = S0.u16 * S1.u16 + S2.u (zext 16 to 32, mul, add).
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 167.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 353.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 380.
//   rdna-shader-instruction-set-architecture.pdf, page 198.
//   rdna2-shader-instruction-set-architecture.pdf, page 193.
//   rdna4-instruction-set-architecture.pdf, page 495.
def V_MAD_U32_U16_e64_sem : InstSISemantic<V_MAD_U32_U16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMZExt (LLVMTrunc (GetNamedOperand $src0),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty),
                         (LLVMZExt (LLVMTrunc (GetNamedOperand $src1),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty)),
               (GetNamedOperand $src2)))
  ];
}

//------------------------------------------------------------------------------
// VOP3 Division Support (complex — empty semantics)
//------------------------------------------------------------------------------

// Sources:
//   gcn3-instruction-set-architecture.pdf, page 315.
//   vega-shader-instruction-set-architecture.pdf, page 168.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 170.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 162.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 161.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 346.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 372.
//   rdna4-instruction-set-architecture.pdf, page 488.
def V_DIV_FIXUP_F32_e64_sem : InstSISemantic<V_DIV_FIXUP_F32_e64> {
  let Semantic = [];
}
def V_DIV_FIXUP_F64_e64_sem : InstSISemantic<V_DIV_FIXUP_F64_e64> {
  let Semantic = [];
}
def V_DIV_FIXUP_F16_e64_sem : InstSISemantic<V_DIV_FIXUP_F16_e64> {
  let Semantic = [];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 242.
//   vega-shader-instruction-set-architecture.pdf, page 172.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 174.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 165.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 164.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 346.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 373.
//   rdna-shader-instruction-set-architecture.pdf, page 187.
//   rdna2-shader-instruction-set-architecture.pdf, page 183.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 424.
//   rdna4-instruction-set-architecture.pdf, page 487.
def V_DIV_FMAS_F32_e64_sem : InstSISemantic<V_DIV_FMAS_F32_e64> {
  let Semantic = [];
}
def V_DIV_FMAS_F64_e64_sem : InstSISemantic<V_DIV_FMAS_F64_e64> {
  let Semantic = [];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 263.
//   vega-shader-instruction-set-architecture.pdf, page 164.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 166.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 158.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 158.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 346.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 372.
//   rdna2-shader-instruction-set-architecture.pdf, page 183.
//   rdna4-instruction-set-architecture.pdf, page 488.
def V_DIV_SCALE_F32_e64_sem : InstSISemantic<V_DIV_SCALE_F32_e64> {
  let Semantic = [];
}
def V_DIV_SCALE_F64_e64_sem : InstSISemantic<V_DIV_SCALE_F64_e64> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// VOP3 Cube Map (complex — empty semantics)
//------------------------------------------------------------------------------

// Sources:
//   gcn3-instruction-set-architecture.pdf, page 235.
//   vega-shader-instruction-set-architecture.pdf, page 165.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 166.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 159.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 158.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 336.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 363.
//   rdna-shader-instruction-set-architecture.pdf, page 179.
//   rdna2-shader-instruction-set-architecture.pdf, page 176.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 415.
//   rdna4-instruction-set-architecture.pdf, page 476.
def V_CUBEID_F32_e64_sem : InstSISemantic<V_CUBEID_F32_e64> {
  let Semantic = [];
}
def V_CUBESC_F32_e64_sem : InstSISemantic<V_CUBESC_F32_e64> {
  let Semantic = [];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 236.
//   vega-shader-instruction-set-architecture.pdf, page 165.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 167.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 159.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 159.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 337.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 364.
//   rdna-shader-instruction-set-architecture.pdf, page 180.
//   rdna2-shader-instruction-set-architecture.pdf, page 177.
//   rdna4-instruction-set-architecture.pdf, page 477.
def V_CUBETC_F32_e64_sem : InstSISemantic<V_CUBETC_F32_e64> {
  let Semantic = [];
}
def V_CUBEMA_F32_e64_sem : InstSISemantic<V_CUBEMA_F32_e64> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// VOP3 3-operand Integer
//------------------------------------------------------------------------------

// D.u = S0.u + S1.u + S2.u.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 176.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 178.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 168.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 356.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 383.
//   rdna-shader-instruction-set-architecture.pdf, page 198.
//   rdna2-shader-instruction-set-architecture.pdf, page 193.
//   rdna4-instruction-set-architecture.pdf, page 494.
def V_ADD3_U32_e64_sem : InstSISemantic<V_ADD3_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMAdd (GetNamedOperand $src0), (GetNamedOperand $src1)),
               (GetNamedOperand $src2)))
  ];
}

// D.u = S0.u | S1.u | S2.u.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 176.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 178.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 168.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 357.
//   rdna-shader-instruction-set-architecture.pdf, page 198.
//   rdna2-shader-instruction-set-architecture.pdf, page 193.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 431.
//   rdna4-instruction-set-architecture.pdf, page 495.
def V_OR3_B32_e64_sem : InstSISemantic<V_OR3_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMOr (LLVMOr (GetNamedOperand $src0), (GetNamedOperand $src1)),
              (GetNamedOperand $src2)))
  ];
}

// D.u = S0.u ^ S1.u ^ S2.u.
// Sources:
//   rdna-shader-instruction-set-architecture.pdf, page 190.
//   rdna2-shader-instruction-set-architecture.pdf, page 186.
def V_XOR3_B32_e64_sem : InstSISemantic<V_XOR3_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMXor (LLVMXor (GetNamedOperand $src0), (GetNamedOperand $src1)),
               (GetNamedOperand $src2)))
  ];
}

// D.u = (S0.u & S1.u) | S2.u.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 176.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 178.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 168.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 356.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 383.
//   rdna-shader-instruction-set-architecture.pdf, page 198.
//   rdna2-shader-instruction-set-architecture.pdf, page 193.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 431.
//   rdna4-instruction-set-architecture.pdf, page 495.
def V_AND_OR_B32_e64_sem : InstSISemantic<V_AND_OR_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMOr (LLVMAnd (GetNamedOperand $src0), (GetNamedOperand $src1)),
              (GetNamedOperand $src2)))
  ];
}

// D.u = (S0.u << S1.u) | S2.u.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 176.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 178.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 168.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 356.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 383.
//   rdna-shader-instruction-set-architecture.pdf, page 198.
//   rdna2-shader-instruction-set-architecture.pdf, page 193.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 431.
//   rdna4-instruction-set-architecture.pdf, page 494.
def V_LSHL_OR_B32_e64_sem : InstSISemantic<V_LSHL_OR_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMOr (LLVMShl (GetNamedOperand $src0),
                        (LLVMAnd (GetNamedOperand $src1),
                                  (ConstantInt si_llvm_i32_ty, 31))),
              (GetNamedOperand $src2)))
  ];
}

// D.u = (S0.u + S1.u) << S2.u.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 176.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 178.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 168.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 356.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 383.
//   rdna-shader-instruction-set-architecture.pdf, page 193.
//   rdna2-shader-instruction-set-architecture.pdf, page 189.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 427.
//   rdna4-instruction-set-architecture.pdf, page 491.
def V_ADD_LSHL_U32_e64_sem : InstSISemantic<V_ADD_LSHL_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMShl (LLVMAdd (GetNamedOperand $src0), (GetNamedOperand $src1)),
               (LLVMAnd (GetNamedOperand $src2),
                         (ConstantInt si_llvm_i32_ty, 31))))
  ];
}

// D.u = (S0.u << S1.u) + S2.u.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 168.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 356.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 382.
//   rdna-shader-instruction-set-architecture.pdf, page 193.
//   rdna2-shader-instruction-set-architecture.pdf, page 189.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 427.
//   rdna4-instruction-set-architecture.pdf, page 491.
def V_LSHL_ADD_U32_e64_sem : InstSISemantic<V_LSHL_ADD_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMShl (GetNamedOperand $src0),
                         (LLVMAnd (GetNamedOperand $src1),
                                   (ConstantInt si_llvm_i32_ty, 31))),
               (GetNamedOperand $src2)))
  ];
}

// D.u64 = (S0.u64 << S1.u) + S2.u64.
// Sources:
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 359.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 386.
def V_LSHL_ADD_U64_e64_sem : InstSISemantic<V_LSHL_ADD_U64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMShl (GetNamedOperand $src0),
                         (LLVMZExt (LLVMAnd (GetNamedOperand $src1),
                                            (ConstantInt si_llvm_i32_ty, 63)),
                                   si_llvm_i64_ty)),
               (GetNamedOperand $src2)))
  ];
}

// D.u = (S0.u ^ S1.u) + S2.u.
def V_XAD_U32_e64_sem : InstSISemantic<V_XAD_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMXor (GetNamedOperand $src0), (GetNamedOperand $src1)),
               (GetNamedOperand $src2)))
  ];
}

//------------------------------------------------------------------------------
// VOP3 MinMax / MaxMin F32
//------------------------------------------------------------------------------

// D.f = min(max(S0.f, S1.f), S2.f).
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 434.
def V_MAXMIN_F32_e64_sem : InstSISemantic<V_MAXMIN_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMinNum
          (LLVMMaxNum (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                      (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = max(min(S0.f, S1.f), S2.f).
def V_MINMAX_F32_e64_sem : InstSISemantic<V_MINMAX_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMaxNum
          (LLVMMinNum (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                      (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f16 = min(max(S0.f16, S1.f16), S2.f16).
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 435.
def V_MAXMIN_F16_e64_sem : InstSISemantic<V_MAXMIN_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMinNum
            (LLVMMaxNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f16 = max(min(S0.f16, S1.f16), S2.f16).
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 435.
def V_MINMAX_F16_e64_sem : InstSISemantic<V_MINMAX_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMaxNum
            (LLVMMinNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP3 MinMax / MaxMin Integer
//------------------------------------------------------------------------------

// D.i = min(max(S0.i, S1.i), S2.i) (signed).
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 436.
//   rdna4-instruction-set-architecture.pdf, page 498.
def V_MAXMIN_I32_e64_sem : InstSISemantic<V_MAXMIN_I32_e64> {
  let Semantic = [
    (DefVal $mx, (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetNamedOperand $src0),
                                              (GetNamedOperand $src1)),
                              (GetNamedOperand $src0),
                              (GetNamedOperand $src1))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $mx), (GetNamedOperand $src2)),
                  (GetVal $mx), (GetNamedOperand $src2)))
  ];
}

// D.u = min(max(S0.u, S1.u), S2.u) (unsigned).
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 435.
//   rdna4-instruction-set-architecture.pdf, page 498.
def V_MAXMIN_U32_e64_sem : InstSISemantic<V_MAXMIN_U32_e64> {
  let Semantic = [
    (DefVal $mx, (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetNamedOperand $src0),
                                               (GetNamedOperand $src1)),
                              (GetNamedOperand $src0),
                              (GetNamedOperand $src1))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $mx), (GetNamedOperand $src2)),
                  (GetVal $mx), (GetNamedOperand $src2)))
  ];
}

// D.i = max(min(S0.i, S1.i), S2.i) (signed).
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 436.
//   rdna4-instruction-set-architecture.pdf, page 499.
def V_MINMAX_I32_e64_sem : InstSISemantic<V_MINMAX_I32_e64> {
  let Semantic = [
    (DefVal $mn, (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetNamedOperand $src0),
                                              (GetNamedOperand $src1)),
                              (GetNamedOperand $src0),
                              (GetNamedOperand $src1))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $mn), (GetNamedOperand $src2)),
                  (GetVal $mn), (GetNamedOperand $src2)))
  ];
}

// D.u = max(min(S0.u, S1.u), S2.u) (unsigned).
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 436.
//   rdna4-instruction-set-architecture.pdf, page 498.
def V_MINMAX_U32_e64_sem : InstSISemantic<V_MINMAX_U32_e64> {
  let Semantic = [
    (DefVal $mn, (LLVMSelect (LLVMCmp LLVMICmpULT, (GetNamedOperand $src0),
                                               (GetNamedOperand $src1)),
                              (GetNamedOperand $src0),
                              (GetNamedOperand $src1))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $mn), (GetNamedOperand $src2)),
                  (GetVal $mn), (GetNamedOperand $src2)))
  ];
}

//------------------------------------------------------------------------------
// VOP3 Min3/Max3/Med3 F32
//------------------------------------------------------------------------------

// D.f = max(max(S0.f, S1.f), S2.f).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 250.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   rdna-shader-instruction-set-architecture.pdf, page 181.
//   rdna2-shader-instruction-set-architecture.pdf, page 178.
def V_MAX3_F32_e64_sem : InstSISemantic<V_MAX3_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMaxNum
          (LLVMMaxNum (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                      (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = min(min(S0.f, S1.f), S2.f).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 254.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   rdna-shader-instruction-set-architecture.pdf, page 181.
//   rdna2-shader-instruction-set-architecture.pdf, page 178.
def V_MIN3_F32_e64_sem : InstSISemantic<V_MIN3_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMinNum
          (LLVMMinNum (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                      (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src2), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}

// D.f = median(S0.f, S1.f, S2.f) — complex, empty semantic.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 252.
//   vega-shader-instruction-set-architecture.pdf, page 167.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   rdna-shader-instruction-set-architecture.pdf, page 182.
//   rdna2-shader-instruction-set-architecture.pdf, page 178.
def V_MED3_F32_e64_sem : InstSISemantic<V_MED3_F32_e64> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// VOP3 Min3/Max3/Med3 I32
//------------------------------------------------------------------------------

// D.i = max(max(S0.i, S1.i), S2.i) (signed).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 251.
//   vega-shader-instruction-set-architecture.pdf, page 167.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   rdna-shader-instruction-set-architecture.pdf, page 181.
//   rdna2-shader-instruction-set-architecture.pdf, page 178.
def V_MAX3_I32_e64_sem : InstSISemantic<V_MAX3_I32_e64> {
  let Semantic = [
    (DefVal $mx, (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetNamedOperand $src0),
                                              (GetNamedOperand $src1)),
                              (GetNamedOperand $src0),
                              (GetNamedOperand $src1))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $mx), (GetNamedOperand $src2)),
                  (GetVal $mx), (GetNamedOperand $src2)))
  ];
}

// D.i = min(min(S0.i, S1.i), S2.i) (signed).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 254.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   rdna-shader-instruction-set-architecture.pdf, page 181.
//   rdna2-shader-instruction-set-architecture.pdf, page 178.
def V_MIN3_I32_e64_sem : InstSISemantic<V_MIN3_I32_e64> {
  let Semantic = [
    (DefVal $mn, (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetNamedOperand $src0),
                                              (GetNamedOperand $src1)),
                              (GetNamedOperand $src0),
                              (GetNamedOperand $src1))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $mn), (GetNamedOperand $src2)),
                  (GetVal $mn), (GetNamedOperand $src2)))
  ];
}

// D.i = median(S0.i, S1.i, S2.i) — complex, empty semantic.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 252.
//   vega-shader-instruction-set-architecture.pdf, page 167.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 161.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   rdna-shader-instruction-set-architecture.pdf, page 182.
//   rdna2-shader-instruction-set-architecture.pdf, page 179.
def V_MED3_I32_e64_sem : InstSISemantic<V_MED3_I32_e64> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// VOP3 Min3/Max3/Med3 U32
//------------------------------------------------------------------------------

// D.u = max(max(S0.u, S1.u), S2.u) (unsigned).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 251.
//   vega-shader-instruction-set-architecture.pdf, page 167.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   rdna-shader-instruction-set-architecture.pdf, page 181.
//   rdna2-shader-instruction-set-architecture.pdf, page 178.
def V_MAX3_U32_e64_sem : InstSISemantic<V_MAX3_U32_e64> {
  let Semantic = [
    (DefVal $mx, (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetNamedOperand $src0),
                                               (GetNamedOperand $src1)),
                              (GetNamedOperand $src0),
                              (GetNamedOperand $src1))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $mx), (GetNamedOperand $src2)),
                  (GetVal $mx), (GetNamedOperand $src2)))
  ];
}

// D.u = min(min(S0.u, S1.u), S2.u) (unsigned).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 254.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   rdna-shader-instruction-set-architecture.pdf, page 181.
//   rdna2-shader-instruction-set-architecture.pdf, page 178.
def V_MIN3_U32_e64_sem : InstSISemantic<V_MIN3_U32_e64> {
  let Semantic = [
    (DefVal $mn, (LLVMSelect (LLVMCmp LLVMICmpULT, (GetNamedOperand $src0),
                                               (GetNamedOperand $src1)),
                              (GetNamedOperand $src0),
                              (GetNamedOperand $src1))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $mn), (GetNamedOperand $src2)),
                  (GetVal $mn), (GetNamedOperand $src2)))
  ];
}

// D.u = median(S0.u, S1.u, S2.u) — complex, empty semantic.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 253.
//   vega-shader-instruction-set-architecture.pdf, page 167.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 161.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   rdna-shader-instruction-set-architecture.pdf, page 182.
//   rdna2-shader-instruction-set-architecture.pdf, page 179.
def V_MED3_U32_e64_sem : InstSISemantic<V_MED3_U32_e64> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// VOP3 Min3/Max3/Med3 F16
//------------------------------------------------------------------------------

// D.f16 = max(max(S0.f16, S1.f16), S2.f16).
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 167.
//   rdna-shader-instruction-set-architecture.pdf, page 194.
//   rdna2-shader-instruction-set-architecture.pdf, page 189.
def V_MAX3_F16_e64_sem : InstSISemantic<V_MAX3_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMaxNum
            (LLVMMaxNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// D.f16 = min(min(S0.f16, S1.f16), S2.f16).
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 167.
//   rdna-shader-instruction-set-architecture.pdf, page 193.
//   rdna2-shader-instruction-set-architecture.pdf, page 189.
def V_MIN3_F16_e64_sem : InstSISemantic<V_MIN3_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMinNum
            (LLVMMinNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 168.
//   rdna-shader-instruction-set-architecture.pdf, page 194.
//   rdna2-shader-instruction-set-architecture.pdf, page 190.
def V_MED3_F16_e64_sem : InstSISemantic<V_MED3_F16_e64> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// VOP3 Min3/Max3/Med3 I16/U16
//------------------------------------------------------------------------------

// D.i16 = max(max(S0.i16, S1.i16), S2.i16) (signed).
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 167.
//   rdna-shader-instruction-set-architecture.pdf, page 194.
//   rdna2-shader-instruction-set-architecture.pdf, page 189.
def V_MAX3_I16_e64_sem : InstSISemantic<V_MAX3_I16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mx, (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $mx), (GetVal $c)),
                    (GetVal $mx), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// D.i16 = min(min(S0.i16, S1.i16), S2.i16) (signed).
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 167.
//   rdna-shader-instruction-set-architecture.pdf, page 194.
//   rdna2-shader-instruction-set-architecture.pdf, page 189.
def V_MIN3_I16_e64_sem : InstSISemantic<V_MIN3_I16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mn, (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $mn), (GetVal $c)),
                    (GetVal $mn), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 168.
//   rdna-shader-instruction-set-architecture.pdf, page 194.
//   rdna2-shader-instruction-set-architecture.pdf, page 190.
def V_MED3_I16_e64_sem : InstSISemantic<V_MED3_I16_e64> {
  let Semantic = [];
}

// D.u16 = max(max(S0.u16, S1.u16), S2.u16) (unsigned).
def V_MAX3_U16_e64_sem : InstSISemantic<V_MAX3_U16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mx, (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $mx), (GetVal $c)),
                    (GetVal $mx), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// D.u16 = min(min(S0.u16, S1.u16), S2.u16) (unsigned).
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 167.
//   rdna-shader-instruction-set-architecture.pdf, page 194.
//   rdna2-shader-instruction-set-architecture.pdf, page 189.
def V_MIN3_U16_e64_sem : InstSISemantic<V_MIN3_U16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mn, (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $mn), (GetVal $c)),
                    (GetVal $mn), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 175.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 177.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 168.
//   rdna-shader-instruction-set-architecture.pdf, page 194.
//   rdna2-shader-instruction-set-architecture.pdf, page 190.
def V_MED3_U16_e64_sem : InstSISemantic<V_MED3_U16_e64> {
  let Semantic = [];
}

//------------------------------------------------------------------------------
// VOP3 Maximum/Minimum (IEEE 754-2019) — empty semantics
//------------------------------------------------------------------------------

foreach inst = [
    V_MAXIMUM3_F32_e64,
    V_MINIMUM3_F32_e64,
    V_MAXIMUM3_F16_e64,
    V_MINIMUM3_F16_e64,
    V_MAXIMUM_F32_e64,
    V_MAXIMUM_F64_e64,
    V_MAXIMUM_F16_e64,
    V_MINIMUMMAXIMUM_F32_e64,
    V_MAXIMUMMINIMUM_F32_e64,
    V_MAXIMUMMINIMUM_F16_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// VOP3 Bit Field Extract / Insert
//------------------------------------------------------------------------------

// D.u = (S0.u >> S1.u[4:0]) & ((1 << S2.u[4:0]) - 1) (unsigned BFE).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 314.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 167.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 159.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 338.
//   rdna-shader-instruction-set-architecture.pdf, page 180.
//   rdna2-shader-instruction-set-architecture.pdf, page 177.
//   rdna4-instruction-set-architecture.pdf, page 478.
def V_BFE_U32_e64_sem : InstSISemantic<V_BFE_U32_e64> {
  let Semantic = [
    (DefVal $offset, (LLVMAnd (GetNamedOperand $src1),
                               (ConstantInt si_llvm_i32_ty, 31))),
    (DefVal $width, (LLVMAnd (GetNamedOperand $src2),
                              (ConstantInt si_llvm_i32_ty, 31))),
    (SetNamedOperand $vdst,
      (LLVMAnd (LLVMLShr (GetNamedOperand $src0), (GetVal $offset)),
               (LLVMSub (LLVMShl (ConstantInt si_llvm_i32_ty, 1),
                                  (GetVal $width)),
                         (ConstantInt si_llvm_i32_ty, 1))))
  ];
}

// D.i = sign_extend((S0.i >> S1.u[4:0]) & ((1 << S2.u[4:0]) - 1)) (signed BFE).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 314.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 167.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 159.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 338.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 365.
//   rdna-shader-instruction-set-architecture.pdf, page 180.
//   rdna2-shader-instruction-set-architecture.pdf, page 177.
//   rdna4-instruction-set-architecture.pdf, page 478.
def V_BFE_I32_e64_sem : InstSISemantic<V_BFE_I32_e64> {
  let Semantic = [
    (DefVal $offset, (LLVMAnd (GetNamedOperand $src1),
                               (ConstantInt si_llvm_i32_ty, 31))),
    (DefVal $width, (LLVMAnd (GetNamedOperand $src2),
                              (ConstantInt si_llvm_i32_ty, 31))),
    (DefVal $extracted,
      (LLVMAnd (LLVMAShr (GetNamedOperand $src0), (GetVal $offset)),
               (LLVMSub (LLVMShl (ConstantInt si_llvm_i32_ty, 1),
                                  (GetVal $width)),
                         (ConstantInt si_llvm_i32_ty, 1)))),
    // Sign extend: shift left by (32-width), then arithmetic shift right
    (DefVal $shamt, (LLVMSub (ConstantInt si_llvm_i32_ty, 32), (GetVal $width))),
    (SetNamedOperand $vdst,
      (LLVMAShr (LLVMShl (GetVal $extracted), (GetVal $shamt)),
                (GetVal $shamt)))
  ];
}

// D.u = (S0.u & S1.u) | (~S0.u & S2.u) (bit field insert).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 314.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 167.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 159.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 338.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 365.
//   rdna-shader-instruction-set-architecture.pdf, page 180.
//   rdna2-shader-instruction-set-architecture.pdf, page 177.
//   rdna4-instruction-set-architecture.pdf, page 478.
def V_BFI_B32_e64_sem : InstSISemantic<V_BFI_B32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMOr (LLVMAnd (GetNamedOperand $src0), (GetNamedOperand $src1)),
              (LLVMAnd LLVMNot<(GetNamedOperand $src0), si_llvm_i32_ty>.ret,
                       (GetNamedOperand $src2))))
  ];
}

// D.u = ({S0, S1} >> S2.u[4:0])[31:0] (align bit).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 231.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 339.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 366.
//   rdna-shader-instruction-set-architecture.pdf, page 181.
//   rdna2-shader-instruction-set-architecture.pdf, page 178.
def V_ALIGNBIT_B32_e64_sem : InstSISemantic<V_ALIGNBIT_B32_e64> {
  let Semantic = [
    (DefVal $concat,
      (LLVMOr (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                        (ConstantInt si_llvm_i64_ty, 32)),
              (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (SetNamedOperand $vdst,
      (LLVMTrunc
        (LLVMLShr (GetVal $concat),
                  (LLVMZExt (LLVMAnd (GetNamedOperand $src2),
                                     (ConstantInt si_llvm_i32_ty, 31)),
                            si_llvm_i64_ty)),
        si_llvm_i32_ty))
  ];
}

// D.u = ({S0, S1} >> (8 * S2.u[1:0]))[31:0] (align byte).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 231.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 340.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 366.
//   rdna-shader-instruction-set-architecture.pdf, page 181.
//   rdna2-shader-instruction-set-architecture.pdf, page 178.
//   rdna4-instruction-set-architecture.pdf, page 480.
def V_ALIGNBYTE_B32_e64_sem : InstSISemantic<V_ALIGNBYTE_B32_e64> {
  let Semantic = [
    (DefVal $concat,
      (LLVMOr (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                        (ConstantInt si_llvm_i64_ty, 32)),
              (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (DefVal $shift,
      (LLVMShl (LLVMZExt (LLVMAnd (GetNamedOperand $src2),
                                   (ConstantInt si_llvm_i32_ty, 3)),
                          si_llvm_i64_ty),
               (ConstantInt si_llvm_i64_ty, 3))),
    (SetNamedOperand $vdst,
      (LLVMTrunc (LLVMLShr (GetVal $concat), (GetVal $shift)),
                 si_llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP3 Pack/Conversion — empty or intrinsic semantics
//------------------------------------------------------------------------------

// D.u = f32_to_f16_rtz(S1.f) << 16 | f32_to_f16_rtz(S0.f).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 239.
//   vega-shader-instruction-set-architecture.pdf, page 266.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 268.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 253.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 247.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 540.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 587.
//   rdna-shader-instruction-set-architecture.pdf, page 263.
//   rdna2-shader-instruction-set-architecture.pdf, page 257.
def V_CVT_PKRTZ_F16_F32_e64_sem : InstSISemantic<V_CVT_PKRTZ_F16_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_v2f16_ty, int_amdgcn_cvt_pkrtz, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))
  ];
}

def V_CVT_PKNORM_I16_F32_e64_sem : InstSISemantic<V_CVT_PKNORM_I16_F32_e64> {
  let Semantic = [];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 239.
//   vega-shader-instruction-set-architecture.pdf, page 266.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 268.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 253.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 247.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 540.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 587.
//   rdna-shader-instruction-set-architecture.pdf, page 274.
//   rdna2-shader-instruction-set-architecture.pdf, page 268.
def V_CVT_PKNORM_U16_F32_e64_sem : InstSISemantic<V_CVT_PKNORM_U16_F32_e64> {
  let Semantic = [];
}

def V_CVT_PK_I16_I32_e64_sem : InstSISemantic<V_CVT_PK_I16_I32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_v2i16_ty, int_amdgcn_cvt_pk_i16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))
  ];
}
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 238.
//   vega-shader-instruction-set-architecture.pdf, page 181.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 183.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 174.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 172.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 540.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 587.
//   rdna-shader-instruction-set-architecture.pdf, page 198.
//   rdna2-shader-instruction-set-architecture.pdf, page 193.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 445.
//   rdna4-instruction-set-architecture.pdf, page 518.
def V_CVT_PK_U16_U32_e64_sem : InstSISemantic<V_CVT_PK_U16_U32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_v2i16_ty, int_amdgcn_cvt_pk_u16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))
  ];
}

// Sources:
//   gcn3-instruction-set-architecture.pdf, page 237.
//   vega-shader-instruction-set-architecture.pdf, page 168.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 161.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 161.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 343.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 370.
//   rdna-shader-instruction-set-architecture.pdf, page 183.
//   rdna2-shader-instruction-set-architecture.pdf, page 179.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 422.
//   rdna4-instruction-set-architecture.pdf, page 483.
def V_CVT_PK_U8_F32_e64_sem : InstSISemantic<V_CVT_PK_U8_F32_e64> {
  let Semantic = [];
}
def V_CVT_PKACCUM_U8_F32_e64_sem : InstSISemantic<V_CVT_PKACCUM_U8_F32_e64> {
  let Semantic = [];
}

foreach inst = [
    V_CVT_PKNORM_I16_F16_e64,
    V_CVT_PKNORM_U16_F16_e64,
    V_CVT_PK_I16_F32_e64,
    V_CVT_PK_U16_F32_e64,
    V_PACK_B32_F16_e64,
    V_SAT_PK_U8_I16_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// VOP3 FP8/BF8 Conversions — empty semantics
//------------------------------------------------------------------------------

foreach inst = [
    V_CVT_F32_FP8_e64,
    V_CVT_F32_BF8_e64,
//    V_CVT_F32_FP8_OP_SEL_e64,
//    V_CVT_F32_BF8_OP_SEL_e64,
    V_CVT_PK_F32_FP8_e64,
    V_CVT_PK_F32_BF8_e64,
    V_CVT_PK_FP8_F32_e64,
    V_CVT_PK_BF8_F32_e64,
    V_CVT_SR_FP8_F32_e64,
    V_CVT_SR_BF8_F32_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// VOP3 SAD (Sum of Absolute Differences) — empty semantics
//------------------------------------------------------------------------------

// Sources:
//   gcn3-instruction-set-architecture.pdf, page 260.
//   vega-shader-instruction-set-architecture.pdf, page 167.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 161.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 348.
//   rdna-shader-instruction-set-architecture.pdf, page 182.
//   rdna2-shader-instruction-set-architecture.pdf, page 179.

// Sources:
//   gcn3-instruction-set-architecture.pdf, page 260.
//   vega-shader-instruction-set-architecture.pdf, page 167.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 169.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 161.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 348.
//   rdna-shader-instruction-set-architecture.pdf, page 182.
//   rdna2-shader-instruction-set-architecture.pdf, page 179.
foreach inst = [
    V_SAD_U8_e64,
    V_SAD_HI_U8_e64,
    V_SAD_U16_e64,
    V_SAD_U32_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// VOP3 Misc — empty semantics
//------------------------------------------------------------------------------

// Sources:
//   gcn3-instruction-set-architecture.pdf, page 245.
//   vega-shader-instruction-set-architecture.pdf, page 166.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 168.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 160.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 160.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 339.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 366.
//   rdna-shader-instruction-set-architecture.pdf, page 181.
//   rdna2-shader-instruction-set-architecture.pdf, page 178.
//   rdna4-instruction-set-architecture.pdf, page 479.
foreach inst = [
    V_LERP_U8_e64,
    V_PERM_B32_e64,
    V_MULLIT_F32_e64,
    V_TRIG_PREOP_F64_e64,
    V_QSAD_PK_U16_U8_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// VOP3 Interpolation — empty semantics
//------------------------------------------------------------------------------

// Sources:
//   gcn3-instruction-set-architecture.pdf, page 98.
//   vega-shader-instruction-set-architecture.pdf, page 91.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 92.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 95.
//   rdna-shader-instruction-set-architecture.pdf, page 100.
//   rdna2-shader-instruction-set-architecture.pdf, page 101.

// Sources:
//   gcn3-instruction-set-architecture.pdf, page 98.
//   vega-shader-instruction-set-architecture.pdf, page 91.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 92.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 95.
//   rdna-shader-instruction-set-architecture.pdf, page 100.
//   rdna2-shader-instruction-set-architecture.pdf, page 101.
foreach inst = [
    V_INTERP_P1_F32_e64,
    V_INTERP_P2_F32_e64,
    V_INTERP_MOV_F32_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// VOP3 Lane Operations — empty semantics
//------------------------------------------------------------------------------

// Sources:
//   rdna-shader-instruction-set-architecture.pdf, page 274.
foreach inst = [
    V_PERMLANE16_B32_e64,
    V_PERMLANEX16_B32_e64,
    V_PERMLANE16_VAR_B32_e64,
    V_PERMLANEX16_VAR_B32_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// VOP3 Dot Product e64 — empty semantics
//------------------------------------------------------------------------------

// Sources (HasDot5Insts):
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 185.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 197.
//   rdna-shader-instruction-set-architecture.pdf, page 143.
//   rdna2-shader-instruction-set-architecture.pdf, page 142.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 266.
//   rdna4-instruction-set-architecture.pdf, page 307.

// Sources (HasDot5Insts):
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 185.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 197.
//   rdna-shader-instruction-set-architecture.pdf, page 143.
//   rdna2-shader-instruction-set-architecture.pdf, page 142.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 266.
//   rdna4-instruction-set-architecture.pdf, page 307.
foreach inst = [
    V_DOT2C_F32_F16_e64,
    V_DOT2C_I32_I16_e64,
    V_DOT4C_I32_I8_e64,
    V_DOT8C_I32_I4_e64,
    V_DOT2_BF16_BF16_e64,
    V_DOT2_F16_F16_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// VOP3 S-prefixed Math (GFX12+) — empty semantics
//------------------------------------------------------------------------------

foreach inst = [
    V_S_EXP_F32_e64,
    V_S_EXP_F16_e64,
    V_S_LOG_F32_e64,
    V_S_LOG_F16_e64,
    V_S_RCP_F32_e64,
    V_S_RCP_F16_e64,
    V_S_RSQ_F32_e64,
    V_S_RSQ_F16_e64,
    V_S_SQRT_F32_e64,
    V_S_SQRT_F16_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// VOP3 CNDMASK B16 e64
//------------------------------------------------------------------------------

// D.u16 = src2[lane] ? S1.u16 : S0.u16. Condition from explicit $src2.
// Sources (GFX11+ only, isGFX11Plus):
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 434.
//   rdna4-instruction-set-architecture.pdf, page 498.
//------------------------------------------------------------------------------
// VOP3 PK_FMAC — empty semantic
//------------------------------------------------------------------------------

// V_PK_FMAC_F16 — fused-multiply-accumulate with tied accumulator:
//   vdst = src0 * src1 + vdst    (per-lane v2f16 FMA)
// $vdst is both def and use; the tied input is synthesized by
// CodeDiscoveryPass's MCID-driven tied-operand fix (see memory:
// reference-mir-tied-operand-synthesis).
def V_PK_FMAC_F16_e64_sem : InstSISemantic<V_PK_FMAC_F16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFMA (GetNamedOperand $src0, si_llvm_v2f16_ty),
               (GetNamedOperand $src1, si_llvm_v2f16_ty),
               (GetNamedOperand $vdst, si_llvm_v2f16_ty)))
  ];
}

//------------------------------------------------------------------------------
// VOP3 SMFMAC (Sparse Matrix) — empty semantics
//------------------------------------------------------------------------------

// All V_SMFMAC_*_e64 variants use empty semantics due to complexity.
foreach inst = [
    V_SMFMAC_F32_16X16X32_F16_e64,
    V_SMFMAC_F32_16X16X32_BF16_e64,
    V_SMFMAC_F32_16X16X64_BF8_BF8_e64,
    V_SMFMAC_F32_16X16X64_BF8_FP8_e64,
    V_SMFMAC_F32_16X16X64_FP8_BF8_e64,
    V_SMFMAC_F32_16X16X64_FP8_FP8_e64,
    V_SMFMAC_F32_16X16X128_BF8_BF8_e64,
    V_SMFMAC_F32_16X16X128_BF8_FP8_e64,
    V_SMFMAC_F32_16X16X128_FP8_BF8_e64,
    V_SMFMAC_F32_16X16X128_FP8_FP8_e64,
    V_SMFMAC_F32_32X32X16_BF16_e64,
    V_SMFMAC_F32_32X32X16_F16_e64,
    V_SMFMAC_F32_32X32X32_BF8_BF8_e64,
    V_SMFMAC_F32_32X32X32_BF8_FP8_e64,
    V_SMFMAC_F32_32X32X32_FP8_BF8_e64,
    V_SMFMAC_F32_32X32X32_FP8_FP8_e64,
    V_SMFMAC_F32_32X32X64_BF8_BF8_e64,
    V_SMFMAC_F32_32X32X64_BF8_FP8_e64,
    V_SMFMAC_F32_32X32X64_FP8_BF8_e64,
    V_SMFMAC_F32_32X32X64_FP8_FP8_e64,
    V_SMFMAC_I32_16X16X64_I8_e64,
    V_SMFMAC_I32_32X32X32_I8_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}

//------------------------------------------------------------------------------
// VOP3 MFMA (Matrix) — empty semantics
//------------------------------------------------------------------------------

// All V_MFMA_*_e64 variants use empty semantics due to complexity.
foreach inst = [
    V_MFMA_F32_4X4X1F32_e64,
    V_MFMA_F32_4X4X2BF16_e64,
    V_MFMA_F32_4X4X4F16_e64,
    V_MFMA_F32_16X16X1F32_e64,
    V_MFMA_F32_16X16X2BF16_e64,
    V_MFMA_F32_16X16X4F16_e64,
    V_MFMA_F32_16X16X4F32_e64,
    V_MFMA_F32_16X16X8BF16_e64,
    V_MFMA_F32_16X16X8XF32_e64,
    V_MFMA_F32_16X16X16F16_e64,
    V_MFMA_F32_16X16X32_BF8_BF8_e64,
    V_MFMA_F32_16X16X32_BF8_FP8_e64,
    V_MFMA_F32_16X16X32_FP8_BF8_e64,
    V_MFMA_F32_16X16X32_FP8_FP8_e64,
    V_MFMA_F32_32X32X1F32_e64,
    V_MFMA_F32_32X32X2BF16_e64,
    V_MFMA_F32_32X32X2F32_e64,
    V_MFMA_F32_32X32X4BF16_e64,
    V_MFMA_F32_32X32X4F16_e64,
    V_MFMA_F32_32X32X4XF32_e64,
    V_MFMA_F32_32X32X8F16_e64,
    V_MFMA_F32_32X32X16_BF8_BF8_e64,
    V_MFMA_F32_32X32X16_BF8_FP8_e64,
    V_MFMA_F32_32X32X16_FP8_BF8_e64,
    V_MFMA_F32_32X32X16_FP8_FP8_e64,
    V_MFMA_I32_4X4X4I8_e64,
    V_MFMA_I32_16X16X4I8_e64,
    V_MFMA_I32_16X16X16I8_e64,
    V_MFMA_I32_16X16X32I8_e64,
    V_MFMA_I32_32X32X4I8_e64,
    V_MFMA_I32_32X32X8I8_e64,
    V_MFMA_I32_32X32X16I8_e64,
    V_MFMA_F64_4X4X4F64_e64,
    V_MFMA_F64_16X16X4F64_e64
] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [];
  }
}


//==============================================================================
// V_ family _t16 / _fake16 register-file-split variants (Phase 7)
// gfx11+ 16-bit register file split — same IR semantics as the base op.
//==============================================================================

// _t16/_fake16 alias of V_ADD_F16_e64
def V_ADD_F16_fake16_e64_sem : InstSISemantic<V_ADD_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFAdd
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_ADD_F16_e64
def V_ADD_F16_t16_e64_sem : InstSISemantic<V_ADD_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFAdd
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_ALIGNBIT_B32_e64
def V_ALIGNBIT_B32_fake16_e64_sem : InstSISemantic<V_ALIGNBIT_B32_fake16_e64> {
  let Semantic = [
    (DefVal $concat,
      (LLVMOr (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                        (ConstantInt si_llvm_i64_ty, 32)),
              (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (SetNamedOperand $vdst,
      (LLVMTrunc
        (LLVMLShr (GetVal $concat),
                  (LLVMZExt (LLVMAnd (GetNamedOperand $src2),
                                     (ConstantInt si_llvm_i32_ty, 31)),
                            si_llvm_i64_ty)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_ALIGNBIT_B32_e64
def V_ALIGNBIT_B32_t16_e64_sem : InstSISemantic<V_ALIGNBIT_B32_t16_e64> {
  let Semantic = [
    (DefVal $concat,
      (LLVMOr (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                        (ConstantInt si_llvm_i64_ty, 32)),
              (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (SetNamedOperand $vdst,
      (LLVMTrunc
        (LLVMLShr (GetVal $concat),
                  (LLVMZExt (LLVMAnd (GetNamedOperand $src2),
                                     (ConstantInt si_llvm_i32_ty, 31)),
                            si_llvm_i64_ty)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_ALIGNBYTE_B32_e64
def V_ALIGNBYTE_B32_fake16_e64_sem : InstSISemantic<V_ALIGNBYTE_B32_fake16_e64> {
  let Semantic = [
    (DefVal $concat,
      (LLVMOr (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                        (ConstantInt si_llvm_i64_ty, 32)),
              (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (DefVal $shift,
      (LLVMShl (LLVMZExt (LLVMAnd (GetNamedOperand $src2),
                                   (ConstantInt si_llvm_i32_ty, 3)),
                          si_llvm_i64_ty),
               (ConstantInt si_llvm_i64_ty, 3))),
    (SetNamedOperand $vdst,
      (LLVMTrunc (LLVMLShr (GetVal $concat), (GetVal $shift)),
                 si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_ALIGNBYTE_B32_e64
def V_ALIGNBYTE_B32_t16_e64_sem : InstSISemantic<V_ALIGNBYTE_B32_t16_e64> {
  let Semantic = [
    (DefVal $concat,
      (LLVMOr (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                        (ConstantInt si_llvm_i64_ty, 32)),
              (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (DefVal $shift,
      (LLVMShl (LLVMZExt (LLVMAnd (GetNamedOperand $src2),
                                   (ConstantInt si_llvm_i32_ty, 3)),
                          si_llvm_i64_ty),
               (ConstantInt si_llvm_i64_ty, 3))),
    (SetNamedOperand $vdst,
      (LLVMTrunc (LLVMLShr (GetVal $concat), (GetVal $shift)),
                 si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_ASHRREV_I16_e64
def V_ASHRREV_I16_fake16_e64_sem : InstSISemantic<V_ASHRREV_I16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAShr (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                  (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                            (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_ASHRREV_I16_e64
def V_ASHRREV_I16_t16_e64_sem : InstSISemantic<V_ASHRREV_I16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAShr (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                  (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                            (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CEIL_F16_e64
def V_CEIL_F16_fake16_e64_sem : InstSISemantic<V_CEIL_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_ceil, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                              si_llvm_i16_ty),
                                  si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CEIL_F16_e64
def V_CEIL_F16_t16_e64_sem : InstSISemantic<V_CEIL_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_ceil, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                              si_llvm_i16_ty),
                                  si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_COS_F16_e64
def V_COS_F16_fake16_e64_sem : InstSISemantic<V_COS_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_cos, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_COS_F16_e64
def V_COS_F16_t16_e64_sem : InstSISemantic<V_COS_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_cos, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_CVT_F16_F32_e64
def V_CVT_F16_F32_fake16_e64_sem : InstSISemantic<V_CVT_F16_F32_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFPTrunc (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_F16_F32_e64
def V_CVT_F16_F32_t16_e64_sem : InstSISemantic<V_CVT_F16_F32_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFPTrunc (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_F16_I16_e64
def V_CVT_F16_I16_fake16_e64_sem : InstSISemantic<V_CVT_F16_I16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMSIToFP (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_F16_I16_e64
def V_CVT_F16_I16_t16_e64_sem : InstSISemantic<V_CVT_F16_I16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMSIToFP (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_F16_U16_e64
def V_CVT_F16_U16_fake16_e64_sem : InstSISemantic<V_CVT_F16_U16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUIToFP (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_F16_U16_e64
def V_CVT_F16_U16_t16_e64_sem : InstSISemantic<V_CVT_F16_U16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUIToFP (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_F32_F16_e64
def V_CVT_F32_F16_fake16_e64_sem : InstSISemantic<V_CVT_F32_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFPExt (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                            si_llvm_i16_ty),
                                 si_llvm_half_ty),
                   si_llvm_float_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_F32_F16_e64
def V_CVT_F32_F16_t16_e64_sem : InstSISemantic<V_CVT_F32_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFPExt (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                            si_llvm_i16_ty),
                                 si_llvm_half_ty),
                   si_llvm_float_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_I16_F16_e64
def V_CVT_I16_F16_fake16_e64_sem : InstSISemantic<V_CVT_I16_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMFPToSI
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_I16_F16_e64
def V_CVT_I16_F16_t16_e64_sem : InstSISemantic<V_CVT_I16_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMFPToSI
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_I32_I16_e64
def V_CVT_I32_I16_fake16_e64_sem : InstSISemantic<V_CVT_I32_I16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSExt (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_I32_I16_e64
def V_CVT_I32_I16_t16_e64_sem : InstSISemantic<V_CVT_I32_I16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSExt (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_NORM_I16_F16_e64
def V_CVT_NORM_I16_F16_fake16_e64_sem : InstSISemantic<V_CVT_NORM_I16_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_pknorm_i16, [(GetNamedOperand $src0)]))
  ];
}

// _t16/_fake16 alias of V_CVT_NORM_I16_F16_e64
def V_CVT_NORM_I16_F16_t16_e64_sem : InstSISemantic<V_CVT_NORM_I16_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_pknorm_i16, [(GetNamedOperand $src0)]))
  ];
}

// _t16/_fake16 alias of V_CVT_NORM_U16_F16_e64
def V_CVT_NORM_U16_F16_fake16_e64_sem : InstSISemantic<V_CVT_NORM_U16_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_pknorm_u16, [(GetNamedOperand $src0)]))
  ];
}

// _t16/_fake16 alias of V_CVT_NORM_U16_F16_e64
def V_CVT_NORM_U16_F16_t16_e64_sem : InstSISemantic<V_CVT_NORM_U16_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_pknorm_u16, [(GetNamedOperand $src0)]))
  ];
}

// _t16/_fake16 alias of V_CVT_U16_F16_e64
def V_CVT_U16_F16_fake16_e64_sem : InstSISemantic<V_CVT_U16_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMFPToUI
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_U16_F16_e64
def V_CVT_U16_F16_t16_e64_sem : InstSISemantic<V_CVT_U16_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMFPToUI
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                       si_llvm_half_ty),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_U32_U16_e64
def V_CVT_U32_U16_fake16_e64_sem : InstSISemantic<V_CVT_U32_U16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_CVT_U32_U16_e64
def V_CVT_U32_U16_t16_e64_sem : InstSISemantic<V_CVT_U32_U16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_EXP_F16_e64
def V_EXP_F16_fake16_e64_sem : InstSISemantic<V_EXP_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_exp2, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_EXP_F16_e64
def V_EXP_F16_t16_e64_sem : InstSISemantic<V_EXP_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_exp2, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_FLOOR_F16_e64
def V_FLOOR_F16_fake16_e64_sem : InstSISemantic<V_FLOOR_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_floor, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                               si_llvm_i16_ty),
                                   si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_FLOOR_F16_e64
def V_FLOOR_F16_t16_e64_sem : InstSISemantic<V_FLOOR_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_floor, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                               si_llvm_i16_ty),
                                   si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_FMAC_F16_e64
def V_FMAC_F16_fake16_e64_sem : InstSISemantic<V_FMAC_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFMA
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_FMAC_F16_e64
def V_FMAC_F16_t16_e64_sem : InstSISemantic<V_FMAC_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFMA
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_FRACT_F16_e64
def V_FRACT_F16_fake16_e64_sem : InstSISemantic<V_FRACT_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_fract, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_FRACT_F16_e64
def V_FRACT_F16_t16_e64_sem : InstSISemantic<V_FRACT_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_fract, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_FREXP_EXP_I16_F16_e64
def V_FREXP_EXP_I16_F16_fake16_e64_sem : InstSISemantic<V_FREXP_EXP_I16_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_frexp_exp,
        [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_FREXP_EXP_I16_F16_e64
def V_FREXP_EXP_I16_F16_t16_e64_sem : InstSISemantic<V_FREXP_EXP_I16_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_frexp_exp,
        [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_FREXP_MANT_F16_e64
def V_FREXP_MANT_F16_fake16_e64_sem : InstSISemantic<V_FREXP_MANT_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_frexp_mant, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_FREXP_MANT_F16_e64
def V_FREXP_MANT_F16_t16_e64_sem : InstSISemantic<V_FREXP_MANT_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_frexp_mant, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_LDEXP_F16_e64
def V_LDEXP_F16_fake16_e64_sem : InstSISemantic<V_LDEXP_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMLdexp
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_LDEXP_F16_e64
def V_LDEXP_F16_t16_e64_sem : InstSISemantic<V_LDEXP_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMLdexp
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_LOG_F16_e64
def V_LOG_F16_fake16_e64_sem : InstSISemantic<V_LOG_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_log, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_LOG_F16_e64
def V_LOG_F16_t16_e64_sem : InstSISemantic<V_LOG_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_log, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_LSHLREV_B16_e64
def V_LSHLREV_B16_fake16_e64_sem : InstSISemantic<V_LSHLREV_B16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMShl (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                 (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_LSHLREV_B16_e64
def V_LSHLREV_B16_t16_e64_sem : InstSISemantic<V_LSHLREV_B16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMShl (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                 (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_LSHRREV_B16_e64
def V_LSHRREV_B16_fake16_e64_sem : InstSISemantic<V_LSHRREV_B16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMLShr (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                  (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                            (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_LSHRREV_B16_e64
def V_LSHRREV_B16_t16_e64_sem : InstSISemantic<V_LSHRREV_B16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMLShr (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                  (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                            (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAD_I32_I16_e64
def V_MAD_I32_I16_fake16_e64_sem : InstSISemantic<V_MAD_I32_I16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMSExt (LLVMTrunc (GetNamedOperand $src0),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty),
                         (LLVMSExt (LLVMTrunc (GetNamedOperand $src1),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty)),
               (GetNamedOperand $src2)))
  ];
}

// _t16/_fake16 alias of V_MAD_I32_I16_e64
def V_MAD_I32_I16_t16_e64_sem : InstSISemantic<V_MAD_I32_I16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMSExt (LLVMTrunc (GetNamedOperand $src0),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty),
                         (LLVMSExt (LLVMTrunc (GetNamedOperand $src1),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty)),
               (GetNamedOperand $src2)))
  ];
}

// _t16/_fake16 alias of V_MAD_U32_U16_e64
def V_MAD_U32_U16_fake16_e64_sem : InstSISemantic<V_MAD_U32_U16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMZExt (LLVMTrunc (GetNamedOperand $src0),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty),
                         (LLVMZExt (LLVMTrunc (GetNamedOperand $src1),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty)),
               (GetNamedOperand $src2)))
  ];
}

// _t16/_fake16 alias of V_MAD_U32_U16_e64
def V_MAD_U32_U16_t16_e64_sem : InstSISemantic<V_MAD_U32_U16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMZExt (LLVMTrunc (GetNamedOperand $src0),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty),
                         (LLVMZExt (LLVMTrunc (GetNamedOperand $src1),
                                               si_llvm_i16_ty),
                                   si_llvm_i32_ty)),
               (GetNamedOperand $src2)))
  ];
}

// _t16/_fake16 alias of V_MAX3_F16_e64
def V_MAX3_F16_fake16_e64_sem : InstSISemantic<V_MAX3_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMaxNum
            (LLVMMaxNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX3_F16_e64
def V_MAX3_F16_t16_e64_sem : InstSISemantic<V_MAX3_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMaxNum
            (LLVMMaxNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX3_I16_e64
def V_MAX3_I16_fake16_e64_sem : InstSISemantic<V_MAX3_I16_fake16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mx, (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $mx), (GetVal $c)),
                    (GetVal $mx), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX3_I16_e64
def V_MAX3_I16_t16_e64_sem : InstSISemantic<V_MAX3_I16_t16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mx, (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $mx), (GetVal $c)),
                    (GetVal $mx), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX3_U16_e64
def V_MAX3_U16_fake16_e64_sem : InstSISemantic<V_MAX3_U16_fake16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mx, (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $mx), (GetVal $c)),
                    (GetVal $mx), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX3_U16_e64
def V_MAX3_U16_t16_e64_sem : InstSISemantic<V_MAX3_U16_t16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mx, (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $mx), (GetVal $c)),
                    (GetVal $mx), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAXMIN_F16_e64
def V_MAXMIN_F16_fake16_e64_sem : InstSISemantic<V_MAXMIN_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMinNum
            (LLVMMaxNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAXMIN_F16_e64
def V_MAXMIN_F16_t16_e64_sem : InstSISemantic<V_MAXMIN_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMinNum
            (LLVMMaxNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX_F16_e64
def V_MAX_F16_fake16_e64_sem : InstSISemantic<V_MAX_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMaxNum
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX_F16_e64
def V_MAX_F16_t16_e64_sem : InstSISemantic<V_MAX_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMaxNum
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX_I16_e64
def V_MAX_I16_fake16_e64_sem : InstSISemantic<V_MAX_I16_fake16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX_I16_e64
def V_MAX_I16_t16_e64_sem : InstSISemantic<V_MAX_I16_t16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX_U16_e64
def V_MAX_U16_fake16_e64_sem : InstSISemantic<V_MAX_U16_fake16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MAX_U16_e64
def V_MAX_U16_t16_e64_sem : InstSISemantic<V_MAX_U16_t16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MED3_F16_e64
def V_MED3_F16_fake16_e64_sem : InstSISemantic<V_MED3_F16_fake16_e64> {
  let Semantic = [];
}

// _t16/_fake16 alias of V_MED3_F16_e64
def V_MED3_F16_t16_e64_sem : InstSISemantic<V_MED3_F16_t16_e64> {
  let Semantic = [];
}

// _t16/_fake16 alias of V_MED3_I16_e64
def V_MED3_I16_fake16_e64_sem : InstSISemantic<V_MED3_I16_fake16_e64> {
  let Semantic = [];
}

// _t16/_fake16 alias of V_MED3_I16_e64
def V_MED3_I16_t16_e64_sem : InstSISemantic<V_MED3_I16_t16_e64> {
  let Semantic = [];
}

// _t16/_fake16 alias of V_MED3_U16_e64
def V_MED3_U16_fake16_e64_sem : InstSISemantic<V_MED3_U16_fake16_e64> {
  let Semantic = [];
}

// _t16/_fake16 alias of V_MED3_U16_e64
def V_MED3_U16_t16_e64_sem : InstSISemantic<V_MED3_U16_t16_e64> {
  let Semantic = [];
}

// _t16/_fake16 alias of V_MIN3_F16_e64
def V_MIN3_F16_fake16_e64_sem : InstSISemantic<V_MIN3_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMinNum
            (LLVMMinNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN3_F16_e64
def V_MIN3_F16_t16_e64_sem : InstSISemantic<V_MIN3_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMinNum
            (LLVMMinNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN3_I16_e64
def V_MIN3_I16_fake16_e64_sem : InstSISemantic<V_MIN3_I16_fake16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mn, (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $mn), (GetVal $c)),
                    (GetVal $mn), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN3_I16_e64
def V_MIN3_I16_t16_e64_sem : InstSISemantic<V_MIN3_I16_t16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mn, (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $mn), (GetVal $c)),
                    (GetVal $mn), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN3_U16_e64
def V_MIN3_U16_fake16_e64_sem : InstSISemantic<V_MIN3_U16_fake16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mn, (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $mn), (GetVal $c)),
                    (GetVal $mn), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN3_U16_e64
def V_MIN3_U16_t16_e64_sem : InstSISemantic<V_MIN3_U16_t16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (DefVal $c, (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
    (DefVal $mn, (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $a), (GetVal $b)),
                              (GetVal $a), (GetVal $b))),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $mn), (GetVal $c)),
                    (GetVal $mn), (GetVal $c)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MINMAX_F16_e64
def V_MINMAX_F16_fake16_e64_sem : InstSISemantic<V_MINMAX_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMaxNum
            (LLVMMinNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MINMAX_F16_e64
def V_MINMAX_F16_t16_e64_sem : InstSISemantic<V_MINMAX_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMaxNum
            (LLVMMinNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN_F16_e64
def V_MIN_F16_fake16_e64_sem : InstSISemantic<V_MIN_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMinNum
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN_F16_e64
def V_MIN_F16_t16_e64_sem : InstSISemantic<V_MIN_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMMinNum
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN_I16_e64
def V_MIN_I16_fake16_e64_sem : InstSISemantic<V_MIN_I16_fake16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN_I16_e64
def V_MIN_I16_t16_e64_sem : InstSISemantic<V_MIN_I16_t16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN_U16_e64
def V_MIN_U16_fake16_e64_sem : InstSISemantic<V_MIN_U16_fake16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MIN_U16_e64
def V_MIN_U16_t16_e64_sem : InstSISemantic<V_MIN_U16_t16_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MUL_F16_e64
def V_MUL_F16_fake16_e64_sem : InstSISemantic<V_MUL_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFMul
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MUL_F16_e64
def V_MUL_F16_t16_e64_sem : InstSISemantic<V_MUL_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFMul
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MUL_LO_U16_e64
def V_MUL_LO_U16_fake16_e64_sem : InstSISemantic<V_MUL_LO_U16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                 (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_MUL_LO_U16_e64
def V_MUL_LO_U16_t16_e64_sem : InstSISemantic<V_MUL_LO_U16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                 (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_NOT_B16_e64
def V_NOT_B16_fake16_e64_sem : InstSISemantic<V_NOT_B16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMXor (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                 (LLVMSExt (ConstantTrue si_llvm_i1_ty), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_NOT_B16_e64
def V_NOT_B16_t16_e64_sem : InstSISemantic<V_NOT_B16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMXor (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                 (LLVMSExt (ConstantTrue si_llvm_i1_ty), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_RCP_F16_e64
def V_RCP_F16_fake16_e64_sem : InstSISemantic<V_RCP_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_rcp, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_RCP_F16_e64
def V_RCP_F16_t16_e64_sem : InstSISemantic<V_RCP_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_rcp, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_RNDNE_F16_e64
def V_RNDNE_F16_fake16_e64_sem : InstSISemantic<V_RNDNE_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_roundeven, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                                   si_llvm_i16_ty),
                                       si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_RNDNE_F16_e64
def V_RNDNE_F16_t16_e64_sem : InstSISemantic<V_RNDNE_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_roundeven, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                                   si_llvm_i16_ty),
                                       si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_RSQ_F16_e64
def V_RSQ_F16_fake16_e64_sem : InstSISemantic<V_RSQ_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_rsq, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_RSQ_F16_e64
def V_RSQ_F16_t16_e64_sem : InstSISemantic<V_RSQ_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_rsq, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_SIN_F16_e64
def V_SIN_F16_fake16_e64_sem : InstSISemantic<V_SIN_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_sin, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_SIN_F16_e64
def V_SIN_F16_t16_e64_sem : InstSISemantic<V_SIN_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_sin, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_SQRT_F16_e64
def V_SQRT_F16_fake16_e64_sem : InstSISemantic<V_SQRT_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_sqrt, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_SQRT_F16_e64
def V_SQRT_F16_t16_e64_sem : InstSISemantic<V_SQRT_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_sqrt, [(GetNamedOperand $src0, si_llvm_half_ty)]))
  ];
}

// _t16/_fake16 alias of V_SUBREV_F16_e64
def V_SUBREV_F16_fake16_e64_sem : InstSISemantic<V_SUBREV_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFSub
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_SUBREV_F16_e64
def V_SUBREV_F16_t16_e64_sem : InstSISemantic<V_SUBREV_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFSub
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_SUB_F16_e64
def V_SUB_F16_fake16_e64_sem : InstSISemantic<V_SUB_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFSub
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_SUB_F16_e64
def V_SUB_F16_t16_e64_sem : InstSISemantic<V_SUB_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFSub
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_TRUNC_F16_e64
def V_TRUNC_F16_fake16_e64_sem : InstSISemantic<V_TRUNC_F16_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_trunc, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                                si_llvm_i16_ty),
                                    si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// _t16/_fake16 alias of V_TRUNC_F16_e64
def V_TRUNC_F16_t16_e64_sem : InstSISemantic<V_TRUNC_F16_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMUnaryIntrinsic int_trunc, (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                                si_llvm_i16_ty),
                                    si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

//==============================================================================
// V_ variant aliases (Phase 7): opsel / gfx<n> / additional t16+fake16
//==============================================================================

// alias of V_ALIGNBIT_B32_e64
def V_ALIGNBIT_B32_opsel_e64_sem : InstSISemantic<V_ALIGNBIT_B32_opsel_e64> {
  let Semantic = [
    (DefVal $concat,
      (LLVMOr (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                        (ConstantInt si_llvm_i64_ty, 32)),
              (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (SetNamedOperand $vdst,
      (LLVMTrunc
        (LLVMLShr (GetVal $concat),
                  (LLVMZExt (LLVMAnd (GetNamedOperand $src2),
                                     (ConstantInt si_llvm_i32_ty, 31)),
                            si_llvm_i64_ty)),
        si_llvm_i32_ty))
  ];
}

// alias of V_ALIGNBYTE_B32_e64
def V_ALIGNBYTE_B32_opsel_e64_sem : InstSISemantic<V_ALIGNBYTE_B32_opsel_e64> {
  let Semantic = [
    (DefVal $concat,
      (LLVMOr (LLVMShl (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                        (ConstantInt si_llvm_i64_ty, 32)),
              (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty))),
    (DefVal $shift,
      (LLVMShl (LLVMZExt (LLVMAnd (GetNamedOperand $src2),
                                   (ConstantInt si_llvm_i32_ty, 3)),
                          si_llvm_i64_ty),
               (ConstantInt si_llvm_i64_ty, 3))),
    (SetNamedOperand $vdst,
      (LLVMTrunc (LLVMLShr (GetVal $concat), (GetVal $shift)),
                 si_llvm_i32_ty))
  ];
}

// alias of V_ASHRREV_I16_e64
def V_ASHRREV_I16_opsel_e64_sem : InstSISemantic<V_ASHRREV_I16_opsel_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAShr (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                  (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                            (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// alias of V_DIV_FIXUP_F16_e64
def V_DIV_FIXUP_F16_gfx9_e64_sem : InstSISemantic<V_DIV_FIXUP_F16_gfx9_e64> {
  let Semantic = [];
}

// alias of V_DIV_FIXUP_F16_e64
def V_DIV_FIXUP_F16_gfx9_fake16_e64_sem : InstSISemantic<V_DIV_FIXUP_F16_gfx9_fake16_e64> {
  let Semantic = [];
}

// alias of V_DIV_FIXUP_F16_e64
def V_DIV_FIXUP_F16_gfx9_t16_e64_sem : InstSISemantic<V_DIV_FIXUP_F16_gfx9_t16_e64> {
  let Semantic = [];
}

// alias of V_FMA_F16_e64
def V_FMA_F16_gfx9_e64_sem : InstSISemantic<V_FMA_F16_gfx9_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFMA
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// alias of V_FMA_F16_e64
def V_FMA_F16_gfx9_fake16_e64_sem : InstSISemantic<V_FMA_F16_gfx9_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFMA
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// alias of V_FMA_F16_e64
def V_FMA_F16_gfx9_t16_e64_sem : InstSISemantic<V_FMA_F16_gfx9_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFMA
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                         si_llvm_half_ty),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// alias of V_LSHLREV_B16_e64
def V_LSHLREV_B16_opsel_e64_sem : InstSISemantic<V_LSHLREV_B16_opsel_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMShl (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                 (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// alias of V_LSHRREV_B16_e64
def V_LSHRREV_B16_opsel_e64_sem : InstSISemantic<V_LSHRREV_B16_opsel_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMLShr (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                  (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                            (ConstantInt si_llvm_i16_ty, 15))),
        si_llvm_i32_ty))
  ];
}

// alias of V_MAD_F16_e64
def V_MAD_F16_gfx9_e64_sem : InstSISemantic<V_MAD_F16_gfx9_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFAdd
            (LLVMFMul
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           si_llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty),
                           si_llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty),
                         si_llvm_half_ty)),
          si_llvm_i16_ty),
        si_llvm_i32_ty))
  ];
}

// alias of V_MAD_I16_e64
def V_MAD_I16_gfx9_e64_sem : InstSISemantic<V_MAD_I16_gfx9_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAdd (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
                 (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// alias of V_MAD_I16_e64
def V_MAD_I16_gfx9_fake16_e64_sem : InstSISemantic<V_MAD_I16_gfx9_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAdd (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
                 (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// alias of V_MAD_I16_e64
def V_MAD_I16_gfx9_t16_e64_sem : InstSISemantic<V_MAD_I16_gfx9_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAdd (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
                 (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// alias of V_MAD_I64_I32_e64
def V_MAD_I64_I32_gfx11_e64_sem : InstSISemantic<V_MAD_I64_I32_gfx11_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMSExt (GetNamedOperand $src0), si_llvm_i64_ty),
                         (LLVMSExt (GetNamedOperand $src1), si_llvm_i64_ty)),
               (GetNamedOperand $src2)))
  ];
}

// alias of V_MAD_U16_e64
def V_MAD_U16_gfx9_e64_sem : InstSISemantic<V_MAD_U16_gfx9_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAdd (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
                 (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// alias of V_MAD_U16_e64
def V_MAD_U16_gfx9_fake16_e64_sem : InstSISemantic<V_MAD_U16_gfx9_fake16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAdd (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
                 (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// alias of V_MAD_U16_e64
def V_MAD_U16_gfx9_t16_e64_sem : InstSISemantic<V_MAD_U16_gfx9_t16_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAdd (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                           (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
                 (LLVMTrunc (GetNamedOperand $src2), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

// alias of V_MAD_U64_U32_e64
def V_MAD_U64_U32_gfx11_e64_sem : InstSISemantic<V_MAD_U64_U32_gfx11_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMMul (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
                         (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty)),
               (GetNamedOperand $src2)))
  ];
}

// alias of V_MAX_I16_e64
def V_MAX_I16_opsel_e64_sem : InstSISemantic<V_MAX_I16_opsel_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSGT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// alias of V_MAX_U16_e64
def V_MAX_U16_opsel_e64_sem : InstSISemantic<V_MAX_U16_opsel_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpUGT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// alias of V_MIN_I16_e64
def V_MIN_I16_opsel_e64_sem : InstSISemantic<V_MIN_I16_opsel_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpSLT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// alias of V_MIN_U16_e64
def V_MIN_U16_opsel_e64_sem : InstSISemantic<V_MIN_U16_opsel_e64> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp LLVMICmpULT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        si_llvm_i32_ty))
  ];
}

// alias of V_MUL_LO_U16_e64
def V_MUL_LO_U16_opsel_e64_sem : InstSISemantic<V_MUL_LO_U16_opsel_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMMul (LLVMTrunc (GetNamedOperand $src0), si_llvm_i16_ty),
                 (LLVMTrunc (GetNamedOperand $src1), si_llvm_i16_ty)),
        si_llvm_i32_ty))
  ];
}

//==============================================================================
// V_ family record-extracted aliases (Phase 7)
// Bodies extracted from expanded tablegen records (--print-records) for ops
// whose base semantics were defined via foreach indirection.
//==============================================================================

// alias of V_ADD_NC_U16_e64
def V_ADD_NC_U16_fake16_e64_sem : InstSISemantic<V_ADD_NC_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMAdd (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)), si_llvm_i32_ty))];
}

// alias of V_ADD_NC_U16_e64
def V_ADD_NC_U16_t16_e64_sem : InstSISemantic<V_ADD_NC_U16_t16_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMAdd (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)), si_llvm_i32_ty))];
}

// alias of V_CMPX_CLASS_F16_e32
def V_CMPX_CLASS_F16_fake16_e32_sem : InstSISemantic<V_CMPX_CLASS_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMPX_CLASS_F16_e64
def V_CMPX_CLASS_F16_fake16_e64_sem : InstSISemantic<V_CMPX_CLASS_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMIntrinsic si_llvm_i1_ty, int_amdgcn_class, [(GetNamedOperand ?:$src0), (GetNamedOperand ?:$src1)]))];
}

// alias of V_CMPX_EQ_F16_e32
def V_CMPX_EQ_F16_fake16_e32_sem : InstSISemantic<V_CMPX_EQ_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpOEQ, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_EQ_F16_e64
def V_CMPX_EQ_F16_fake16_e64_sem : InstSISemantic<V_CMPX_EQ_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpOEQ, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_EQ_I16_e32
def V_CMPX_EQ_I16_fake16_e32_sem : InstSISemantic<V_CMPX_EQ_I16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_EQ_I16_e64
def V_CMPX_EQ_I16_fake16_e64_sem : InstSISemantic<V_CMPX_EQ_I16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_EQ_U16_e32
def V_CMPX_EQ_U16_fake16_e32_sem : InstSISemantic<V_CMPX_EQ_U16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_EQ_U16_e64
def V_CMPX_EQ_U16_fake16_e64_sem : InstSISemantic<V_CMPX_EQ_U16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_F_F16_e32
def V_CMPX_F_F16_fake16_e32_sem : InstSISemantic<V_CMPX_F_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMPX_F_F16_e64
def V_CMPX_F_F16_fake16_e64_sem : InstSISemantic<V_CMPX_F_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantFalse si_llvm_i1_ty))];
}

// alias of V_CMPX_F_I16_e32
def V_CMPX_F_I16_fake16_e32_sem : InstSISemantic<V_CMPX_F_I16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMPX_F_I16_e64
def V_CMPX_F_I16_fake16_e64_sem : InstSISemantic<V_CMPX_F_I16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantFalse si_llvm_i1_ty))];
}

// alias of V_CMPX_F_U16_e32
def V_CMPX_F_U16_fake16_e32_sem : InstSISemantic<V_CMPX_F_U16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMPX_F_U16_e64
def V_CMPX_F_U16_fake16_e64_sem : InstSISemantic<V_CMPX_F_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantFalse si_llvm_i1_ty))];
}

// alias of V_CMPX_GE_F16_e32
def V_CMPX_GE_F16_fake16_e32_sem : InstSISemantic<V_CMPX_GE_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpOGE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_GE_F16_e64
def V_CMPX_GE_F16_fake16_e64_sem : InstSISemantic<V_CMPX_GE_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpOGE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_GE_I16_e32
def V_CMPX_GE_I16_fake16_e32_sem : InstSISemantic<V_CMPX_GE_I16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpSGE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_GE_I16_e64
def V_CMPX_GE_I16_fake16_e64_sem : InstSISemantic<V_CMPX_GE_I16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpSGE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_GE_U16_e32
def V_CMPX_GE_U16_fake16_e32_sem : InstSISemantic<V_CMPX_GE_U16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpUGE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_GE_U16_e64
def V_CMPX_GE_U16_fake16_e64_sem : InstSISemantic<V_CMPX_GE_U16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpUGE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_GT_F16_e32
def V_CMPX_GT_F16_fake16_e32_sem : InstSISemantic<V_CMPX_GT_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpOGT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_GT_F16_e64
def V_CMPX_GT_F16_fake16_e64_sem : InstSISemantic<V_CMPX_GT_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpOGT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_GT_I16_e32
def V_CMPX_GT_I16_fake16_e32_sem : InstSISemantic<V_CMPX_GT_I16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpSGT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_GT_I16_e64
def V_CMPX_GT_I16_fake16_e64_sem : InstSISemantic<V_CMPX_GT_I16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpSGT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_GT_U16_e32
def V_CMPX_GT_U16_fake16_e32_sem : InstSISemantic<V_CMPX_GT_U16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpUGT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_GT_U16_e64
def V_CMPX_GT_U16_fake16_e64_sem : InstSISemantic<V_CMPX_GT_U16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpUGT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_LE_F16_e32
def V_CMPX_LE_F16_fake16_e32_sem : InstSISemantic<V_CMPX_LE_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpOLE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_LE_F16_e64
def V_CMPX_LE_F16_fake16_e64_sem : InstSISemantic<V_CMPX_LE_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpOLE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_LE_I16_e32
def V_CMPX_LE_I16_fake16_e32_sem : InstSISemantic<V_CMPX_LE_I16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpSLE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_LE_I16_e64
def V_CMPX_LE_I16_fake16_e64_sem : InstSISemantic<V_CMPX_LE_I16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpSLE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_LE_U16_e32
def V_CMPX_LE_U16_fake16_e32_sem : InstSISemantic<V_CMPX_LE_U16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpULE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_LE_U16_e64
def V_CMPX_LE_U16_fake16_e64_sem : InstSISemantic<V_CMPX_LE_U16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpULE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_LG_F16_e32
def V_CMPX_LG_F16_fake16_e32_sem : InstSISemantic<V_CMPX_LG_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpONE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_LG_F16_e64
def V_CMPX_LG_F16_fake16_e64_sem : InstSISemantic<V_CMPX_LG_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpONE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_LT_F16_e32
def V_CMPX_LT_F16_fake16_e32_sem : InstSISemantic<V_CMPX_LT_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpOLT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_LT_F16_e64
def V_CMPX_LT_F16_fake16_e64_sem : InstSISemantic<V_CMPX_LT_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpOLT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_LT_I16_e32
def V_CMPX_LT_I16_fake16_e32_sem : InstSISemantic<V_CMPX_LT_I16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpSLT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_LT_I16_e64
def V_CMPX_LT_I16_fake16_e64_sem : InstSISemantic<V_CMPX_LT_I16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpSLT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_LT_U16_e32
def V_CMPX_LT_U16_fake16_e32_sem : InstSISemantic<V_CMPX_LT_U16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpULT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_LT_U16_e64
def V_CMPX_LT_U16_fake16_e64_sem : InstSISemantic<V_CMPX_LT_U16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpULT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_NEQ_F16_e32
def V_CMPX_NEQ_F16_fake16_e32_sem : InstSISemantic<V_CMPX_NEQ_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpUNE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_NEQ_F16_e64
def V_CMPX_NEQ_F16_fake16_e64_sem : InstSISemantic<V_CMPX_NEQ_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpUNE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_NE_I16_e32
def V_CMPX_NE_I16_fake16_e32_sem : InstSISemantic<V_CMPX_NE_I16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_NE_I16_e64
def V_CMPX_NE_I16_fake16_e64_sem : InstSISemantic<V_CMPX_NE_I16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_NE_U16_e32
def V_CMPX_NE_U16_fake16_e32_sem : InstSISemantic<V_CMPX_NE_U16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMPX_NE_U16_e64
def V_CMPX_NE_U16_fake16_e64_sem : InstSISemantic<V_CMPX_NE_U16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_NGE_F16_e32
def V_CMPX_NGE_F16_fake16_e32_sem : InstSISemantic<V_CMPX_NGE_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpULT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_NGE_F16_e64
def V_CMPX_NGE_F16_fake16_e64_sem : InstSISemantic<V_CMPX_NGE_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpULT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_NGT_F16_e32
def V_CMPX_NGT_F16_fake16_e32_sem : InstSISemantic<V_CMPX_NGT_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpULE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_NGT_F16_e64
def V_CMPX_NGT_F16_fake16_e64_sem : InstSISemantic<V_CMPX_NGT_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpULE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_NLE_F16_e32
def V_CMPX_NLE_F16_fake16_e32_sem : InstSISemantic<V_CMPX_NLE_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpUGT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_NLE_F16_e64
def V_CMPX_NLE_F16_fake16_e64_sem : InstSISemantic<V_CMPX_NLE_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpUGT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_NLG_F16_e32
def V_CMPX_NLG_F16_fake16_e32_sem : InstSISemantic<V_CMPX_NLG_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpUEQ, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_NLG_F16_e64
def V_CMPX_NLG_F16_fake16_e64_sem : InstSISemantic<V_CMPX_NLG_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpUEQ, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_NLT_F16_e32
def V_CMPX_NLT_F16_fake16_e32_sem : InstSISemantic<V_CMPX_NLT_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpUGE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_NLT_F16_e64
def V_CMPX_NLT_F16_fake16_e64_sem : InstSISemantic<V_CMPX_NLT_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$result, (LLVMFCmp LLVMFCmpUGE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$result))];
}

// alias of V_CMPX_O_F16_e32
def V_CMPX_O_F16_fake16_e32_sem : InstSISemantic<V_CMPX_O_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMFCmp LLVMFCmpOLE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_O_F16_e64
def V_CMPX_O_F16_fake16_e64_sem : InstSISemantic<V_CMPX_O_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMFCmp LLVMFCmpOLE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMPX_TRU_F16_e32
def V_CMPX_TRU_F16_fake16_e32_sem : InstSISemantic<V_CMPX_TRU_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMPX_TRU_F16_e64
def V_CMPX_TRU_F16_fake16_e64_sem : InstSISemantic<V_CMPX_TRU_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantTrue si_llvm_i1_ty))];
}

// alias of V_CMPX_T_I16_e32
def V_CMPX_T_I16_fake16_e32_sem : InstSISemantic<V_CMPX_T_I16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMPX_T_I16_e64
def V_CMPX_T_I16_fake16_e64_sem : InstSISemantic<V_CMPX_T_I16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantTrue si_llvm_i1_ty))];
}

// alias of V_CMPX_T_U16_e32
def V_CMPX_T_U16_fake16_e32_sem : InstSISemantic<V_CMPX_T_U16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMPX_T_U16_e64
def V_CMPX_T_U16_fake16_e64_sem : InstSISemantic<V_CMPX_T_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantTrue si_llvm_i1_ty))];
}

// alias of V_CMPX_U_F16_e32
def V_CMPX_U_F16_fake16_e32_sem : InstSISemantic<V_CMPX_U_F16_fake16_e32> {
  let Semantic = [(DefVal ?:$r, (LLVMFCmp LLVMFCmpUNO, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMPX_U_F16_e64
def V_CMPX_U_F16_fake16_e64_sem : InstSISemantic<V_CMPX_U_F16_fake16_e64> {
  let Semantic = [(DefVal ?:$r, (LLVMFCmp LLVMFCmpUNO, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty))), (SetNamedOperand ?:$sdst, (GetVal ?:$r))];
}

// alias of V_CMP_CLASS_F16_e32
def V_CMP_CLASS_F16_fake16_e32_sem : InstSISemantic<V_CMP_CLASS_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_CLASS_F16_e64
def V_CMP_CLASS_F16_fake16_e64_sem : InstSISemantic<V_CMP_CLASS_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMIntrinsic si_llvm_i1_ty, int_amdgcn_class, [(GetNamedOperand ?:$src0), (GetNamedOperand ?:$src1)]))];
}

// alias of V_CMP_EQ_F16_e32
def V_CMP_EQ_F16_fake16_e32_sem : InstSISemantic<V_CMP_EQ_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_EQ_F16_e64
def V_CMP_EQ_F16_fake16_e64_sem : InstSISemantic<V_CMP_EQ_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpOEQ, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_EQ_I16_e32
def V_CMP_EQ_I16_fake16_e32_sem : InstSISemantic<V_CMP_EQ_I16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_EQ_I16_e64
def V_CMP_EQ_I16_fake16_e64_sem : InstSISemantic<V_CMP_EQ_I16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_EQ_U16_e32
def V_CMP_EQ_U16_fake16_e32_sem : InstSISemantic<V_CMP_EQ_U16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_EQ_U16_e64
def V_CMP_EQ_U16_fake16_e64_sem : InstSISemantic<V_CMP_EQ_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_F_F16_e32
def V_CMP_F_F16_fake16_e32_sem : InstSISemantic<V_CMP_F_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_F_F16_e64
def V_CMP_F_F16_fake16_e64_sem : InstSISemantic<V_CMP_F_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantFalse si_llvm_i1_ty))];
}

// alias of V_CMP_F_I16_e32
def V_CMP_F_I16_fake16_e32_sem : InstSISemantic<V_CMP_F_I16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_F_I16_e64
def V_CMP_F_I16_fake16_e64_sem : InstSISemantic<V_CMP_F_I16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantFalse si_llvm_i1_ty))];
}

// alias of V_CMP_F_U16_e32
def V_CMP_F_U16_fake16_e32_sem : InstSISemantic<V_CMP_F_U16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_F_U16_e64
def V_CMP_F_U16_fake16_e64_sem : InstSISemantic<V_CMP_F_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantFalse si_llvm_i1_ty))];
}

// alias of V_CMP_GE_F16_e32
def V_CMP_GE_F16_fake16_e32_sem : InstSISemantic<V_CMP_GE_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_GE_F16_e64
def V_CMP_GE_F16_fake16_e64_sem : InstSISemantic<V_CMP_GE_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpOGE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_GE_I16_e32
def V_CMP_GE_I16_fake16_e32_sem : InstSISemantic<V_CMP_GE_I16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_GE_I16_e64
def V_CMP_GE_I16_fake16_e64_sem : InstSISemantic<V_CMP_GE_I16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpSGE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_GE_U16_e32
def V_CMP_GE_U16_fake16_e32_sem : InstSISemantic<V_CMP_GE_U16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_GE_U16_e64
def V_CMP_GE_U16_fake16_e64_sem : InstSISemantic<V_CMP_GE_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpUGE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_GT_F16_e32
def V_CMP_GT_F16_fake16_e32_sem : InstSISemantic<V_CMP_GT_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_GT_F16_e64
def V_CMP_GT_F16_fake16_e64_sem : InstSISemantic<V_CMP_GT_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpOGT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_GT_I16_e32
def V_CMP_GT_I16_fake16_e32_sem : InstSISemantic<V_CMP_GT_I16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_GT_I16_e64
def V_CMP_GT_I16_fake16_e64_sem : InstSISemantic<V_CMP_GT_I16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpSGT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_GT_U16_e32
def V_CMP_GT_U16_fake16_e32_sem : InstSISemantic<V_CMP_GT_U16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_GT_U16_e64
def V_CMP_GT_U16_fake16_e64_sem : InstSISemantic<V_CMP_GT_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpUGT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_LE_F16_e32
def V_CMP_LE_F16_fake16_e32_sem : InstSISemantic<V_CMP_LE_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_LE_F16_e64
def V_CMP_LE_F16_fake16_e64_sem : InstSISemantic<V_CMP_LE_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpOLE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_LE_I16_e32
def V_CMP_LE_I16_fake16_e32_sem : InstSISemantic<V_CMP_LE_I16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_LE_I16_e64
def V_CMP_LE_I16_fake16_e64_sem : InstSISemantic<V_CMP_LE_I16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpSLE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_LE_U16_e32
def V_CMP_LE_U16_fake16_e32_sem : InstSISemantic<V_CMP_LE_U16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_LE_U16_e64
def V_CMP_LE_U16_fake16_e64_sem : InstSISemantic<V_CMP_LE_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpULE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_LG_F16_e32
def V_CMP_LG_F16_fake16_e32_sem : InstSISemantic<V_CMP_LG_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_LG_F16_e64
def V_CMP_LG_F16_fake16_e64_sem : InstSISemantic<V_CMP_LG_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpONE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_LT_F16_e32
def V_CMP_LT_F16_fake16_e32_sem : InstSISemantic<V_CMP_LT_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_LT_F16_e64
def V_CMP_LT_F16_fake16_e64_sem : InstSISemantic<V_CMP_LT_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpOLT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_LT_I16_e32
def V_CMP_LT_I16_fake16_e32_sem : InstSISemantic<V_CMP_LT_I16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_LT_I16_e64
def V_CMP_LT_I16_fake16_e64_sem : InstSISemantic<V_CMP_LT_I16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpSLT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_LT_U16_e32
def V_CMP_LT_U16_fake16_e32_sem : InstSISemantic<V_CMP_LT_U16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_LT_U16_e64
def V_CMP_LT_U16_fake16_e64_sem : InstSISemantic<V_CMP_LT_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpULT, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_NEQ_F16_e32
def V_CMP_NEQ_F16_fake16_e32_sem : InstSISemantic<V_CMP_NEQ_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_NEQ_F16_e64
def V_CMP_NEQ_F16_fake16_e64_sem : InstSISemantic<V_CMP_NEQ_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpUNE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_NE_I16_e32
def V_CMP_NE_I16_fake16_e32_sem : InstSISemantic<V_CMP_NE_I16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_NE_I16_e64
def V_CMP_NE_I16_fake16_e64_sem : InstSISemantic<V_CMP_NE_I16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_NE_U16_e32
def V_CMP_NE_U16_fake16_e32_sem : InstSISemantic<V_CMP_NE_U16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_NE_U16_e64
def V_CMP_NE_U16_fake16_e64_sem : InstSISemantic<V_CMP_NE_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)))];
}

// alias of V_CMP_NGE_F16_e32
def V_CMP_NGE_F16_fake16_e32_sem : InstSISemantic<V_CMP_NGE_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_NGE_F16_e64
def V_CMP_NGE_F16_fake16_e64_sem : InstSISemantic<V_CMP_NGE_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpULT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_NGT_F16_e32
def V_CMP_NGT_F16_fake16_e32_sem : InstSISemantic<V_CMP_NGT_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_NGT_F16_e64
def V_CMP_NGT_F16_fake16_e64_sem : InstSISemantic<V_CMP_NGT_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpULE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_NLE_F16_e32
def V_CMP_NLE_F16_fake16_e32_sem : InstSISemantic<V_CMP_NLE_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_NLE_F16_e64
def V_CMP_NLE_F16_fake16_e64_sem : InstSISemantic<V_CMP_NLE_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpUGT, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_NLG_F16_e32
def V_CMP_NLG_F16_fake16_e32_sem : InstSISemantic<V_CMP_NLG_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_NLG_F16_e64
def V_CMP_NLG_F16_fake16_e64_sem : InstSISemantic<V_CMP_NLG_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpUEQ, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_NLT_F16_e32
def V_CMP_NLT_F16_fake16_e32_sem : InstSISemantic<V_CMP_NLT_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_NLT_F16_e64
def V_CMP_NLT_F16_fake16_e64_sem : InstSISemantic<V_CMP_NLT_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpUGE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_O_F16_e32
def V_CMP_O_F16_fake16_e32_sem : InstSISemantic<V_CMP_O_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_O_F16_e64
def V_CMP_O_F16_fake16_e64_sem : InstSISemantic<V_CMP_O_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpOLE, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CMP_TRU_F16_e32
def V_CMP_TRU_F16_fake16_e32_sem : InstSISemantic<V_CMP_TRU_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_TRU_F16_e64
def V_CMP_TRU_F16_fake16_e64_sem : InstSISemantic<V_CMP_TRU_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantTrue si_llvm_i1_ty))];
}

// alias of V_CMP_T_I16_e32
def V_CMP_T_I16_fake16_e32_sem : InstSISemantic<V_CMP_T_I16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_T_I16_e64
def V_CMP_T_I16_fake16_e64_sem : InstSISemantic<V_CMP_T_I16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantTrue si_llvm_i1_ty))];
}

// alias of V_CMP_T_U16_e32
def V_CMP_T_U16_fake16_e32_sem : InstSISemantic<V_CMP_T_U16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_T_U16_e64
def V_CMP_T_U16_fake16_e64_sem : InstSISemantic<V_CMP_T_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (ConstantTrue si_llvm_i1_ty))];
}

// alias of V_CMP_U_F16_e32
def V_CMP_U_F16_fake16_e32_sem : InstSISemantic<V_CMP_U_F16_fake16_e32> {
  let Semantic = [];
}

// alias of V_CMP_U_F16_e64
def V_CMP_U_F16_fake16_e64_sem : InstSISemantic<V_CMP_U_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$sdst, (LLVMFCmp LLVMFCmpUNO, (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty)))];
}

// alias of V_CVT_F16_F32_e32
def V_CVT_F16_F32_fake16_e32_sem : InstSISemantic<V_CVT_F16_F32_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMBitCast (LLVMFPTrunc (LLVMBitCast (GetNamedOperand ?:$src0), si_llvm_float_ty), si_llvm_half_ty), si_llvm_i16_ty), si_llvm_i32_ty))];
}

// alias of V_CVT_F16_I16_e32
def V_CVT_F16_I16_fake16_e32_sem : InstSISemantic<V_CVT_F16_I16_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMBitCast (LLVMSIToFP (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), si_llvm_i16_ty), si_llvm_i32_ty))];
}

// alias of V_CVT_F16_U16_e32
def V_CVT_F16_U16_fake16_e32_sem : InstSISemantic<V_CVT_F16_U16_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMBitCast (LLVMUIToFP (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), si_llvm_i16_ty), si_llvm_i32_ty))];
}

// alias of V_CVT_F32_F16_e32
def V_CVT_F32_F16_fake16_e32_sem : InstSISemantic<V_CVT_F32_F16_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMBitCast (LLVMFPExt (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), si_llvm_float_ty), si_llvm_i32_ty))];
}

// alias of V_CVT_F32_FP8_e64
def V_CVT_F32_FP8_gfx1250_e64_sem : InstSISemantic<V_CVT_F32_FP8_gfx1250_e64> {
  let Semantic = [];
}

// alias of V_CVT_I16_F16_e32
def V_CVT_I16_F16_fake16_e32_sem : InstSISemantic<V_CVT_I16_F16_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMFPToSI (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), si_llvm_i16_ty), si_llvm_i32_ty))];
}

// alias of V_CVT_NORM_I16_F16_e32
def V_CVT_NORM_I16_F16_fake16_e32_sem : InstSISemantic<V_CVT_NORM_I16_F16_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_pknorm_i16, [(GetNamedOperand ?:$src0)]))];
}

// alias of V_CVT_NORM_U16_F16_e32
def V_CVT_NORM_U16_F16_fake16_e32_sem : InstSISemantic<V_CVT_NORM_U16_F16_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_pknorm_u16, [(GetNamedOperand ?:$src0)]))];
}

// alias of V_CVT_PKNORM_I16_F16_e64
def V_CVT_PKNORM_I16_F16_fake16_e64_sem : InstSISemantic<V_CVT_PKNORM_I16_F16_fake16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PKNORM_I16_F16_e64
def V_CVT_PKNORM_I16_F16_t16_e64_sem : InstSISemantic<V_CVT_PKNORM_I16_F16_t16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PKNORM_U16_F16_e64
def V_CVT_PKNORM_U16_F16_fake16_e64_sem : InstSISemantic<V_CVT_PKNORM_U16_F16_fake16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PKNORM_U16_F16_e64
def V_CVT_PKNORM_U16_F16_t16_e64_sem : InstSISemantic<V_CVT_PKNORM_U16_F16_t16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_BF8_F32_e64
def V_CVT_PK_BF8_F32_fake16_e64_sem : InstSISemantic<V_CVT_PK_BF8_F32_fake16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_BF8_F32_e64
def V_CVT_PK_BF8_F32_t16_e64_sem : InstSISemantic<V_CVT_PK_BF8_F32_t16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_F32_BF8_e64
def V_CVT_PK_F32_BF8_fake16_e64_sem : InstSISemantic<V_CVT_PK_F32_BF8_fake16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_F32_BF8_e64
def V_CVT_PK_F32_BF8_t16_e64_sem : InstSISemantic<V_CVT_PK_F32_BF8_t16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_F32_FP8_e64
def V_CVT_PK_F32_FP8_fake16_e64_sem : InstSISemantic<V_CVT_PK_F32_FP8_fake16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_F32_FP8_e64
def V_CVT_PK_F32_FP8_t16_e64_sem : InstSISemantic<V_CVT_PK_F32_FP8_t16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_FP8_F32_e64
def V_CVT_PK_FP8_F32_fake16_e64_sem : InstSISemantic<V_CVT_PK_FP8_F32_fake16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_FP8_F32_e64
def V_CVT_PK_FP8_F32_gfx1250_e64_sem : InstSISemantic<V_CVT_PK_FP8_F32_gfx1250_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_FP8_F32_e64
def V_CVT_PK_FP8_F32_gfx1250_fake16_e64_sem : InstSISemantic<V_CVT_PK_FP8_F32_gfx1250_fake16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_FP8_F32_e64
def V_CVT_PK_FP8_F32_gfx1250_t16_e64_sem : InstSISemantic<V_CVT_PK_FP8_F32_gfx1250_t16_e64> {
  let Semantic = [];
}

// alias of V_CVT_PK_FP8_F32_e64
def V_CVT_PK_FP8_F32_t16_e64_sem : InstSISemantic<V_CVT_PK_FP8_F32_t16_e64> {
  let Semantic = [];
}

// alias of V_CVT_SR_FP8_F32_e64
def V_CVT_SR_FP8_F32_gfx1250_e64_sem : InstSISemantic<V_CVT_SR_FP8_F32_gfx1250_e64> {
  let Semantic = [];
}

// alias of V_CVT_U16_F16_e32
def V_CVT_U16_F16_fake16_e32_sem : InstSISemantic<V_CVT_U16_F16_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMFPToUI (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), si_llvm_i16_ty), si_llvm_i32_ty))];
}

// alias of V_DOT2_BF16_BF16_e64
def V_DOT2_BF16_BF16_fake16_e64_sem : InstSISemantic<V_DOT2_BF16_BF16_fake16_e64> {
  let Semantic = [];
}

// alias of V_DOT2_BF16_BF16_e64
def V_DOT2_BF16_BF16_t16_e64_sem : InstSISemantic<V_DOT2_BF16_BF16_t16_e64> {
  let Semantic = [];
}

// alias of V_DOT2_F16_F16_e64
def V_DOT2_F16_F16_fake16_e64_sem : InstSISemantic<V_DOT2_F16_F16_fake16_e64> {
  let Semantic = [];
}

// alias of V_DOT2_F16_F16_e64
def V_DOT2_F16_F16_t16_e64_sem : InstSISemantic<V_DOT2_F16_F16_t16_e64> {
  let Semantic = [];
}

// alias of V_FMAC_F16_e32
def V_FMAC_F16_fake16_e32_sem : InstSISemantic<V_FMAC_F16_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMBitCast (LLVMFMA (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty), si_llvm_half_ty), (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src2), si_llvm_i16_ty), si_llvm_half_ty)), si_llvm_i16_ty), si_llvm_i32_ty))];
}

// alias of V_FREXP_EXP_I16_F16_e32
def V_FREXP_EXP_I16_F16_fake16_e32_sem : InstSISemantic<V_FREXP_EXP_I16_F16_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_frexp_exp, [(GetNamedOperand ?:$src0, si_llvm_half_ty)]))];
}

// alias of V_LDEXP_F16_e32
def V_LDEXP_F16_fake16_e32_sem : InstSISemantic<V_LDEXP_F16_fake16_e32> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMBitCast (LLVMLdexp (LLVMBitCast (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), si_llvm_half_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)), si_llvm_i16_ty), si_llvm_i32_ty))];
}

// alias of V_MAXIMUM3_F16_e64
def V_MAXIMUM3_F16_fake16_e64_sem : InstSISemantic<V_MAXIMUM3_F16_fake16_e64> {
  let Semantic = [];
}

// alias of V_MAXIMUM3_F16_e64
def V_MAXIMUM3_F16_t16_e64_sem : InstSISemantic<V_MAXIMUM3_F16_t16_e64> {
  let Semantic = [];
}

// alias of V_MAXIMUMMINIMUM_F16_e64
def V_MAXIMUMMINIMUM_F16_fake16_e64_sem : InstSISemantic<V_MAXIMUMMINIMUM_F16_fake16_e64> {
  let Semantic = [];
}

// alias of V_MAXIMUMMINIMUM_F16_e64
def V_MAXIMUMMINIMUM_F16_t16_e64_sem : InstSISemantic<V_MAXIMUMMINIMUM_F16_t16_e64> {
  let Semantic = [];
}

// alias of V_MAXIMUM_F16_e64
def V_MAXIMUM_F16_fake16_e64_sem : InstSISemantic<V_MAXIMUM_F16_fake16_e64> {
  let Semantic = [];
}

// alias of V_MAXIMUM_F16_e64
def V_MAXIMUM_F16_t16_e64_sem : InstSISemantic<V_MAXIMUM_F16_t16_e64> {
  let Semantic = [];
}

// alias of V_MINIMUM3_F16_e64
def V_MINIMUM3_F16_fake16_e64_sem : InstSISemantic<V_MINIMUM3_F16_fake16_e64> {
  let Semantic = [];
}

// alias of V_MINIMUM3_F16_e64
def V_MINIMUM3_F16_t16_e64_sem : InstSISemantic<V_MINIMUM3_F16_t16_e64> {
  let Semantic = [];
}

// alias of V_PACK_B32_F16_e64
def V_PACK_B32_F16_fake16_e64_sem : InstSISemantic<V_PACK_B32_F16_fake16_e64> {
  let Semantic = [];
}

// alias of V_PACK_B32_F16_e64
def V_PACK_B32_F16_t16_e64_sem : InstSISemantic<V_PACK_B32_F16_t16_e64> {
  let Semantic = [];
}

// alias of V_SAT_PK_U8_I16_e64
def V_SAT_PK_U8_I16_fake16_e64_sem : InstSISemantic<V_SAT_PK_U8_I16_fake16_e64> {
  let Semantic = [];
}

// alias of V_SAT_PK_U8_I16_e64
def V_SAT_PK_U8_I16_t16_e64_sem : InstSISemantic<V_SAT_PK_U8_I16_t16_e64> {
  let Semantic = [];
}

// alias of V_SUB_NC_U16_e64
def V_SUB_NC_U16_fake16_e64_sem : InstSISemantic<V_SUB_NC_U16_fake16_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMSub (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)), si_llvm_i32_ty))];
}

// alias of V_SUB_NC_U16_e64
def V_SUB_NC_U16_t16_e64_sem : InstSISemantic<V_SUB_NC_U16_t16_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMZExt (LLVMSub (LLVMTrunc (GetNamedOperand ?:$src0), si_llvm_i16_ty), (LLVMTrunc (GetNamedOperand ?:$src1), si_llvm_i16_ty)), si_llvm_i32_ty))];
}

//==============================================================================
// V_ cross-product alias variants (Phase 8 generic)
//==============================================================================

// alias of V_CVT_SR_BF8_F32_e64
def V_CVT_SR_BF8_F32_gfx12_e64_sem : InstSISemantic<V_CVT_SR_BF8_F32_gfx12_e64> {
  let Semantic = [];
}

// alias of V_CVT_SR_FP8_F32_e64
def V_CVT_SR_FP8_F32_gfx12_e64_sem : InstSISemantic<V_CVT_SR_FP8_F32_gfx12_e64> {
  let Semantic = [];
}

//==============================================================================
// V_ cross-product alias variants (Phase 8 generic)
//==============================================================================

// alias of V_CMPX_CLASS_F16_nosdst_e32
def V_CMPX_CLASS_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_CLASS_F16_fake16_nosdst_e32> {
  let Semantic = [(LLVMIntrinsic si_llvm_i1_ty, int_amdgcn_class, [(GetNamedOperand ?:$src0), (GetNamedOperand ?:$src1)])];
}

// alias of V_CMPX_EQ_F16_nosdst_e32
def V_CMPX_EQ_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_EQ_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_EQ_I16_nosdst_e32
def V_CMPX_EQ_I16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_EQ_I16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_EQ_U16_nosdst_e32
def V_CMPX_EQ_U16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_EQ_U16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_F_F16_nosdst_e32
def V_CMPX_F_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_F_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_F_I16_nosdst_e32
def V_CMPX_F_I16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_F_I16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_F_U16_nosdst_e32
def V_CMPX_F_U16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_F_U16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_GE_F16_nosdst_e32
def V_CMPX_GE_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_GE_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_GE_I16_nosdst_e32
def V_CMPX_GE_I16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_GE_I16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_GE_U16_nosdst_e32
def V_CMPX_GE_U16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_GE_U16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_GT_F16_nosdst_e32
def V_CMPX_GT_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_GT_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_GT_I16_nosdst_e32
def V_CMPX_GT_I16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_GT_I16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_GT_U16_nosdst_e32
def V_CMPX_GT_U16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_GT_U16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_LE_F16_nosdst_e32
def V_CMPX_LE_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_LE_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_LE_I16_nosdst_e32
def V_CMPX_LE_I16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_LE_I16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_LE_U16_nosdst_e32
def V_CMPX_LE_U16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_LE_U16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_LG_F16_nosdst_e32
def V_CMPX_LG_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_LG_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_LT_F16_nosdst_e32
def V_CMPX_LT_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_LT_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_LT_I16_nosdst_e32
def V_CMPX_LT_I16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_LT_I16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_LT_U16_nosdst_e32
def V_CMPX_LT_U16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_LT_U16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_NEQ_F16_nosdst_e32
def V_CMPX_NEQ_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_NEQ_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_NE_I16_nosdst_e32
def V_CMPX_NE_I16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_NE_I16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_NE_U16_nosdst_e32
def V_CMPX_NE_U16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_NE_U16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_NGE_F16_nosdst_e32
def V_CMPX_NGE_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_NGE_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_NGT_F16_nosdst_e32
def V_CMPX_NGT_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_NGT_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_NLE_F16_nosdst_e32
def V_CMPX_NLE_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_NLE_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_NLG_F16_nosdst_e32
def V_CMPX_NLG_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_NLG_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_NLT_F16_nosdst_e32
def V_CMPX_NLT_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_NLT_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_O_F16_nosdst_e32
def V_CMPX_O_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_O_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_TRU_F16_nosdst_e32
def V_CMPX_TRU_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_TRU_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_T_I16_nosdst_e32
def V_CMPX_T_I16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_T_I16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_T_U16_nosdst_e32
def V_CMPX_T_U16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_T_U16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_CMPX_U_F16_nosdst_e32
def V_CMPX_U_F16_fake16_nosdst_e32_sem : InstSISemantic<V_CMPX_U_F16_fake16_nosdst_e32> {
  let Semantic = [];
}

// alias of V_INTERP_P2_F16
def V_INTERP_P2_F16_opsel_sem : InstSISemantic<V_INTERP_P2_F16_opsel> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_interp_p2_f16, [(GetNamedOperand ?:$src0), (GetNamedOperand ?:$attrchan), (GetNamedOperand ?:$attr), (GetNamedOperand ?:$high), (ImplicitUse M0)]))];
}

//==============================================================================
// V_ cross-product alias variants (Phase 8 generic)
//==============================================================================

// alias of V_ADD_F64_pseudo_e32
def V_ADD_F64_pseudo_e64_sem : InstSISemantic<V_ADD_F64_pseudo_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMBitCast (LLVMFAdd (LLVMBitCast (GetNamedOperand ?:$src0), si_llvm_double_ty), (LLVMBitCast (GetNamedOperand ?:$src1), si_llvm_double_ty)), si_llvm_i64_ty))];
}

// alias of V_CMPX_CLASS_F16_fake16_nosdst_e32
def V_CMPX_CLASS_F16_fake16_nosdst_e64_sem : InstSISemantic<V_CMPX_CLASS_F16_fake16_nosdst_e64> {
  let Semantic = [(LLVMIntrinsic si_llvm_i1_ty, int_amdgcn_class, [(GetNamedOperand ?:$src0), (GetNamedOperand ?:$src1)])];
}

// alias of V_CMPX_CLASS_F16_nosdst_e32
def V_CMPX_CLASS_F16_nosdst_e64_sem : InstSISemantic<V_CMPX_CLASS_F16_nosdst_e64> {
  let Semantic = [(LLVMIntrinsic si_llvm_i1_ty, int_amdgcn_class, [(GetNamedOperand ?:$src0), (GetNamedOperand ?:$src1)])];
}

// alias of V_CMPX_CLASS_F16_t16_nosdst_e32
def V_CMPX_CLASS_F16_t16_nosdst_e64_sem : InstSISemantic<V_CMPX_CLASS_F16_t16_nosdst_e64> {
  let Semantic = [(LLVMIntrinsic si_llvm_i1_ty, int_amdgcn_class, [(GetNamedOperand ?:$src0), (GetNamedOperand ?:$src1)])];
}

// alias of V_CMPX_CLASS_F32_nosdst_e32
def V_CMPX_CLASS_F32_nosdst_e64_sem : InstSISemantic<V_CMPX_CLASS_F32_nosdst_e64> {
  let Semantic = [(LLVMIntrinsic si_llvm_i1_ty, int_amdgcn_class, [(GetNamedOperand ?:$src0), (GetNamedOperand ?:$src1)])];
}

// alias of V_CMPX_CLASS_F64_nosdst_e32
def V_CMPX_CLASS_F64_nosdst_e64_sem : InstSISemantic<V_CMPX_CLASS_F64_nosdst_e64> {
  let Semantic = [(LLVMIntrinsic si_llvm_i1_ty, int_amdgcn_class, [(GetNamedOperand ?:$src0), (GetNamedOperand ?:$src1)])];
}

// alias of V_LSHLREV_B64_pseudo_e32
def V_LSHLREV_B64_pseudo_e64_sem : InstSISemantic<V_LSHLREV_B64_pseudo_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMShl (GetNamedOperand ?:$src1), (LLVMZExt (LLVMAnd (GetNamedOperand ?:$src0), (ConstantInt si_llvm_i32_ty, 63)), si_llvm_i64_ty)))];
}

// alias of V_MIN_NUM_F64_e32
def V_MIN_NUM_F64_e64_sem : InstSISemantic<V_MIN_NUM_F64_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMBitCast (LLVMMinNum (LLVMBitCast (GetNamedOperand ?:$src0), si_llvm_double_ty), (LLVMBitCast (GetNamedOperand ?:$src1), si_llvm_double_ty)), si_llvm_i64_ty))];
}

// alias of V_MOVRELD_B32_e32
def V_MOVRELD_B32_e64_sem : InstSISemantic<V_MOVRELD_B32_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (GetNamedOperand ?:$src0))];
}

// alias of V_MOVRELSD_2_B32_e32
def V_MOVRELSD_2_B32_e64_sem : InstSISemantic<V_MOVRELSD_2_B32_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (GetNamedOperand ?:$src0))];
}

// alias of V_MOVRELSD_B32_e32
def V_MOVRELSD_B32_e64_sem : InstSISemantic<V_MOVRELSD_B32_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (GetNamedOperand ?:$src0))];
}

// alias of V_MOVRELS_B32_e32
def V_MOVRELS_B32_e64_sem : InstSISemantic<V_MOVRELS_B32_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (GetNamedOperand ?:$src0))];
}

// alias of V_MOV_B16_t16_e32
def V_MOV_B16_t16_e64_sem : InstSISemantic<V_MOV_B16_t16_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (GetNamedOperand ?:$src0))];
}

// alias of V_MOV_B64_e32
def V_MOV_B64_e64_sem : InstSISemantic<V_MOV_B64_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (GetNamedOperand ?:$src0))];
}

// alias of V_MUL_F64_pseudo_e32
def V_MUL_F64_pseudo_e64_sem : InstSISemantic<V_MUL_F64_pseudo_e64> {
  let Semantic = [(SetNamedOperand ?:$vdst, (LLVMBitCast (LLVMFMul (LLVMBitCast (GetNamedOperand ?:$src0), si_llvm_double_ty), (LLVMBitCast (GetNamedOperand ?:$src1), si_llvm_double_ty)), si_llvm_i64_ty))];
}

//==============================================================================
// V_TANH — Hyperbolic tangent (Phase 11)
//==============================================================================
// D.f = tanh(S0.f). Maps to int_amdgcn_tanh per type.
// Sources:
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 360.
//   rdna4-instruction-set-architecture.pdf, page 326.

// F32
foreach inst = [V_TANH_F32_e32, V_TANH_F32_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_tanh, [(GetNamedOperand $src0)]))
    ];
  }
}

// F16 — base + t16 + fake16 encodings; bitcast i16→half→i16 to keep $vdst i32-shaped
foreach inst = [V_TANH_F16_e32, V_TANH_F16_e64,
                V_TANH_F16_t16_e32, V_TANH_F16_t16_e64,
                V_TANH_F16_fake16_e32, V_TANH_F16_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_tanh, [(GetNamedOperand $src0)]))
    ];
  }
}

// BF16
foreach inst = [V_TANH_BF16_e32, V_TANH_BF16_e64,
                V_TANH_BF16_t16_e32, V_TANH_BF16_t16_e64,
                V_TANH_BF16_fake16_e32, V_TANH_BF16_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMIntrinsic si_llvm_bfloat_ty, int_amdgcn_tanh, [(GetNamedOperand $src0)]))
    ];
  }
}

//==============================================================================
// V_SAT_PK4 — saturating pack 4×{i8,u8} -> 4×{i4,u4} packed into i16 (Phase 11)
//==============================================================================
// D.i16 = pack4(sat(S0.i32 as 4×i8)) — saturate each byte to {i,u}4 range,
// pack into the low nibbles of an i16. Maps to int_amdgcn_sat_pk4_{i4_i8,u4_u8}.
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 326.
//   rdna4-instruction-set-architecture.pdf, page 326.

foreach inst = [V_SAT_PK4_I4_I8_e32, V_SAT_PK4_I4_I8_e64,
                V_SAT_PK4_I4_I8_t16_e32, V_SAT_PK4_I4_I8_t16_e64,
                V_SAT_PK4_I4_I8_fake16_e32, V_SAT_PK4_I4_I8_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_sat_pk4_i4_i8, [(GetNamedOperand $src0)]))
    ];
  }
}

foreach inst = [V_SAT_PK4_U4_U8_e32, V_SAT_PK4_U4_U8_e64,
                V_SAT_PK4_U4_U8_t16_e32, V_SAT_PK4_U4_U8_t16_e64,
                V_SAT_PK4_U4_U8_fake16_e32, V_SAT_PK4_U4_U8_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_sat_pk4_u4_u8, [(GetNamedOperand $src0)]))
    ];
  }
}

//==============================================================================
// V_MOV_B16 — 16-bit move (Phase 11)
//==============================================================================
// D.b16 = S0.b16. Plain register move.
foreach inst = [V_MOV_B16_e32, V_MOV_B16_e64,
                V_MOV_B16_fake16_e32, V_MOV_B16_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst, (GetNamedOperand $src0))
    ];
  }
}

//==============================================================================
// V_MINIMUM / V_MAXIMUM — IEEE 754-2019 minimumNumber / maximumNumber (Phase 11)
//==============================================================================
// Unlike V_MIN/V_MAX which use minNum/maxNum semantics, the V_MINIMUM family
// implements the new IEEE 754-2019 minimum/maximum (NaN propagation, -0 < +0).
// Maps to LLVM's int_minimum / int_maximum intrinsic.
def V_MINIMUM_F32_e64_sem : InstSISemantic<V_MINIMUM_F32_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMBinaryIntrinsic int_minimum,
          (LLVMBitCast (GetNamedOperand $src0), si_llvm_float_ty),
          (LLVMBitCast (GetNamedOperand $src1), si_llvm_float_ty)),
        si_llvm_i32_ty))
  ];
}
def V_MINIMUM_F64_e64_sem : InstSISemantic<V_MINIMUM_F64_e64> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMBinaryIntrinsic int_minimum,
          (LLVMBitCast (GetNamedOperand $src0), si_llvm_double_ty),
          (LLVMBitCast (GetNamedOperand $src1), si_llvm_double_ty)),
        si_llvm_i64_ty))
  ];
}
foreach inst = [V_MINIMUM_F16_e64, V_MINIMUM_F16_t16_e64, V_MINIMUM_F16_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMBinaryIntrinsic int_minimum,
          (GetNamedOperand $src0, si_llvm_half_ty),
          (GetNamedOperand $src1, si_llvm_half_ty)))
    ];
  }
}

//==============================================================================
// V_MFMA / V_WMMA / V_SWMMAC / V_SMFMAC semantics (Phase 12)
//==============================================================================

def V_MFMA_F32_16X16X128_F8F6F4_f4_f4_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f4_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f4_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f4_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f4_f6_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f4_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f4_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f4_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f4_f8_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f4_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f4_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f4_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f6_f4_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f6_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f6_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f6_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f6_f6_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f6_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f6_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f6_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f6_f8_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f6_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f6_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f6_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f8_f4_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f8_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f8_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f8_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f8_f6_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f8_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f8_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f8_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f8_f8_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f8_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X16BF16_1K_e64_sem : InstSISemantic<V_MFMA_F32_16X16X16BF16_1K_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x16bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X16BF16_1K_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X16BF16_1K_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x16bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X16F16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X16F16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x16f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X1F32_mac_e64_sem : InstSISemantic<V_MFMA_F32_16X16X1F32_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x1f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X1F32_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X1F32_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x1f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X1F32_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X1F32_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x1f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X2BF16_mac_e64_sem : InstSISemantic<V_MFMA_F32_16X16X2BF16_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x2bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X2BF16_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X2BF16_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x2bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X2BF16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X2BF16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x2bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X32_BF16_e64_sem : InstSISemantic<V_MFMA_F32_16X16X32_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X32_BF16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X32_BF16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X32_BF8_BF8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X32_BF8_BF8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x32_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X32_BF8_FP8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X32_BF8_FP8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x32_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X32_F16_e64_sem : InstSISemantic<V_MFMA_F32_16X16X32_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X32_F16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X32_F16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X32_FP8_BF8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X32_FP8_BF8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x32_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X32_FP8_FP8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X32_FP8_FP8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x32_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X4BF16_1K_e64_sem : InstSISemantic<V_MFMA_F32_16X16X4BF16_1K_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x4bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X4BF16_1K_mac_e64_sem : InstSISemantic<V_MFMA_F32_16X16X4BF16_1K_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x4bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X4BF16_1K_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X4BF16_1K_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x4bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X4BF16_1K_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X4BF16_1K_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x4bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X4F16_mac_e64_sem : InstSISemantic<V_MFMA_F32_16X16X4F16_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x4f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X4F16_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X4F16_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x4f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X4F16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X4F16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_16x16x4f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X4F32_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X4F32_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x4f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X8BF16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X8BF16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x8bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_16X16X8XF32_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_16X16X8XF32_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_16x16x8_xf32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_BF16_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_BF16_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_BF16_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_BF16_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_BF16_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_BF16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_BF16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_BF8_BF8_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_BF8_BF8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_BF8_BF8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_BF8_BF8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_BF8_BF8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_BF8_BF8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_BF8_FP8_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_BF8_FP8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_BF8_FP8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_BF8_FP8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_BF8_FP8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_BF8_FP8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_F16_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_F16_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_F16_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_F16_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_F16_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_F16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_F16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_FP8_BF8_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_FP8_BF8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_FP8_BF8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_FP8_BF8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_FP8_BF8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_FP8_BF8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_FP8_FP8_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_FP8_FP8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_FP8_FP8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_FP8_FP8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X16_FP8_FP8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X16_FP8_FP8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x16_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X1F32_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X1F32_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x1f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X1F32_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X1F32_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x1f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X1F32_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X1F32_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x1f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X2BF16_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X2BF16_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x2bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X2BF16_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X2BF16_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x2bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X2BF16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X2BF16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x2bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X2F32_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X2F32_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x2f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X2F32_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X2F32_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x2f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X2F32_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X2F32_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x2f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4BF16_1K_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4BF16_1K_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x4bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4BF16_1K_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4BF16_1K_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x4bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4BF16_1K_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4BF16_1K_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x4bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4BF16_1K_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4BF16_1K_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x4bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4BF16_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4BF16_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x4bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4BF16_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4BF16_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x4bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4BF16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4BF16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x4bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4F16_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4F16_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x4f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4F16_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4F16_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x4f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4F16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4F16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_mfma_f32_32x32x4f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4XF32_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4XF32_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x4_xf32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4XF32_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4XF32_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x4_xf32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X4XF32_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X4XF32_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x4_xf32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f4_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f4_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f4_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f4_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f4_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f6_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f6_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f6_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f6_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f6_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f8_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f8_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f4_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f4_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f4_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f4_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f4_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f4_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f4_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f6_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f6_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f6_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f6_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f6_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f8_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f8_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f6_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f6_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f4_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f4_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f4_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f4_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f4_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f6_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f6_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f6_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f6_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f6_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f8_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f8_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X64_F8F6F4_f8_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X64_F8F6F4_f8_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X8BF16_1K_e64_sem : InstSISemantic<V_MFMA_F32_32X32X8BF16_1K_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x8bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X8BF16_1K_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X8BF16_1K_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x8bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X8BF16_1K_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X8BF16_1K_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x8bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X8BF16_1K_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X8BF16_1K_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x8bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X8F16_mac_e64_sem : InstSISemantic<V_MFMA_F32_32X32X8F16_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x8f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X8F16_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X8F16_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x8f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_32X32X8F16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_32X32X8F16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_f32_32x32x8f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_4X4X1F32_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_4X4X1F32_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_4x4x1f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_4X4X2BF16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_4X4X2BF16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_4x4x2bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_4X4X4BF16_1K_e64_sem : InstSISemantic<V_MFMA_F32_4X4X4BF16_1K_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_4x4x4bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_4X4X4BF16_1K_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_4X4X4BF16_1K_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_4x4x4bf16_1k, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F32_4X4X4F16_vgprcd_e64_sem : InstSISemantic<V_MFMA_F32_4X4X4F16_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_f32_4x4x4f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F64_16X16X4F64_mac_e64_sem : InstSISemantic<V_MFMA_F64_16X16X4F64_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f64_ty, int_amdgcn_mfma_f64_16x16x4f64, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F64_16X16X4F64_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_F64_16X16X4F64_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f64_ty, int_amdgcn_mfma_f64_16x16x4f64, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F64_16X16X4F64_vgprcd_e64_sem : InstSISemantic<V_MFMA_F64_16X16X4F64_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f64_ty, int_amdgcn_mfma_f64_16x16x4f64, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_F64_4X4X4F64_vgprcd_e64_sem : InstSISemantic<V_MFMA_F64_4X4X4F64_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_double_ty, int_amdgcn_mfma_f64_4x4x4f64, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_16X16X16I8_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_16X16X16I8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4i32_ty, int_amdgcn_mfma_i32_16x16x16i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_16X16X32I8_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_16X16X32I8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4i32_ty, int_amdgcn_mfma_i32_16x16x32_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_16X16X4I8_mac_e64_sem : InstSISemantic<V_MFMA_I32_16X16X4I8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_16x16x4i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_16X16X4I8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_16X16X4I8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_16x16x4i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_16X16X4I8_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_16X16X4I8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_16x16x4i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_16X16X64_I8_e64_sem : InstSISemantic<V_MFMA_I32_16X16X64_I8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4i32_ty, int_amdgcn_mfma_i32_16x16x64_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_16X16X64_I8_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_16X16X64_I8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4i32_ty, int_amdgcn_mfma_i32_16x16x64_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X16I8_mac_e64_sem : InstSISemantic<V_MFMA_I32_32X32X16I8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_32x32x16_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X16I8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_32X32X16I8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_32x32x16_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X16I8_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_32X32X16I8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_32x32x16_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X32_I8_e64_sem : InstSISemantic<V_MFMA_I32_32X32X32_I8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_32x32x32_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X32_I8_mac_e64_sem : InstSISemantic<V_MFMA_I32_32X32X32_I8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_32x32x32_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X32_I8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_32X32X32_I8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_32x32x32_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X32_I8_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_32X32X32_I8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_32x32x32_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X4I8_mac_e64_sem : InstSISemantic<V_MFMA_I32_32X32X4I8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32i32_ty, int_amdgcn_mfma_i32_32x32x4i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X4I8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_32X32X4I8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32i32_ty, int_amdgcn_mfma_i32_32x32x4i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X4I8_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_32X32X4I8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32i32_ty, int_amdgcn_mfma_i32_32x32x4i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X8I8_mac_e64_sem : InstSISemantic<V_MFMA_I32_32X32X8I8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_32x32x8i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X8I8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_32X32X8I8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_32x32x8i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_32X32X8I8_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_32X32X8I8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_mfma_i32_32x32x8i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_I32_4X4X4I8_vgprcd_e64_sem : InstSISemantic<V_MFMA_I32_4X4X4I8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4i32_ty, int_amdgcn_mfma_i32_4x4x4i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_mac_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_mac_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_mac_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f4_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_mac_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_mac_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_mac_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f6_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_mac_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f4_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_mac_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f6_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_mac_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_mac_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_mac_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_mac_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_vgprcd_e64_sem : InstSISemantic<V_MFMA_SCALE_F32_32X32X64_F8F6F4_f8_f8_vgprcd_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_mfma_scale_f32_32x32x64_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $blgp, si_llvm_i32_ty), (GetNamedOperand $src0_modifiers), (GetNamedOperand $scale_src0), (GetNamedOperand $src1_modifiers), (GetNamedOperand $scale_src1)]))];
}

def V_SMFMAC_F32_16X16X64_BF16_e64_sem : InstSISemantic<V_SMFMAC_F32_16X16X64_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_smfmac_f32_16x16x64_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $idx), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty)]))];
}

def V_SMFMAC_F32_16X16X64_F16_e64_sem : InstSISemantic<V_SMFMAC_F32_16X16X64_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_smfmac_f32_16x16x64_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $idx), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty)]))];
}

def V_SMFMAC_F32_32X32X32_BF16_e64_sem : InstSISemantic<V_SMFMAC_F32_32X32X32_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_smfmac_f32_32x32x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $idx), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty)]))];
}

def V_SMFMAC_F32_32X32X32_F16_e64_sem : InstSISemantic<V_SMFMAC_F32_32X32X32_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_smfmac_f32_32x32x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $idx), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty)]))];
}

def V_SMFMAC_I32_16X16X128_I8_e64_sem : InstSISemantic<V_SMFMAC_I32_16X16X128_I8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4i32_ty, int_amdgcn_smfmac_i32_16x16x128_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $idx), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty)]))];
}

def V_SMFMAC_I32_32X32X64_I8_e64_sem : InstSISemantic<V_SMFMAC_I32_32X32X64_I8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16i32_ty, int_amdgcn_smfmac_i32_32x32x64_i8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $idx), (GetNamedOperand $cbsz, si_llvm_i32_ty), (GetNamedOperand $abid, si_llvm_i32_ty)]))];
}

def V_SWMMAC_BF16F32_16X16X64_BF16_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_BF16F32_16X16X64_BF16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_bf16f32_16x16x64_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_BF16_16X16X32_BF16_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_BF16_16X16X32_BF16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_bf16_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_BF16_16X16X32_BF16_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_BF16_16X16X32_BF16_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_bf16_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_8bit), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_BF16_16X16X64_BF16_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_BF16_16X16X64_BF16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_bf16_16x16x64_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_F16_16X16X128_BF8_BF8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F16_16X16X128_BF8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f16_16x16x128_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_32bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse)]))];
}

def V_SWMMAC_F16_16X16X128_BF8_FP8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F16_16X16X128_BF8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f16_16x16x128_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_32bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse)]))];
}

def V_SWMMAC_F16_16X16X128_FP8_BF8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F16_16X16X128_FP8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f16_16x16x128_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_32bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse)]))];
}

def V_SWMMAC_F16_16X16X128_FP8_FP8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F16_16X16X128_FP8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f16_16x16x128_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_32bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse)]))];
}

def V_SWMMAC_F16_16X16X32_F16_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F16_16X16X32_F16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f16_16x16x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_F16_16X16X32_F16_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_F16_16X16X32_F16_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f16_16x16x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_8bit), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_F16_16X16X64_F16_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F16_16X16X64_F16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f16_16x16x64_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_F32_16X16X128_BF8_BF8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X128_BF8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x128_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_32bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse)]))];
}

def V_SWMMAC_F32_16X16X128_BF8_FP8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X128_BF8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x128_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_32bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse)]))];
}

def V_SWMMAC_F32_16X16X128_FP8_BF8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X128_FP8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x128_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_32bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse)]))];
}

def V_SWMMAC_F32_16X16X128_FP8_FP8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X128_FP8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x128_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_32bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse)]))];
}

def V_SWMMAC_F32_16X16X32_BF16_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_BF16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_F32_16X16X32_BF16_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_BF16_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_8bit), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_F32_16X16X32_BF8_BF8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_BF8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit)]))];
}

def V_SWMMAC_F32_16X16X32_BF8_BF8_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_BF8_BF8_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_8bit)]))];
}

def V_SWMMAC_F32_16X16X32_BF8_FP8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_BF8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit)]))];
}

def V_SWMMAC_F32_16X16X32_BF8_FP8_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_BF8_FP8_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_8bit)]))];
}

def V_SWMMAC_F32_16X16X32_F16_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_F16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_F32_16X16X32_F16_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_F16_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_8bit), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_F32_16X16X32_FP8_BF8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_FP8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit)]))];
}

def V_SWMMAC_F32_16X16X32_FP8_BF8_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_FP8_BF8_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_8bit)]))];
}

def V_SWMMAC_F32_16X16X32_FP8_FP8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_FP8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit)]))];
}

def V_SWMMAC_F32_16X16X32_FP8_FP8_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X32_FP8_FP8_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x32_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_8bit)]))];
}

def V_SWMMAC_F32_16X16X64_BF16_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X64_BF16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x64_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_F32_16X16X64_F16_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_F32_16X16X64_F16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_f32_16x16x64_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_SWMMAC_I32_16X16X128_IU8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_I32_16X16X128_IU8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_i32_16x16x128_iu8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_32bit), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo)]))];
}

def V_SWMMAC_I32_16X16X32_IU4_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_I32_16X16X32_IU4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_i32_16x16x32_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $neg_lo)]))];
}

def V_SWMMAC_I32_16X16X32_IU4_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_I32_16X16X32_IU4_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_i32_16x16x32_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $neg_lo)]))];
}

def V_SWMMAC_I32_16X16X32_IU8_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_I32_16X16X32_IU8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_i32_16x16x32_iu8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $neg_lo)]))];
}

def V_SWMMAC_I32_16X16X32_IU8_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_I32_16X16X32_IU8_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_i32_16x16x32_iu8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_8bit), (GetNamedOperand $neg_lo)]))];
}

def V_SWMMAC_I32_16X16X64_IU4_w32_twoaddr_sem : InstSISemantic<V_SWMMAC_I32_16X16X64_IU4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_i32_16x16x64_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_SWMMAC_I32_16X16X64_IU4_w64_twoaddr_sem : InstSISemantic<V_SWMMAC_I32_16X16X64_IU4_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_swmmac_i32_16x16x64_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $srcTiedDef), (GetNamedOperand $src2), (GetNamedOperand $index_key_16bit), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_BF16F32_16X16X32_BF16_w32_threeaddr_sem : InstSISemantic<V_WMMA_BF16F32_16X16X32_BF16_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_bf16f32_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_BF16F32_16X16X32_BF16_w32_twoaddr_sem : InstSISemantic<V_WMMA_BF16F32_16X16X32_BF16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_bf16f32_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_BF16_16X16X16_BF16_w32_threeaddr_sem : InstSISemantic<V_WMMA_BF16_16X16X16_BF16_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_bf16_16x16x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_BF16_16X16X16_BF16_w32_twoaddr_sem : InstSISemantic<V_WMMA_BF16_16X16X16_BF16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_bf16_16x16x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_BF16_16X16X16_BF16_w64_threeaddr_sem : InstSISemantic<V_WMMA_BF16_16X16X16_BF16_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_bf16_16x16x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_BF16_16X16X16_BF16_w64_twoaddr_sem : InstSISemantic<V_WMMA_BF16_16X16X16_BF16_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_bf16_16x16x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_BF16_16X16X32_BF16_w32_threeaddr_sem : InstSISemantic<V_WMMA_BF16_16X16X32_BF16_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_bf16_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_BF16_16X16X32_BF16_w32_twoaddr_sem : InstSISemantic<V_WMMA_BF16_16X16X32_BF16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_bf16_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X128_BF8_BF8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X128_BF8_BF8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x128_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X128_BF8_BF8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X128_BF8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x128_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X128_BF8_FP8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X128_BF8_FP8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x128_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X128_BF8_FP8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X128_BF8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x128_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X128_FP8_BF8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X128_FP8_BF8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x128_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X128_FP8_BF8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X128_FP8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x128_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X128_FP8_FP8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X128_FP8_FP8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x128_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X128_FP8_FP8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X128_FP8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x128_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X16_F16_w32_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X16_F16_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X16_F16_w32_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X16_F16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X16_F16_w64_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X16_F16_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X16_F16_w64_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X16_F16_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X32_F16_w32_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X32_F16_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X32_F16_w32_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X32_F16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X64_BF8_BF8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X64_BF8_BF8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x64_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X64_BF8_BF8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X64_BF8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x64_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X64_BF8_FP8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X64_BF8_FP8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x64_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X64_BF8_FP8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X64_BF8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x64_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X64_FP8_BF8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X64_FP8_BF8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x64_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X64_FP8_BF8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X64_FP8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x64_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X64_FP8_FP8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F16_16X16X64_FP8_FP8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x64_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F16_16X16X64_FP8_FP8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F16_16X16X64_FP8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f16_16x16x64_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_BF8_BF8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_BF8_BF8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x128_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_BF8_BF8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_BF8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x128_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_BF8_FP8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_BF8_FP8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x128_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_BF8_FP8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_BF8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x128_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f4_f4_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f4_f4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f4_f4_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f4_f4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f4_f6_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f4_f6_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f4_f6_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f4_f6_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f4_f8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f4_f8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f4_f8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f4_f8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f6_f4_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f6_f4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f6_f4_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f6_f4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f6_f6_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f6_f6_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f6_f6_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f6_f6_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f6_f8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f6_f8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f6_f8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f6_f8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f8_f4_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f8_f4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f8_f4_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f8_f4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f8_f6_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f8_f6_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f8_f6_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f8_f6_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_F8F6F4_f8_f8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_F8F6F4_f8_f8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_FP8_BF8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_FP8_BF8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x128_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_FP8_BF8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_FP8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x128_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_FP8_FP8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_FP8_FP8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x128_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X128_FP8_FP8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X128_FP8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x128_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF16_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF16_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF16_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF16_w64_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF16_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF16_w64_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF16_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF8_BF8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF8_BF8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF8_BF8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF8_BF8_w64_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF8_BF8_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF8_BF8_w64_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF8_BF8_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF8_FP8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF8_FP8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF8_FP8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF8_FP8_w64_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF8_FP8_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_BF8_FP8_w64_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_BF8_FP8_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_F16_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_F16_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_F16_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_F16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_F16_w64_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_F16_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_F16_w64_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_F16_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_FP8_BF8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_FP8_BF8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_FP8_BF8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_FP8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_FP8_BF8_w64_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_FP8_BF8_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_FP8_BF8_w64_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_FP8_BF8_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_FP8_FP8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_FP8_FP8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_FP8_FP8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_FP8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_FP8_FP8_w64_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_FP8_FP8_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X16_FP8_FP8_w64_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X16_FP8_FP8_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x16_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X32_BF16_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X32_BF16_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X32_BF16_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X32_BF16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x32_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X32_F16_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X32_F16_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X32_F16_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X32_F16_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x32_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X4_F32_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X4_F32_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x4_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X4_F32_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X4_F32_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x4_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X64_BF8_BF8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X64_BF8_BF8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x64_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X64_BF8_BF8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X64_BF8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x64_bf8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X64_BF8_FP8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X64_BF8_FP8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x64_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X64_BF8_FP8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X64_BF8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x64_bf8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X64_FP8_BF8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X64_FP8_BF8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x64_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X64_FP8_BF8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X64_FP8_BF8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x64_fp8_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X64_FP8_FP8_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_16X16X64_FP8_FP8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x64_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_16X16X64_FP8_FP8_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_16X16X64_FP8_FP8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_16x16x64_fp8_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_32X16X128_F4_w32_threeaddr_sem : InstSISemantic<V_WMMA_F32_32X16X128_F4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_32x16x128_f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_F32_32X16X128_F4_w32_twoaddr_sem : InstSISemantic<V_WMMA_F32_32X16X128_F4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_f32_32x16x128_f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_I32_16X16X16_IU4_w32_threeaddr_sem : InstSISemantic<V_WMMA_I32_16X16X16_IU4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x16_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X16_IU4_w32_twoaddr_sem : InstSISemantic<V_WMMA_I32_16X16X16_IU4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x16_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X16_IU4_w64_threeaddr_sem : InstSISemantic<V_WMMA_I32_16X16X16_IU4_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x16_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X16_IU4_w64_twoaddr_sem : InstSISemantic<V_WMMA_I32_16X16X16_IU4_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x16_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X16_IU8_w32_threeaddr_sem : InstSISemantic<V_WMMA_I32_16X16X16_IU8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x16_iu8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X16_IU8_w32_twoaddr_sem : InstSISemantic<V_WMMA_I32_16X16X16_IU8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x16_iu8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X16_IU8_w64_threeaddr_sem : InstSISemantic<V_WMMA_I32_16X16X16_IU8_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x16_iu8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X16_IU8_w64_twoaddr_sem : InstSISemantic<V_WMMA_I32_16X16X16_IU8_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x16_iu8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X32_IU4_w32_threeaddr_sem : InstSISemantic<V_WMMA_I32_16X16X32_IU4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x32_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X32_IU4_w32_twoaddr_sem : InstSISemantic<V_WMMA_I32_16X16X32_IU4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x32_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X32_IU4_w64_threeaddr_sem : InstSISemantic<V_WMMA_I32_16X16X32_IU4_w64_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x32_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X32_IU4_w64_twoaddr_sem : InstSISemantic<V_WMMA_I32_16X16X32_IU4_w64_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x32_iu4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X64_IU8_w32_threeaddr_sem : InstSISemantic<V_WMMA_I32_16X16X64_IU8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x64_iu8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_I32_16X16X64_IU8_w32_twoaddr_sem : InstSISemantic<V_WMMA_I32_16X16X64_IU8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_i32_16x16x64_iu8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f4_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f4_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f6_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f6_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f6_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f6_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f8_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f8_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f4_f8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f4_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f4_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f6_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f6_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f6_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f6_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f8_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f8_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f6_f8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f4_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f4_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f6_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f6_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f6_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f6_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f8_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_16X16X128_F8F6F4_f8_f8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_32X16X128_F4_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_32X16X128_F4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_32x16x128_f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE16_F32_32X16X128_F4_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE16_F32_32X16X128_F4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale16_f32_32x16x128_f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f6_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f4_f8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f6_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f6_f8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f6_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_fmt), (GetNamedOperand $matrix_b_fmt), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_32X16X128_F4_w32_threeaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_32X16X128_F4_w32_threeaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_32x16x128_f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

def V_WMMA_SCALE_F32_32X16X128_F4_w32_twoaddr_sem : InstSISemantic<V_WMMA_SCALE_F32_32X16X128_F4_w32_twoaddr> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v4f32_ty, int_amdgcn_wmma_scale_f32_32x16x128_f4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $scale_src0), (GetNamedOperand $scale_src1), (GetNamedOperand $matrix_a_scale), (GetNamedOperand $matrix_b_scale), (GetNamedOperand $matrix_a_scale_fmt), (GetNamedOperand $matrix_b_scale_fmt), (GetNamedOperand $matrix_a_reuse), (GetNamedOperand $matrix_b_reuse), (GetNamedOperand $neg_lo), (GetNamedOperand $neg_hi)]))];
}

//==============================================================================
// V_MFMA / V_WMMA / V_SWMMAC / V_SMFMAC semantics (Phase 12)
//==============================================================================



//==============================================================================
// V_{RCP,RSQ,SQRT,SIN,COS,LOG,EXP}_BF16 (Phase 15, re-add 2)
//==============================================================================
foreach base = ["RCP", "RSQ", "SQRT", "SIN", "COS", "LOG", "EXP"] in {
  foreach inst = [!cast<InstSI>("V_"#base#"_BF16_e32"),
                  !cast<InstSI>("V_"#base#"_BF16_e64"),
                  !cast<InstSI>("V_"#base#"_BF16_t16_e32"),
                  !cast<InstSI>("V_"#base#"_BF16_t16_e64"),
                  !cast<InstSI>("V_"#base#"_BF16_fake16_e32"),
                  !cast<InstSI>("V_"#base#"_BF16_fake16_e64")] in {
    def inst#_sem : InstSISemantic<inst> {
      let Semantic = [
        (SetNamedOperand $vdst,
          (LLVMIntrinsic si_llvm_bfloat_ty,
            !cast<Intrinsic>("int_amdgcn_"#!tolower(!subst("EXP", "exp2", base))),
            [(GetNamedOperand $src0)]))
      ];
    }
  }
}

//==============================================================================
// V_{ADD,PK_ADD}_{MIN,MAX}_{I,U}{16,32} (Phase 15, re-add)
//==============================================================================
foreach base = ["MIN", "MAX"] in {
  foreach sign = ["I", "U"] in {
    def !cast<InstSI>("V_ADD_"#base#"_"#sign#"32_e64")#_sem
        : InstSISemantic<!cast<InstSI>("V_ADD_"#base#"_"#sign#"32_e64")> {
      let Semantic = [
        (SetNamedOperand $vdst,
          (LLVMIntrinsic si_llvm_i32_ty,
            !cast<Intrinsic>("int_amdgcn_add_"#!tolower(base)#"_"#!tolower(sign)#"32"),
            [(GetNamedOperand $src0), (GetNamedOperand $src1),
             (GetNamedOperand $src2),
             (LLVMZExtOrTrunc (GetNamedOperand $clamp), si_llvm_i1_ty)]))
      ];
    }
    def !cast<InstSI>("V_PK_ADD_"#base#"_"#sign#"16")#_sem
        : InstSISemantic<!cast<InstSI>("V_PK_ADD_"#base#"_"#sign#"16")> {
      let Semantic = [
        (SetNamedOperand $vdst,
          (LLVMIntrinsic si_llvm_v2i16_ty,
            !cast<Intrinsic>("int_amdgcn_pk_add_"#!tolower(base)#"_"#!tolower(sign)#"16"),
            [(GetNamedOperand $src0), (GetNamedOperand $src1),
             (GetNamedOperand $src2),
             (LLVMZExtOrTrunc (GetNamedOperand $clamp), si_llvm_i1_ty)]))
      ];
    }
  }
}

//==============================================================================
// V_ADD/SUB integer + V_MUL_U64 (Phase 17, re-add)
//==============================================================================
foreach inst = [V_ADD_I32_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMAdd (GetNamedOperand $src0), (GetNamedOperand $src1)))];
  }
}
foreach inst = [V_SUB_I32_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMSub (GetNamedOperand $src0), (GetNamedOperand $src1)))];
  }
}
foreach inst = [V_ADD_I16_e64, V_ADD_I16_t16_e64, V_ADD_I16_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMAdd (GetNamedOperand $src0, si_llvm_i16_ty),
               (GetNamedOperand $src1, si_llvm_i16_ty)))];
  }
}
foreach inst = [V_SUB_I16_e64, V_SUB_I16_t16_e64, V_SUB_I16_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMSub (GetNamedOperand $src0, si_llvm_i16_ty),
               (GetNamedOperand $src1, si_llvm_i16_ty)))];
  }
}
foreach inst = [V_ADD_U64_e32, V_ADD_U64_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMAdd (GetNamedOperand $src0, si_llvm_i64_ty),
               (GetNamedOperand $src1, si_llvm_i64_ty)))];
  }
}
foreach inst = [V_SUB_U64_e32, V_SUB_U64_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMSub (GetNamedOperand $src0, si_llvm_i64_ty),
               (GetNamedOperand $src1, si_llvm_i64_ty)))];
  }
}
foreach inst = [V_MUL_U64_e32, V_MUL_U64_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMMul (GetNamedOperand $src0, si_llvm_i64_ty),
               (GetNamedOperand $src1, si_llvm_i64_ty)))];
  }
}
def V_FMAAK_F64_sem : InstSISemantic<V_FMAAK_F64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMFMA (GetNamedOperand $src0, si_llvm_double_ty),
             (GetNamedOperand $src1, si_llvm_double_ty),
             (GetNamedOperand $imm, si_llvm_double_ty)))];
}
def V_FMAMK_F64_sem : InstSISemantic<V_FMAMK_F64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMFMA (GetNamedOperand $src0, si_llvm_double_ty),
             (GetNamedOperand $imm, si_llvm_double_ty),
             (GetNamedOperand $src1, si_llvm_double_ty)))];
}
def V_MAD_U32_e64_sem : InstSISemantic<V_MAD_U32_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMAdd (LLVMMul (GetNamedOperand $src0), (GetNamedOperand $src1)),
             (GetNamedOperand $src2)))];
}
def V_MAD_NC_I64_I32_e64_sem : InstSISemantic<V_MAD_NC_I64_I32_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMAdd
      (LLVMMul (LLVMSExt (GetNamedOperand $src0), si_llvm_i64_ty),
               (LLVMSExt (GetNamedOperand $src1), si_llvm_i64_ty)),
      (GetNamedOperand $src2, si_llvm_i64_ty)))];
}
def V_MAD_NC_U64_U32_e64_sem : InstSISemantic<V_MAD_NC_U64_U32_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMAdd
      (LLVMMul (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
               (LLVMZExt (GetNamedOperand $src1), si_llvm_i64_ty)),
      (GetNamedOperand $src2, si_llvm_i64_ty)))];
}

//==============================================================================
// V_BITOP3 (Phase 18, re-add)
//==============================================================================
def V_BITOP3_B32_e64_sem : InstSISemantic<V_BITOP3_B32_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_bitop3,
      [(GetNamedOperand $src0), (GetNamedOperand $src1),
       (GetNamedOperand $src2),
       (GetNamedOperand $bitop3, si_llvm_i32_ty)]))];
}
foreach inst = [V_BITOP3_B16_e64, V_BITOP3_B16_gfx1250_e64,
                V_BITOP3_B16_gfx1250_t16_e64, V_BITOP3_B16_gfx1250_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_bitop3,
        [(GetNamedOperand $src0, si_llvm_i16_ty),
         (GetNamedOperand $src1, si_llvm_i16_ty),
         (GetNamedOperand $src2, si_llvm_i16_ty),
         (GetNamedOperand $bitop3, si_llvm_i32_ty)]))];
  }
}
foreach pair = [["AND", "LLVMAnd"], ["OR", "LLVMOr"], ["XOR", "LLVMXor"]] in {
  defvar opName = pair[0];
  defvar Op = !cast<LLVMOp>(pair[1]);
  foreach inst = [!cast<InstSI>("V_"#opName#"_B16_t16_e64"),
                  !cast<InstSI>("V_"#opName#"_B16_fake16_e64")] in {
    def inst#_sem : InstSISemantic<inst> {
      let Semantic = [(SetNamedOperand $vdst,
        (Op (GetNamedOperand $src0, si_llvm_i16_ty),
            (GetNamedOperand $src1, si_llvm_i16_ty)))];
    }
  }
}
foreach inst = [V_CNDMASK_B16_t16_e32, V_CNDMASK_B16_t16_e64,
                V_CNDMASK_B16_fake16_e32, V_CNDMASK_B16_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMSelect (ImplicitUse VCC),
                  (GetNamedOperand $src1, si_llvm_i16_ty),
                  (GetNamedOperand $src0, si_llvm_i16_ty)))];
  }
}
def V_MIN_I64_e64_sem : InstSISemantic<V_MIN_I64_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMSelect (LLVMCmp LLVMICmpSLT,
                  (GetNamedOperand $src0, si_llvm_i64_ty),
                  (GetNamedOperand $src1, si_llvm_i64_ty)),
                (GetNamedOperand $src0, si_llvm_i64_ty),
                (GetNamedOperand $src1, si_llvm_i64_ty)))];
}
def V_MAX_I64_e64_sem : InstSISemantic<V_MAX_I64_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMSelect (LLVMCmp LLVMICmpSGT,
                  (GetNamedOperand $src0, si_llvm_i64_ty),
                  (GetNamedOperand $src1, si_llvm_i64_ty)),
                (GetNamedOperand $src0, si_llvm_i64_ty),
                (GetNamedOperand $src1, si_llvm_i64_ty)))];
}
def V_MIN_U64_e64_sem : InstSISemantic<V_MIN_U64_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMSelect (LLVMCmp LLVMICmpULT,
                  (GetNamedOperand $src0, si_llvm_i64_ty),
                  (GetNamedOperand $src1, si_llvm_i64_ty)),
                (GetNamedOperand $src0, si_llvm_i64_ty),
                (GetNamedOperand $src1, si_llvm_i64_ty)))];
}
def V_MAX_U64_e64_sem : InstSISemantic<V_MAX_U64_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMSelect (LLVMCmp LLVMICmpUGT,
                  (GetNamedOperand $src0, si_llvm_i64_ty),
                  (GetNamedOperand $src1, si_llvm_i64_ty)),
                (GetNamedOperand $src0, si_llvm_i64_ty),
                (GetNamedOperand $src1, si_llvm_i64_ty)))];
}
foreach inst = [V_MINIMUMMAXIMUM_F16_e64,
                V_MINIMUMMAXIMUM_F16_t16_e64,
                V_MINIMUMMAXIMUM_F16_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMBinaryIntrinsic int_minimum,
        (GetNamedOperand $src0, si_llvm_half_ty),
        (GetNamedOperand $src1, si_llvm_half_ty)))];
  }
}
def V_PERMLANE16_SWAP_B32_e64_sem : InstSISemantic<V_PERMLANE16_SWAP_B32_e64> {
  let Semantic = [
    (DefVal $r, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_permlane16_swap,
      [(GetNamedOperand $src0), (GetNamedOperand $src1),
       (ConstantInt si_llvm_i1_ty, 0), (ConstantInt si_llvm_i1_ty, 0)])),
    (SetNamedOperand $vdst, (LLVMExtractElement (GetVal $r), (ConstantInt si_llvm_i32_ty, 0)))
  ];
}
def V_PERMLANE16_SWAP_B32_e32_sem : InstSISemantic<V_PERMLANE16_SWAP_B32_e32> {
  let Semantic = [
    (DefVal $r, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_permlane16_swap,
      [(GetNamedOperand $src0), (GetNamedOperand $src1),
       (ConstantInt si_llvm_i1_ty, 0), (ConstantInt si_llvm_i1_ty, 0)])),
    (SetNamedOperand $vdst, (LLVMExtractElement (GetVal $r), (ConstantInt si_llvm_i32_ty, 0)))
  ];
}
def V_PERMLANE32_SWAP_B32_e64_sem : InstSISemantic<V_PERMLANE32_SWAP_B32_e64> {
  let Semantic = [
    (DefVal $r, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_permlane32_swap,
      [(GetNamedOperand $src0), (GetNamedOperand $src1),
       (ConstantInt si_llvm_i1_ty, 0), (ConstantInt si_llvm_i1_ty, 0)])),
    (SetNamedOperand $vdst, (LLVMExtractElement (GetVal $r), (ConstantInt si_llvm_i32_ty, 0)))
  ];
}
def V_PERMLANE32_SWAP_B32_e32_sem : InstSISemantic<V_PERMLANE32_SWAP_B32_e32> {
  let Semantic = [
    (DefVal $r, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_permlane32_swap,
      [(GetNamedOperand $src0), (GetNamedOperand $src1),
       (ConstantInt si_llvm_i1_ty, 0), (ConstantInt si_llvm_i1_ty, 0)])),
    (SetNamedOperand $vdst, (LLVMExtractElement (GetVal $r), (ConstantInt si_llvm_i32_ty, 0)))
  ];
}
def V_PERMLANE_BCAST_B32_e64_sem : InstSISemantic<V_PERMLANE_BCAST_B32_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_permlane_bcast,
      [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

//==============================================================================
// V_FMA_MIX + V_MAD_MIX + V_DOT2C + V_PRNG + V_MSAD/MQSAD + V_PERMLANE_UP/DOWN/XOR/IDX_GEN + V_SWAP_B16 + V_SET_INACTIVE (Phase 19, re-add)
//==============================================================================
def V_FMA_MIX_F32_sem : InstSISemantic<V_FMA_MIX_F32> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMFMA (GetNamedOperand $src0, si_llvm_float_ty),
             (GetNamedOperand $src1, si_llvm_float_ty),
             (GetNamedOperand $src2, si_llvm_float_ty)))];
}
def V_FMA_MIX_F32_BF16_sem : InstSISemantic<V_FMA_MIX_F32_BF16> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMFMA (GetNamedOperand $src0, si_llvm_float_ty),
             (GetNamedOperand $src1, si_llvm_float_ty),
             (GetNamedOperand $src2, si_llvm_float_ty)))];
}
def V_MAD_MIX_F32_sem : InstSISemantic<V_MAD_MIX_F32> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMFAdd (LLVMFMul (GetNamedOperand $src0, si_llvm_float_ty),
                        (GetNamedOperand $src1, si_llvm_float_ty)),
              (GetNamedOperand $src2, si_llvm_float_ty)))];
}
foreach inst = [V_FMA_MIXHI_F16, V_FMA_MIXLO_F16,
                V_MAD_MIXHI_F16, V_MAD_MIXLO_F16] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMFPTrunc (LLVMFMA (GetNamedOperand $src0, si_llvm_float_ty),
                            (GetNamedOperand $src1, si_llvm_float_ty),
                            (GetNamedOperand $src2, si_llvm_float_ty)),
                   si_llvm_half_ty))];
  }
}
foreach inst = [V_FMA_MIXHI_BF16, V_FMA_MIXLO_BF16] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMFPTrunc (LLVMFMA (GetNamedOperand $src0, si_llvm_float_ty),
                            (GetNamedOperand $src1, si_llvm_float_ty),
                            (GetNamedOperand $src2, si_llvm_float_ty)),
                   si_llvm_bfloat_ty))];
  }
}
foreach inst = [V_DOT2C_F32_BF16_e32, V_DOT2C_F32_BF16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (DefVal $a, (GetNamedOperand $src0, si_llvm_v2bf16_ty)),
      (DefVal $b, (GetNamedOperand $src1, si_llvm_v2bf16_ty)),
      (DefVal $a0, (LLVMFPExt (LLVMExtractElement (GetVal $a), (ConstantInt si_llvm_i32_ty, 0)), si_llvm_float_ty)),
      (DefVal $a1, (LLVMFPExt (LLVMExtractElement (GetVal $a), (ConstantInt si_llvm_i32_ty, 1)), si_llvm_float_ty)),
      (DefVal $b0, (LLVMFPExt (LLVMExtractElement (GetVal $b), (ConstantInt si_llvm_i32_ty, 0)), si_llvm_float_ty)),
      (DefVal $b1, (LLVMFPExt (LLVMExtractElement (GetVal $b), (ConstantInt si_llvm_i32_ty, 1)), si_llvm_float_ty)),
      (SetNamedOperand $vdst,
        (LLVMFAdd (LLVMFMul (GetVal $a0), (GetVal $b0)),
                  (LLVMFAdd (LLVMFMul (GetVal $a1), (GetVal $b1)),
                            (GetNamedOperand $vdst, si_llvm_float_ty))))
    ];
  }
}
foreach inst = [V_PRNG_B32_e32, V_PRNG_B32_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_prng_b32,
        [(GetNamedOperand $src0)]))];
  }
}
def V_MSAD_U8_e64_sem : InstSISemantic<V_MSAD_U8_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_msad_u8,
      [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}
def V_MQSAD_U32_U8_e64_sem : InstSISemantic<V_MQSAD_U32_U8_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_v4i32_ty, int_amdgcn_mqsad_u32_u8,
      [(GetNamedOperand $src0, si_llvm_i64_ty),
       (GetNamedOperand $src1),
       (GetNamedOperand $src2, si_llvm_v4i32_ty)]))];
}
def V_MQSAD_PK_U16_U8_e64_sem : InstSISemantic<V_MQSAD_PK_U16_U8_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_i64_ty, int_amdgcn_mqsad_pk_u16_u8,
      [(GetNamedOperand $src0, si_llvm_i64_ty),
       (GetNamedOperand $src1),
       (GetNamedOperand $src2, si_llvm_i64_ty)]))];
}
def V_PERMLANE_UP_B32_e64_sem : InstSISemantic<V_PERMLANE_UP_B32_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_permlane_up,
      [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}
def V_PERMLANE_DOWN_B32_e64_sem : InstSISemantic<V_PERMLANE_DOWN_B32_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_permlane_down,
      [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}
def V_PERMLANE_XOR_B32_e64_sem : InstSISemantic<V_PERMLANE_XOR_B32_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_permlane_xor,
      [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}
def V_PERMLANE_IDX_GEN_B32_e64_sem : InstSISemantic<V_PERMLANE_IDX_GEN_B32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (GetNamedOperand $src0))];
}
def V_SWAP_B16_sem : InstSISemantic<V_SWAP_B16> {
  let Semantic = [(SetNamedOperand $vdst, (GetNamedOperand $src0, si_llvm_i16_ty))];
}
def V_SET_INACTIVE_B32_sem : InstSISemantic<V_SET_INACTIVE_B32> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_set_inactive,
      [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

//==============================================================================
// V_CVT_F32_BF16 / V_CVT_F16_FP8/BF8 / V_CVT_PK_F16_FP8/BF8 / V_CVT_PK_*_F16 / V_CVT_PK_F16/BF16_F32 (Phase 20, re-add)
//==============================================================================
foreach inst = [V_CVT_F32_BF16_e32, V_CVT_F32_BF16_e64,
                V_CVT_F32_BF16_t16_e32, V_CVT_F32_BF16_t16_e64,
                V_CVT_F32_BF16_fake16_e32, V_CVT_F32_BF16_fake16_e64,
                V_CVT_F32_BF16_gfx1250_t16_e32, V_CVT_F32_BF16_gfx1250_t16_e64,
                V_CVT_F32_BF16_gfx1250_fake16_e32, V_CVT_F32_BF16_gfx1250_fake16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst,
      (LLVMFPExt (GetNamedOperand $src0, si_llvm_bfloat_ty), si_llvm_float_ty))];
  }
}
foreach fpFmt = ["FP8", "BF8"] in {
  foreach inst = [!cast<InstSI>("V_CVT_F16_"#fpFmt#"_e32"),
                  !cast<InstSI>("V_CVT_F16_"#fpFmt#"_e64"),
                  !cast<InstSI>("V_CVT_F16_"#fpFmt#"_t16_e32"),
                  !cast<InstSI>("V_CVT_F16_"#fpFmt#"_t16_e64"),
                  !cast<InstSI>("V_CVT_F16_"#fpFmt#"_fake16_e32"),
                  !cast<InstSI>("V_CVT_F16_"#fpFmt#"_fake16_e64")] in {
    def inst#_sem : InstSISemantic<inst> {
      let Semantic = [(SetNamedOperand $vdst,
        (LLVMIntrinsic si_llvm_half_ty,
          !cast<Intrinsic>("int_amdgcn_cvt_f16_"#!tolower(fpFmt)),
          [(GetNamedOperand $src0), (ConstantInt si_llvm_i32_ty, 0)]))];
    }
  }
}
foreach fpFmt = ["FP8", "BF8"] in {
  foreach inst = [!cast<InstSI>("V_CVT_PK_F16_"#fpFmt#"_e32"),
                  !cast<InstSI>("V_CVT_PK_F16_"#fpFmt#"_e64"),
                  !cast<InstSI>("V_CVT_PK_F16_"#fpFmt#"_t16_e32"),
                  !cast<InstSI>("V_CVT_PK_F16_"#fpFmt#"_t16_e64"),
                  !cast<InstSI>("V_CVT_PK_F16_"#fpFmt#"_fake16_e32"),
                  !cast<InstSI>("V_CVT_PK_F16_"#fpFmt#"_fake16_e64")] in {
    def inst#_sem : InstSISemantic<inst> {
      let Semantic = [(SetNamedOperand $vdst,
        (LLVMIntrinsic si_llvm_v2f16_ty,
          !cast<Intrinsic>("int_amdgcn_cvt_pk_f16_"#!tolower(fpFmt)),
          [(GetNamedOperand $src0, si_llvm_i16_ty)]))];
    }
  }
}
//foreach fpFmt = ["FP8", "BF8"] in {
//  foreach inst = [
////  !cast<InstSI>("V_CVT_PK_"#fpFmt#"_F16_gfx1250_e64"),
////                  !cast<InstSI>("V_CVT_PK_"#fpFmt#"_F16_gfx1250_t16_e64"),
////                  !cast<InstSI>("V_CVT_PK_"#fpFmt#"_F16_gfx1250_fake16_e64")
//                  ] in {
//    def inst#_sem : InstSISemantic<inst> {
//      let Semantic = [(SetNamedOperand $vdst,
//        (LLVMIntrinsic si_llvm_i16_ty,
//          !cast<Intrinsic>("int_amdgcn_cvt_pk_"#!tolower(fpFmt)#"_f16"),
//          [(GetNamedOperand $src0, si_llvm_v2f16_ty)]))];
//    }
//  }
//}
def V_CVT_PK_F16_F32_e64_sem : InstSISemantic<V_CVT_PK_F16_F32_e64> {
  let Semantic = [
    (DefVal $lo, (LLVMFPTrunc (GetNamedOperand $src0, si_llvm_float_ty), si_llvm_half_ty)),
    (DefVal $hi, (LLVMFPTrunc (GetNamedOperand $src1, si_llvm_float_ty), si_llvm_half_ty)),
    (DefVal $vec, (LLVMInsertElement
      (LLVMInsertElement
        (PoisionVal si_llvm_v2f16_ty),
        (GetVal $lo), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $hi), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $vec))
  ];
}
def V_CVT_PK_BF16_F32_e64_sem : InstSISemantic<V_CVT_PK_BF16_F32_e64> {
  let Semantic = [
    (DefVal $lo, (LLVMFPTrunc (GetNamedOperand $src0, si_llvm_float_ty), si_llvm_bfloat_ty)),
    (DefVal $hi, (LLVMFPTrunc (GetNamedOperand $src1, si_llvm_float_ty), si_llvm_bfloat_ty)),
    (DefVal $vec, (LLVMInsertElement
      (LLVMInsertElement
        (PoisionVal si_llvm_v2bf16_ty),
        (GetVal $lo), (ConstantInt si_llvm_i32_ty, 0)),
      (GetVal $hi), (ConstantInt si_llvm_i32_ty, 1))),
    (SetNamedOperand $vdst, (GetVal $vec))
  ];
}

//==============================================================================
// V_ASHR_PK + V_MAX_BF16_PSEUDO + V_INTERP_inreg (Phase 22, re-add)
//==============================================================================
def V_ASHR_PK_I8_I32_e64_sem : InstSISemantic<V_ASHR_PK_I8_I32_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_ashr_pk_i8_i32,
      [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}
def V_ASHR_PK_U8_I32_e64_sem : InstSISemantic<V_ASHR_PK_U8_I32_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_ashr_pk_u8_i32,
      [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}
def V_MAX_BF16_PSEUDO_e64_sem : InstSISemantic<V_MAX_BF16_PSEUDO_e64> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMBinaryIntrinsic int_maxnum,
      (GetNamedOperand $src0, si_llvm_bfloat_ty),
      (GetNamedOperand $src1, si_llvm_bfloat_ty)))];
}
def V_INTERP_P10_F32_inreg_sem : InstSISemantic<V_INTERP_P10_F32_inreg> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_interp_inreg_p10,
      [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}
def V_INTERP_P2_F32_inreg_sem : InstSISemantic<V_INTERP_P2_F32_inreg> {
  let Semantic = [(SetNamedOperand $vdst,
    (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_interp_inreg_p2,
      [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

//==============================================================================
// V_CVT_SCALEF32_* family (Phase 21) — auto-generated from opcode-derived
// int_amdgcn_cvt_scalef32_* intrinsics.
//==============================================================================

def V_CVT_SCALEF32_2XPK16_BF6_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_2XPK16_BF6_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_2xpk16_bf6_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_2XPK16_FP6_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_2XPK16_FP6_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_2xpk16_fp6_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_F16_BF8_e64_sem : InstSISemantic<V_CVT_SCALEF32_F16_BF8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_cvt_scalef32_f16_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_F16_FP8_e64_sem : InstSISemantic<V_CVT_SCALEF32_F16_FP8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_cvt_scalef32_f16_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_F32_BF8_e64_sem : InstSISemantic<V_CVT_SCALEF32_F32_BF8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_cvt_scalef32_f32_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_F32_FP8_e64_sem : InstSISemantic<V_CVT_SCALEF32_F32_FP8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_float_ty, int_amdgcn_cvt_scalef32_f32_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK16_BF6_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK16_BF6_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_pk16_bf6_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK16_BF6_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK16_BF6_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_pk16_bf6_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK16_BF6_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK16_BF6_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_pk16_bf6_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK16_FP6_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK16_FP6_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_pk16_fp6_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK16_FP6_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK16_FP6_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_pk16_fp6_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK16_FP6_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK16_FP6_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_pk16_fp6_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK32_BF16_BF6_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK32_BF16_BF6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32bf16_ty, int_amdgcn_cvt_scalef32_pk32_bf16_bf6, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK32_BF16_FP6_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK32_BF16_FP6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32bf16_ty, int_amdgcn_cvt_scalef32_pk32_bf16_fp6, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK32_BF6_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK32_BF6_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_pk32_bf6_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK32_BF6_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK32_BF6_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_pk32_bf6_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK32_F16_BF6_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK32_F16_BF6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f16_ty, int_amdgcn_cvt_scalef32_pk32_f16_bf6, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK32_F16_FP6_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK32_F16_FP6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f16_ty, int_amdgcn_cvt_scalef32_pk32_f16_fp6, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK32_F32_BF6_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK32_F32_BF6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_cvt_scalef32_pk32_f32_bf6, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK32_F32_FP6_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK32_F32_FP6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v32f32_ty, int_amdgcn_cvt_scalef32_pk32_f32_fp6, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK32_FP6_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK32_FP6_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_pk32_fp6_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK32_FP6_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK32_FP6_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_pk32_fp6_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK8_BF8_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK8_BF8_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_pk8_bf8_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK8_BF8_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK8_BF8_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_pk8_bf8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK8_BF8_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK8_BF8_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_pk8_bf8_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK8_FP4_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK8_FP4_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_pk8_fp4_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK8_FP4_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK8_FP4_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_pk8_fp4_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK8_FP4_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK8_FP4_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_pk8_fp4_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK8_FP8_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK8_FP8_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_pk8_fp8_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK8_FP8_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK8_FP8_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_pk8_fp8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK8_FP8_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK8_FP8_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_pk8_fp8_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK_BF16_BF8_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_BF16_BF8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2bf16_ty, int_amdgcn_cvt_scalef32_pk_bf16_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK_BF16_FP4_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_BF16_FP4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2bf16_ty, int_amdgcn_cvt_scalef32_pk_bf16_fp4, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK_BF16_FP8_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_BF16_FP8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2bf16_ty, int_amdgcn_cvt_scalef32_pk_bf16_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK_BF8_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_BF8_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_scalef32_pk_bf8_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_PK_BF8_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_BF8_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_scalef32_pk_bf8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_PK_BF8_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_BF8_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_scalef32_pk_bf8_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_PK_F16_BF8_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_F16_BF8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2f16_ty, int_amdgcn_cvt_scalef32_pk_f16_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK_F16_FP4_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_F16_FP4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2f16_ty, int_amdgcn_cvt_scalef32_pk_f16_fp4, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK_F16_FP8_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_F16_FP8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2f16_ty, int_amdgcn_cvt_scalef32_pk_f16_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK_F32_BF8_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_F32_BF8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2f32_ty, int_amdgcn_cvt_scalef32_pk_f32_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK_F32_FP4_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_F32_FP4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2f32_ty, int_amdgcn_cvt_scalef32_pk_f32_fp4, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK_F32_FP8_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_F32_FP8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2f32_ty, int_amdgcn_cvt_scalef32_pk_f32_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))];
}

def V_CVT_SCALEF32_PK_FP4_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_FP4_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_scalef32_pk_fp4_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_PK_FP4_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_FP4_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_scalef32_pk_fp4_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_PK_FP4_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_FP4_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_scalef32_pk_fp4_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_PK_FP8_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_FP8_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_scalef32_pk_fp8_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_PK_FP8_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_FP8_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_scalef32_pk_fp8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_PK_FP8_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_PK_FP8_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i16_ty, int_amdgcn_cvt_scalef32_pk_fp8_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_SR_BF8_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_BF8_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_scalef32_sr_bf8_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_SR_BF8_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_BF8_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_scalef32_sr_bf8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_SR_BF8_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_BF8_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_scalef32_sr_bf8_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_SR_FP8_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_FP8_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_scalef32_sr_fp8_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_SR_FP8_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_FP8_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_scalef32_sr_fp8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_SR_FP8_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_FP8_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_scalef32_sr_fp8_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SCALEF32_SR_PK16_BF6_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK16_BF6_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_sr_pk16_bf6_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK16_BF6_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK16_BF6_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_sr_pk16_bf6_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK16_BF6_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK16_BF6_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_sr_pk16_bf6_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK16_FP6_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK16_FP6_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_sr_pk16_fp6_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK16_FP6_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK16_FP6_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_sr_pk16_fp6_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK16_FP6_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK16_FP6_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v3i32_ty, int_amdgcn_cvt_scalef32_sr_pk16_fp6_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK32_BF6_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK32_BF6_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_sr_pk32_bf6_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK32_BF6_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK32_BF6_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_sr_pk32_bf6_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK32_BF6_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK32_BF6_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_sr_pk32_bf6_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK32_FP6_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK32_FP6_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_sr_pk32_fp6_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK32_FP6_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK32_FP6_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_sr_pk32_fp6_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK32_FP6_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK32_FP6_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v6i32_ty, int_amdgcn_cvt_scalef32_sr_pk32_fp6_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK8_BF8_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK8_BF8_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_sr_pk8_bf8_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK8_BF8_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK8_BF8_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_sr_pk8_bf8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK8_BF8_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK8_BF8_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_sr_pk8_bf8_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK8_FP4_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK8_FP4_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_sr_pk8_fp4_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK8_FP4_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK8_FP4_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_sr_pk8_fp4_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK8_FP4_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK8_FP4_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_sr_pk8_fp4_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK8_FP8_BF16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK8_FP8_BF16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_sr_pk8_fp8_bf16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK8_FP8_F16_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK8_FP8_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_sr_pk8_fp8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SCALEF32_SR_PK8_FP8_F32_e64_sem : InstSISemantic<V_CVT_SCALEF32_SR_PK8_FP8_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2i32_ty, int_amdgcn_cvt_scalef32_sr_pk8_fp8_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

//==============================================================================
// V_CVT_SR_* / V_CVT_SCALE_PK* (Phase 22)
//==============================================================================

def V_CVT_SCALE_PK16_BF16_BF6_e64_sem : InstSISemantic<V_CVT_SCALE_PK16_BF16_BF6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16bf16_ty, int_amdgcn_cvt_scale_pk16_bf16_bf6, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK16_BF16_FP6_e64_sem : InstSISemantic<V_CVT_SCALE_PK16_BF16_FP6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16bf16_ty, int_amdgcn_cvt_scale_pk16_bf16_fp6, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK16_F16_BF6_e64_sem : InstSISemantic<V_CVT_SCALE_PK16_F16_BF6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f16_ty, int_amdgcn_cvt_scale_pk16_f16_bf6, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK16_F16_FP6_e64_sem : InstSISemantic<V_CVT_SCALE_PK16_F16_FP6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f16_ty, int_amdgcn_cvt_scale_pk16_f16_fp6, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK16_F32_BF6_e64_sem : InstSISemantic<V_CVT_SCALE_PK16_F32_BF6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_cvt_scale_pk16_f32_bf6, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK16_F32_FP6_e64_sem : InstSISemantic<V_CVT_SCALE_PK16_F32_FP6_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v16f32_ty, int_amdgcn_cvt_scale_pk16_f32_fp6, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK8_BF16_BF8_e64_sem : InstSISemantic<V_CVT_SCALE_PK8_BF16_BF8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v8bf16_ty, int_amdgcn_cvt_scale_pk8_bf16_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK8_BF16_FP4_e64_sem : InstSISemantic<V_CVT_SCALE_PK8_BF16_FP4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v8bf16_ty, int_amdgcn_cvt_scale_pk8_bf16_fp4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK8_BF16_FP8_e64_sem : InstSISemantic<V_CVT_SCALE_PK8_BF16_FP8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v8bf16_ty, int_amdgcn_cvt_scale_pk8_bf16_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK8_F16_BF8_e64_sem : InstSISemantic<V_CVT_SCALE_PK8_F16_BF8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v8f16_ty, int_amdgcn_cvt_scale_pk8_f16_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK8_F16_FP4_e64_sem : InstSISemantic<V_CVT_SCALE_PK8_F16_FP4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v8f16_ty, int_amdgcn_cvt_scale_pk8_f16_fp4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK8_F16_FP8_e64_sem : InstSISemantic<V_CVT_SCALE_PK8_F16_FP8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v8f16_ty, int_amdgcn_cvt_scale_pk8_f16_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK8_F32_BF8_e64_sem : InstSISemantic<V_CVT_SCALE_PK8_F32_BF8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v8f32_ty, int_amdgcn_cvt_scale_pk8_f32_bf8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK8_F32_FP4_e64_sem : InstSISemantic<V_CVT_SCALE_PK8_F32_FP4_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v8f32_ty, int_amdgcn_cvt_scale_pk8_f32_fp4, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SCALE_PK8_F32_FP8_e64_sem : InstSISemantic<V_CVT_SCALE_PK8_F32_FP8_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v8f32_ty, int_amdgcn_cvt_scale_pk8_f32_fp8, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $scale_sel)]))];
}

def V_CVT_SR_BF16_F32_e64_sem : InstSISemantic<V_CVT_SR_BF16_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_bfloat_ty, int_amdgcn_cvt_sr_bf16_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SR_BF8_F16_e64_sem : InstSISemantic<V_CVT_SR_BF8_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_sr_bf8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in), (GetNamedOperand $byte_sel)]))];
}

def V_CVT_SR_BF8_F16_fake16_e64_sem : InstSISemantic<V_CVT_SR_BF8_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_sr_bf8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in), (GetNamedOperand $byte_sel)]))];
}

def V_CVT_SR_BF8_F16_t16_e64_sem : InstSISemantic<V_CVT_SR_BF8_F16_t16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_sr_bf8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in), (GetNamedOperand $byte_sel)]))];
}

def V_CVT_SR_F16_F32_e64_sem : InstSISemantic<V_CVT_SR_F16_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_half_ty, int_amdgcn_cvt_sr_f16_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in)]))];
}

def V_CVT_SR_FP8_F16_e64_sem : InstSISemantic<V_CVT_SR_FP8_F16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_sr_fp8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in), (GetNamedOperand $byte_sel)]))];
}

def V_CVT_SR_FP8_F16_fake16_e64_sem : InstSISemantic<V_CVT_SR_FP8_F16_fake16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_sr_fp8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in), (GetNamedOperand $byte_sel)]))];
}

def V_CVT_SR_FP8_F16_t16_e64_sem : InstSISemantic<V_CVT_SR_FP8_F16_t16_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_i8_ty, int_amdgcn_cvt_sr_fp8_f16, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $vdst_in), (GetNamedOperand $byte_sel)]))];
}

def V_CVT_SR_PK_BF16_F32_e64_sem : InstSISemantic<V_CVT_SR_PK_BF16_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2bf16_ty, int_amdgcn_cvt_sr_pk_bf16_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

def V_CVT_SR_PK_F16_F32_e64_sem : InstSISemantic<V_CVT_SR_PK_F16_F32_e64> {
  let Semantic = [(SetNamedOperand $vdst, (LLVMIntrinsic si_llvm_v2f16_ty, int_amdgcn_cvt_sr_pk_f16_f32, [(GetNamedOperand $src0), (GetNamedOperand $src1), (GetNamedOperand $src2)]))];
}

//==============================================================================
// V tail final remnants (Phase 23)
//==============================================================================
// V_CVT_SCALEF32_SR_PK_FP4_<output> — SR-rounded PK FP4 conversion. Map to
// the corresponding non-SR intrinsic since SR rounding has no IR equivalent;
// passes through src0 with scale.
foreach inst = [V_CVT_SCALEF32_SR_PK_FP4_BF16_e64,
                V_CVT_SCALEF32_SR_PK_FP4_F16_e64,
                V_CVT_SCALEF32_SR_PK_FP4_F32_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
    ];
  }
}

// V_MFMA_LD_SCALE_B32 / V_WMMA_LD_SCALE_*: load matrix scale from MFMA reg.
// No clean intrinsic; emit fence placeholders.
def V_MFMA_LD_SCALE_B32_sem : InstSISemantic<V_MFMA_LD_SCALE_B32> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}
def V_WMMA_LD_SCALE_PAIRED_B32_sem : InstSISemantic<V_WMMA_LD_SCALE_PAIRED_B32> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}
def V_WMMA_LD_SCALE16_PAIRED_B64_sem : InstSISemantic<V_WMMA_LD_SCALE16_PAIRED_B64> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

// V_MOV_B64_DPP_PSEUDO: 64-bit move with DPP. Model as plain move.
def V_MOV_B64_DPP_PSEUDO_sem : InstSISemantic<V_MOV_B64_DPP_PSEUDO> {
  let Semantic = [(SetNamedOperand $vdst, (GetNamedOperand $src0, si_llvm_i64_ty))];
}

// V_PERM_PK16_B{4,6,8}_U4 — packed byte permutation. Approximate as plain src0 forward.
foreach inst = [V_PERM_PK16_B4_U4_e64, V_PERM_PK16_B6_U4_e64, V_PERM_PK16_B8_U4_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [(SetNamedOperand $vdst, (GetNamedOperand $src0))];
  }
}
