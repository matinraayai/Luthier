//===- VOPSem.td -------------------------------------------------*- tblgen -*-===//
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
// VOP2 Instructions
//==============================================================================

//------------------------------------------------------------------------------
// VOP2 Float Arithmetic (F32)
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
def V_ADD_F32_e32_sem : InstSISemantic<V_ADD_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
        llvm_i32_ty))
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
def V_SUB_F32_e32_sem : InstSISemantic<V_SUB_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFSub (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
        llvm_i32_ty))
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
def V_SUBREV_F32_e32_sem : InstSISemantic<V_SUBREV_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFSub (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty)),
        llvm_i32_ty))
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
def V_MUL_F32_e32_sem : InstSISemantic<V_MUL_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

// D.f = S0.f * S1.f (DX9 mul_legacy: 0.0 * anything = 0.0).
// Sources (GFX6-GFX10 only, removed in RDNA3+):
//   gcn3-instruction-set-architecture.pdf, page 181.
//   vega-shader-instruction-set-architecture.pdf, page 129.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 126.
//   rdna-shader-instruction-set-architecture.pdf, page 140.
//   rdna2-shader-instruction-set-architecture.pdf, page 139.
def V_MUL_LEGACY_F32_e32_sem : InstSISemantic<V_MUL_LEGACY_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                  (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Min/Max Float (F32)
//------------------------------------------------------------------------------

// D.f = max(S0.f, S1.f) (IEEE 754-2008 maxNum: if one operand is NaN,
// return the other).
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
def V_MAX_F32_e32_sem : InstSISemantic<V_MAX_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMaxNum (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

// D.f = min(S0.f, S1.f) (IEEE 754-2008 minNum: if one operand is NaN,
// return the other).
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
def V_MIN_F32_e32_sem : InstSISemantic<V_MIN_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMinNum (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

// D.f = max(S0.f, S1.f) (DX10 legacy: if S0 >= S1, D = S0, else D = S1;
// NaN propagation depends on operand order).
// GFX6/GFX7 only (isGFX6GFX7).
// LLVM-only pseudo (no separate ISA opcode in PDFs; shares opcode with V_MAX_F32).
// Defined in llvm-project: llvm/lib/Target/AMDGPU/VOP2Instructions.td.
def V_MAX_LEGACY_F32_e32_sem : InstSISemantic<V_MAX_LEGACY_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMSelect (LLVMFCmp SETOGE,
                      (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                      (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
                    (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

// D.f = min(S0.f, S1.f) (DX10 legacy: if S0 < S1, D = S0, else D = S1;
// NaN propagation depends on operand order).
// GFX6/GFX7 only (isGFX6GFX7).
// LLVM-only pseudo (no separate ISA opcode in PDFs; shares opcode with V_MIN_F32).
// Defined in llvm-project: llvm/lib/Target/AMDGPU/VOP2Instructions.td.
def V_MIN_LEGACY_F32_e32_sem : InstSISemantic<V_MIN_LEGACY_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMSelect (LLVMFCmp SETOLT,
                      (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                      (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
                    (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Integer Arithmetic
//------------------------------------------------------------------------------

// D.u = S0.u + S1.u (no carry out).
// Sources (GFX9+ only, HasAddNoCarryInsts):
//   vega-shader-instruction-set-architecture.pdf, page 134.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 183.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 195.
def V_ADD_U32_e32_sem : InstSISemantic<V_ADD_U32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

// D.u = S0.u - S1.u (no carry out).
// Sources (GFX9+ only, HasAddNoCarryInsts):
//   vega-shader-instruction-set-architecture.pdf, page 134.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 135.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 130.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 129.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 183.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 195.
def V_SUB_U32_e32_sem : InstSISemantic<V_SUB_U32_e32> {
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
def V_SUBREV_U32_e32_sem : InstSISemantic<V_SUBREV_U32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSub (GetNamedOperand $src1), (GetNamedOperand $src0)))
  ];
}

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
def V_MUL_I32_I24_e32_sem : InstSISemantic<V_MUL_I32_I24_e32> {
  let Semantic = [
    // Sign-extend bits [23:0] by shifting left 8 then arithmetic right 8
    (DefVal $a, (LLVMAShr (LLVMShl (GetNamedOperand $src0),
                                    (ConstantInt llvm_i32_ty, 8)),
                           (ConstantInt llvm_i32_ty, 8))),
    (DefVal $b, (LLVMAShr (LLVMShl (GetNamedOperand $src1),
                                    (ConstantInt llvm_i32_ty, 8)),
                           (ConstantInt llvm_i32_ty, 8))),
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
def V_MUL_HI_I32_I24_e32_sem : InstSISemantic<V_MUL_HI_I32_I24_e32> {
  let Semantic = [
    (DefVal $a, (LLVMSExt (LLVMAShr (LLVMShl (GetNamedOperand $src0),
                                              (ConstantInt llvm_i32_ty, 8)),
                                     (ConstantInt llvm_i32_ty, 8)),
                           llvm_i64_ty)),
    (DefVal $b, (LLVMSExt (LLVMAShr (LLVMShl (GetNamedOperand $src1),
                                              (ConstantInt llvm_i32_ty, 8)),
                                     (ConstantInt llvm_i32_ty, 8)),
                           llvm_i64_ty)),
    (SetNamedOperand $vdst,
      (LLVMTrunc (LLVMAShr (LLVMMul (GetVal $a), (GetVal $b)),
                            (ConstantInt llvm_i64_ty, 32)),
                 llvm_i32_ty))
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
def V_MUL_U32_U24_e32_sem : InstSISemantic<V_MUL_U32_U24_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMMul (LLVMAnd (GetNamedOperand $src0),
                         (ConstantInt llvm_i32_ty, 16777215)),
               (LLVMAnd (GetNamedOperand $src1),
                         (ConstantInt llvm_i32_ty, 16777215))))
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
def V_MUL_HI_U32_U24_e32_sem : InstSISemantic<V_MUL_HI_U32_U24_e32> {
  let Semantic = [
    (DefVal $a, (LLVMZExt (LLVMAnd (GetNamedOperand $src0),
                                    (ConstantInt llvm_i32_ty, 16777215)),
                           llvm_i64_ty)),
    (DefVal $b, (LLVMZExt (LLVMAnd (GetNamedOperand $src1),
                                    (ConstantInt llvm_i32_ty, 16777215)),
                           llvm_i64_ty)),
    (SetNamedOperand $vdst,
      (LLVMTrunc (LLVMLShr (LLVMMul (GetVal $a), (GetVal $b)),
                            (ConstantInt llvm_i64_ty, 32)),
                 llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Min/Max Integer
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
def V_MAX_I32_e32_sem : InstSISemantic<V_MAX_I32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp SETGT, (GetNamedOperand $src0),
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
def V_MIN_I32_e32_sem : InstSISemantic<V_MIN_I32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp SETLT, (GetNamedOperand $src0),
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
def V_MAX_U32_e32_sem : InstSISemantic<V_MAX_U32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp SETUGT, (GetNamedOperand $src0),
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
def V_MIN_U32_e32_sem : InstSISemantic<V_MIN_U32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp SETULT, (GetNamedOperand $src0),
                                    (GetNamedOperand $src1)),
                  (GetNamedOperand $src0), (GetNamedOperand $src1)))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Add/Sub with Carry Out (VCC)
//------------------------------------------------------------------------------

// D.u = S0.u + S1.u; VCC = unsigned carry out.
// In GCN3/Vega this is V_ADD_I32 (later renamed V_ADD_CO_U32).
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
def V_ADD_CO_U32_e32_sem : InstSISemantic<V_ADD_CO_U32_e32> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMAdd (LLVMZExt (GetNamedOperand $src0), llvm_i64_ty),
               (LLVMZExt (GetNamedOperand $src1), llvm_i64_ty))),
    (ImplicitDef VCC, (LLVMCmp SETNE,
      (LLVMAnd (LLVMLShr (GetVal $tmp), (ConstantInt llvm_i64_ty, 32)),
               (ConstantInt llvm_i64_ty, 1)),
      (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), llvm_i32_ty))
  ];
}

// D.u = S0.u - S1.u; VCC = unsigned borrow out.
// In GCN3 this is V_SUB_I32 (later renamed V_SUB_CO_U32).
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
def V_SUB_CO_U32_e32_sem : InstSISemantic<V_SUB_CO_U32_e32> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMSub (LLVMZExt (GetNamedOperand $src0), llvm_i64_ty),
               (LLVMZExt (GetNamedOperand $src1), llvm_i64_ty))),
    (ImplicitDef VCC, (LLVMCmp SETNE,
      (LLVMAnd (LLVMLShr (GetVal $tmp), (ConstantInt llvm_i64_ty, 32)),
               (ConstantInt llvm_i64_ty, 1)),
      (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), llvm_i32_ty))
  ];
}

// D.u = S1.u - S0.u; VCC = unsigned borrow out (reversed operands).
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
def V_SUBREV_CO_U32_e32_sem : InstSISemantic<V_SUBREV_CO_U32_e32> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMSub (LLVMZExt (GetNamedOperand $src1), llvm_i64_ty),
               (LLVMZExt (GetNamedOperand $src0), llvm_i64_ty))),
    (ImplicitDef VCC, (LLVMCmp SETNE,
      (LLVMAnd (LLVMLShr (GetVal $tmp), (ConstantInt llvm_i64_ty, 32)),
               (ConstantInt llvm_i64_ty, 1)),
      (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Add/Sub with Carry In/Out (VCC)
//------------------------------------------------------------------------------

// D.u = S0.u + S1.u + VCC; VCC = carry out.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 165 (as V_ADDC_U32).
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 175.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 187.
def V_ADDC_U32_e32_sem : InstSISemantic<V_ADDC_U32_e32> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMAdd (LLVMAdd (LLVMZExt (GetNamedOperand $src0), llvm_i64_ty),
                         (LLVMZExt (GetNamedOperand $src1), llvm_i64_ty)),
               (LLVMZExt (ImplicitUse VCC), llvm_i64_ty))),
    (ImplicitDef VCC, (LLVMCmp SETNE,
      (LLVMAnd (LLVMLShr (GetVal $tmp), (ConstantInt llvm_i64_ty, 32)),
               (ConstantInt llvm_i64_ty, 1)),
      (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), llvm_i32_ty))
  ];
}

// D.u = S0.u - S1.u - VCC; VCC = borrow out.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 184 (as V_SUBB_U32).
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 175.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 188.
def V_SUBB_U32_e32_sem : InstSISemantic<V_SUBB_U32_e32> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMSub (LLVMSub (LLVMZExt (GetNamedOperand $src0), llvm_i64_ty),
                         (LLVMZExt (GetNamedOperand $src1), llvm_i64_ty)),
               (LLVMZExt (ImplicitUse VCC), llvm_i64_ty))),
    (ImplicitDef VCC, (LLVMCmp SETNE,
      (LLVMAnd (LLVMLShr (GetVal $tmp), (ConstantInt llvm_i64_ty, 32)),
               (ConstantInt llvm_i64_ty, 1)),
      (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), llvm_i32_ty))
  ];
}

// D.u = S1.u - S0.u - VCC; VCC = borrow out (reversed operands).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 185 (as V_SUBBREV_U32).
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 176.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 188.
def V_SUBBREV_U32_e32_sem : InstSISemantic<V_SUBBREV_U32_e32> {
  let Semantic = [
    (DefVal $tmp,
      (LLVMSub (LLVMSub (LLVMZExt (GetNamedOperand $src1), llvm_i64_ty),
                         (LLVMZExt (GetNamedOperand $src0), llvm_i64_ty)),
               (LLVMZExt (ImplicitUse VCC), llvm_i64_ty))),
    (ImplicitDef VCC, (LLVMCmp SETNE,
      (LLVMAnd (LLVMLShr (GetVal $tmp), (ConstantInt llvm_i64_ty, 32)),
               (ConstantInt llvm_i64_ty, 1)),
      (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $vdst, (LLVMTrunc (GetVal $tmp), llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Logical Operations
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
def V_AND_B32_e32_sem : InstSISemantic<V_AND_B32_e32> {
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
def V_OR_B32_e32_sem : InstSISemantic<V_OR_B32_e32> {
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
def V_XOR_B32_e32_sem : InstSISemantic<V_XOR_B32_e32> {
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
def V_XNOR_B32_e32_sem : InstSISemantic<V_XNOR_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      LLVMNot<(LLVMXor (GetNamedOperand $src0),
                        (GetNamedOperand $src1)), llvm_i32_ty>.ret)
  ];
}

//------------------------------------------------------------------------------
// VOP2 Shift Operations
//------------------------------------------------------------------------------

// D.u = S0.u << S1.u[4:0].
// GFX6/GFX7 only (isGFX6GFX7). Non-reversed operand order.
// Defined in llvm-project: llvm/lib/Target/AMDGPU/VOP2Instructions.td.
def V_LSHL_B32_e32_sem : InstSISemantic<V_LSHL_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMShl (GetNamedOperand $src0),
               (LLVMAnd (GetNamedOperand $src1),
                         (ConstantInt llvm_i32_ty, 31))))
  ];
}

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
def V_LSHLREV_B32_e32_sem : InstSISemantic<V_LSHLREV_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMShl (GetNamedOperand $src1),
               (LLVMAnd (GetNamedOperand $src0),
                         (ConstantInt llvm_i32_ty, 31))))
  ];
}

// D.u = S0.u >> S1.u[4:0] (logical shift right).
// GFX6/GFX7 only (isGFX6GFX7). Non-reversed operand order.
// Defined in llvm-project: llvm/lib/Target/AMDGPU/VOP2Instructions.td.
def V_LSHR_B32_e32_sem : InstSISemantic<V_LSHR_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMLShr (GetNamedOperand $src0),
                (LLVMAnd (GetNamedOperand $src1),
                          (ConstantInt llvm_i32_ty, 31))))
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
def V_LSHRREV_B32_e32_sem : InstSISemantic<V_LSHRREV_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMLShr (GetNamedOperand $src1),
                (LLVMAnd (GetNamedOperand $src0),
                          (ConstantInt llvm_i32_ty, 31))))
  ];
}

// D.i = S0.i >> S1.u[4:0] (arithmetic shift right).
// GFX6/GFX7 only (isGFX6GFX7). Non-reversed operand order.
// Defined in llvm-project: llvm/lib/Target/AMDGPU/VOP2Instructions.td.
def V_ASHR_I32_e32_sem : InstSISemantic<V_ASHR_I32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAShr (GetNamedOperand $src0),
                (LLVMAnd (GetNamedOperand $src1),
                          (ConstantInt llvm_i32_ty, 31))))
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
def V_ASHRREV_I32_e32_sem : InstSISemantic<V_ASHRREV_I32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAShr (GetNamedOperand $src1),
                (LLVMAnd (GetNamedOperand $src0),
                          (ConstantInt llvm_i32_ty, 31))))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Conditional Move
//------------------------------------------------------------------------------

// D.u = VCC[lane] ? S1.u : S0.u. Conditional mask on VCC.
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
def V_CNDMASK_B32_e32_sem : InstSISemantic<V_CNDMASK_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSelect (ImplicitUse VCC),
                  (GetNamedOperand $src1), (GetNamedOperand $src0)))
  ];
}

// D.u16 = VCC[lane] ? S1.u16 : S0.u16. Conditional mask on VCC (16-bit).
// Sources (GFX11+ only, isGFX11Plus):
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 434.
//   rdna4-instruction-set-architecture.pdf, page 498.
def V_CNDMASK_B16_e32_sem : InstSISemantic<V_CNDMASK_B16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMSelect (ImplicitUse VCC),
                  (GetNamedOperand $src1), (GetNamedOperand $src0)))
  ];
}

//------------------------------------------------------------------------------
// VOP2 Bit Operations
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
def V_BFM_B32_e32_sem : InstSISemantic<V_BFM_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMShl
        (LLVMSub (LLVMShl (ConstantInt llvm_i32_ty, 1),
                           (LLVMAnd (GetNamedOperand $src0),
                                     (ConstantInt llvm_i32_ty, 31))),
                 (ConstantInt llvm_i32_ty, 1)),
        (LLVMAnd (GetNamedOperand $src1), (ConstantInt llvm_i32_ty, 31))))
  ];
}

// D.u = CountOneBits(S0.u) + S1.u. Bit count.
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
def V_BCNT_U32_B32_e32_sem : InstSISemantic<V_BCNT_U32_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMAdd (LLVMCtPop (GetNamedOperand $src0)),
               (GetNamedOperand $src1)))
  ];
}

// D.u = MaskedBitCount(S0.u, EXEC[31:0]) + S1.u.
// Count bits set in S0 for lanes below current lane in lower 32-bit EXEC mask.
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
def V_MBCNT_LO_U32_B32_e32_sem : InstSIIntrinsic<V_MBCNT_LO_U32_B32_e32,
                                                    int_amdgcn_mbcnt_lo>;

// D.u = MaskedBitCount(S0.u, EXEC[63:32]) + S1.u.
// Count bits set in S0 for lanes below current lane in upper 32-bit EXEC mask.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 174.
//   vega-shader-instruction-set-architecture.pdf, page 180.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 173.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 171.
//   rdna-shader-instruction-set-architecture.pdf, page 197.
//   rdna2-shader-instruction-set-architecture.pdf, page 192.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 444.
//   rdna4-instruction-set-architecture.pdf, page 516.
def V_MBCNT_HI_U32_B32_e32_sem : InstSIIntrinsic<V_MBCNT_HI_U32_B32_e32,
                                                    int_amdgcn_mbcnt_hi>;

//------------------------------------------------------------------------------
// VOP2 LDEXP
//------------------------------------------------------------------------------

// D.f = S0.f * 2^S1.i (C math ldexp function).
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
def V_LDEXP_F32_e32_sem : InstSISemantic<V_LDEXP_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMLdexp (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                   (GetNamedOperand $src1)),
        llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 F16 Arithmetic
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
foreach inst = [V_ADD_F16_e32, V_ADD_F16_t16_e32, V_ADD_F16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFAdd
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
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
foreach inst = [V_SUB_F16_e32, V_SUB_F16_t16_e32, V_SUB_F16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFSub
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

// D.f16 = S1.f16 - S0.f16 (reversed operands).
// Sources: same as V_SUB_F16 above.
foreach inst = [V_SUBREV_F16_e32, V_SUBREV_F16_t16_e32,
                V_SUBREV_F16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFSub
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
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
foreach inst = [V_MUL_F16_e32, V_MUL_F16_t16_e32, V_MUL_F16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFMul
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
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
foreach inst = [V_MAX_F16_e32, V_MAX_F16_t16_e32, V_MAX_F16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMMaxNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
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
foreach inst = [V_MIN_F16_e32, V_MIN_F16_t16_e32, V_MIN_F16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMMinNum
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

// D.f16 = S0.f16 * 2^S1.i (ldexp for half precision).
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
foreach inst = [V_LDEXP_F16_e32, V_LDEXP_F16_t16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMLdexp
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

//------------------------------------------------------------------------------
// VOP2 I16 Arithmetic
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
def V_ADD_U16_e32_sem : InstSISemantic<V_ADD_U16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAdd (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                 (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)),
        llvm_i32_ty))
  ];
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
def V_SUB_U16_e32_sem : InstSISemantic<V_SUB_U16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSub (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                 (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)),
        llvm_i32_ty))
  ];
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
def V_SUBREV_U16_e32_sem : InstSISemantic<V_SUBREV_U16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSub (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                 (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
        llvm_i32_ty))
  ];
}

// D.u16 = S0.u16 * S1.u16 (low 16 bits of unsigned multiply).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 181.
//   vega-shader-instruction-set-architecture.pdf, page 133.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 180.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 192.
def V_MUL_LO_U16_e32_sem : InstSISemantic<V_MUL_LO_U16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMMul (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                 (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)),
        llvm_i32_ty))
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
def V_MAX_I16_e32_sem : InstSISemantic<V_MAX_I16_e32> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp SETGT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        llvm_i32_ty))
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
def V_MIN_I16_e32_sem : InstSISemantic<V_MIN_I16_e32> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp SETLT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        llvm_i32_ty))
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
def V_MAX_U16_e32_sem : InstSISemantic<V_MAX_U16_e32> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp SETUGT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        llvm_i32_ty))
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
def V_MIN_U16_e32_sem : InstSISemantic<V_MIN_U16_e32> {
  let Semantic = [
    (DefVal $a, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
    (DefVal $b, (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)),
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMSelect (LLVMCmp SETULT, (GetVal $a), (GetVal $b)),
                    (GetVal $a), (GetVal $b)),
        llvm_i32_ty))
  ];
}

// D.u16 = S1.u16 << S0.u16 (reversed operand order, 16-bit shift left).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 168.
//   vega-shader-instruction-set-architecture.pdf, page 133.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 134.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 180.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 192.
def V_LSHLREV_B16_e32_sem : InstSISemantic<V_LSHLREV_B16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMShl (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                 (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           (ConstantInt llvm_i16_ty, 15))),
        llvm_i32_ty))
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
def V_LSHRREV_B16_e32_sem : InstSISemantic<V_LSHRREV_B16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMLShr (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                  (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                            (ConstantInt llvm_i16_ty, 15))),
        llvm_i32_ty))
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
def V_ASHRREV_I16_e32_sem : InstSISemantic<V_ASHRREV_I16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMAShr (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                  (LLVMAnd (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                            (ConstantInt llvm_i16_ty, 15))),
        llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 F64 Arithmetic (GFX12+ only, isGFX12Plus)
//------------------------------------------------------------------------------

// D.d = S0.d + S1.d.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 299.
def V_ADD_F64_pseudo_e32_sem : InstSISemantic<V_ADD_F64_pseudo_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)),
        llvm_i64_ty))
  ];
}

// D.d = S0.d * S1.d.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 300.
def V_MUL_F64_pseudo_e32_sem : InstSISemantic<V_MUL_F64_pseudo_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
                  (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)),
        llvm_i64_ty))
  ];
}

// D.d = min(S0.d, S1.d) (IEEE minNum).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 302.
def V_MIN_NUM_F64_e32_sem : InstSISemantic<V_MIN_NUM_F64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMinNum (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
                    (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)),
        llvm_i64_ty))
  ];
}

// D.d = max(S0.d, S1.d) (IEEE maxNum).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 303.
def V_MAX_NUM_F64_e32_sem : InstSISemantic<V_MAX_NUM_F64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMMaxNum (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
                    (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)),
        llvm_i64_ty))
  ];
}

// D.u64 = S1.u64 << S0.u[5:0] (64-bit shift left, reversed operand order).
// Sources (GFX12+ only):
//   rdna4-instruction-set-architecture.pdf, page 307.
def V_LSHLREV_B64_pseudo_e32_sem : InstSISemantic<V_LSHLREV_B64_pseudo_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMShl (GetNamedOperand $src1),
               (LLVMZExt (LLVMAnd (GetNamedOperand $src0),
                                   (ConstantInt llvm_i32_ty, 63)),
                          llvm_i64_ty)))
  ];
}

//------------------------------------------------------------------------------
// VOP2 MAC (Multiply-Accumulate, non-fused)
//------------------------------------------------------------------------------

// D.f += S0.f * S1.f (non-fused multiply-accumulate).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 170.
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 143.
def V_MAC_F32_e32_sem : InstSISemantic<V_MAC_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd
          (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src2), llvm_float_ty)),
        llvm_i32_ty))
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
def V_MAC_F16_e32_sem : InstSISemantic<V_MAC_F16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFAdd
            (LLVMFMul
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), llvm_i16_ty),
                         llvm_half_ty)),
          llvm_i16_ty),
        llvm_i32_ty))
  ];
}

// D.f += S0.f * S1.f (DX9 legacy non-fused multiply-accumulate: 0.0 * x = 0.0).
// Sources (isGFX6GFX7GFX10):
//   rdna-shader-instruction-set-architecture.pdf, page 140.
def V_MAC_LEGACY_F32_e32_sem : InstSISemantic<V_MAC_LEGACY_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd
          (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src2), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 FMAC (Fused Multiply-Accumulate)
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
def V_FMAC_F32_e32_sem : InstSISemantic<V_FMAC_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src2), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

// D.f16 = FMA(S0.f16, S1.f16, D.f16) (fused multiply-accumulate, f16).
// Sources (isGFX10Plus):
//   rdna-shader-instruction-set-architecture.pdf, page 145.
//   rdna2-shader-instruction-set-architecture.pdf, page 143.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 270.
//   rdna4-instruction-set-architecture.pdf, page 313.
foreach inst = [V_FMAC_F16_e32, V_FMAC_F16_t16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFMA
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src2), llvm_i16_ty),
                           llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

// D.d = FMA(S0.d, S1.d, D.d) (fused multiply-accumulate, f64).
// Sources (HasFmacF64Inst):
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 125.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 170.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 182.
def V_FMAC_F64_e32_sem : InstSISemantic<V_FMAC_F64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
                 (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty),
                 (LLVMBitCast (GetNamedOperand $src2), llvm_double_ty)),
        llvm_i64_ty))
  ];
}

// D.f = FMA(S0.f, S1.f, D.f) (DX9 legacy fused multiply-accumulate).
// Sources (HasFmaLegacy32):
//   rdna2-shader-instruction-set-architecture.pdf, page 257.
def V_FMAC_LEGACY_F32_e32_sem : InstSISemantic<V_FMAC_LEGACY_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src2), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 MAD with Inline Constant (MADAK/MADMK, non-fused)
//------------------------------------------------------------------------------

// D.f = S0.f * S1.f + K (MAD with Add Konstant, non-fused).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 171.
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 143.
def V_MADAK_F32_sem : InstSISemantic<V_MADAK_F32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd
          (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $imm), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

// D.f = S0.f * K + S1.f (MAD with Multiply Konstant, non-fused).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 171.
//   vega-shader-instruction-set-architecture.pdf, page 131.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 132.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 128.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 143.
def V_MADMK_F32_sem : InstSISemantic<V_MADMK_F32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFAdd
          (LLVMFMul (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                    (LLVMBitCast (GetNamedOperand $imm), llvm_float_ty)),
          (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

// D.f16 = S0.f16 * S1.f16 + K (MAD with Add Konstant, f16, non-fused).
// Sources (Has16BitInsts):
//   gcn3-instruction-set-architecture.pdf, page 170.
//   vega-shader-instruction-set-architecture.pdf, page 132.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 179.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 191.
def V_MADAK_F16_sem : InstSISemantic<V_MADAK_F16> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFAdd
            (LLVMFMul
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty)),
            (LLVMBitCast (GetNamedOperand $imm), llvm_half_ty)),
          llvm_i16_ty),
        llvm_i32_ty))
  ];
}

// D.f16 = S0.f16 * K + S1.f16 (MAD with Multiply Konstant, f16, non-fused).
// Sources (Has16BitInsts):
//   gcn3-instruction-set-architecture.pdf, page 171.
//   vega-shader-instruction-set-architecture.pdf, page 132.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 133.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 129.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 128.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 178.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 191.
def V_MADMK_F16_sem : InstSISemantic<V_MADMK_F16> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMZExt
        (LLVMBitCast
          (LLVMFAdd
            (LLVMFMul
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (GetNamedOperand $imm), llvm_half_ty)),
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                         llvm_half_ty)),
          llvm_i16_ty),
        llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP2 FMA with Inline Constant (FMAAK/FMAMK, fused)
//------------------------------------------------------------------------------

// D.f = FMA(S0.f, S1.f, K) (FMA with Add Konstant).
// Sources (HasFmaakFmamkF32Insts):
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 175.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 187.
//   rdna-shader-instruction-set-architecture.pdf, page 144.
//   rdna2-shader-instruction-set-architecture.pdf, page 142.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 268.
//   rdna4-instruction-set-architecture.pdf, page 310.
def V_FMAAK_F32_sem : InstSISemantic<V_FMAAK_F32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $imm), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

// D.f = FMA(S0.f, K, S1.f) (FMA with Multiply Konstant).
// Sources (HasFmaakFmamkF32Insts):
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 174.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 187.
//   rdna-shader-instruction-set-architecture.pdf, page 144.
//   rdna2-shader-instruction-set-architecture.pdf, page 142.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 268.
//   rdna4-instruction-set-architecture.pdf, page 310.
def V_FMAMK_F32_sem : InstSISemantic<V_FMAMK_F32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $imm), llvm_float_ty),
                 (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)),
        llvm_i32_ty))
  ];
}

// D.f16 = FMA(S0.f16, S1.f16, K) (FMA with Add Konstant, f16).
// Sources (isGFX10Plus):
//   rdna-shader-instruction-set-architecture.pdf, page 145.
//   rdna2-shader-instruction-set-architecture.pdf, page 143.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 270.
//   rdna4-instruction-set-architecture.pdf, page 313.
foreach inst = [V_FMAAK_F16, V_FMAAK_F16_t16] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFMA
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (GetNamedOperand $imm), llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

// D.f16 = FMA(S0.f16, K, S1.f16) (FMA with Multiply Konstant, f16).
// Sources (isGFX10Plus):
//   rdna-shader-instruction-set-architecture.pdf, page 145.
//   rdna2-shader-instruction-set-architecture.pdf, page 143.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 270.
//   rdna4-instruction-set-architecture.pdf, page 313.
foreach inst = [V_FMAMK_F16, V_FMAMK_F16_t16] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFMA
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                           llvm_half_ty),
              (LLVMBitCast (GetNamedOperand $imm), llvm_half_ty),
              (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                           llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

//------------------------------------------------------------------------------
// VOP2 Conversion / Pack Instructions
//------------------------------------------------------------------------------

// D.u = f32_to_f16_rtz(S1.f) << 16 | f32_to_f16_rtz(S0.f).
// Convert two F32 values to F16 with round-toward-zero and pack.
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
def V_CVT_PKRTZ_F16_F32_e32_sem : InstSIIntrinsic<V_CVT_PKRTZ_F16_F32_e32,
                                                    int_amdgcn_cvt_pkrtz>;

// D.u = snorm16(S0.f) | (snorm16(S1.f) << 16).
// Convert two F32 to signed normalized I16 and pack.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 238.
//   vega-shader-instruction-set-architecture.pdf, page 266.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 268.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 253.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 247.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 540.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 587.
//   rdna-shader-instruction-set-architecture.pdf, page 274.
//   rdna2-shader-instruction-set-architecture.pdf, page 268.
def V_CVT_PKNORM_I16_F32_e32_sem : InstSIIntrinsic<V_CVT_PKNORM_I16_F32_e32,
                                                     int_amdgcn_cvt_pknorm_i16>;

// D.u = unorm16(S0.f) | (unorm16(S1.f) << 16).
// Convert two F32 to unsigned normalized U16 and pack.
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
def V_CVT_PKNORM_U16_F32_e32_sem : InstSIIntrinsic<V_CVT_PKNORM_U16_F32_e32,
                                                     int_amdgcn_cvt_pknorm_u16>;

// D.u = i16_sat(S0.i) | (i16_sat(S1.i) << 16).
// Pack two I32 to I16 with saturation.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 237.
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
def V_CVT_PK_I16_I32_e32_sem : InstSIIntrinsic<V_CVT_PK_I16_I32_e32,
                                                 int_amdgcn_cvt_pk_i16>;

// D.u = u16_sat(S0.u) | (u16_sat(S1.u) << 16).
// Pack two U32 to U16 with saturation.
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
def V_CVT_PK_U16_U32_e32_sem : InstSIIntrinsic<V_CVT_PK_U16_U32_e32,
                                                 int_amdgcn_cvt_pk_u16>;

// Convert F32 to U8 and accumulate into byte position.
// D.u[byte_sel] = f32_to_u8(S0.f), other bytes unchanged.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 238.
//   vega-shader-instruction-set-architecture.pdf, page 265.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 267.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 252.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 246.
def V_CVT_PKACCUM_U8_F32_e32_sem : InstSIIntrinsic<V_CVT_PKACCUM_U8_F32_e32,
                                                     int_amdgcn_cvt_pk_u8_f32>;

//------------------------------------------------------------------------------
// VOP2 DOT Product Instructions
//------------------------------------------------------------------------------

// D.f += dot2(S0.f16x2, S1.f16x2).
// Two-element dot product of f16 vectors accumulated into f32.
// Sources (HasDot5Insts):
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 184.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 196.
//   rdna2-shader-instruction-set-architecture.pdf, page 139.
def V_DOT2C_F32_F16_e32_sem : InstSIIntrinsic<V_DOT2C_F32_F16_e32,
                                                int_amdgcn_fdot2>;

// D.i += dot2(S0.i16x2, S1.i16x2).
// Two-element dot product of i16 vectors accumulated into i32.
// Sources (HasDot4Insts):
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 184.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 196.
def V_DOT2C_I32_I16_e32_sem : InstSIIntrinsic<V_DOT2C_I32_I16_e32,
                                                int_amdgcn_sdot2>;

// D.i += dot4(S0.i8x4, S1.i8x4).
// Four-element dot product of i8 vectors accumulated into i32.
// Sources (HasDot6Insts):
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 184.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 196.
//   rdna2-shader-instruction-set-architecture.pdf, page 140.
def V_DOT4C_I32_I8_e32_sem : InstSIIntrinsic<V_DOT4C_I32_I8_e32,
                                               int_amdgcn_sdot4>;

// D.i += dot8(S0.i4x8, S1.i4x8).
// Eight-element dot product of i4 vectors accumulated into i32.
// Sources (HasDot3Insts):
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 184.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 197.
def V_DOT8C_I32_I4_e32_sem : InstSIIntrinsic<V_DOT8C_I32_I4_e32,
                                               int_amdgcn_sdot8>;

//------------------------------------------------------------------------------
// VOP2 Lane Operations
//------------------------------------------------------------------------------

// D.u = S0.u[S1.u]. Read value from a specific VGPR lane.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 259.
//   vega-shader-instruction-set-architecture.pdf, page 179.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 181.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 172.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 170.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 362.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 408.
//   rdna-shader-instruction-set-architecture.pdf, page 196.
//   rdna2-shader-instruction-set-architecture.pdf, page 191.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 451.
//   rdna4-instruction-set-architecture.pdf, page 523.
def V_READLANE_B32_sem : InstSIIntrinsic<V_READLANE_B32,
                                          int_amdgcn_readlane>;

// D.u[S1.u] = S0.u. Write value to a specific VGPR lane.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 187.
//   vega-shader-instruction-set-architecture.pdf, page 180.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 182.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 172.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 171.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 363.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 408.
//   rdna-shader-instruction-set-architecture.pdf, page 197.
//   rdna2-shader-instruction-set-architecture.pdf, page 192.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 452.
//   rdna4-instruction-set-architecture.pdf, page 524.
def V_WRITELANE_B32_sem : InstSIIntrinsic<V_WRITELANE_B32,
                                           int_amdgcn_writelane>;

//------------------------------------------------------------------------------
// VOP2 Packed Operations
//------------------------------------------------------------------------------

// D.f16x2 = S0.f16x2 * S1.f16x2 + D.f16x2.
// Packed 2xf16 fused multiply-accumulate.
// Sources (HasPkFmacF16Inst):
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 131.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 130.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 185.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 197.
//   rdna-shader-instruction-set-architecture.pdf, page 146.
//   rdna2-shader-instruction-set-architecture.pdf, page 144.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 273.
//   rdna4-instruction-set-architecture.pdf, page 314.
def V_PK_FMAC_F16_e32_sem : InstSISemantic<V_PK_FMAC_F16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFMA (LLVMBitCast (GetNamedOperand $src0), llvm_v2f16_ty),
                 (LLVMBitCast (GetNamedOperand $src1), llvm_v2f16_ty),
                 (LLVMBitCast (GetNamedOperand $vdst), llvm_v2f16_ty)),
        llvm_i32_ty))
  ];
}

//==============================================================================
// VOP1 Instructions
//==============================================================================

//------------------------------------------------------------------------------
// VOP1 Move
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
def V_MOV_B32_e32_sem : InstSISemantic<V_MOV_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
}

// D.u64 = S0.u64. 64-bit vector move.
// Sources (GFX12+ only):
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 203.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 216.
def V_MOV_B64_e32_sem : InstSISemantic<V_MOV_B64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
}

// D.u16 = S0.u16. 16-bit vector move (True16).
// Sources (GFX11+):
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 303.
//   rdna4-instruction-set-architecture.pdf, page 344.
def V_MOV_B16_t16_e32_sem : InstSISemantic<V_MOV_B16_t16_e32> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
}

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
def V_NOP_e32_sem : InstSISemantic<V_NOP_e32> {
  let Semantic = [];
}

// D.u = S0.u. Read value from first active lane of VGPR into SGPR.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 219.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 187.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 200.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 274.
//   rdna4-instruction-set-architecture.pdf, page 315.
def V_READFIRSTLANE_B32_sem : InstSIIntrinsic<V_READFIRSTLANE_B32,
                                                int_amdgcn_readfirstlane>;

//------------------------------------------------------------------------------
// VOP1 Float-to-Int / Int-to-Float Conversion (F32/I32/U32)
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
def V_CVT_F32_I32_e32_sem : InstSISemantic<V_CVT_F32_I32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMSIToFP (GetNamedOperand $src0), llvm_float_ty),
                   llvm_i32_ty))
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
def V_CVT_F32_U32_e32_sem : InstSISemantic<V_CVT_F32_U32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUIToFP (GetNamedOperand $src0), llvm_float_ty),
                   llvm_i32_ty))
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
def V_CVT_I32_F32_e32_sem : InstSISemantic<V_CVT_I32_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToSI (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                   llvm_i32_ty))
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
def V_CVT_U32_F32_e32_sem : InstSISemantic<V_CVT_U32_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToUI (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                   llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Float Precision Conversion (F16<->F32, F32<->F64)
//------------------------------------------------------------------------------

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
foreach inst = [V_CVT_F16_F32_e32, V_CVT_F16_F32_t16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFPTrunc (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                         llvm_half_ty),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
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
foreach inst = [V_CVT_F32_F16_e32, V_CVT_F32_F16_t16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMBitCast
          (LLVMFPExt (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                              llvm_i16_ty),
                                   llvm_half_ty),
                     llvm_float_ty),
          llvm_i32_ty))
    ];
  }
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
def V_CVT_F64_F32_e32_sem : InstSISemantic<V_CVT_F64_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFPExt (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                   llvm_double_ty),
        llvm_i64_ty))
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
def V_CVT_F32_F64_e32_sem : InstSISemantic<V_CVT_F32_F64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast
        (LLVMFPTrunc (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
                     llvm_float_ty),
        llvm_i32_ty))
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
def V_CVT_F64_I32_e32_sem : InstSISemantic<V_CVT_F64_I32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMSIToFP (GetNamedOperand $src0), llvm_double_ty),
                   llvm_i64_ty))
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
def V_CVT_F64_U32_e32_sem : InstSISemantic<V_CVT_F64_U32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMUIToFP (GetNamedOperand $src0), llvm_double_ty),
                   llvm_i64_ty))
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
def V_CVT_I32_F64_e32_sem : InstSISemantic<V_CVT_I32_F64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToSI (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
                   llvm_i32_ty))
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
def V_CVT_U32_F64_e32_sem : InstSISemantic<V_CVT_U32_F64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToUI (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
                   llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Unsigned Byte to Float Conversion
//------------------------------------------------------------------------------

// D.f = (float)S0.u[7:0]. Convert unsigned byte 0 to float.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 195.
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
foreach i = 0-3 in {
  defvar inst = ["V_CVT_F32_UBYTE0_e32", "V_CVT_F32_UBYTE1_e32",
                 "V_CVT_F32_UBYTE2_e32", "V_CVT_F32_UBYTE3_e32"][i];
  defvar shift = [0, 8, 16, 24][i];

  def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMBitCast
          (LLVMUIToFP
            (LLVMAnd (LLVMLShr (GetNamedOperand $src0),
                               (ConstantInt llvm_i32_ty, shift)),
                     (ConstantInt llvm_i32_ty, 255)),
            llvm_float_ty),
          llvm_i32_ty))
    ];
  }
}

//------------------------------------------------------------------------------
// VOP1 Special Float-to-Int Conversions
//------------------------------------------------------------------------------

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
def V_CVT_FLR_I32_F32_e32_sem : InstSISemantic<V_CVT_FLR_I32_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToSI
        (LLVMFFloor (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty)),
        llvm_i32_ty))
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
def V_CVT_RPI_I32_F32_e32_sem : InstSISemantic<V_CVT_RPI_I32_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMFPToSI
        (LLVMFFloor
          (LLVMFAdd (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
                    (ConstantFP llvm_float_ty, 0.5))),
        llvm_i32_ty))
  ];
}

// D.f = 4-bit signed int to float. Used for texture sampling offsets.
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
def V_CVT_OFF_F32_I4_e32_sem : InstSISemantic<V_CVT_OFF_F32_I4_e32> {
  let Semantic = [
    // Sign-extend 4-bit value by left-shift 28 then arithmetic right-shift 28
    (DefVal $sext, (LLVMAShr (LLVMShl (GetNamedOperand $src0),
                                       (ConstantInt llvm_i32_ty, 28)),
                              (ConstantInt llvm_i32_ty, 28))),
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMSIToFP (GetVal $sext), llvm_float_ty),
                   llvm_i32_ty))
  ];
}

//------------------------------------------------------------------------------
// VOP1 F16 Conversion Instructions
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
foreach inst = [V_CVT_F16_I16_e32, V_CVT_F16_I16_t16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMSIToFP (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                         llvm_half_ty),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
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
foreach inst = [V_CVT_F16_U16_e32, V_CVT_F16_U16_t16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMUIToFP (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                         llvm_half_ty),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
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
foreach inst = [V_CVT_I16_F16_e32, V_CVT_I16_F16_t16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMFPToSI
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                         llvm_half_ty),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
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
foreach inst = [V_CVT_U16_F16_e32, V_CVT_U16_F16_t16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMFPToUI
            (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                         llvm_half_ty),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

// D.i = sign_extend(S0.i16). I16 to I32 sign extension.
// Sources (GFX11+):
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 302.
//   rdna4-instruction-set-architecture.pdf, page 344.
foreach inst = [V_CVT_I32_I16_e32, V_CVT_I32_I16_t16_e32,
                V_CVT_I32_I16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMSExt (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                  llvm_i32_ty))
    ];
  }
}

// D.u = zero_extend(S0.u16). U16 to U32 zero extension.
// Sources (GFX11+):
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 303.
//   rdna4-instruction-set-architecture.pdf, page 344.
foreach inst = [V_CVT_U32_U16_e32, V_CVT_U32_U16_t16_e32,
                V_CVT_U32_U16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                  llvm_i32_ty))
    ];
  }
}

// Normalized F16 to I16/U16 conversions (GFX11+ intrinsics).
// Sources:
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 210.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 223.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 300.
//   rdna4-instruction-set-architecture.pdf, page 341.
foreach inst = [V_CVT_NORM_I16_F16_e32, V_CVT_NORM_I16_F16_t16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_cvt_pknorm_i16>;
}
foreach inst = [V_CVT_NORM_U16_F16_e32, V_CVT_NORM_U16_F16_t16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_cvt_pknorm_u16>;
}

// BF8/FP8 to F32 conversions (GFX9+ intrinsics).
// Sources:
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 540.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 587.
def V_CVT_F32_BF8_e32_sem : InstSIIntrinsic<V_CVT_F32_BF8_e32,
                                              int_amdgcn_cvt_f32_bf8>;
def V_CVT_F32_BF8_OP_SEL_e32_sem : InstSIIntrinsic<V_CVT_F32_BF8_OP_SEL_e32,
                                                      int_amdgcn_cvt_f32_bf8>;
def V_CVT_F32_FP8_e32_sem : InstSIIntrinsic<V_CVT_F32_FP8_e32,
                                              int_amdgcn_cvt_f32_fp8>;
def V_CVT_F32_FP8_OP_SEL_e32_sem : InstSIIntrinsic<V_CVT_F32_FP8_OP_SEL_e32,
                                                      int_amdgcn_cvt_f32_fp8>;
def V_CVT_PK_F32_BF8_e32_sem : InstSIIntrinsic<V_CVT_PK_F32_BF8_e32,
                                                  int_amdgcn_cvt_pk_f32_bf8>;
def V_CVT_PK_F32_BF8_OP_SEL_e32_sem : InstSIIntrinsic<V_CVT_PK_F32_BF8_OP_SEL_e32,
                                                          int_amdgcn_cvt_pk_f32_bf8>;
def V_CVT_PK_F32_FP8_e32_sem : InstSIIntrinsic<V_CVT_PK_F32_FP8_e32,
                                                  int_amdgcn_cvt_pk_f32_fp8>;
def V_CVT_PK_F32_FP8_OP_SEL_e32_sem : InstSIIntrinsic<V_CVT_PK_F32_FP8_OP_SEL_e32,
                                                          int_amdgcn_cvt_pk_f32_fp8>;

//------------------------------------------------------------------------------
// VOP1 Rounding Instructions
//------------------------------------------------------------------------------

// D.f = ceil(S0.f). Floating-point ceiling.
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
def V_CEIL_F32_e32_sem : InstSISemantic<V_CEIL_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMFCeil (LLVMBitCast (GetNamedOperand $src0),
                                            llvm_float_ty)),
                   llvm_i32_ty))
  ];
}

def V_CEIL_F64_e32_sem : InstSISemantic<V_CEIL_F64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMFCeil (LLVMBitCast (GetNamedOperand $src0),
                                            llvm_double_ty)),
                   llvm_i64_ty))
  ];
}

foreach inst = [V_CEIL_F16_e32, V_CEIL_F16_t16_e32,
                V_CEIL_F16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFCeil (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                                llvm_i16_ty),
                                    llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

// D.f = floor(S0.f). Floating-point floor.
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
def V_FLOOR_F32_e32_sem : InstSISemantic<V_FLOOR_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMFFloor (LLVMBitCast (GetNamedOperand $src0),
                                             llvm_float_ty)),
                   llvm_i32_ty))
  ];
}

def V_FLOOR_F64_e32_sem : InstSISemantic<V_FLOOR_F64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMFFloor (LLVMBitCast (GetNamedOperand $src0),
                                             llvm_double_ty)),
                   llvm_i64_ty))
  ];
}

foreach inst = [V_FLOOR_F16_e32, V_FLOOR_F16_t16_e32,
                V_FLOOR_F16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFFloor (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                                 llvm_i16_ty),
                                     llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

// D.f = trunc(S0.f). Floating-point truncation toward zero.
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
def V_TRUNC_F32_e32_sem : InstSISemantic<V_TRUNC_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMFTrunc (LLVMBitCast (GetNamedOperand $src0),
                                             llvm_float_ty)),
                   llvm_i32_ty))
  ];
}

def V_TRUNC_F64_e32_sem : InstSISemantic<V_TRUNC_F64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMFTrunc (LLVMBitCast (GetNamedOperand $src0),
                                             llvm_double_ty)),
                   llvm_i64_ty))
  ];
}

foreach inst = [V_TRUNC_F16_e32, V_TRUNC_F16_t16_e32,
                V_TRUNC_F16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMFTrunc (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                                  llvm_i16_ty),
                                      llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

// D.f = roundeven(S0.f). Round to nearest even.
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
def V_RNDNE_F32_e32_sem : InstSISemantic<V_RNDNE_F32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMRoundEven (LLVMBitCast (GetNamedOperand $src0),
                                                llvm_float_ty)),
                   llvm_i32_ty))
  ];
}

def V_RNDNE_F64_e32_sem : InstSISemantic<V_RNDNE_F64_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      (LLVMBitCast (LLVMRoundEven (LLVMBitCast (GetNamedOperand $src0),
                                                llvm_double_ty)),
                   llvm_i64_ty))
  ];
}

foreach inst = [V_RNDNE_F16_e32, V_RNDNE_F16_t16_e32,
                V_RNDNE_F16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMBitCast
            (LLVMRoundEven (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0),
                                                     llvm_i16_ty),
                                         llvm_half_ty)),
            llvm_i16_ty),
          llvm_i32_ty))
    ];
  }
}

//------------------------------------------------------------------------------
// VOP1 Fractional Part
//------------------------------------------------------------------------------

// D.f = fract(S0.f) = S0.f - floor(S0.f).
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
def V_FRACT_F32_e32_sem : InstSIIntrinsic<V_FRACT_F32_e32,
                                            int_amdgcn_fract>;

def V_FRACT_F64_e32_sem : InstSIIntrinsic<V_FRACT_F64_e32,
                                            int_amdgcn_fract>;

foreach inst = [V_FRACT_F16_e32, V_FRACT_F16_t16_e32,
                V_FRACT_F16_fake16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_fract>;
}

//------------------------------------------------------------------------------
// VOP1 FREXP (Exponent and Mantissa Extraction)
//------------------------------------------------------------------------------

// D.i = frexp_exp(S0.f). Extract exponent of float.
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
def V_FREXP_EXP_I32_F32_e32_sem : InstSIIntrinsic<V_FREXP_EXP_I32_F32_e32,
                                                    int_amdgcn_frexp_exp>;

def V_FREXP_EXP_I32_F64_e32_sem : InstSIIntrinsic<V_FREXP_EXP_I32_F64_e32,
                                                    int_amdgcn_frexp_exp>;

foreach inst = [V_FREXP_EXP_I16_F16_e32, V_FREXP_EXP_I16_F16_t16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_frexp_exp>;
}

// D.f = frexp_mant(S0.f). Extract mantissa of float.
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
def V_FREXP_MANT_F32_e32_sem : InstSIIntrinsic<V_FREXP_MANT_F32_e32,
                                                 int_amdgcn_frexp_mant>;

def V_FREXP_MANT_F64_e32_sem : InstSIIntrinsic<V_FREXP_MANT_F64_e32,
                                                 int_amdgcn_frexp_mant>;

foreach inst = [V_FREXP_MANT_F16_e32, V_FREXP_MANT_F16_t16_e32,
                V_FREXP_MANT_F16_fake16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_frexp_mant>;
}

//------------------------------------------------------------------------------
// VOP1 Transcendental Math (Hardware Approximations)
//------------------------------------------------------------------------------

// D.f = 1.0 / S0.f. Reciprocal (hardware approximation).
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
def V_RCP_F32_e32_sem : InstSIIntrinsic<V_RCP_F32_e32, int_amdgcn_rcp>;
def V_RCP_F64_e32_sem : InstSIIntrinsic<V_RCP_F64_e32, int_amdgcn_rcp>;

foreach inst = [V_RCP_F16_e32, V_RCP_F16_t16_e32,
                V_RCP_F16_fake16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_rcp>;
}

// Clamped reciprocal (GFX6/GFX7 only).
// Defined in llvm-project: llvm/lib/Target/AMDGPU/VOP1Instructions.td.
def V_RCP_CLAMP_F32_e32_sem : InstSIIntrinsic<V_RCP_CLAMP_F32_e32,
                                                int_amdgcn_rcp>;
def V_RCP_CLAMP_F64_e32_sem : InstSIIntrinsic<V_RCP_CLAMP_F64_e32,
                                                int_amdgcn_rcp>;

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
def V_RCP_IFLAG_F32_e32_sem : InstSIIntrinsic<V_RCP_IFLAG_F32_e32,
                                                int_amdgcn_rcp>;

// Legacy reciprocal (GFX6/GFX7 only, DX9 rules).
// Defined in llvm-project: llvm/lib/Target/AMDGPU/VOP1Instructions.td.
def V_RCP_LEGACY_F32_e32_sem : InstSIIntrinsic<V_RCP_LEGACY_F32_e32,
                                                 int_amdgcn_rcp_legacy>;

// D.f = 1.0 / sqrt(S0.f). Reciprocal square root (hardware approximation).
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
def V_RSQ_F32_e32_sem : InstSIIntrinsic<V_RSQ_F32_e32, int_amdgcn_rsq>;
def V_RSQ_F64_e32_sem : InstSIIntrinsic<V_RSQ_F64_e32, int_amdgcn_rsq>;

foreach inst = [V_RSQ_F16_e32, V_RSQ_F16_t16_e32,
                V_RSQ_F16_fake16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_rsq>;
}

// Clamped RSQ and legacy RSQ (GFX6/GFX7 only).
def V_RSQ_CLAMP_F32_e32_sem : InstSIIntrinsic<V_RSQ_CLAMP_F32_e32,
                                                int_amdgcn_rsq_clamp>;
def V_RSQ_CLAMP_F64_e32_sem : InstSIIntrinsic<V_RSQ_CLAMP_F64_e32,
                                                int_amdgcn_rsq_clamp>;
def V_RSQ_LEGACY_F32_e32_sem : InstSIIntrinsic<V_RSQ_LEGACY_F32_e32,
                                                 int_amdgcn_rsq_legacy>;

// D.f = sqrt(S0.f). Square root (hardware approximation).
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
def V_SQRT_F32_e32_sem : InstSIIntrinsic<V_SQRT_F32_e32, int_amdgcn_sqrt>;
def V_SQRT_F64_e32_sem : InstSIIntrinsic<V_SQRT_F64_e32, int_amdgcn_sqrt>;

foreach inst = [V_SQRT_F16_e32, V_SQRT_F16_t16_e32,
                V_SQRT_F16_fake16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_sqrt>;
}

// D.f = sin(S0.f * 2 * PI). Hardware sine (input is in turns, not radians).
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
def V_SIN_F32_e32_sem : InstSIIntrinsic<V_SIN_F32_e32, int_amdgcn_sin>;

foreach inst = [V_SIN_F16_e32, V_SIN_F16_t16_e32,
                V_SIN_F16_fake16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_sin>;
}

// D.f = cos(S0.f * 2 * PI). Hardware cosine (input is in turns, not radians).
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
def V_COS_F32_e32_sem : InstSIIntrinsic<V_COS_F32_e32, int_amdgcn_cos>;

foreach inst = [V_COS_F16_e32, V_COS_F16_t16_e32,
                V_COS_F16_fake16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_cos>;
}

// D.f = 2^S0.f. Hardware base-2 exponential.
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
def V_EXP_F32_e32_sem : InstSIIntrinsic<V_EXP_F32_e32, int_amdgcn_exp2>;

foreach inst = [V_EXP_F16_e32, V_EXP_F16_t16_e32,
                V_EXP_F16_fake16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_exp2>;
}

// Legacy exp2 (GFX6-GFX9 only).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 205.
//   vega-shader-instruction-set-architecture.pdf, page 148.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 149.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 142.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 142.
def V_EXP_LEGACY_F32_e32_sem : InstSIIntrinsic<V_EXP_LEGACY_F32_e32,
                                                 int_amdgcn_exp2>;

// D.f = log2(S0.f). Hardware base-2 logarithm.
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
def V_LOG_F32_e32_sem : InstSIIntrinsic<V_LOG_F32_e32, int_amdgcn_log>;

foreach inst = [V_LOG_F16_e32, V_LOG_F16_t16_e32,
                V_LOG_F16_fake16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_log>;
}

// Clamped log2 (GFX6/GFX7 only).
def V_LOG_CLAMP_F32_e32_sem : InstSIIntrinsic<V_LOG_CLAMP_F32_e32,
                                                int_amdgcn_log_clamp>;

// Legacy log2 (GFX6-GFX9 only).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 214.
//   vega-shader-instruction-set-architecture.pdf, page 148.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 149.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 142.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 142.
def V_LOG_LEGACY_F32_e32_sem : InstSIIntrinsic<V_LOG_LEGACY_F32_e32,
                                                 int_amdgcn_log>;

//------------------------------------------------------------------------------
// VOP1 Bit Operations
//------------------------------------------------------------------------------

// D.u = bitreverse(S0.u). Reverse bit order.
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
def V_BFREV_B32_e32_sem : InstSISemantic<V_BFREV_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst, (LLVMBitReverse (GetNamedOperand $src0)))
  ];
}

// D.u = ~S0.u. Bitwise NOT (32-bit).
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
def V_NOT_B32_e32_sem : InstSISemantic<V_NOT_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst,
      LLVMNot<(GetNamedOperand $src0), llvm_i32_ty>.ret)
  ];
}

// D.u16 = ~S0.u16. Bitwise NOT (16-bit).
// Sources (GFX11+):
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 302.
//   rdna4-instruction-set-architecture.pdf, page 343.
foreach inst = [V_NOT_B16_e32, V_NOT_B16_t16_e32,
                V_NOT_B16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $vdst,
        (LLVMZExt
          (LLVMXor (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                   (LLVMSExt (ConstantTrue llvm_i1_ty), llvm_i16_ty)),
          llvm_i32_ty))
    ];
  }
}

//------------------------------------------------------------------------------
// VOP1 Find First Bit
//------------------------------------------------------------------------------

// D.u = position of first 1 bit in S0.u from MSB (count leading zeros).
// Returns 0xFFFFFFFF if S0 == 0.
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
def V_FFBH_U32_e32_sem : InstSISemantic<V_FFBH_U32_e32> {
  let Semantic = [
    (DefVal $clz, (LLVMCtlz (GetNamedOperand $src0),
                             (ConstantFalse llvm_i1_ty))),
    // ctlz returns 32 when input is 0; FFBH returns 0xFFFFFFFF
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp SETEQ, (GetNamedOperand $src0),
                                   (ConstantInt llvm_i32_ty, 0)),
                  (ConstantSignedInt llvm_i32_ty, -1),
                  (GetVal $clz)))
  ];
}

// D.u = position of first bit different from sign bit in S0.i from MSB.
// Returns 0xFFFFFFFF if S0 == 0 or S0 == -1.
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
def V_FFBH_I32_e32_sem : InstSISemantic<V_FFBH_I32_e32> {
  let Semantic = [
    // XOR with sign-extended sign bit to find first differing bit
    (DefVal $xored, (LLVMXor (GetNamedOperand $src0),
                              (LLVMAShr (GetNamedOperand $src0),
                                        (ConstantInt llvm_i32_ty, 31)))),
    (DefVal $clz, (LLVMCtlz (GetVal $xored), (ConstantFalse llvm_i1_ty))),
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp SETEQ, (GetVal $xored),
                                   (ConstantInt llvm_i32_ty, 0)),
                  (ConstantSignedInt llvm_i32_ty, -1),
                  (GetVal $clz)))
  ];
}

// D.u = position of first 1 bit in S0.u from LSB (count trailing zeros).
// Returns 0xFFFFFFFF if S0 == 0.
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
def V_FFBL_B32_e32_sem : InstSISemantic<V_FFBL_B32_e32> {
  let Semantic = [
    (DefVal $ctz, (LLVMCttz (GetNamedOperand $src0),
                             (ConstantFalse llvm_i1_ty))),
    // cttz returns 32 when input is 0; FFBL returns 0xFFFFFFFF
    (SetNamedOperand $vdst,
      (LLVMSelect (LLVMCmp SETEQ, (GetNamedOperand $src0),
                                   (ConstantInt llvm_i32_ty, 0)),
                  (ConstantSignedInt llvm_i32_ty, -1),
                  (GetVal $ctz)))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Lane / Permute Operations
//------------------------------------------------------------------------------

// D.u = S0.u from lane (lane_id XOR 63). Permute lane 64.
// Sources (GFX11+):
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 301.
//   rdna4-instruction-set-architecture.pdf, page 342.
def V_PERMLANE64_B32_sem : InstSIIntrinsic<V_PERMLANE64_B32,
                                            int_amdgcn_permlane64>;

//------------------------------------------------------------------------------
// VOP1 Saturated Pack
//------------------------------------------------------------------------------

// D.u = {pack_u8(S0.i16[31:16]), pack_u8(S0.i16[15:0])}.
// Saturated pack of two signed I16 to two unsigned U8.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 148.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 149.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 142.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 142.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 211.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 224.
//   rdna-shader-instruction-set-architecture.pdf, page 159.
//   rdna2-shader-instruction-set-architecture.pdf, page 157.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 299.
//   rdna4-instruction-set-architecture.pdf, page 341.
foreach inst = [V_SAT_PK_U8_I16_e32, V_SAT_PK_U8_I16_t16_e32,
                V_SAT_PK_U8_I16_fake16_e32] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      // Extract low and high i16 values
      (DefVal $lo, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
      (DefVal $hi, (LLVMTrunc (LLVMLShr (GetNamedOperand $src0),
                                         (ConstantInt llvm_i32_ty, 16)),
                               llvm_i16_ty)),
      // Saturate each to [0, 255] range
      (DefVal $lo_sat, (LLVMSelect
        (LLVMCmp SETLT, (GetVal $lo), (ConstantInt llvm_i16_ty, 0)),
        (ConstantInt llvm_i16_ty, 0),
        (LLVMSelect
          (LLVMCmp SETGT, (GetVal $lo), (ConstantInt llvm_i16_ty, 255)),
          (ConstantInt llvm_i16_ty, 255), (GetVal $lo)))),
      (DefVal $hi_sat, (LLVMSelect
        (LLVMCmp SETLT, (GetVal $hi), (ConstantInt llvm_i16_ty, 0)),
        (ConstantInt llvm_i16_ty, 0),
        (LLVMSelect
          (LLVMCmp SETGT, (GetVal $hi), (ConstantInt llvm_i16_ty, 255)),
          (ConstantInt llvm_i16_ty, 255), (GetVal $hi)))),
      // Pack into result: hi_byte << 8 | lo_byte
      (SetNamedOperand $vdst,
        (LLVMOr (LLVMZExt (GetVal $lo_sat), llvm_i32_ty),
                (LLVMShl (LLVMZExt (GetVal $hi_sat), llvm_i32_ty),
                         (ConstantInt llvm_i32_ty, 8))))
    ];
  }
}

//------------------------------------------------------------------------------
// VOP1 Register-Relative Move Operations
//------------------------------------------------------------------------------

// VGPR register-relative addressing operations.
// These use indirect addressing through M0 register.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 215.
//   rdna-shader-instruction-set-architecture.pdf, page 155.
//   rdna2-shader-instruction-set-architecture.pdf, page 153.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 291.
//   rdna4-instruction-set-architecture.pdf, page 333.

// VGPR[D + M0.u] = S0.u. Write VGPR at offset from M0.
def V_MOVRELD_B32_e32_sem : InstSISemantic<V_MOVRELD_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
}

// D.u = VGPR[S0 + M0.u]. Read VGPR at offset from M0.
def V_MOVRELS_B32_e32_sem : InstSISemantic<V_MOVRELS_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
}

// VGPR[D + M0.u] = VGPR[S0 + M0.u]. Move with relative src and dst.
def V_MOVRELSD_B32_e32_sem : InstSISemantic<V_MOVRELSD_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
}

// VGPR[D + M0.u] = VGPR[S0 + M0.u + 1]. Move with relative src and dst (offset +1).
// Sources (GFX10+):
//   rdna-shader-instruction-set-architecture.pdf, page 156.
//   rdna2-shader-instruction-set-architecture.pdf, page 154.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 293.
//   rdna4-instruction-set-architecture.pdf, page 334.
def V_MOVRELSD_2_B32_e32_sem : InstSISemantic<V_MOVRELSD_2_B32_e32> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
}

//------------------------------------------------------------------------------
// VOP1 Swap / ACCVGPR / Pipeline / Misc
//------------------------------------------------------------------------------

// Swap VGPR contents between two registers.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 148.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 149.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 143.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 143.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 211.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 224.
//   rdna-shader-instruction-set-architecture.pdf, page 160.
//   rdna2-shader-instruction-set-architecture.pdf, page 157.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 300.
//   rdna4-instruction-set-architecture.pdf, page 342.
def V_SWAP_B32_sem : InstSISemantic<V_SWAP_B32> {
  let Semantic = [];
}

// Register-relative swap.
// Sources (GFX10+):
//   rdna-shader-instruction-set-architecture.pdf, page 160.
//   rdna2-shader-instruction-set-architecture.pdf, page 158.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 302.
//   rdna4-instruction-set-architecture.pdf, page 343.
def V_SWAPREL_B32_sem : InstSISemantic<V_SWAPREL_B32> {
  let Semantic = [];
}

// Move between VGPR and accumulator VGPR (CDNA only).
// Sources:
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 211.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 224.
def V_ACCVGPR_MOV_B32_sem : InstSISemantic<V_ACCVGPR_MOV_B32> {
  let Semantic = [
    (SetNamedOperand $vdst, (GetNamedOperand $src0))
  ];
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
def V_CLREXCP_e32_sem : InstSISemantic<V_CLREXCP_e32> {
  let Semantic = [];
}

// Flush VALU pipeline.
// Sources (GFX10+):
//   rdna-shader-instruction-set-architecture.pdf, page 150.
//   rdna2-shader-instruction-set-architecture.pdf, page 148.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 281.
//   rdna4-instruction-set-architecture.pdf, page 322.
def V_PIPEFLUSH_e32_sem : InstSISemantic<V_PIPEFLUSH_e32> {
  let Semantic = [];
}

// Screen partition (GFX9 only).
// LLVM-only pseudo.
// Defined in llvm-project: llvm/lib/Target/AMDGPU/VOP1Instructions.td.
def V_SCREEN_PARTITION_4SE_B32_e32_sem : InstSISemantic<V_SCREEN_PARTITION_4SE_B32_e32> {
  let Semantic = [];
}
