//===- VOPCSem.td ------------------------------------------------*- tblgen -*-===//
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
// VOPC Instructions
//==============================================================================
//
// VOPC instructions perform per-lane comparisons and write the result to VCC
// (V_CMP) or EXEC (V_CMPX). Each lane gets a 1-bit comparison result.
//
// Variants:
//   V_CMP_*    : result → VCC
//   V_CMPS_*   : signaling float compare, result → VCC (GFX6/GFX7 only)
//   V_CMPX_*   : result → VCC and EXEC
//   V_CMPX_*_nosdst : result → EXEC only (GFX10+, HasNoSdstCMPX)
//   V_CMPSX_*  : signaling float compare, result → VCC and EXEC (GFX6/GFX7)
//   V_CMPSX_*_nosdst : signaling, result → EXEC only (GFX6/GFX7)
//   *_t16      : True16 encoding variant (same semantics)
//
// PDF sources for VOPC instructions (VCC = S0 <cmp> S1):
//   gcn3-instruction-set-architecture.pdf, pages 226-260.
//   vega-shader-instruction-set-architecture.pdf, pages 151-164.
//   vega-7nm-shader-instruction-set-architecture.pdf, pages 152-165.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, pages 145-158.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, pages 145-158.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, pages 216-260.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, pages 230-274.
//   rdna-shader-instruction-set-architecture.pdf, pages 161-178.
//   rdna2-shader-instruction-set-architecture.pdf, pages 158-175.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, pages 305-370.
//   rdna4-instruction-set-architecture.pdf, pages 347-410.

//------------------------------------------------------------------------------
// VOPC F32 Ordered Float Comparisons
//------------------------------------------------------------------------------

// VCC = S0.f <cmp> S1.f (ordered: both operands must not be NaN).
foreach i = 0-5 in {
  defvar cmpName = ["EQ", "LG", "GT", "GE", "LT", "LE"][i];
  defvar cmpPred = [SETOEQ, SETONE, SETOGT, SETOGE, SETOLT, SETOLE][i];

  // V_CMP → VCC
  def V_CMP_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  // V_CMPS → VCC (signaling, GFX6/GFX7 only, same semantics)
  def V_CMPS_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  // V_CMPX → VCC + EXEC
  def V_CMPX_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (ImplicitDef VCC, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }

  // V_CMPX_nosdst → EXEC only
  def V_CMPX_#cmpName#_F32_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_nosdst_e32")> {
    let Semantic = [
      (ImplicitDef EXEC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  // V_CMPSX → VCC + EXEC (signaling, GFX6/GFX7 only)
  def V_CMPSX_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (ImplicitDef VCC, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }

  // V_CMPSX_nosdst → EXEC only (signaling, GFX6/GFX7 only)
  def V_CMPSX_#cmpName#_F32_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_nosdst_e32")> {
    let Semantic = [
      (ImplicitDef EXEC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }
}

//------------------------------------------------------------------------------
// VOPC F32 Unordered Float Comparisons
//------------------------------------------------------------------------------

// VCC = !(S0.f <cmp> S1.f) (unordered: true if either operand is NaN).
foreach i = 0-5 in {
  defvar cmpName = ["NEQ", "NLG", "NGT", "NGE", "NLT", "NLE"][i];
  defvar cmpPred = [SETUNE, SETUEQ, SETULE, SETULT, SETUGE, SETUGT][i];

  def V_CMP_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  def V_CMPS_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  def V_CMPX_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (ImplicitDef VCC, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }

  def V_CMPX_#cmpName#_F32_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_nosdst_e32")> {
    let Semantic = [
      (ImplicitDef EXEC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  def V_CMPSX_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (ImplicitDef VCC, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }

  def V_CMPSX_#cmpName#_F32_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_nosdst_e32")> {
    let Semantic = [
      (ImplicitDef EXEC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }
}

//------------------------------------------------------------------------------
// VOPC F32 Special Float Comparisons (Ordered/Unordered/False/True)
//------------------------------------------------------------------------------

// O = ordered (both not NaN), U = unordered (either is NaN)
foreach i = 0-1 in {
  defvar cmpName = ["O", "U"][i];
  defvar cmpPred = [SETO, SETUO][i];

  def V_CMP_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  def V_CMPS_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  def V_CMPX_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (ImplicitDef VCC, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }

  def V_CMPX_#cmpName#_F32_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_nosdst_e32")> {
    let Semantic = [
      (ImplicitDef EXEC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  def V_CMPSX_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (ImplicitDef VCC, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }

  def V_CMPSX_#cmpName#_F32_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_nosdst_e32")> {
    let Semantic = [
      (ImplicitDef EXEC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }
}

// F = always false, TRU = always true (F32)
foreach i = 0-1 in {
  defvar cmpName = ["F", "TRU"][i];
  defvar constVal = [ConstantFalse, ConstantTrue][i];

  def V_CMP_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F32_e32")> {
    let Semantic = [(ImplicitDef VCC, (constVal llvm_i1_ty))];
  }
  def V_CMPS_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F32_e32")> {
    let Semantic = [(ImplicitDef VCC, (constVal llvm_i1_ty))];
  }
  def V_CMPX_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (constVal llvm_i1_ty)),
      (ImplicitDef EXEC, (constVal llvm_i1_ty))
    ];
  }
  def V_CMPX_#cmpName#_F32_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_nosdst_e32")> {
    let Semantic = [(ImplicitDef EXEC, (constVal llvm_i1_ty))];
  }
  def V_CMPSX_#cmpName#_F32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (constVal llvm_i1_ty)),
      (ImplicitDef EXEC, (constVal llvm_i1_ty))
    ];
  }
  def V_CMPSX_#cmpName#_F32_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_nosdst_e32")> {
    let Semantic = [(ImplicitDef EXEC, (constVal llvm_i1_ty))];
  }
}

//------------------------------------------------------------------------------
// VOPC F64 Float Comparisons (all predicates)
// Same patterns as F32 but with llvm_double_ty. No CMPS/CMPSX t16 variants.
//------------------------------------------------------------------------------

// F64 ordered comparisons
foreach i = 0-5 in {
  defvar cmpName = ["EQ", "LG", "GT", "GE", "LT", "LE"][i];
  defvar cmpPred = [SETOEQ, SETONE, SETOGT, SETOGE, SETOLT, SETOLE][i];

  def V_CMP_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F64_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPS_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F64_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPX_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F64_e32")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))),
      (ImplicitDef VCC, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
  def V_CMPX_#cmpName#_F64_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F64_nosdst_e32")> {
    let Semantic = [
      (ImplicitDef EXEC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPSX_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F64_e32")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))),
      (ImplicitDef VCC, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
  def V_CMPSX_#cmpName#_F64_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F64_nosdst_e32")> {
    let Semantic = [
      (ImplicitDef EXEC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
}

// F64 unordered comparisons
foreach i = 0-5 in {
  defvar cmpName = ["NEQ", "NLG", "NGT", "NGE", "NLT", "NLE"][i];
  defvar cmpPred = [SETUNE, SETUEQ, SETULE, SETULT, SETUGE, SETUGT][i];

  def V_CMP_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F64_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPS_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F64_e32")> {
    let Semantic = [
      (ImplicitDef VCC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPX_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F64_e32")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))),
      (ImplicitDef VCC, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
  def V_CMPX_#cmpName#_F64_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F64_nosdst_e32")> {
    let Semantic = [
      (ImplicitDef EXEC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPSX_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F64_e32")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))),
      (ImplicitDef VCC, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
  def V_CMPSX_#cmpName#_F64_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F64_nosdst_e32")> {
    let Semantic = [
      (ImplicitDef EXEC, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
}

// F64 special: O, U, F, TRU
foreach i = 0-3 in {
  defvar cmpName = ["O", "U", "F", "TRU"][i];
  defvar cmpSem = !cond(
    !eq(i, 0) : [(ImplicitDef VCC, (LLVMFCmp LLVMFCmpOLE,
      (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
      (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))],
    !eq(i, 1) : [(ImplicitDef VCC, (LLVMFCmp LLVMFCmpUNO,
      (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
      (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))],
    !eq(i, 2) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty))],
    !eq(i, 3) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty))]
  );

  def V_CMP_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F64_e32"), cmpSem>;
  def V_CMPS_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F64_e32"), cmpSem>;

  defvar cmpxSemF64 = !cond(
    !eq(i, 0) : [(DefVal $r, (LLVMFCmp LLVMFCmpOLE, (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty), (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
    !eq(i, 1) : [(DefVal $r, (LLVMFCmp LLVMFCmpUNO, (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty), (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
    !eq(i, 2) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
    !eq(i, 3) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
  );
  defvar nosdstSemF64 = !cond(
    !eq(i, 0) : [(ImplicitDef EXEC, (LLVMFCmp LLVMFCmpOLE, (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty), (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))],
    !eq(i, 1) : [(ImplicitDef EXEC, (LLVMFCmp LLVMFCmpUNO, (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty), (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))],
    !eq(i, 2) : [(ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
    !eq(i, 3) : [(ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
  );

  def V_CMPX_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F64_e32"), cmpxSemF64>;
  def V_CMPX_#cmpName#_F64_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F64_nosdst_e32"), nosdstSemF64>;
  def V_CMPSX_#cmpName#_F64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F64_e32"), cmpxSemF64>;
  def V_CMPSX_#cmpName#_F64_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F64_nosdst_e32"), nosdstSemF64>;
}

//------------------------------------------------------------------------------
// VOPC F16 Float Comparisons
// F16 operands: trunc to i16, bitcast to half. No CMPS/CMPSX variants.
// Both base and t16 encoding variants.
//------------------------------------------------------------------------------

// F16 ordered comparisons
foreach i = 0-5 in {
  defvar cmpName = ["EQ", "LG", "GT", "GE", "LT", "LE"][i];
  defvar cmpPred = [SETOEQ, SETONE, SETOGT, SETOGE, SETOLT, SETOLE][i];

  foreach inst = ["V_CMP_"#cmpName#"_F16_e32",
                  "V_CMP_"#cmpName#"_F16_t16_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = [
        (ImplicitDef VCC, (LLVMFCmp cmpPred,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                       llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                       llvm_half_ty)))
      ];
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_F16_e32",
                  "V_CMPX_"#cmpName#"_F16_t16_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = [
        (DefVal $result, (LLVMFCmp cmpPred,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                       llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                       llvm_half_ty))),
        (ImplicitDef VCC, (GetVal $result)),
        (ImplicitDef EXEC, (GetVal $result))
      ];
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_F16_nosdst_e32",
                  "V_CMPX_"#cmpName#"_F16_t16_nosdst_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = [
        (ImplicitDef EXEC, (LLVMFCmp cmpPred,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                       llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                       llvm_half_ty)))
      ];
    }
  }
}

// F16 unordered comparisons
foreach i = 0-5 in {
  defvar cmpName = ["NEQ", "NLG", "NGT", "NGE", "NLT", "NLE"][i];
  defvar cmpPred = [SETUNE, SETUEQ, SETULE, SETULT, SETUGE, SETUGT][i];

  foreach inst = ["V_CMP_"#cmpName#"_F16_e32",
                  "V_CMP_"#cmpName#"_F16_t16_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = [
        (ImplicitDef VCC, (LLVMFCmp cmpPred,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                       llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                       llvm_half_ty)))
      ];
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_F16_e32",
                  "V_CMPX_"#cmpName#"_F16_t16_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = [
        (DefVal $result, (LLVMFCmp cmpPred,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                       llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                       llvm_half_ty))),
        (ImplicitDef VCC, (GetVal $result)),
        (ImplicitDef EXEC, (GetVal $result))
      ];
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_F16_nosdst_e32",
                  "V_CMPX_"#cmpName#"_F16_t16_nosdst_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = [
        (ImplicitDef EXEC, (LLVMFCmp cmpPred,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                       llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                       llvm_half_ty)))
      ];
    }
  }
}

// F16 special: O, U, F, TRU
foreach i = 0-3 in {
  defvar cmpName = ["O", "U", "F", "TRU"][i];

  foreach inst = ["V_CMP_"#cmpName#"_F16_e32",
                  "V_CMP_"#cmpName#"_F16_t16_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(ImplicitDef VCC, (LLVMFCmp LLVMFCmpOLE,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty)))],
        !eq(i, 1) : [(ImplicitDef VCC, (LLVMFCmp LLVMFCmpUNO,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty)))],
        !eq(i, 2) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 3) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_F16_e32",
                  "V_CMPX_"#cmpName#"_F16_t16_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(DefVal $r, (LLVMFCmp LLVMFCmpOLE,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty))),
          (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 1) : [(DefVal $r, (LLVMFCmp LLVMFCmpUNO,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty))),
          (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 2) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty)),
                      (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 3) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty)),
                      (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_F16_nosdst_e32",
                  "V_CMPX_"#cmpName#"_F16_t16_nosdst_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(ImplicitDef EXEC, (LLVMFCmp LLVMFCmpOLE,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty)))],
        !eq(i, 1) : [(ImplicitDef EXEC, (LLVMFCmp LLVMFCmpUNO,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty)))],
        !eq(i, 2) : [(ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 3) : [(ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }
}

//------------------------------------------------------------------------------
// VOPC I32 Signed Integer Comparisons
//------------------------------------------------------------------------------

foreach i = 0-7 in {
  defvar cmpName = ["EQ", "NE", "GT", "GE", "LT", "LE", "F", "T"][i];
  defvar cmpSem = !cond(
    !eq(i, 0) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpEQ, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 1) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpNE, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 2) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSGT, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 3) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSGE, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 4) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSLT, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 5) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSLE, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty))],
    !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty))]
  );

  def V_CMP_#cmpName#_I32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_I32_e32"), cmpSem>;

  defvar cmpPredI32 = [SETEQ, SETNE, SETGT, SETGE, SETLT, SETLE, SETEQ, SETEQ][i];

  def V_CMPX_#cmpName#_I32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_I32_e32")> {
    let Semantic = !cond(
      !le(i, 5) : [(DefVal $r, (LLVMCmp cmpPredI32, (GetNamedOperand $src0), (GetNamedOperand $src1))),
                    (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
      !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }
  def V_CMPX_#cmpName#_I32_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_I32_nosdst_e32")> {
    let Semantic = !cond(
      !le(i, 5) : [(ImplicitDef EXEC, (LLVMCmp cmpPredI32, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
      !eq(i, 6) : [(ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }
}

//------------------------------------------------------------------------------
// VOPC U32 Unsigned Integer Comparisons
//------------------------------------------------------------------------------

foreach i = 0-7 in {
  defvar cmpName = ["EQ", "NE", "GT", "GE", "LT", "LE", "F", "T"][i];
  defvar cmpSem = !cond(
    !eq(i, 0) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpEQ, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 1) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpNE, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 2) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpUGT, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 3) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpUGE, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 4) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpULT, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 5) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpULE, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty))],
    !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty))]
  );

  def V_CMP_#cmpName#_U32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_U32_e32"), cmpSem>;

  defvar cmpPredU32 = [SETEQ, SETNE, SETUGT, SETUGE, SETULT, SETULE, SETEQ, SETEQ][i];

  def V_CMPX_#cmpName#_U32_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_U32_e32")> {
    let Semantic = !cond(
      !le(i, 5) : [(DefVal $r, (LLVMCmp cmpPredU32, (GetNamedOperand $src0), (GetNamedOperand $src1))),
                    (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
      !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }
  def V_CMPX_#cmpName#_U32_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_U32_nosdst_e32")> {
    let Semantic = !cond(
      !le(i, 5) : [(ImplicitDef EXEC, (LLVMCmp cmpPredU32, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
      !eq(i, 6) : [(ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }
}

//------------------------------------------------------------------------------
// VOPC I64/U64 Integer Comparisons
//------------------------------------------------------------------------------

foreach i = 0-7 in {
  defvar cmpName = ["EQ", "NE", "GT", "GE", "LT", "LE", "F", "T"][i];

  // I64 signed
  defvar i64SemSigned = !cond(
    !eq(i, 0) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpEQ, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 1) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpNE, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 2) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSGT, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 3) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSGE, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 4) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSLT, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 5) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSLE, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
    !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty))],
    !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty))]
  );

  defvar cmpPredI64 = [SETEQ, SETNE, SETGT, SETGE, SETLT, SETLE, SETEQ, SETEQ][i];

  def V_CMP_#cmpName#_I64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_I64_e32"), i64SemSigned>;
  def V_CMPX_#cmpName#_I64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_I64_e32")> {
    let Semantic = !cond(
      !le(i, 5) : [(DefVal $r, (LLVMCmp cmpPredI64, (GetNamedOperand $src0), (GetNamedOperand $src1))),
                    (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
      !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }
  def V_CMPX_#cmpName#_I64_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_I64_nosdst_e32")> {
    let Semantic = !cond(
      !le(i, 5) : [(ImplicitDef EXEC, (LLVMCmp cmpPredI64, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
      !eq(i, 6) : [(ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }

  // U64 unsigned
  defvar cmpPredU64 = [SETEQ, SETNE, SETUGT, SETUGE, SETULT, SETULE, SETEQ, SETEQ][i];

  def V_CMP_#cmpName#_U64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_U64_e32")> {
    let Semantic = !cond(
      !le(i, 5) : [(ImplicitDef VCC, (LLVMCmp cmpPredU64, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
      !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty))]
    );
  }
  def V_CMPX_#cmpName#_U64_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_U64_e32")> {
    let Semantic = !cond(
      !le(i, 5) : [(DefVal $r, (LLVMCmp cmpPredU64, (GetNamedOperand $src0), (GetNamedOperand $src1))),
                    (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
      !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }
  def V_CMPX_#cmpName#_U64_nosdst_e32_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_U64_nosdst_e32")> {
    let Semantic = !cond(
      !le(i, 5) : [(ImplicitDef EXEC, (LLVMCmp cmpPredU64, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
      !eq(i, 6) : [(ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }
}

//------------------------------------------------------------------------------
// VOPC I16/U16 Integer Comparisons (with t16 variants)
//------------------------------------------------------------------------------

foreach i = 0-7 in {
  defvar cmpName = ["EQ", "NE", "GT", "GE", "LT", "LE", "F", "T"][i];

  // I16 signed
  foreach inst = ["V_CMP_"#cmpName#"_I16_e32",
                  "V_CMP_"#cmpName#"_I16_t16_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 1) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 2) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSGT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 3) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSGE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 4) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSLT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 5) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpSLE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_I16_e32",
                  "V_CMPX_"#cmpName#"_I16_t16_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(DefVal $r, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 1) : [(DefVal $r, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 2) : [(DefVal $r, (LLVMCmp LLVMICmpSGT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 3) : [(DefVal $r, (LLVMCmp LLVMICmpSGE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 4) : [(DefVal $r, (LLVMCmp LLVMICmpSLT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 5) : [(DefVal $r, (LLVMCmp LLVMICmpSLE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_I16_nosdst_e32",
                  "V_CMPX_"#cmpName#"_I16_t16_nosdst_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 1) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 2) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpSGT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 3) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpSGE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 4) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpSLT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 5) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpSLE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 6) : [(ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 7) : [(ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  // U16 unsigned - same structure, different predicates for GT/GE/LT/LE
  foreach inst = ["V_CMP_"#cmpName#"_U16_e32",
                  "V_CMP_"#cmpName#"_U16_t16_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 1) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 2) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpUGT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 3) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpUGE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 4) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpULT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 5) : [(ImplicitDef VCC, (LLVMCmp LLVMICmpULE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_U16_e32",
                  "V_CMPX_"#cmpName#"_U16_t16_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(DefVal $r, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 1) : [(DefVal $r, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 2) : [(DefVal $r, (LLVMCmp LLVMICmpUGT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 3) : [(DefVal $r, (LLVMCmp LLVMICmpUGE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 4) : [(DefVal $r, (LLVMCmp LLVMICmpULT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 5) : [(DefVal $r, (LLVMCmp LLVMICmpULE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (ImplicitDef VCC, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 6) : [(ImplicitDef VCC, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 7) : [(ImplicitDef VCC, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_U16_nosdst_e32",
                  "V_CMPX_"#cmpName#"_U16_t16_nosdst_e32"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 1) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 2) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpUGT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 3) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpUGE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 4) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpULT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 5) : [(ImplicitDef EXEC, (LLVMCmp LLVMICmpULE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 6) : [(ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 7) : [(ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }
}

//------------------------------------------------------------------------------
// VOPC CLASS Instructions (float class test)
//------------------------------------------------------------------------------

// VCC[lane] = isClassOf(S0.f, S1.u). Tests float for NaN, Inf, zero, etc.
// S1 is a bitmask specifying which classes to test for.

// V_CMP_CLASS → VCC
foreach inst = [V_CMP_CLASS_F32_e32, V_CMP_CLASS_F64_e32,
                V_CMP_CLASS_F16_e32, V_CMP_CLASS_F16_t16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_class>;
}

// V_CMPX_CLASS → VCC + EXEC
foreach inst = [V_CMPX_CLASS_F32_e32, V_CMPX_CLASS_F64_e32,
                V_CMPX_CLASS_F16_e32, V_CMPX_CLASS_F16_t16_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_class>;
}

// V_CMPX_CLASS_nosdst → EXEC only
foreach inst = [V_CMPX_CLASS_F32_nosdst_e32, V_CMPX_CLASS_F64_nosdst_e32,
                V_CMPX_CLASS_F16_nosdst_e32,
                V_CMPX_CLASS_F16_t16_nosdst_e32] in {
  def inst#_sem : InstSIIntrinsic<inst, int_amdgcn_class>;
}
