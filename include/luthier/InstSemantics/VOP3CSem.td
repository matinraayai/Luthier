//===- VOP3CSem.td ------------------------------------------------*- tblgen -*-===//
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
// VOPC e64 (VOP3) comparison instruction semantic translations.
//
// e64 encoding differences from e32:
//   - Explicit $sdst output operand (instead of implicit VCC def)
//   - Float comparisons have $src0_modifiers, $src1_modifiers, $clamp operands
//   - No _nosdst variants (those are e32-only on GFX10+)
//   - No V_CMPSX_* variants without e64 (those are GFX6/GFX7 e32-only)
//   - V_CMPX_*_e64 still writes EXEC implicitly in addition to $sdst
//
// PDF sources for VOPC e64 instructions (same pages as e32 encodings):
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
//===----------------------------------------------------------------------===//

//==============================================================================
// VOPC e64 Instructions
//==============================================================================

//------------------------------------------------------------------------------
// VOPC F32 Ordered Float Comparisons (e64)
//------------------------------------------------------------------------------
//
// Sources:
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

foreach i = 0-5 in {
  defvar cmpName = ["EQ", "LG", "GT", "GE", "LT", "LE"][i];
  defvar cmpPred = [SETOEQ, SETONE, SETOGT, SETOGE, SETOLT, SETOLE][i];

  // V_CMP → $sdst
  def V_CMP_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  // V_CMPS → $sdst (signaling, GFX6/GFX7 only, same semantics)
  def V_CMPS_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  // V_CMPX → $sdst + EXEC
  def V_CMPX_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }

  // V_CMPSX → $sdst + EXEC (signaling, GFX6/GFX7 only)
  def V_CMPSX_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
}

//------------------------------------------------------------------------------
// VOPC F32 Unordered Float Comparisons (e64)
//------------------------------------------------------------------------------
//
// Sources:
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

foreach i = 0-5 in {
  defvar cmpName = ["NEQ", "NLG", "NGT", "NGE", "NLT", "NLE"][i];
  defvar cmpPred = [SETUNE, SETUEQ, SETULE, SETULT, SETUGE, SETUGT][i];

  def V_CMP_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  def V_CMPS_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  def V_CMPX_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }

  def V_CMPSX_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
}

//------------------------------------------------------------------------------
// VOPC F32 Special Float Comparisons (e64): O, U, F, TRU, T
//------------------------------------------------------------------------------
//
// Sources:
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

foreach i = 0-1 in {
  defvar cmpName = ["O", "U"][i];
  defvar cmpPred = [SETO, SETUO][i];

  def V_CMP_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  def V_CMPS_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty)))
    ];
  }

  def V_CMPX_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }

  def V_CMPSX_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_float_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
}

// F = always false, TRU/T = always true (F32 e64)
foreach i = 0-2 in {
  defvar cmpName = ["F", "TRU", "T"][i];
  defvar constVal = [ConstantFalse, ConstantTrue, ConstantTrue][i];

  def V_CMP_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F32_e64")> {
    let Semantic = [(SetNamedOperand $sdst, (constVal llvm_i1_ty))];
  }
  def V_CMPS_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F32_e64")> {
    let Semantic = [(SetNamedOperand $sdst, (constVal llvm_i1_ty))];
  }
  def V_CMPX_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (constVal llvm_i1_ty)),
      (ImplicitDef EXEC, (constVal llvm_i1_ty))
    ];
  }
  def V_CMPSX_#cmpName#_F32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F32_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (constVal llvm_i1_ty)),
      (ImplicitDef EXEC, (constVal llvm_i1_ty))
    ];
  }
}

//------------------------------------------------------------------------------
// VOPC F64 Ordered Float Comparisons (e64)
//------------------------------------------------------------------------------
//
// Sources:
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

foreach i = 0-5 in {
  defvar cmpName = ["EQ", "LG", "GT", "GE", "LT", "LE"][i];
  defvar cmpPred = [SETOEQ, SETONE, SETOGT, SETOGE, SETOLT, SETOLE][i];

  def V_CMP_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPS_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPX_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
  def V_CMPSX_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
}

//------------------------------------------------------------------------------
// VOPC F64 Unordered Float Comparisons (e64)
//------------------------------------------------------------------------------
//
// Sources:
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

foreach i = 0-5 in {
  defvar cmpName = ["NEQ", "NLG", "NGT", "NGE", "NLT", "NLE"][i];
  defvar cmpPred = [SETUNE, SETUEQ, SETULE, SETULT, SETUGE, SETUGT][i];

  def V_CMP_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPS_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPX_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
  def V_CMPSX_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
}

//------------------------------------------------------------------------------
// VOPC F64 Special Float Comparisons (e64): O, U, F, TRU, T
//------------------------------------------------------------------------------
//
// Sources:
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

foreach i = 0-1 in {
  defvar cmpName = ["O", "U"][i];
  defvar cmpPred = [SETO, SETUO][i];

  def V_CMP_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPS_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty)))
    ];
  }
  def V_CMPX_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
  def V_CMPSX_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (DefVal $result, (LLVMFCmp cmpPred,
        (LLVMBitCast (GetNamedOperand $src0), llvm_double_ty),
        (LLVMBitCast (GetNamedOperand $src1), llvm_double_ty))),
      (SetNamedOperand $sdst, (GetVal $result)),
      (ImplicitDef EXEC, (GetVal $result))
    ];
  }
}

foreach i = 0-2 in {
  defvar cmpName = ["F", "TRU", "T"][i];
  defvar constVal = [ConstantFalse, ConstantTrue, ConstantTrue][i];

  def V_CMP_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_F64_e64")> {
    let Semantic = [(SetNamedOperand $sdst, (constVal llvm_i1_ty))];
  }
  def V_CMPS_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPS_"#cmpName#"_F64_e64")> {
    let Semantic = [(SetNamedOperand $sdst, (constVal llvm_i1_ty))];
  }
  def V_CMPX_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (constVal llvm_i1_ty)),
      (ImplicitDef EXEC, (constVal llvm_i1_ty))
    ];
  }
  def V_CMPSX_#cmpName#_F64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPSX_"#cmpName#"_F64_e64")> {
    let Semantic = [
      (SetNamedOperand $sdst, (constVal llvm_i1_ty)),
      (ImplicitDef EXEC, (constVal llvm_i1_ty))
    ];
  }
}

//------------------------------------------------------------------------------
// VOPC F16 Ordered Float Comparisons (e64)
// F16 operands: trunc to i16, bitcast to half. No CMPS/CMPSX variants.
//------------------------------------------------------------------------------
//
// Sources (GFX8+, not present in gcn3):
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
// _t16 variants: RDNA3+/RDNA4 only.

foreach i = 0-5 in {
  defvar cmpName = ["EQ", "LG", "GT", "GE", "LT", "LE"][i];
  defvar cmpPred = [SETOEQ, SETONE, SETOGT, SETOGE, SETOLT, SETOLE][i];

  foreach inst = ["V_CMP_"#cmpName#"_F16_e64",
                  "V_CMP_"#cmpName#"_F16_t16_e64"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = [
        (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                       llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                       llvm_half_ty)))
      ];
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_F16_e64",
                  "V_CMPX_"#cmpName#"_F16_t16_e64"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = [
        (DefVal $result, (LLVMFCmp cmpPred,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                       llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                       llvm_half_ty))),
        (SetNamedOperand $sdst, (GetVal $result)),
        (ImplicitDef EXEC, (GetVal $result))
      ];
    }
  }
}

//------------------------------------------------------------------------------
// VOPC F16 Unordered Float Comparisons (e64)
//------------------------------------------------------------------------------
//
// Sources (GFX8+, not present in gcn3):
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
// _t16 variants: RDNA3+/RDNA4 only.

foreach i = 0-5 in {
  defvar cmpName = ["NEQ", "NLG", "NGT", "NGE", "NLT", "NLE"][i];
  defvar cmpPred = [SETUNE, SETUEQ, SETULE, SETULT, SETUGE, SETUGT][i];

  foreach inst = ["V_CMP_"#cmpName#"_F16_e64",
                  "V_CMP_"#cmpName#"_F16_t16_e64"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = [
        (SetNamedOperand $sdst, (LLVMFCmp cmpPred,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                       llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                       llvm_half_ty)))
      ];
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_F16_e64",
                  "V_CMPX_"#cmpName#"_F16_t16_e64"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = [
        (DefVal $result, (LLVMFCmp cmpPred,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty),
                       llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty),
                       llvm_half_ty))),
        (SetNamedOperand $sdst, (GetVal $result)),
        (ImplicitDef EXEC, (GetVal $result))
      ];
    }
  }
}

//------------------------------------------------------------------------------
// VOPC F16 Special Float Comparisons (e64): O, U, F, TRU
//------------------------------------------------------------------------------
//
// Sources (GFX8+, not present in gcn3):
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
// _t16 variants: RDNA3+/RDNA4 only.

foreach i = 0-3 in {
  defvar cmpName = ["O", "U", "F", "TRU"][i];

  foreach inst = ["V_CMP_"#cmpName#"_F16_e64",
                  "V_CMP_"#cmpName#"_F16_t16_e64"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(SetNamedOperand $sdst, (LLVMFCmp LLVMFCmpOLE,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty)))],
        !eq(i, 1) : [(SetNamedOperand $sdst, (LLVMFCmp LLVMFCmpUNO,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty)))],
        !eq(i, 2) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty))],
        !eq(i, 3) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_F16_e64",
                  "V_CMPX_"#cmpName#"_F16_t16_e64"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(DefVal $r, (LLVMFCmp LLVMFCmpOLE,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty))),
          (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 1) : [(DefVal $r, (LLVMFCmp LLVMFCmpUNO,
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), llvm_half_ty),
          (LLVMBitCast (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty), llvm_half_ty))),
          (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 2) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty)),
                      (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 3) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty)),
                      (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }
}

//------------------------------------------------------------------------------
// VOPC I32 Signed Integer Comparisons (e64)
//------------------------------------------------------------------------------
//
// Sources:
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

foreach i = 0-7 in {
  defvar cmpName = ["EQ", "NE", "GT", "GE", "LT", "LE", "F", "T"][i];
  defvar cmpPredI32 = [SETEQ, SETNE, SETGT, SETGE, SETLT, SETLE, SETEQ, SETEQ][i];

  def V_CMP_#cmpName#_I32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_I32_e64")> {
    let Semantic = !cond(
      !le(i, 5) : [(SetNamedOperand $sdst, (LLVMCmp cmpPredI32, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
      !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty))]
    );
  }

  def V_CMPX_#cmpName#_I32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_I32_e64")> {
    let Semantic = !cond(
      !le(i, 5) : [(DefVal $r, (LLVMCmp cmpPredI32, (GetNamedOperand $src0), (GetNamedOperand $src1))),
                    (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
      !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }
}

//------------------------------------------------------------------------------
// VOPC U32 Unsigned Integer Comparisons (e64)
//------------------------------------------------------------------------------
//
// Sources:
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

foreach i = 0-7 in {
  defvar cmpName = ["EQ", "NE", "GT", "GE", "LT", "LE", "F", "T"][i];
  defvar cmpPredU32 = [SETEQ, SETNE, SETUGT, SETUGE, SETULT, SETULE, SETEQ, SETEQ][i];

  def V_CMP_#cmpName#_U32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_U32_e64")> {
    let Semantic = !cond(
      !le(i, 5) : [(SetNamedOperand $sdst, (LLVMCmp cmpPredU32, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
      !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty))]
    );
  }

  def V_CMPX_#cmpName#_U32_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_U32_e64")> {
    let Semantic = !cond(
      !le(i, 5) : [(DefVal $r, (LLVMCmp cmpPredU32, (GetNamedOperand $src0), (GetNamedOperand $src1))),
                    (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
      !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }
}

//------------------------------------------------------------------------------
// VOPC I64/U64 Integer Comparisons (e64)
//------------------------------------------------------------------------------
//
// Sources:
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

foreach i = 0-7 in {
  defvar cmpName = ["EQ", "NE", "GT", "GE", "LT", "LE", "F", "T"][i];

  // I64 signed
  defvar cmpPredI64 = [SETEQ, SETNE, SETGT, SETGE, SETLT, SETLE, SETEQ, SETEQ][i];

  def V_CMP_#cmpName#_I64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_I64_e64")> {
    let Semantic = !cond(
      !le(i, 5) : [(SetNamedOperand $sdst, (LLVMCmp cmpPredI64, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
      !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty))]
    );
  }
  def V_CMPX_#cmpName#_I64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_I64_e64")> {
    let Semantic = !cond(
      !le(i, 5) : [(DefVal $r, (LLVMCmp cmpPredI64, (GetNamedOperand $src0), (GetNamedOperand $src1))),
                    (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
      !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }

  // U64 unsigned
  defvar cmpPredU64 = [SETEQ, SETNE, SETUGT, SETUGE, SETULT, SETULE, SETEQ, SETEQ][i];

  def V_CMP_#cmpName#_U64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMP_"#cmpName#"_U64_e64")> {
    let Semantic = !cond(
      !le(i, 5) : [(SetNamedOperand $sdst, (LLVMCmp cmpPredU64, (GetNamedOperand $src0), (GetNamedOperand $src1)))],
      !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty))]
    );
  }
  def V_CMPX_#cmpName#_U64_e64_sem
    : InstSISemantic<!cast<InstSI>("V_CMPX_"#cmpName#"_U64_e64")> {
    let Semantic = !cond(
      !le(i, 5) : [(DefVal $r, (LLVMCmp cmpPredU64, (GetNamedOperand $src0), (GetNamedOperand $src1))),
                    (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
      !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
      !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
    );
  }
}

//------------------------------------------------------------------------------
// VOPC I16/U16 Integer Comparisons (e64, with t16 variants)
//------------------------------------------------------------------------------
//
// Sources (GFX8+, not present in gcn3):
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
// _t16 variants: RDNA3+/RDNA4 only.

foreach i = 0-7 in {
  defvar cmpName = ["EQ", "NE", "GT", "GE", "LT", "LE", "F", "T"][i];

  // I16 signed
  foreach inst = ["V_CMP_"#cmpName#"_I16_e64",
                  "V_CMP_"#cmpName#"_I16_t16_e64"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 1) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 2) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpSGT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 3) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpSGE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 4) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpSLT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 5) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpSLE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty))],
        !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_I16_e64",
                  "V_CMPX_"#cmpName#"_I16_t16_e64"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(DefVal $r, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 1) : [(DefVal $r, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 2) : [(DefVal $r, (LLVMCmp LLVMICmpSGT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 3) : [(DefVal $r, (LLVMCmp LLVMICmpSGE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 4) : [(DefVal $r, (LLVMCmp LLVMICmpSLT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 5) : [(DefVal $r, (LLVMCmp LLVMICmpSLE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  // U16 unsigned
  foreach inst = ["V_CMP_"#cmpName#"_U16_e64",
                  "V_CMP_"#cmpName#"_U16_t16_e64"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 1) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 2) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpUGT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 3) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpUGE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 4) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpULT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 5) : [(SetNamedOperand $sdst, (LLVMCmp LLVMICmpULE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty)))],
        !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty))],
        !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty))]
      );
    }
  }

  foreach inst = ["V_CMPX_"#cmpName#"_U16_e64",
                  "V_CMPX_"#cmpName#"_U16_t16_e64"] in {
    def inst#_sem : InstSISemantic<!cast<InstSI>(inst)> {
      let Semantic = !cond(
        !eq(i, 0) : [(DefVal $r, (LLVMCmp LLVMICmpEQ, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 1) : [(DefVal $r, (LLVMCmp LLVMICmpNE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 2) : [(DefVal $r, (LLVMCmp LLVMICmpUGT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 3) : [(DefVal $r, (LLVMCmp LLVMICmpUGE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 4) : [(DefVal $r, (LLVMCmp LLVMICmpULT, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 5) : [(DefVal $r, (LLVMCmp LLVMICmpULE, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty), (LLVMTrunc (GetNamedOperand $src1), llvm_i16_ty))), (SetNamedOperand $sdst, (GetVal $r)), (ImplicitDef EXEC, (GetVal $r))],
        !eq(i, 6) : [(SetNamedOperand $sdst, (ConstantFalse llvm_i1_ty)), (ImplicitDef EXEC, (ConstantFalse llvm_i1_ty))],
        !eq(i, 7) : [(SetNamedOperand $sdst, (ConstantTrue llvm_i1_ty)), (ImplicitDef EXEC, (ConstantTrue llvm_i1_ty))]
      );
    }
  }
}

//------------------------------------------------------------------------------
// VOPC CLASS Instructions (e64) - float class tests
//------------------------------------------------------------------------------
//
// Sources:
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
// _t16 variants: RDNA3+/RDNA4 only.

// V_CMP_CLASS → $sdst
foreach inst = [V_CMP_CLASS_F32_e64, V_CMP_CLASS_F64_e64,
                V_CMP_CLASS_F16_e64, V_CMP_CLASS_F16_t16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $sdst,
      (LLVMIntrinsic llvm_i1_ty, int_amdgcn_class, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))
    ];
  }
}

// V_CMPX_CLASS → $sdst + EXEC
foreach inst = [V_CMPX_CLASS_F32_e64, V_CMPX_CLASS_F64_e64,
                V_CMPX_CLASS_F16_e64, V_CMPX_CLASS_F16_t16_e64] in {
  def inst#_sem : InstSISemantic<inst> {
    let Semantic = [
      (SetNamedOperand $sdst,
      (LLVMIntrinsic llvm_i1_ty, int_amdgcn_class, [(GetNamedOperand $src0), (GetNamedOperand $src1)]))
    ];
  }
}
