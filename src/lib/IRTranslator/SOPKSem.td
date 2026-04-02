//===- SOPKSem.td -----------------------------------------------------*- tblgen -*-===//
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
// SOPK Instructions
//------------------------------------------------------------------------------

// D.i = D.i + signext(SIMM16); SCC = unsigned overflow.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 119.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 111.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 123.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 116.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 212.
//   rdna4-instruction-set-architecture.pdf, page 175.
def S_ADDK_I32_sem : InstSISemantic<S_ADDK_I32> {
  let Semantic = [
    (DefVal $tmp_add, (LLVMAdd (GetNamedOperand $src0), (GetNamedOperand $simm16))),
    (ImplicitDef SCC, (LLVMCmp SETUGE, (GetVal $tmp_add), (ConstantInt llvm_i32_ty, 0x100000000))),
    (SetNamedOperand $sdst, (LLVMTrunc (GetVal $tmp_add), llvm_i32_ty))
  ];
}

// if (SCC) D.i = signext(SIMM16); else NOP.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 119.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 109.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 121.
//   rdna-shader-instruction-set-architecture.pdf, page 115.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 209.
//   rdna4-instruction-set-architecture.pdf, page 176.
def S_CMOVK_I32_sem : InstSISemantic<S_CMOVK_I32> {
  let Semantic = [
    (DefVal $tmp, (LLVMSelect (ImplicitUse SCC), 
                            (GetNamedOperand $simm16), 
                            (GetNamedOperand $sdst))),
    (SetNamedOperand $sdst, (GetVal $tmp))
  ];
}

// D.i = signext(SIMM16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 123.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 109.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 121.
//   rdna-shader-instruction-set-architecture.pdf, page 115.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 209.
//   rdna4-instruction-set-architecture.pdf, page 176.
def S_MOVK_I32_sem : InstSISemantic<S_MOVK_I32> {
  let Semantic = [
    (SetNamedOperand $sdst, (GetNamedOperand $simm16))
  ];
}

// D.i = D.i * signext(SIMM16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 123.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 111.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 123.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 116.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 212.
//   rdna4-instruction-set-architecture.pdf, page 176.
def S_MULK_I32_sem : InstSISemantic<S_MULK_I32> {
  let Semantic = [
    (DefVal $tmp, (LLVMMul (GetNamedOperand $src0), (GetNamedOperand $simm16))),
    (SetNamedOperand $sdst, (GetVal $tmp))
  ];
}

// SCC = (D.i == signext(SIMM16)).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 119.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 109.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 121.
//   rdna-shader-instruction-set-architecture.pdf, page 115.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_EQ_I32_sem : InstSISemantic<S_CMPK_EQ_I32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETEQ, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.u == SIMM16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 120.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 110.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 122.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_EQ_U32_sem : InstSISemantic<S_CMPK_EQ_U32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETEQ, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.i >= signext(SIMM16)).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 120.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 110.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 122.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_GE_I32_sem : InstSISemantic<S_CMPK_GE_I32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETGE, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.u >= SIMM16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 120.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 111.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 123.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_GE_U32_sem : InstSISemantic<S_CMPK_GE_U32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETUGE, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.i > signext(SIMM16)).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 120.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 109.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 121.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_GT_I32_sem : InstSISemantic<S_CMPK_GT_I32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETGT, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.u > SIMM16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 121.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 111.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 123.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_GT_U32_sem : InstSISemantic<S_CMPK_GT_U32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETUGT, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.i <= signext(SIMM16)).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 121.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 110.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 122.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_LE_I32_sem : InstSISemantic<S_CMPK_LE_I32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETLE, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.u <= SIMM16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 121.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 111.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 123.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_LE_U32_sem : InstSISemantic<S_CMPK_LE_U32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETULE, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.i != signext(SIMM16)).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 121.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 109.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 121.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_LG_I32_sem : InstSISemantic<S_CMPK_LG_I32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETNE, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.u != SIMM16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 122.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 110.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 122.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_LG_U32_sem : InstSISemantic<S_CMPK_LG_U32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETNE, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.i < signext(SIMM16)).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 122.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 110.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 122.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_LT_I32_sem : InstSISemantic<S_CMPK_LT_I32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETLT, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

// SCC = (D.u < SIMM16).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 122.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 104.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 111.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 123.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
def S_CMPK_LT_U32_sem : InstSISemantic<S_CMPK_LT_U32> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMCmp SETULT, (GetNamedOperand $sdst), (GetNamedOperand $simm16)))
  ];
}

//------------------------------------------------------------------------------
// S_GETREG_B32
//------------------------------------------------------------------------------

// D = HWREG[SIMM16]. Read a hardware register.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 122.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 105.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 112.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 124.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 116.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 212.
//   rdna4-instruction-set-architecture.pdf, page 176.
def S_GETREG_B32_sem : InstSISemantic<S_GETREG_B32> {
  let Semantic = [
    (SetNamedOperand $sdst, (int_amdgcn_s_getreg (GetNamedOperand $simm16)))
  ];
}

//------------------------------------------------------------------------------
// S_SETREG_B32
//------------------------------------------------------------------------------

// HWREG[SIMM16] = S0. Write a hardware register.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 123.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 105.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 112.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 124.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 116.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 213.
//   rdna4-instruction-set-architecture.pdf, page 176.
def S_SETREG_B32_sem : InstSISemantic<S_SETREG_B32> {
  let Semantic = [
    (int_amdgcn_s_setreg (GetNamedOperand $simm16), (GetNamedOperand $sdst))
  ];
}
