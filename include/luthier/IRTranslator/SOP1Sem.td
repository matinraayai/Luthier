//===- SOP1Sem.td -------------------------------------------------*- tblgen -*-===//
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
// SOP1 Instructions
//------------------------------------------------------------------------------

// D.u = S0.u. Move scalar value.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 131.
//   vega-shader-instruction-set-architecture.pdf, page 106.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 115.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 119.
//   rdna2-shader-instruction-set-architecture.pdf, page 119.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 217.
//   rdna4-instruction-set-architecture.pdf, page 230.
def S_MOV_B32_sem : InstSISemantic<S_MOV_B32> {
  let Semantic = [
    (SetNamedOperand $sdst, (GetNamedOperand $src0))
  ];
}

// D.u = S0.u. Move scalar value (64-bit).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 131.
//   vega-shader-instruction-set-architecture.pdf, page 106.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 115.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 119.
//   rdna2-shader-instruction-set-architecture.pdf, page 119.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 217.
//   rdna4-instruction-set-architecture.pdf, page 230.
def S_MOV_B64_sem : InstSISemantic<S_MOV_B64> {
  let Semantic = [
    (SetNamedOperand $sdst, (GetNamedOperand $src0))
  ];
}

// Same as S_MOV_B32 but with side effects (e.g. SRC_POPS_EXITING_WAVE_ID).
// LLVM-only pseudo (no distinct ISA opcode); same encoding as S_MOV_B32.
// Defined in llvm-project: llvm/lib/Target/AMDGPU/SOPInstructions.td, line 226.
def S_MOV_B32_sideeffects_sem : InstSISemantic<S_MOV_B32_sideeffects> {
  let Semantic = [
    (SetNamedOperand $sdst, (GetNamedOperand $src0))
  ];
}

// D.u = ~S0.u; SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 133.
//   vega-shader-instruction-set-architecture.pdf, page 106.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 115.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 120.
//   rdna2-shader-instruction-set-architecture.pdf, page 119.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 226.
//   rdna4-instruction-set-architecture.pdf, page 239.
def S_NOT_B32_sem : InstSISemantic<S_NOT_B32> {
  let Semantic = [
    (DefVal $result, (LLVMXor (GetNamedOperand $src0), (LLVMSExt (ConstantTrue llvm_i1_ty), llvm_i32_ty))),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, $result, (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.u = ~S0.u; SCC = (D != 0). (64-bit)
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 133.
//   vega-shader-instruction-set-architecture.pdf, page 107.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 116.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 128.
//   rdna-shader-instruction-set-architecture.pdf, page 120.
//   rdna2-shader-instruction-set-architecture.pdf, page 119.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 226.
//   rdna4-instruction-set-architecture.pdf, page 239.
def S_NOT_B64_sem : InstSISemantic<S_NOT_B64> {
  let Semantic = [
    (DefVal $result, (LLVMXor (GetNamedOperand $src0), (LLVMSExt (ConstantTrue llvm_i1_ty), llvm_i64_ty))),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, $result, (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.i = abs(S0.i); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 125.
//   vega-shader-instruction-set-architecture.pdf, page 115.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 116.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 113.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 129.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 141.
//   rdna-shader-instruction-set-architecture.pdf, page 128.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 222.
//   rdna4-instruction-set-architecture.pdf, page 235.
def S_ABS_I32_sem : InstSISemantic<S_ABS_I32> {
  let Semantic = [
    (DefVal $sign, (LLVMAnd (GetNamedOperand $src0), (ConstantInt llvm_i32_ty, 0x80000000))),
    (DefVal $neg, (LLVMSub (ConstantInt llvm_i32_ty, 0), (GetNamedOperand $src0))),
    (DefVal $result, (LLVMSelect (LLVMCmp LLVMICmpNE, (GetVal $sign), (ConstantInt llvm_i32_ty, 0)), (GetVal $neg), (GetNamedOperand $src0))),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.u = WholeQuadMode(S0.u); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 136.
//   vega-shader-instruction-set-architecture.pdf, page 107.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 116.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 128.
//   rdna-shader-instruction-set-architecture.pdf, page 120.
//   rdna2-shader-instruction-set-architecture.pdf, page 119.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 225.
//   rdna4-instruction-set-architecture.pdf, page 238.
def S_WQM_B32_sem : InstSISemantic<S_WQM_B32> {
  let Semantic = [
    (DefVal $result, (LLVMUnaryIntrinsic int_amdgcn_s_wqm, (GetNamedOperand $src0))),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.u = WholeQuadMode(S0.u); SCC = (D != 0). (64-bit)
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 136.
//   vega-shader-instruction-set-architecture.pdf, page 107.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 116.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 128.
//   rdna-shader-instruction-set-architecture.pdf, page 120.
//   rdna2-shader-instruction-set-architecture.pdf, page 119.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 225.
//   rdna4-instruction-set-architecture.pdf, page 238.
def S_WQM_B64_sem : InstSISemantic<S_WQM_B64> {
  let Semantic = [
    (DefVal $result, (LLVMUnaryIntrinsic int_amdgcn_s_wqm, (GetNamedOperand $src0))),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.i = signext(S0.i[15:0]).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 135.
//   vega-shader-instruction-set-architecture.pdf, page 112.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 113.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 110.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 122.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 134.
//   rdna-shader-instruction-set-architecture.pdf, page 125.
//   rdna2-shader-instruction-set-architecture.pdf, page 125.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 221.
//   rdna4-instruction-set-architecture.pdf, page 234.
def S_SEXT_I32_I16_sem : InstSISemantic<S_SEXT_I32_I16> {
  let Semantic = [
    (DefVal $result, (LLVMSExt (GetNamedOperand $src0), llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.i = signext(S0.i[7:0]).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 135.
//   vega-shader-instruction-set-architecture.pdf, page 112.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 113.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 110.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 122.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 134.
//   rdna-shader-instruction-set-architecture.pdf, page 125.
//   rdna2-shader-instruction-set-architecture.pdf, page 125.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 221.
//   rdna4-instruction-set-architecture.pdf, page 234.
def S_SEXT_I32_I8_sem : InstSISemantic<S_SEXT_I32_I8> {
  let Semantic = [
    (DefVal $result, (LLVMSExt (GetNamedOperand $src0), llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// if (SCC) D.u = S0.u; else NOP.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 128.
//   vega-shader-instruction-set-architecture.pdf, page 106.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 115.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 119.
//   rdna2-shader-instruction-set-architecture.pdf, page 119.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 217.
//   rdna4-instruction-set-architecture.pdf, page 230.
def S_CMOV_B32_sem : InstSISemantic<S_CMOV_B32> {
  let Semantic = [
    (DefVal $result, (LLVMSelect (ImplicitUse SCC), (GetNamedOperand $src0), (GetNamedOperand $sdst))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// if (SCC) D.u = S0.u; else NOP. (64-bit)
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 128.
//   vega-shader-instruction-set-architecture.pdf, page 106.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 115.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 120.
//   rdna2-shader-instruction-set-architecture.pdf, page 119.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 217.
//   rdna4-instruction-set-architecture.pdf, page 230.
def S_CMOV_B64_sem : InstSISemantic<S_CMOV_B64> {
  let Semantic = [
    (DefVal $result, (LLVMSelect (ImplicitUse SCC), (GetNamedOperand $src0), (GetNamedOperand $sdst))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.u = QuadMask(S0.u). Reduce pixel mask to quad mask.
// Sources (introduced in GFX9/Vega):
//   vega-shader-instruction-set-architecture.pdf, page 113.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 114.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 112.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 111.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 127.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 139.
//   rdna-shader-instruction-set-architecture.pdf, page 127.
//   rdna2-shader-instruction-set-architecture.pdf, page 126.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 224.
//   rdna4-instruction-set-architecture.pdf, page 237.
def S_QUADMASK_B32_sem : InstSISemantic<S_QUADMASK_B32> {
  let Semantic = [
    (DefVal $result, (LLVMUnaryIntrinsic int_amdgcn_s_quadmask, (GetNamedOperand $src0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.u = QuadMask(S0.u). Reduce pixel mask to quad mask. (64-bit)
// Sources (introduced in GFX9/Vega):
//   vega-shader-instruction-set-architecture.pdf, page 114.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 115.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 113.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 112.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 127.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 139.
//   rdna-shader-instruction-set-architecture.pdf, page 127.
//   rdna2-shader-instruction-set-architecture.pdf, page 126.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 225.
//   rdna4-instruction-set-architecture.pdf, page 238.
def S_QUADMASK_B64_sem : InstSISemantic<S_QUADMASK_B64> {
  let Semantic = [
    (DefVal $result, (LLVMUnaryIntrinsic int_amdgcn_s_quadmask, (GetNamedOperand $src0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// SAVEEXEC Instructions
// For all SAVEEXEC instructions:
//   D = EXEC (old value); EXEC = <op>(S0, EXEC); SCC = (EXEC != 0)
// The destination register receives the OLD EXEC value before the operation.
//------------------------------------------------------------------------------

// D = EXEC; EXEC = S0 & EXEC; SCC = (EXEC != 0).
// Sources (B64 since GCN3; B32 added in GFX10/RDNA):
//   gcn3-instruction-set-architecture.pdf, page 125 (B64).
//   vega-shader-instruction-set-architecture.pdf, page 113 (B64).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 114 (B64).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 111 (B64).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 124 (B64).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 136 (B64).
//   rdna-shader-instruction-set-architecture.pdf, page 126 (B64), page 130 (B32).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 226.
//   rdna4-instruction-set-architecture.pdf, page 239 (B32), page 240 (B64).
def S_AND_SAVEEXEC_B32_sem : InstSISemantic<S_AND_SAVEEXEC_B32> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMAnd (GetNamedOperand $src0), (GetVal $old_exec))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

def S_AND_SAVEEXEC_B64_sem : InstSISemantic<S_AND_SAVEEXEC_B64> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMAnd (GetNamedOperand $src0), (GetVal $old_exec))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

// D = EXEC; EXEC = S0 | EXEC; SCC = (EXEC != 0).
// Sources (B64 since GCN3; B32 added in GFX10/RDNA):
//   gcn3-instruction-set-architecture.pdf, page 133 (B64).
//   vega-shader-instruction-set-architecture.pdf, page 113 (B64).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 114 (B64).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 111 (B64).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 125 (B64).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 137 (B64).
//   rdna-shader-instruction-set-architecture.pdf, page 126 (B64), page 130 (B32).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 227.
//   rdna4-instruction-set-architecture.pdf, page 240.
def S_OR_SAVEEXEC_B32_sem : InstSISemantic<S_OR_SAVEEXEC_B32> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMOr (GetNamedOperand $src0), (GetVal $old_exec))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

def S_OR_SAVEEXEC_B64_sem : InstSISemantic<S_OR_SAVEEXEC_B64> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMOr (GetNamedOperand $src0), (GetVal $old_exec))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

// D = EXEC; EXEC = S0 ^ EXEC; SCC = (EXEC != 0).
// Sources (B64 since GCN3; B32 added in GFX10/RDNA):
//   gcn3-instruction-set-architecture.pdf, page 137 (B64).
//   vega-shader-instruction-set-architecture.pdf, page 113 (B64).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 114 (B64).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 111 (B64).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 125 (B64).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 137 (B64).
//   rdna-shader-instruction-set-architecture.pdf, page 126 (B64), page 130 (B32).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 227 (B32), page 228 (B64).
//   rdna4-instruction-set-architecture.pdf, page 240 (B32), page 241 (B64).
def S_XOR_SAVEEXEC_B32_sem : InstSISemantic<S_XOR_SAVEEXEC_B32> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMXor (GetNamedOperand $src0), (GetVal $old_exec))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

def S_XOR_SAVEEXEC_B64_sem : InstSISemantic<S_XOR_SAVEEXEC_B64> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMXor (GetNamedOperand $src0), (GetVal $old_exec))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

// D = EXEC; EXEC = ~(S0 & EXEC); SCC = (EXEC != 0).
// Sources (B64 since GCN3; B32 added in GFX10/RDNA):
//   gcn3-instruction-set-architecture.pdf, page 132 (B64).
//   vega-shader-instruction-set-architecture.pdf, page 113 (B64).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 114 (B64).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 111 (B64).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 126 (B64).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 138 (B64).
//   rdna-shader-instruction-set-architecture.pdf, page 127 (B64), page 130 (B32).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 228.
//   rdna4-instruction-set-architecture.pdf, page 241.
def S_NAND_SAVEEXEC_B32_sem : InstSISemantic<S_NAND_SAVEEXEC_B32> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMXor (LLVMAnd (GetNamedOperand $src0), (GetVal $old_exec)), (ConstantInt llvm_i32_ty, -1))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

def S_NAND_SAVEEXEC_B64_sem : InstSISemantic<S_NAND_SAVEEXEC_B64> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMXor (LLVMAnd (GetNamedOperand $src0), (GetVal $old_exec)), (ConstantInt llvm_i64_ty, -1))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

// D = EXEC; EXEC = ~(S0 | EXEC); SCC = (EXEC != 0).
// Sources (B64 since GCN3; B32 added in GFX10/RDNA):
//   gcn3-instruction-set-architecture.pdf, page 133 (B64).
//   vega-shader-instruction-set-architecture.pdf, page 113 (B64).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 114 (B64).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 111 (B64).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 126 (B64).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 138 (B64).
//   rdna-shader-instruction-set-architecture.pdf, page 127 (B64), page 131 (B32).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 228 (B32), page 229 (B64).
//   rdna4-instruction-set-architecture.pdf, page 242.
def S_NOR_SAVEEXEC_B32_sem : InstSISemantic<S_NOR_SAVEEXEC_B32> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMXor (LLVMOr (GetNamedOperand $src0), (GetVal $old_exec)), (ConstantInt llvm_i32_ty, -1))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

def S_NOR_SAVEEXEC_B64_sem : InstSISemantic<S_NOR_SAVEEXEC_B64> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMXor (LLVMOr (GetNamedOperand $src0), (GetVal $old_exec)), (ConstantInt llvm_i64_ty, -1))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

// D = EXEC; EXEC = ~(S0 ^ EXEC); SCC = (EXEC != 0).
// Sources (B64 since GCN3; B32 added in GFX10/RDNA):
//   gcn3-instruction-set-architecture.pdf, page 136 (B64).
//   vega-shader-instruction-set-architecture.pdf, page 113 (B64).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 115 (B64).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 111 (B64).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 126 (B64).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 138 (B64).
//   rdna-shader-instruction-set-architecture.pdf, page 127 (B64), page 131 (B32).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 229.
//   rdna4-instruction-set-architecture.pdf, page 242.
def S_XNOR_SAVEEXEC_B32_sem : InstSISemantic<S_XNOR_SAVEEXEC_B32> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMXor (LLVMXor (GetNamedOperand $src0), (GetVal $old_exec)), (ConstantInt llvm_i32_ty, -1))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

def S_XNOR_SAVEEXEC_B64_sem : InstSISemantic<S_XNOR_SAVEEXEC_B64> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMXor (LLVMXor (GetNamedOperand $src0), (GetVal $old_exec)), (ConstantInt llvm_i64_ty, -1))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

// D = EXEC; EXEC = ~S0 & EXEC; SCC = (EXEC != 0).
// "ANDN1" = AND with NOT of operand 1 (S0).
// Sources (introduced in GFX9/Vega; B32 variant added in GFX10/RDNA):
//   vega-shader-instruction-set-architecture.pdf, page 115 (B64).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 116 (B64).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 113 (B64).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 130 (B64).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 142 (B64).
//   rdna-shader-instruction-set-architecture.pdf, page 128 (B64), page 131 (B32).
//   (Renamed to S_AND_NOT0_SAVEEXEC in RDNA3+:)
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 230.
//   rdna4-instruction-set-architecture.pdf, page 243.
def S_ANDN1_SAVEEXEC_B32_sem : InstSISemantic<S_ANDN1_SAVEEXEC_B32> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMAnd (LLVMXor (GetNamedOperand $src0), (ConstantInt llvm_i32_ty, -1)), (GetVal $old_exec))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

def S_ANDN1_SAVEEXEC_B64_sem : InstSISemantic<S_ANDN1_SAVEEXEC_B64> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMAnd (LLVMXor (GetNamedOperand $src0), (ConstantInt llvm_i64_ty, -1)), (GetVal $old_exec))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

// D = EXEC; EXEC = S0 & ~EXEC; SCC = (EXEC != 0).
// "ANDN2" = AND with NOT of operand 2 (EXEC).
// Sources (B64 since GCN3; B32 added in GFX10/RDNA):
//   gcn3-instruction-set-architecture.pdf, page 125 (B64).
//   vega-shader-instruction-set-architecture.pdf, page 113 (B64).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 114 (B64).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 111 (B64).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 125 (B64).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 137 (B64).
//   rdna-shader-instruction-set-architecture.pdf, page 127 (B64), page 130 (B32).
//   (Renamed to S_AND_NOT1_SAVEEXEC in RDNA3+:)
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 231.
//   rdna4-instruction-set-architecture.pdf, page 244.
def S_ANDN2_SAVEEXEC_B32_sem : InstSISemantic<S_ANDN2_SAVEEXEC_B32> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMAnd (GetNamedOperand $src0), (LLVMXor (GetVal $old_exec), (ConstantInt llvm_i32_ty, -1)))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

def S_ANDN2_SAVEEXEC_B64_sem : InstSISemantic<S_ANDN2_SAVEEXEC_B64> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMAnd (GetNamedOperand $src0), (LLVMXor (GetVal $old_exec), (ConstantInt llvm_i64_ty, -1)))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

// D = EXEC; EXEC = ~S0 | EXEC; SCC = (EXEC != 0).
// "ORN1" = OR with NOT of operand 1 (S0).
// Sources (introduced in GFX9/Vega; B32 variant added in GFX10/RDNA):
//   vega-shader-instruction-set-architecture.pdf, page 115 (B64).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 116 (B64).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 113 (B64).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 130 (B64).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 142 (B64).
//   rdna-shader-instruction-set-architecture.pdf, page 128 (B64), page 131 (B32).
//   (Renamed to S_OR_NOT0_SAVEEXEC in RDNA3+:)
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 230 (B32), page 230 (B64).
//   rdna4-instruction-set-architecture.pdf, page 243 (B32), page 244 (B64).
def S_ORN1_SAVEEXEC_B32_sem : InstSISemantic<S_ORN1_SAVEEXEC_B32> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMOr (LLVMXor (GetNamedOperand $src0), (ConstantInt llvm_i32_ty, -1)), (GetVal $old_exec))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

def S_ORN1_SAVEEXEC_B64_sem : InstSISemantic<S_ORN1_SAVEEXEC_B64> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMOr (LLVMXor (GetNamedOperand $src0), (ConstantInt llvm_i64_ty, -1)), (GetVal $old_exec))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

// D = EXEC; EXEC = S0 | ~EXEC; SCC = (EXEC != 0).
// "ORN2" = OR with NOT of operand 2 (EXEC).
// Sources (B64 since GCN3; B32 added in GFX10/RDNA):
//   gcn3-instruction-set-architecture.pdf, page 134 (B64).
//   vega-shader-instruction-set-architecture.pdf, page 113 (B64).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 114 (B64).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 111 (B64).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 126 (B64).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 138 (B64).
//   rdna-shader-instruction-set-architecture.pdf, page 127 (B64), page 130 (B32).
//   (Renamed to S_OR_NOT1_SAVEEXEC in RDNA3+:)
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 231 (B32), page 232 (B64).
//   rdna4-instruction-set-architecture.pdf, page 245.
def S_ORN2_SAVEEXEC_B32_sem : InstSISemantic<S_ORN2_SAVEEXEC_B32> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMOr (GetNamedOperand $src0), (LLVMXor (GetVal $old_exec), (ConstantInt llvm_i32_ty, -1)))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

def S_ORN2_SAVEEXEC_B64_sem : InstSISemantic<S_ORN2_SAVEEXEC_B64> {
  let Semantic = [
    (DefVal $old_exec, (ImplicitUse EXEC)),
    (DefVal $new_exec, (LLVMOr (GetNamedOperand $src0), (LLVMXor (GetVal $old_exec), (ConstantInt llvm_i64_ty, -1)))),
    (ImplicitDef EXEC, (GetVal $new_exec)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $new_exec), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $old_exec))
  ];
}

//------------------------------------------------------------------------------
// Bit Count Instructions
//------------------------------------------------------------------------------

// D.i = CountZeroBits(S0.u) = 32 - popcount(S0); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 125.
//   vega-shader-instruction-set-architecture.pdf, page 107.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 107.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 117.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 129.
//   rdna-shader-instruction-set-architecture.pdf, page 120.
//   rdna2-shader-instruction-set-architecture.pdf, page 120.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 223.
//   rdna4-instruction-set-architecture.pdf, page 236.
def S_BCNT0_I32_B32_sem : InstSISemantic<S_BCNT0_I32_B32> {
  let Semantic = [
    (DefVal $popcnt, (LLVMUnaryIntrinsic int_ctpop, (GetNamedOperand $src0))),
    (DefVal $result, (LLVMSub (ConstantInt llvm_i32_ty, 32), $popcnt)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.i = CountZeroBits(S0.u) = 64 - popcount(S0); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 126.
//   vega-shader-instruction-set-architecture.pdf, page 107.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 107.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 117.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 129.
//   rdna-shader-instruction-set-architecture.pdf, page 121.
//   rdna2-shader-instruction-set-architecture.pdf, page 120.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 223.
//   rdna4-instruction-set-architecture.pdf, page 236.
def S_BCNT0_I32_B64_sem : InstSISemantic<S_BCNT0_I32_B64> {
  let Semantic = [
    (DefVal $popcnt, (LLVMUnaryIntrinsic int_ctpop, (GetNamedOperand $src0))),
    (DefVal $result, (LLVMSub (ConstantInt llvm_i64_ty, 64), $popcnt)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.i = CountOneBits(S0.u) = popcount(S0); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 126.
//   vega-shader-instruction-set-architecture.pdf, page 108.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 109.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 107.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 118.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 130.
//   rdna-shader-instruction-set-architecture.pdf, page 121.
//   rdna2-shader-instruction-set-architecture.pdf, page 120.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 223.
//   rdna4-instruction-set-architecture.pdf, page 237.
def S_BCNT1_I32_B32_sem : InstSISemantic<S_BCNT1_I32_B32> {
  let Semantic = [
    (DefVal $result, (LLVMUnaryIntrinsic int_ctpop, (GetNamedOperand $src0))),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.i = CountOneBits(S0.u) = popcount(S0); SCC = (D != 0).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 126.
//   vega-shader-instruction-set-architecture.pdf, page 108.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 109.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 107.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 118.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 130.
//   rdna-shader-instruction-set-architecture.pdf, page 121.
//   rdna2-shader-instruction-set-architecture.pdf, page 120.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 224.
//   rdna4-instruction-set-architecture.pdf, page 237.
def S_BCNT1_I32_B64_sem : InstSISemantic<S_BCNT1_I32_B64> {
  let Semantic = [
    (DefVal $result, (LLVMUnaryIntrinsic int_ctpop, (GetNamedOperand $src0))),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $result), (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_BREV_B32 / S_BREV_B64
//------------------------------------------------------------------------------

// D.u = BITREVERSE(S0.u) (reverse bits of 32-bit input).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 127.
//   vega-shader-instruction-set-architecture.pdf, page 99.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 100.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 99.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 98.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 109.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 121.
//   rdna-shader-instruction-set-architecture.pdf, page 112.
//   rdna2-shader-instruction-set-architecture.pdf, page 111.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 217.
//   rdna4-instruction-set-architecture.pdf, page 230.
def S_BREV_B32_sem : InstSISemantic<S_BREV_B32> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMUnaryIntrinsic int_bitreverse, (GetNamedOperand $src0)))
  ];
}

// D.u = BITREVERSE(S0.u) (reverse bits of 64-bit input).
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 128.
//   vega-shader-instruction-set-architecture.pdf, page 99.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 100.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 99.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 98.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 109.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 121.
//   rdna-shader-instruction-set-architecture.pdf, page 112.
//   rdna2-shader-instruction-set-architecture.pdf, page 112.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 218.
//   rdna4-instruction-set-architecture.pdf, page 231.
def S_BREV_B64_sem : InstSISemantic<S_BREV_B64> {
  let Semantic = [
    (SetNamedOperand $sdst, (LLVMUnaryIntrinsic int_bitreverse, (GetNamedOperand $src0)))
  ];
}

//------------------------------------------------------------------------------
// S_FF0_I32_B32 / S_FF0_I32_B64
//------------------------------------------------------------------------------

// D.i = FindFirstZero(S0.u) from LSB; if no zeros, return -1.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 129.
//   vega-shader-instruction-set-architecture.pdf, page 100.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 100.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 110.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 122.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 113.
//   (Removed in RDNA3+; no gfx11/gfx12 real encodings.)
def S_FF0_I32_B32_sem : InstSISemantic<S_FF0_I32_B32> {
  let Semantic = [
    (DefVal $not_src, (LLVMXor (GetNamedOperand $src0),
                                (ConstantInt llvm_i32_ty, -1))),
    (DefVal $ctz, (LLVMUnaryIntrinsic int_cttz, (GetVal $not_src))),
    (DefVal $result, (LLVMSelect
      (LLVMCmp LLVMICmpEQ, (GetVal $not_src), (ConstantInt llvm_i32_ty, 0)),
      (ConstantSignedInt llvm_i32_ty, -1),
      (GetVal $ctz))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.i = FindFirstZero(S0.u) from LSB; if no zeros, return -1.
// Input is 64-bit, output is 32-bit.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 129.
//   vega-shader-instruction-set-architecture.pdf, page 101.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 100.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 111.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 123.
//   rdna-shader-instruction-set-architecture.pdf, page 114.
//   rdna2-shader-instruction-set-architecture.pdf, page 113.
//   (Removed in RDNA3+; no gfx11/gfx12 real encodings.)
def S_FF0_I32_B64_sem : InstSISemantic<S_FF0_I32_B64> {
  let Semantic = [
    (DefVal $not_src, (LLVMXor (GetNamedOperand $src0),
                                (ConstantInt llvm_i64_ty, -1))),
    (DefVal $ctz, (LLVMUnaryIntrinsic int_cttz, (GetVal $not_src))),
    (DefVal $ctz32, (LLVMTrunc (GetVal $ctz), llvm_i32_ty)),
    (DefVal $result, (LLVMSelect
      (LLVMCmp LLVMICmpEQ, (GetVal $not_src), (ConstantInt llvm_i64_ty, 0)),
      (ConstantSignedInt llvm_i32_ty, -1),
      (GetVal $ctz32))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_FF1_I32_B32 / S_FF1_I32_B64
//------------------------------------------------------------------------------

// D.i = FindFirstOne(S0.u) from LSB; if no ones, return -1.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 129.
//   vega-shader-instruction-set-architecture.pdf, page 101.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 101.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 100.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 111.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 123.
//   rdna-shader-instruction-set-architecture.pdf, page 115.
//   rdna2-shader-instruction-set-architecture.pdf, page 113.
//   (Renamed to S_CTZ_I32_B32 in RDNA3+: rdna4-instruction-set-architecture.pdf, page 231.)
def S_FF1_I32_B32_sem : InstSISemantic<S_FF1_I32_B32> {
  let Semantic = [
    (DefVal $ctz, (LLVMUnaryIntrinsic int_cttz, (GetNamedOperand $src0))),
    (DefVal $result, (LLVMSelect
      (LLVMCmp LLVMICmpEQ, (GetNamedOperand $src0), (ConstantInt llvm_i32_ty, 0)),
      (ConstantSignedInt llvm_i32_ty, -1),
      (GetVal $ctz))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.i = FindFirstOne(S0.u) from LSB; if no ones, return -1.
// Input is 64-bit, output is 32-bit.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 129.
//   vega-shader-instruction-set-architecture.pdf, page 102.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 112.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 124.
//   rdna-shader-instruction-set-architecture.pdf, page 115.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   (Renamed to S_CTZ_I32_B64 in RDNA3+: rdna4-instruction-set-architecture.pdf, page 231.)
def S_FF1_I32_B64_sem : InstSISemantic<S_FF1_I32_B64> {
  let Semantic = [
    (DefVal $ctz, (LLVMUnaryIntrinsic int_cttz, (GetNamedOperand $src0))),
    (DefVal $ctz32, (LLVMTrunc (GetVal $ctz), llvm_i32_ty)),
    (DefVal $result, (LLVMSelect
      (LLVMCmp LLVMICmpEQ, (GetNamedOperand $src0), (ConstantInt llvm_i64_ty, 0)),
      (ConstantSignedInt llvm_i32_ty, -1),
      (GetVal $ctz32))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_FLBIT_I32_B32 / S_FLBIT_I32_B64
//------------------------------------------------------------------------------

// D.i = FindFirstOne(S0.u) from MSB (count leading zeros); if no ones, return -1.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 130.
//   vega-shader-instruction-set-architecture.pdf, page 102.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 112.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 124.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 114.
//   (Renamed to S_CLZ_I32_U32 in RDNA3+: rdna4-instruction-set-architecture.pdf, page 232.)
def S_FLBIT_I32_B32_sem : InstSISemantic<S_FLBIT_I32_B32> {
  let Semantic = [
    (DefVal $clz, (LLVMUnaryIntrinsic int_ctlz, (GetNamedOperand $src0))),
    (DefVal $result, (LLVMSelect
      (LLVMCmp LLVMICmpEQ, (GetNamedOperand $src0), (ConstantInt llvm_i32_ty, 0)),
      (ConstantSignedInt llvm_i32_ty, -1),
      (GetVal $clz))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.i = FindFirstOne(S0.u) from MSB (count leading zeros); if no ones, return -1.
// Input is 64-bit, output is 32-bit.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 130.
//   vega-shader-instruction-set-architecture.pdf, page 103.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 102.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 101.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 113.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 125.
//   rdna-shader-instruction-set-architecture.pdf, page 116.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
//   (Renamed to S_CLZ_I32_U64 in RDNA3+: rdna4-instruction-set-architecture.pdf, page 232.)
def S_FLBIT_I32_B64_sem : InstSISemantic<S_FLBIT_I32_B64> {
  let Semantic = [
    (DefVal $clz, (LLVMUnaryIntrinsic int_ctlz, (GetNamedOperand $src0))),
    (DefVal $clz32, (LLVMTrunc (GetVal $clz), llvm_i32_ty)),
    (DefVal $result, (LLVMSelect
      (LLVMCmp LLVMICmpEQ, (GetNamedOperand $src0), (ConstantInt llvm_i64_ty, 0)),
      (ConstantSignedInt llvm_i32_ty, -1),
      (GetVal $clz32))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_FLBIT_I32 / S_FLBIT_I32_I64
//------------------------------------------------------------------------------

// D.i = Find first bit opposite of sign bit from MSB.
// If S0 == 0 or S0 == -1, return -1.
// Computed as ctlz(S0 ^ ashr(S0, 31)); return -1 if result == 32.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 130.
//   vega-shader-instruction-set-architecture.pdf, page 103.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 102.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 113.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 125.
//   rdna-shader-instruction-set-architecture.pdf, page 117.
//   rdna2-shader-instruction-set-architecture.pdf, page 115.
//   (Renamed to S_CLS_I32 in RDNA3+: rdna4-instruction-set-architecture.pdf, page 233.)
def S_FLBIT_I32_sem : InstSISemantic<S_FLBIT_I32> {
  let Semantic = [
    (DefVal $src, (GetNamedOperand $src0)),
    (DefVal $sign_ext, (LLVMAShr (GetVal $src),
                                  (ConstantInt llvm_i32_ty, 31))),
    (DefVal $tmp, (LLVMXor (GetVal $src), (GetVal $sign_ext))),
    (DefVal $clz, (LLVMUnaryIntrinsic int_ctlz, (GetVal $tmp))),
    (DefVal $result, (LLVMSelect
      (LLVMCmp LLVMICmpEQ, (GetVal $clz), (ConstantInt llvm_i32_ty, 32)),
      (ConstantSignedInt llvm_i32_ty, -1),
      (GetVal $clz))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.i = Find first bit opposite of sign bit from MSB (64-bit input).
// If S0 == 0 or S0 == -1, return -1.
// Computed as ctlz(S0 ^ ashr(S0, 63)); return -1 if result == 64.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 130.
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 102.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 114.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 126.
//   rdna-shader-instruction-set-architecture.pdf, page 117.
//   rdna2-shader-instruction-set-architecture.pdf, page 116.
//   (Renamed to S_CLS_I32_I64 in RDNA3+: rdna4-instruction-set-architecture.pdf, page 233.)
def S_FLBIT_I32_I64_sem : InstSISemantic<S_FLBIT_I32_I64> {
  let Semantic = [
    (DefVal $src, (GetNamedOperand $src0)),
    (DefVal $sign_ext, (LLVMAShr (GetVal $src),
                                  (ConstantInt llvm_i64_ty, 63))),
    (DefVal $tmp, (LLVMXor (GetVal $src), (GetVal $sign_ext))),
    (DefVal $clz, (LLVMUnaryIntrinsic int_ctlz, (GetVal $tmp))),
    (DefVal $clz32, (LLVMTrunc (GetVal $clz), llvm_i32_ty)),
    (DefVal $result, (LLVMSelect
      (LLVMCmp LLVMICmpEQ, (GetVal $clz32), (ConstantInt llvm_i32_ty, 64)),
      (ConstantSignedInt llvm_i32_ty, -1),
      (GetVal $clz32))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_BITSET0_B32 / S_BITSET0_B64
//------------------------------------------------------------------------------

// D.u[S0.u[4:0]] = 0. Clear the bit at position S0[4:0] in D.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 126.
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 102.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 114.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 126.
//   rdna-shader-instruction-set-architecture.pdf, page 117.
//   rdna2-shader-instruction-set-architecture.pdf, page 116.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 221.
//   rdna4-instruction-set-architecture.pdf, page 234.
def S_BITSET0_B32_sem : InstSISemantic<S_BITSET0_B32> {
  let Semantic = [
    (DefVal $bitpos, (LLVMAnd (GetNamedOperand $src0),
                               (ConstantInt llvm_i32_ty, 31))),
    (DefVal $mask, (LLVMXor (LLVMShl (ConstantInt llvm_i32_ty, 1),
                                      (GetVal $bitpos)),
                             (ConstantInt llvm_i32_ty, -1))),
    (DefVal $result, (LLVMAnd (GetNamedOperand $sdst_in), (GetVal $mask))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.u[S0.u[5:0]] = 0. Clear the bit at position S0[5:0] in D.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 127.
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 102.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 115.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 118.
//   rdna2-shader-instruction-set-architecture.pdf, page 116.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 221.
//   rdna4-instruction-set-architecture.pdf, page 234.
def S_BITSET0_B64_sem : InstSISemantic<S_BITSET0_B64> {
  let Semantic = [
    (DefVal $bitpos, (LLVMZExt (LLVMAnd (GetNamedOperand $src0),
                                         (ConstantInt llvm_i32_ty, 63)),
                                llvm_i64_ty)),
    (DefVal $mask, (LLVMXor (LLVMShl (ConstantInt llvm_i64_ty, 1),
                                      (GetVal $bitpos)),
                             (ConstantInt llvm_i64_ty, -1))),
    (DefVal $result, (LLVMAnd (GetNamedOperand $sdst_in), (GetVal $mask))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_BITSET1_B32 / S_BITSET1_B64
//------------------------------------------------------------------------------

// D.u[S0.u[4:0]] = 1. Set the bit at position S0[4:0] in D.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 127.
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 102.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 115.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 118.
//   rdna2-shader-instruction-set-architecture.pdf, page 116.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 221.
//   rdna4-instruction-set-architecture.pdf, page 234.
def S_BITSET1_B32_sem : InstSISemantic<S_BITSET1_B32> {
  let Semantic = [
    (DefVal $bitpos, (LLVMAnd (GetNamedOperand $src0),
                               (ConstantInt llvm_i32_ty, 31))),
    (DefVal $bit, (LLVMShl (ConstantInt llvm_i32_ty, 1), (GetVal $bitpos))),
    (DefVal $result, (LLVMOr (GetNamedOperand $sdst_in), (GetVal $bit))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.u[S0.u[5:0]] = 1. Set the bit at position S0[5:0] in D.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 127.
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 103.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 102.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 115.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 118.
//   rdna2-shader-instruction-set-architecture.pdf, page 116.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 222.
//   rdna4-instruction-set-architecture.pdf, page 235.
def S_BITSET1_B64_sem : InstSISemantic<S_BITSET1_B64> {
  let Semantic = [
    (DefVal $bitpos, (LLVMZExt (LLVMAnd (GetNamedOperand $src0),
                                         (ConstantInt llvm_i32_ty, 63)),
                                llvm_i64_ty)),
    (DefVal $bit, (LLVMShl (ConstantInt llvm_i64_ty, 1), (GetVal $bitpos))),
    (DefVal $result, (LLVMOr (GetNamedOperand $sdst_in), (GetVal $bit))),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_BITREPLICATE_B64_B32
//------------------------------------------------------------------------------

// Replicate each bit of a 32-bit input to two consecutive bits in a 64-bit
// output. D0.u64[i*2] = S0.u32[i]; D0.u64[i*2+1] = S0.u32[i] for i in 0:31.
// Sources:
//   vega-shader-instruction-set-architecture.pdf, page 107.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 106.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 105.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 123.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 135.
//   rdna-shader-instruction-set-architecture.pdf, page 122.
//   rdna2-shader-instruction-set-architecture.pdf, page 120.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 222.
//   rdna4-instruction-set-architecture.pdf, page 235.
//   (Not present in GCN3; introduced in GFX9/Vega.)
def S_BITREPLICATE_B64_B32_sem : InstSIIntrinsic<S_BITREPLICATE_B64_B32,
                                                  int_amdgcn_s_bitreplicate>;

//------------------------------------------------------------------------------
// S_ANDN1_WREXEC_B32 / S_ANDN1_WREXEC_B64
//------------------------------------------------------------------------------

// EXEC = ~S0 & EXEC; D = EXEC; SCC = EXEC != 0.
// "ANDN1" = AND with NOT of operand 1 (S0).
// Sources (B64 only in GFX9; B32 added in GFX10/RDNA):
//   vega-shader-instruction-set-architecture.pdf, page 107 (B64 only).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 108 (B64 only).
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 106 (B64 only).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 105 (B64 only).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 122 (B64 only).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 134 (B64 only).
//   rdna-shader-instruction-set-architecture.pdf, page 121 (B64), page 123 (B32).
//   rdna2-shader-instruction-set-architecture.pdf, page 119 (B64), page 122 (B32).
//   (Renamed to S_AND_NOT0_WREXEC in RDNA3+:)
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 232.
//   rdna4-instruction-set-architecture.pdf, page 245.
def S_ANDN1_WREXEC_B32_sem : InstSISemantic<S_ANDN1_WREXEC_B32> {
  let Semantic = [
    (DefVal $temp, (LLVMAnd (LLVMXor (GetNamedOperand $src0),
                                      (ConstantInt llvm_i32_ty, -1)),
                             (ImplicitUse EXEC))),
    (ImplicitDef EXEC, (GetVal $temp)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $temp),
                                       (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $temp))
  ];
}

def S_ANDN1_WREXEC_B64_sem : InstSISemantic<S_ANDN1_WREXEC_B64> {
  let Semantic = [
    (DefVal $temp, (LLVMAnd (LLVMXor (GetNamedOperand $src0),
                                      (ConstantInt llvm_i64_ty, -1)),
                             (ImplicitUse EXEC))),
    (ImplicitDef EXEC, (GetVal $temp)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $temp),
                                       (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $temp))
  ];
}

//------------------------------------------------------------------------------
// S_ANDN2_WREXEC_B32 / S_ANDN2_WREXEC_B64
//------------------------------------------------------------------------------

// EXEC = S0 & ~EXEC; D = EXEC; SCC = EXEC != 0.
// "ANDN2" = AND with NOT of operand 2 (EXEC).
// Sources (B64 only in GFX9; B32 added in GFX10/RDNA):
//   vega-shader-instruction-set-architecture.pdf, page 107 (B64 only).
//   vega-7nm-shader-instruction-set-architecture.pdf, page 108 (B64 only).
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 106 (B64 only).
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 105 (B64 only).
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 123 (B64 only).
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 135 (B64 only).
//   rdna-shader-instruction-set-architecture.pdf, page 121 (B64), page 123 (B32).
//   rdna2-shader-instruction-set-architecture.pdf, page 120 (B64), page 122 (B32).
//   (Renamed to S_AND_NOT1_WREXEC in RDNA3+:)
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 233.
//   rdna4-instruction-set-architecture.pdf, page 246.
def S_ANDN2_WREXEC_B32_sem : InstSISemantic<S_ANDN2_WREXEC_B32> {
  let Semantic = [
    (DefVal $temp, (LLVMAnd (GetNamedOperand $src0),
                    (LLVMXor (ImplicitUse EXEC),
                              (ConstantInt llvm_i32_ty, -1)))),
    (ImplicitDef EXEC, (GetVal $temp)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $temp),
                                       (ConstantInt llvm_i32_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $temp))
  ];
}

def S_ANDN2_WREXEC_B64_sem : InstSISemantic<S_ANDN2_WREXEC_B64> {
  let Semantic = [
    (DefVal $temp, (LLVMAnd (GetNamedOperand $src0),
                    (LLVMXor (ImplicitUse EXEC),
                              (ConstantInt llvm_i64_ty, -1)))),
    (ImplicitDef EXEC, (GetVal $temp)),
    (ImplicitDef SCC, (LLVMCmp LLVMICmpNE, (GetVal $temp),
                                       (ConstantInt llvm_i64_ty, 0))),
    (SetNamedOperand $sdst, (GetVal $temp))
  ];
}

//------------------------------------------------------------------------------
// S_CVT_F32_I32 / S_CVT_F32_U32
//------------------------------------------------------------------------------

// D.f32 = (float)S0.i32. Convert signed 32-bit integer to float.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 254.
def S_CVT_F32_I32_sem : InstSISemantic<S_CVT_F32_I32> {
  let Semantic = [
    (DefVal $result, (LLVMBitCast
      (LLVMSIToFP (GetNamedOperand $src0), llvm_float_ty),
      llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.f32 = (float)S0.u32. Convert unsigned 32-bit integer to float.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 254.
def S_CVT_F32_U32_sem : InstSISemantic<S_CVT_F32_U32> {
  let Semantic = [
    (DefVal $result, (LLVMBitCast
      (LLVMUIToFP (GetNamedOperand $src0), llvm_float_ty),
      llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_CVT_I32_F32 / S_CVT_U32_F32
//------------------------------------------------------------------------------

// D.i32 = (int)S0.f32. Convert float to signed 32-bit integer.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 254.
def S_CVT_I32_F32_sem : InstSISemantic<S_CVT_I32_F32> {
  let Semantic = [
    (DefVal $result, (LLVMFPToSI
      (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
      llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.u32 = (uint)S0.f32. Convert float to unsigned 32-bit integer.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 254.
def S_CVT_U32_F32_sem : InstSISemantic<S_CVT_U32_F32> {
  let Semantic = [
    (DefVal $result, (LLVMFPToUI
      (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty),
      llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_CVT_F16_F32 / S_CVT_F32_F16 / S_CVT_HI_F32_F16
//------------------------------------------------------------------------------

// D.f16 = f32_to_f16(S0.f32). Convert float to half (lower 16 bits of D).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 255.
def S_CVT_F16_F32_sem : InstSISemantic<S_CVT_F16_F32> {
  let Semantic = [
    (DefVal $float_val, (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty)),
    (DefVal $half_val, (LLVMFPTrunc (GetVal $float_val), llvm_half_ty)),
    (DefVal $result, (LLVMZExt (LLVMBitCast (GetVal $half_val), llvm_i16_ty),
                                llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.f32 = f16_to_f32(S0.f16). Convert half (lower 16 bits) to float.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 255.
def S_CVT_F32_F16_sem : InstSISemantic<S_CVT_F32_F16> {
  let Semantic = [
    (DefVal $src_i16, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
    (DefVal $half_val, (LLVMBitCast (GetVal $src_i16), llvm_half_ty)),
    (DefVal $float_val, (LLVMFPExt (GetVal $half_val), llvm_float_ty)),
    (DefVal $result, (LLVMBitCast (GetVal $float_val), llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.f32 = f16_to_f32(S0[31:16].f16). Convert high half to float.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 255.
def S_CVT_HI_F32_F16_sem : InstSISemantic<S_CVT_HI_F32_F16> {
  let Semantic = [
    (DefVal $hi16, (LLVMTrunc (LLVMLShr (GetNamedOperand $src0),
                                         (ConstantInt llvm_i32_ty, 16)),
                               llvm_i16_ty)),
    (DefVal $half_val, (LLVMBitCast (GetVal $hi16), llvm_half_ty)),
    (DefVal $float_val, (LLVMFPExt (GetVal $half_val), llvm_float_ty)),
    (DefVal $result, (LLVMBitCast (GetVal $float_val), llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_CEIL_F32 / S_CEIL_F16
//------------------------------------------------------------------------------

// D.f32 = ceil(S0.f32). Round float up to next integer.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 253.
def S_CEIL_F32_sem : InstSISemantic<S_CEIL_F32> {
  let Semantic = [
    (DefVal $result, (LLVMBitCast
      (LLVMUnaryIntrinsic int_ceil, (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty)),
      llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.f16 = ceil(S0.f16). Round half up to next integer.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 255.
def S_CEIL_F16_sem : InstSISemantic<S_CEIL_F16> {
  let Semantic = [
    (DefVal $src_i16, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
    (DefVal $half_val, (LLVMBitCast (GetVal $src_i16), llvm_half_ty)),
    (DefVal $result_half, (LLVMUnaryIntrinsic int_ceil, (GetVal $half_val))),
    (DefVal $result, (LLVMZExt (LLVMBitCast (GetVal $result_half), llvm_i16_ty),
                                llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_FLOOR_F32 / S_FLOOR_F16
//------------------------------------------------------------------------------

// D.f32 = floor(S0.f32). Round float down to previous integer.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 253.
def S_FLOOR_F32_sem : InstSISemantic<S_FLOOR_F32> {
  let Semantic = [
    (DefVal $result, (LLVMBitCast
      (LLVMUnaryIntrinsic int_floor, (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty)),
      llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.f16 = floor(S0.f16). Round half down to previous integer.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 255.
def S_FLOOR_F16_sem : InstSISemantic<S_FLOOR_F16> {
  let Semantic = [
    (DefVal $src_i16, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
    (DefVal $half_val, (LLVMBitCast (GetVal $src_i16), llvm_half_ty)),
    (DefVal $result_half, (LLVMUnaryIntrinsic int_floor, (GetVal $half_val))),
    (DefVal $result, (LLVMZExt (LLVMBitCast (GetVal $result_half), llvm_i16_ty),
                                llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_TRUNC_F32 / S_TRUNC_F16
//------------------------------------------------------------------------------

// D.f32 = trunc(S0.f32). Round float toward zero.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 253.
def S_TRUNC_F32_sem : InstSISemantic<S_TRUNC_F32> {
  let Semantic = [
    (DefVal $result, (LLVMBitCast
      (LLVMUnaryIntrinsic int_trunc, (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty)),
      llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.f16 = trunc(S0.f16). Round half toward zero.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 256.
def S_TRUNC_F16_sem : InstSISemantic<S_TRUNC_F16> {
  let Semantic = [
    (DefVal $src_i16, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
    (DefVal $half_val, (LLVMBitCast (GetVal $src_i16), llvm_half_ty)),
    (DefVal $result_half, (LLVMUnaryIntrinsic int_trunc, (GetVal $half_val))),
    (DefVal $result, (LLVMZExt (LLVMBitCast (GetVal $result_half), llvm_i16_ty),
                                llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_RNDNE_F32 / S_RNDNE_F16
//------------------------------------------------------------------------------

// D.f32 = roundeven(S0.f32). Round float to nearest even integer.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 254.
def S_RNDNE_F32_sem : InstSISemantic<S_RNDNE_F32> {
  let Semantic = [
    (DefVal $result, (LLVMBitCast
      (LLVMUnaryIntrinsic int_roundeven, (LLVMBitCast (GetNamedOperand $src0), llvm_float_ty)),
      llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

// D.f16 = roundeven(S0.f16). Round half to nearest even integer.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 256.
def S_RNDNE_F16_sem : InstSISemantic<S_RNDNE_F16> {
  let Semantic = [
    (DefVal $src_i16, (LLVMTrunc (GetNamedOperand $src0), llvm_i16_ty)),
    (DefVal $half_val, (LLVMBitCast (GetVal $src_i16), llvm_half_ty)),
    (DefVal $result_half, (LLVMUnaryIntrinsic int_roundeven, (GetVal $half_val))),
    (DefVal $result, (LLVMZExt (LLVMBitCast (GetVal $result_half), llvm_i16_ty),
                                llvm_i32_ty)),
    (SetNamedOperand $sdst, (GetVal $result))
  ];
}

//------------------------------------------------------------------------------
// S_GETPC_B64 / S_GETPC_B64_pseudo
//------------------------------------------------------------------------------

// D.u = PC + 4. Get the byte address of the next instruction.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 131.
//   vega-shader-instruction-set-architecture.pdf, page 104.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 105.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 104.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 103.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 115.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 127.
//   rdna-shader-instruction-set-architecture.pdf, page 118.
//   rdna2-shader-instruction-set-architecture.pdf, page 116.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 236.
//   rdna4-instruction-set-architecture.pdf, page 249.
def S_GETPC_B64_sem : InstSIIntrinsic<S_GETPC_B64, int_amdgcn_s_getpc>;

// LLVM-only pseudo (no physical opcode in ISA docs); zero-extends PC result.
// Defined in llvm-project: llvm/lib/Target/AMDGPU/SOPInstructions.td, line 316.
def S_GETPC_B64_pseudo_sem : InstSIIntrinsic<S_GETPC_B64_pseudo,
                                              int_amdgcn_s_getpc>;

//------------------------------------------------------------------------------
// S_SENDMSG_RTN_B32 / S_SENDMSG_RTN_B64
//------------------------------------------------------------------------------

// Send a message and return a response into the destination register.
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 237.
//   rdna4-instruction-set-architecture.pdf, page 250.
// Defined in llvm-project: llvm/lib/Target/AMDGPU/SOPInstructions.td, line 424 (B32), line 428 (B64).
def S_SENDMSG_RTN_B32_sem : InstSIIntrinsic<S_SENDMSG_RTN_B32,
                                             int_amdgcn_s_sendmsg_rtn>;

def S_SENDMSG_RTN_B64_sem : InstSIIntrinsic<S_SENDMSG_RTN_B64,
                                             int_amdgcn_s_sendmsg_rtn>;

//------------------------------------------------------------------------------
// S_SLEEP_VAR
//------------------------------------------------------------------------------

// Cause a wave to sleep for approximately (64*(S0[6:0]-1) ... 64*S0[6:0])
// clocks. No sleep when S0[6:0] is zero.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 253.
// Defined in llvm-project: llvm/lib/Target/AMDGPU/SOPInstructions.td, line 1744.
def S_SLEEP_VAR_sem : InstSIIntrinsic<S_SLEEP_VAR, int_amdgcn_s_sleep_var>;

//------------------------------------------------------------------------------
// S_BARRIER_SIGNAL_IMM / S_BARRIER_SIGNAL_M0
//------------------------------------------------------------------------------

// Signal that a wave has arrived at a barrier.
// In RDNA4, S_BARRIER_SIGNAL is a unified instruction (not split into _IMM/_M0);
// the LLVM backend splits them into separate pseudo records for the
// immediate vs M0 operand paths.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 250 (unified S_BARRIER_SIGNAL).
// Defined in llvm-project: llvm/lib/Target/AMDGPU/SOPInstructions.td, line 536 (IMM), line 496 (M0).
def S_BARRIER_SIGNAL_IMM_sem : InstSIIntrinsic<S_BARRIER_SIGNAL_IMM,
                                                int_amdgcn_s_barrier_signal>;

def S_BARRIER_SIGNAL_M0_sem : InstSIIntrinsic<S_BARRIER_SIGNAL_M0,
                                               int_amdgcn_s_barrier_signal>;

//------------------------------------------------------------------------------
// S_BARRIER_SIGNAL_ISFIRST_IMM / S_BARRIER_SIGNAL_ISFIRST_M0
//------------------------------------------------------------------------------

// Signal barrier arrival and set SCC if this is the first wave to signal.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 251 (unified S_BARRIER_SIGNAL_ISFIRST).
// Defined in llvm-project: llvm/lib/Target/AMDGPU/SOPInstructions.td, line 542 (IMM), line 502 (M0).
def S_BARRIER_SIGNAL_ISFIRST_IMM_sem :
  InstSIIntrinsic<S_BARRIER_SIGNAL_ISFIRST_IMM,
                   int_amdgcn_s_barrier_signal_isfirst>;

def S_BARRIER_SIGNAL_ISFIRST_M0_sem :
  InstSIIntrinsic<S_BARRIER_SIGNAL_ISFIRST_M0,
                   int_amdgcn_s_barrier_signal_isfirst>;

//------------------------------------------------------------------------------
// S_BARRIER_INIT_IMM / S_BARRIER_INIT_M0
//------------------------------------------------------------------------------

// Initialize a barrier with a member count.
// Not found in any ISA PDF (gfx12+ only, not documented in available ISA manuals).
// Defined in llvm-project: llvm/lib/Target/AMDGPU/SOPInstructions.td, line 516 (IMM), line 510 (M0).
// Has gfx12 real encodings (opcode 0x051).
def S_BARRIER_INIT_IMM_sem : InstSIIntrinsic<S_BARRIER_INIT_IMM,
                                              int_amdgcn_s_barrier_init>;

def S_BARRIER_INIT_M0_sem : InstSIIntrinsic<S_BARRIER_INIT_M0,
                                             int_amdgcn_s_barrier_init>;

//------------------------------------------------------------------------------
// S_BARRIER_JOIN_IMM / S_BARRIER_JOIN_M0
//------------------------------------------------------------------------------

// Join a barrier.
// Not found in any ISA PDF (gfx12+ only, not documented in available ISA manuals).
// Defined in llvm-project: llvm/lib/Target/AMDGPU/SOPInstructions.td, line 551 (IMM), line 522 (M0).
// Has gfx12 real encodings (opcode 0x052).
def S_BARRIER_JOIN_IMM_sem : InstSIIntrinsic<S_BARRIER_JOIN_IMM,
                                              int_amdgcn_s_barrier_join>;

def S_BARRIER_JOIN_M0_sem : InstSIIntrinsic<S_BARRIER_JOIN_M0,
                                             int_amdgcn_s_barrier_join>;

//------------------------------------------------------------------------------
// S_GET_BARRIER_STATE_IMM / S_GET_BARRIER_STATE_M0
//------------------------------------------------------------------------------

// Read out the current barrier state.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 251 (unified S_GET_BARRIER_STATE).
// Defined in llvm-project: llvm/lib/Target/AMDGPU/SOPInstructions.td, line 565 (IMM), line 571 (M0).
def S_GET_BARRIER_STATE_IMM_sem :
  InstSIIntrinsic<S_GET_BARRIER_STATE_IMM, int_amdgcn_s_get_barrier_state>;

def S_GET_BARRIER_STATE_M0_sem :
  InstSIIntrinsic<S_GET_BARRIER_STATE_M0, int_amdgcn_s_get_barrier_state>;

//------------------------------------------------------------------------------
// S_WAKEUP_BARRIER_IMM / S_WAKEUP_BARRIER_M0
//------------------------------------------------------------------------------

// Wake up waves sleeping on a barrier.
// Not found in any ISA PDF (gfx12+ only, not documented in available ISA manuals).
// Defined in llvm-project: llvm/lib/Target/AMDGPU/SOPInstructions.td, line 557 (IMM), line 528 (M0).
// Has gfx12 real encodings (opcode 0x057).
def S_WAKEUP_BARRIER_IMM_sem : InstSIIntrinsic<S_WAKEUP_BARRIER_IMM,
                                                int_amdgcn_s_wakeup_barrier>;

def S_WAKEUP_BARRIER_M0_sem : InstSIIntrinsic<S_WAKEUP_BARRIER_M0,
                                               int_amdgcn_s_wakeup_barrier>;
