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
    (DefVal $tmp_add,
      (LLVMAdd (LLVMZExt (GetNamedOperand $src0), si_llvm_i64_ty),
               (LLVMZExt (GetNamedOperand $simm16), si_llvm_i64_ty))),
    SetSCCImplicit<(LLVMCmp LLVMICmpUGE, (GetVal $tmp_add), (ConstantInt si_llvm_i64_ty, 0x100000000))>.ret,
    (SetNamedOperand $sdst, (LLVMTrunc (GetVal $tmp_add), si_llvm_i32_ty))
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
    (DefVal $tmp, (LLVMSelect GetSCCImplicit<>.ret,
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
    SetSCCImplicit<(LLVMCmp LLVMICmpEQ, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
    SetSCCImplicit<(LLVMCmp LLVMICmpEQ, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
    SetSCCImplicit<(LLVMCmp LLVMICmpSGE, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
    SetSCCImplicit<(LLVMCmp LLVMICmpUGE, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
    SetSCCImplicit<(LLVMCmp LLVMICmpSGT, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
     SetSCCImplicit<(LLVMCmp LLVMICmpUGT, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
    SetSCCImplicit<(LLVMCmp LLVMICmpSLE, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
    SetSCCImplicit<(LLVMCmp LLVMICmpULE, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
    SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
    SetSCCImplicit<(LLVMCmp LLVMICmpNE, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
    SetSCCImplicit<(LLVMCmp LLVMICmpSLT, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
    SetSCCImplicit<(LLVMCmp LLVMICmpULT, (GetNamedOperand $sdst), (GetNamedOperand $simm16))>.ret
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
  // `int_amdgcn_s_getreg` returns i32 with an immediate hwreg encoding arg;
  // `LLVMUnaryIntrinsic` calls `IRBuilder::CreateUnaryIntrinsic`, which
  // asserts on non-overloaded intrinsics that need explicit result types.
  // Use the explicit `LLVMIntrinsic` form to pass the return type.
  let Semantic = [
    (SetNamedOperand $sdst,
      (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_s_getreg,
                     [(GetNamedOperand $simm16)]<dag>))
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
  // Same caveat as S_GETREG_B32: use the explicit `LLVMIntrinsic` form so
  // the void return type and the two arguments are passed correctly.
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_setreg,
                   [(GetNamedOperand $simm16),
                    (GetNamedOperand $sdst)]<dag>)
  ];
}

//------------------------------------------------------------------------------
// S_CBRANCH_I_FORK
//------------------------------------------------------------------------------

// S_CBRANCH_I_FORK: same as S_CBRANCH_G_FORK but the comparison mask comes
// from the operand misleadingly named $sdst (it's actually an input — see
// SOPInstructions.td:1147 — `(ins SReg_64:$sdst, ...)`), and the branch
// target is a PC-relative 16-bit displacement in $simm16. We don't model
// the actual target address here: $simm16 is a `SOPPBrTarget` operand
// that lowers to MO_MachineBasicBlock at MIR time, so we use `pc + 4` as a
// conservative saved-PC placeholder. Downstream analyses still see the
// correct register-define set (the 6 SGPR quads, EXEC, M0[5:2]).
//
// Reuses CSPStackSlot<> from SOP2Sem.td.
//
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 112.
def S_CBRANCH_I_FORK_sem : InstSISemantic<S_CBRANCH_I_FORK> {
  let Semantic = !listconcat(
    [
      (DefVal $mask_in, (GetNamedOperand $sdst)),
      (DefVal $cur_pc,  (LLVMIntrinsic si_llvm_i64_ty,
                                       int_amdgcn_s_getpc, []<dag>)),
      (DefVal $exec_mask, (ImplicitUse EXEC)),

      (DefVal $mask_pass, (LLVMAnd (GetVal $exec_mask), (GetVal $mask_in))),
      (DefVal $all_ones, (LLVMSExt (ConstantTrue si_llvm_i1_ty),
                                   (LLVMIntNTy (GetWavefrontSize)))),
      (DefVal $not_mask_in, (LLVMXor (GetVal $mask_in), (GetVal $all_ones))),
      (DefVal $mask_fail, (LLVMAnd (GetVal $exec_mask),
                                   (GetVal $not_mask_in))),

      (DefVal $bc_pass, (LLVMUnaryIntrinsic int_ctpop, (GetVal $mask_pass))),
      (DefVal $bc_fail, (LLVMUnaryIntrinsic int_ctpop, (GetVal $mask_fail))),
      (DefVal $fail_smaller,
              (LLVMCmp LLVMICmpULT, (GetVal $bc_fail), (GetVal $bc_pass))),

      (DefVal $all_pass,
              (LLVMCmp LLVMICmpEQ, (GetVal $mask_pass), (GetVal $exec_mask))),
      (DefVal $all_fail,
              (LLVMCmp LLVMICmpEQ, (GetVal $mask_fail), (GetVal $exec_mask))),
      (DefVal $no_fork, (LLVMOr (GetVal $all_pass), (GetVal $all_fail))),
      (DefVal $should_push, (LLVMXor (GetVal $no_fork),
                                     (ConstantTrue si_llvm_i1_ty))),

      // Saved PC: conservative `pc + 4` for both branches of the select
      // (the actual fork-taken target is in $simm16, which we don't lift).
      (DefVal $pc_plus_4, (LLVMAdd (GetVal $cur_pc),
                                   (ConstantInt si_llvm_i64_ty, 4))),
      (DefVal $saved_pc, (GetVal $pc_plus_4)),
      (DefVal $saved_exec, (LLVMSelect (GetVal $fail_smaller),
                                       (GetVal $mask_pass),
                                       (GetVal $mask_fail))),
      (DefVal $saved_pc_128, (LLVMZExt (GetVal $saved_pc), si_llvm_i128_ty)),
      (DefVal $saved_exec_64, (LLVMZExt (GetVal $saved_exec), si_llvm_i64_ty)),
      (DefVal $saved_exec_128, (LLVMZExt (GetVal $saved_exec_64),
                                         si_llvm_i128_ty)),
      (DefVal $saved_exec_shifted, (LLVMShl (GetVal $saved_exec_128),
                                            (ConstantInt si_llvm_i128_ty, 64))),
      (DefVal $entry, (LLVMOr (GetVal $saved_pc_128),
                              (GetVal $saved_exec_shifted))),

      (DefVal $csp, (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_s_getreg,
                                   [(ConstantInt si_llvm_i32_ty, 5953)]<dag>))
    ],
    [CSPStackSlot<0, [SGPR0,  SGPR1,  SGPR2,  SGPR3 ]>.ret,
     CSPStackSlot<1, [SGPR4,  SGPR5,  SGPR6,  SGPR7 ]>.ret,
     CSPStackSlot<2, [SGPR8,  SGPR9,  SGPR10, SGPR11]>.ret,
     CSPStackSlot<3, [SGPR12, SGPR13, SGPR14, SGPR15]>.ret,
     CSPStackSlot<4, [SGPR16, SGPR17, SGPR18, SGPR19]>.ret,
     CSPStackSlot<5, [SGPR20, SGPR21, SGPR22, SGPR23]>.ret],
    [
      (DefVal $forked_exec, (LLVMSelect (GetVal $fail_smaller),
                                        (GetVal $mask_fail),
                                        (GetVal $mask_pass))),
      (DefVal $new_exec, (LLVMSelect (GetVal $no_fork),
                                     (GetVal $exec_mask),
                                     (GetVal $forked_exec))),
      (ImplicitDef EXEC, (GetVal $new_exec)),

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
//def S_CBRANCH_I_FORK_sem_draft : InstSISemantic<S_CBRANCH_I_FORK> {
//  let Semantic = [
//    // Get current PC for branch stack entry
//    (DefVal $current_pc, (LLVMIntrinsic si_llvm_i64_ty, int_amdgcn_s_getpc, []<dag>)),
//    // Store current EXEC to sdst
//    (SetNamedOperand $sdst, (ImplicitUse EXEC)),
//    // Compute pass and fail masks
//    (DefVal $pass_mask, (LLVMAnd (ImplicitUse EXEC), (GetNamedOperand $sdst))),
//    (DefVal $fail_mask, (LLVMAnd (ImplicitUse EXEC),
//                               (LLVMXor (GetNamedOperand $sdst),
//                                        (ConstantSignedInt si_llvm_i64_ty, -1)))),
//    // Get CSP from MODE register (hwmode = 5953, returns i32)
//    (DefVal $csp, (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_s_getreg,
//      [(ConstantInt si_llvm_i16_ty, 5953)]<dag>)),
//    // Read SGPR quads and bitcast to 128-bit vectors
//    (DefVal $sgpr0_3, (LLVMBitCast (ImplicitUse SGPR0_SGPR1_SGPR2_SGPR3), si_llvm_i128_ty)),
//    (DefVal $sgpr4_7, (LLVMBitCast (ImplicitUse SGPR4_SGPR5_SGPR6_SGPR7), si_llvm_i128_ty)),
//    (DefVal $sgpr8_11, (LLVMBitCast (ImplicitUse SGPR8_SGPR9_SGPR10_SGPR11), si_llvm_i128_ty)),
//    (DefVal $sgpr12_15, (LLVMBitCast (ImplicitUse SGPR12_SGPR13_SGPR14_SGPR15), si_llvm_i128_ty)),
//    (DefVal $sgpr16_19, (LLVMBitCast (ImplicitUse SGPR16_SGPR17_SGPR18_SGPR19), si_llvm_i128_ty)),
//    (DefVal $sgpr20_23, (LLVMBitCast (ImplicitUse SGPR20_SGPR21_SGPR22_SGPR23), si_llvm_i128_ty)),
//    // Create 6x128-bit vector of SGPR values
//    (DefVal $sgpr_vec, (LLVMInsertElement
//      (LLVMInsertElement
//        (LLVMInsertElement
//          (LLVMInsertElement
//            (LLVMInsertElement
//              (LLVMInsertElement (ConstantPoison si_llvm_v6i128_ty),
//                                 (GetVal $sgpr0_3), (ConstantInt si_llvm_i32_ty, 0)),
//                                 (GetVal $sgpr4_7), (ConstantInt si_llvm_i32_ty, 1)),
//                                 (GetVal $sgpr8_11), (ConstantInt si_llvm_i32_ty, 2)),
//                                 (GetVal $sgpr12_15), (ConstantInt si_llvm_i32_ty, 3)),
//                                 (GetVal $sgpr16_19), (ConstantInt si_llvm_i32_ty, 4)),
//                                 (GetVal $sgpr20_23), (ConstantInt si_llvm_i32_ty, 5))),
//    // Extract element at CSP index
//    (DefVal $csp_val, (LLVMExtractElement (GetVal $sgpr_vec), (GetVal $csp))),
//    // Branch to target (simm16 is already a BasicBlock)
//    (LLVMBranch (GetNamedOperand $simm16))
//  ];
//}
//
//def S_CBRANCH_I_FORK_gfx6_gfx7_sem : InstSISemantic<S_CBRANCH_I_FORK_gfx6_gfx7> {
//  let Semantic = [
//    (DefVal $current_pc, (LLVMIntrinsic si_llvm_i64_ty, int_amdgcn_s_getpc, []<dag>)),
//    (SetNamedOperand $sdst, (ImplicitUse EXEC)),
//    (DefVal $pass_mask, (LLVMAnd (ImplicitUse EXEC), (GetNamedOperand $sdst))),
//    (DefVal $fail_mask, (LLVMAnd (ImplicitUse EXEC),
//                               (LLVMXor (GetNamedOperand $sdst),
//                                        (ConstantSignedInt si_llvm_i64_ty, -1)))),
//    (DefVal $csp, (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_s_getreg,
//      [(ConstantInt si_llvm_i16_ty, 5953)]<dag>)),
//    (DefVal $sgpr0_3, (LLVMBitCast (ImplicitUse SGPR0_SGPR1_SGPR2_SGPR3), si_llvm_i128_ty)),
//    (DefVal $sgpr4_7, (LLVMBitCast (ImplicitUse SGPR4_SGPR5_SGPR6_SGPR7), si_llvm_i128_ty)),
//    (DefVal $sgpr8_11, (LLVMBitCast (ImplicitUse SGPR8_SGPR9_SGPR10_SGPR11), si_llvm_i128_ty)),
//    (DefVal $sgpr12_15, (LLVMBitCast (ImplicitUse SGPR12_SGPR13_SGPR14_SGPR15), si_llvm_i128_ty)),
//    (DefVal $sgpr16_19, (LLVMBitCast (ImplicitUse SGPR16_SGPR17_SGPR18_SGPR19), si_llvm_i128_ty)),
//    (DefVal $sgpr20_23, (LLVMBitCast (ImplicitUse SGPR20_SGPR21_SGPR22_SGPR23), si_llvm_i128_ty)),
//    (DefVal $sgpr_vec, (LLVMInsertElement
//      (LLVMInsertElement
//        (LLVMInsertElement
//          (LLVMInsertElement
//            (LLVMInsertElement
//              (LLVMInsertElement (ConstantPoison si_llvm_v6i128_ty),
//                                 (GetVal $sgpr0_3), (ConstantInt si_llvm_i32_ty, 0)),
//                                 (GetVal $sgpr4_7), (ConstantInt si_llvm_i32_ty, 1)),
//                                 (GetVal $sgpr8_11), (ConstantInt si_llvm_i32_ty, 2)),
//                                 (GetVal $sgpr12_15), (ConstantInt si_llvm_i32_ty, 3)),
//                                 (GetVal $sgpr16_19), (ConstantInt si_llvm_i32_ty, 4)),
//                                 (GetVal $sgpr20_23), (ConstantInt si_llvm_i32_ty, 5))),
//    (DefVal $csp_val, (LLVMExtractElement (GetVal $sgpr_vec), (GetVal $csp))),
//    (LLVMBranch (GetNamedOperand $simm16))
//  ];
//}
//
//def S_CBRANCH_I_FORK_vi_sem : InstSISemantic<S_CBRANCH_I_FORK_vi> {
//  let Semantic = [
//    (DefVal $current_pc, (LLVMIntrinsic si_llvm_i64_ty, int_amdgcn_s_getpc, []<dag>)),
//    (SetNamedOperand $sdst, (ImplicitUse EXEC)),
//    (DefVal $pass_mask, (LLVMAnd (ImplicitUse EXEC), (GetNamedOperand $sdst))),
//    (DefVal $fail_mask, (LLVMAnd (ImplicitUse EXEC),
//                               (LLVMXor (GetNamedOperand $sdst),
//                                        (ConstantSignedInt si_llvm_i64_ty, -1)))),
//    (DefVal $csp, (LLVMIntrinsic si_llvm_i32_ty, int_amdgcn_s_getreg,
//      [(ConstantInt si_llvm_i16_ty, 5953)]<dag>)),
//    (DefVal $sgpr0_3, (LLVMBitCast (ImplicitUse SGPR0_SGPR1_SGPR2_SGPR3), si_llvm_i128_ty)),
//    (DefVal $sgpr4_7, (LLVMBitCast (ImplicitUse SGPR4_SGPR5_SGPR6_SGPR7), si_llvm_i128_ty)),
//    (DefVal $sgpr8_11, (LLVMBitCast (ImplicitUse SGPR8_SGPR9_SGPR10_SGPR11), si_llvm_i128_ty)),
//    (DefVal $sgpr12_15, (LLVMBitCast (ImplicitUse SGPR12_SGPR13_SGPR14_SGPR15), si_llvm_i128_ty)),
//    (DefVal $sgpr16_19, (LLVMBitCast (ImplicitUse SGPR16_SGPR17_SGPR18_SGPR19), si_llvm_i128_ty)),
//    (DefVal $sgpr20_23, (LLVMBitCast (ImplicitUse SGPR20_SGPR21_SGPR22_SGPR23), si_llvm_i128_ty)),
//    (DefVal $sgpr_vec, (LLVMInsertElement
//      (LLVMInsertElement
//        (LLVMInsertElement
//          (LLVMInsertElement
//            (LLVMInsertElement
//              (LLVMInsertElement (ConstantPoison si_llvm_v6i128_ty),
//                                 (GetVal $sgpr0_3), (ConstantInt si_llvm_i32_ty, 0)),
//                                 (GetVal $sgpr4_7), (ConstantInt si_llvm_i32_ty, 1)),
//                                 (GetVal $sgpr8_11), (ConstantInt si_llvm_i32_ty, 2)),
//                                 (GetVal $sgpr12_15), (ConstantInt si_llvm_i32_ty, 3)),
//                                 (GetVal $sgpr16_19), (ConstantInt si_llvm_i32_ty, 4)),
//                                 (GetVal $sgpr20_23), (ConstantInt si_llvm_i32_ty, 5))),
//    (DefVal $csp_val, (LLVMExtractElement (GetVal $sgpr_vec), (GetVal $csp))),
//    (LLVMBranch (GetNamedOperand $simm16))
//  ];
//}

//==============================================================================
// S_CALL_B64 — direct subroutine call (Phase 26)
//==============================================================================
// Hardware semantics: $sdst gets the PC immediately after this instruction
// (the return address); control jumps to the target encoded in $simm16
// (a 16-bit signed offset, sign-extended and shifted left by 2, added to PC+4).
//
// IR-level model:
//   $sdst = int_amdgcn_s_getpc()   // post-PC == return address
//   call <callee>()                // direct call to resolved Function*
//
// The $simm16 operand is rewritten by CodeDiscoveryPass from the raw 16-bit
// displacement (MO_Immediate) to MO_GlobalAddress(<callee Function>) after
// the callgraph pass resolves the target. If resolution failed the operand
// stays as Imm; getOperandAsFunction returns nullptr and the call emission
// silently no-ops.
def S_CALL_B64_sem : InstSISemantic<S_CALL_B64> {
  let Semantic = [
    (SetNamedOperand $sdst,
      (LLVMIntrinsic si_llvm_i64_ty, int_amdgcn_s_getpc, []<dag>)),
    (DirectCall (GetNamedOperand $sdst), (GetNamedOperand $simm16, si_llvm_i16_ty))
  ];
}

//==============================================================================
// S_SUBVECTOR_LOOP_BEGIN / S_SUBVECTOR_LOOP_END — wave64 subvector iteration
//==============================================================================
// RDNA2/RDNA3 wave64 idiom: split a wave64 body into two wave32 passes (low
// then high lanes) for cache locality. EXEC and `$sdst` together encode the
// state machine; the ISA itself says "internally it is equivalent to an
// S_CBRANCH with extra math" — no hidden subvector state.
//
// We model the dataflow only — every `$sdst` write and EXEC mutation flows
// through a three-way `select` chain matching the ISA pseudocode. The actual
// branch to `$simm16` is not modeled; the MIR-level branch is preserved as
// the trace and the IR-level CFG is left to the underlying MBB structure
// (same approach used for S_CBRANCH_G/I_FORK).
//
// Wave32 caveat: ISA says these are "well-defined but pointless" on wave32
// because `EXEC_HI` is always 0 there. Our model assumes EXEC is i64; on
// wave32 the lift still emits the math, but the conditions trivialize so
// the optimizer collapses the chain.
//
// Sources:
//   rdna2-shader-instruction-set-architecture.pdf — section 12.2 (SOPK opcodes).
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf — equivalent.

//------------------------------------------------------------------------------
// S_SUBVECTOR_LOOP_BEGIN $sdst, $simm16:
//   if (EXEC == 0):                 jump LABEL
//   elif (EXEC_LO == 0):            $sdst = EXEC_LO       # = 0
//   else:                           $sdst = EXEC_HI ; EXEC_HI = 0
//------------------------------------------------------------------------------
def S_SUBVECTOR_LOOP_BEGIN_sem : InstSISemantic<S_SUBVECTOR_LOOP_BEGIN> {
  let Semantic = [
    (DefVal $exec, (ImplicitUse EXEC)),
    (DefVal $exec_lo64,
      (LLVMAnd (GetVal $exec),
               (ConstantInt si_llvm_i64_ty, 0xFFFFFFFF))),
    (DefVal $exec_hi64,
      (LLVMLShr (GetVal $exec),
                (ConstantInt si_llvm_i64_ty, 32))),
    (DefVal $exec_lo, (LLVMTrunc (GetVal $exec_lo64), si_llvm_i32_ty)),
    (DefVal $exec_hi, (LLVMTrunc (GetVal $exec_hi64), si_llvm_i32_ty)),

    (DefVal $exec_full_zero,
      (LLVMCmp LLVMICmpEQ, (GetVal $exec),
                            (ConstantInt si_llvm_i64_ty, 0))),
    (DefVal $exec_lo_zero,
      (LLVMCmp LLVMICmpEQ, (GetVal $exec_lo),
                            (ConstantInt si_llvm_i32_ty, 0))),

    // $sdst := full_zero ? old_sdst : (lo_zero ? EXEC_LO : EXEC_HI)
    (DefVal $sdst_old, (GetNamedOperand $sdst)),
    (DefVal $sdst_inner,
      (LLVMSelect (GetVal $exec_lo_zero),
                  (GetVal $exec_lo), (GetVal $exec_hi))),
    (SetNamedOperand $sdst,
      (LLVMSelect (GetVal $exec_full_zero),
                  (GetVal $sdst_old), (GetVal $sdst_inner))),

    // EXEC_HI := full_zero ? EXEC_HI : (lo_zero ? EXEC_HI : 0)
    // Equivalently: zero EXEC_HI only when both halves are nonzero.
    (DefVal $hi_keep,
      (LLVMOr (GetVal $exec_full_zero), (GetVal $exec_lo_zero))),
    (DefVal $new_exec_hi,
      (LLVMSelect (GetVal $hi_keep),
                  (GetVal $exec_hi),
                  (ConstantInt si_llvm_i32_ty, 0))),
    (DefVal $new_exec_hi64,
      (LLVMShl (LLVMZExt (GetVal $new_exec_hi), si_llvm_i64_ty),
               (ConstantInt si_llvm_i64_ty, 32))),
    (ImplicitDef EXEC,
      (LLVMOr (GetVal $exec_lo64), (GetVal $new_exec_hi64)))
  ];
}

//------------------------------------------------------------------------------
// S_SUBVECTOR_LOOP_END $sdst, $simm16:
//   if (EXEC_HI != 0):              EXEC_LO = $sdst        # exit from high pass
//   elif ($sdst == 0):              nop                    # done
//   else:                           EXEC_HI = $sdst ;      # start second pass
//                                   $sdst = EXEC_LO ;
//                                   EXEC_LO = 0 ;
//                                   jump LABEL
//------------------------------------------------------------------------------
def S_SUBVECTOR_LOOP_END_sem : InstSISemantic<S_SUBVECTOR_LOOP_END> {
  let Semantic = [
    (DefVal $exec, (ImplicitUse EXEC)),
    (DefVal $exec_lo64,
      (LLVMAnd (GetVal $exec),
               (ConstantInt si_llvm_i64_ty, 0xFFFFFFFF))),
    (DefVal $exec_hi64,
      (LLVMLShr (GetVal $exec),
                (ConstantInt si_llvm_i64_ty, 32))),
    (DefVal $exec_lo, (LLVMTrunc (GetVal $exec_lo64), si_llvm_i32_ty)),
    (DefVal $exec_hi, (LLVMTrunc (GetVal $exec_hi64), si_llvm_i32_ty)),

    (DefVal $exec_hi_nonzero,
      (LLVMCmp LLVMICmpNE, (GetVal $exec_hi),
                            (ConstantInt si_llvm_i32_ty, 0))),
    (DefVal $sdst_old, (GetNamedOperand $sdst)),
    (DefVal $sdst_zero,
      (LLVMCmp LLVMICmpEQ, (GetVal $sdst_old),
                            (ConstantInt si_llvm_i32_ty, 0))),

    // new EXEC_LO:
    //   exec_hi_nonzero ? $sdst : (sdst_zero ? EXEC_LO : 0)
    (DefVal $new_exec_lo_inner,
      (LLVMSelect (GetVal $sdst_zero),
                  (GetVal $exec_lo),
                  (ConstantInt si_llvm_i32_ty, 0))),
    (DefVal $new_exec_lo,
      (LLVMSelect (GetVal $exec_hi_nonzero),
                  (GetVal $sdst_old),
                  (GetVal $new_exec_lo_inner))),

    // new EXEC_HI:
    //   exec_hi_nonzero ? EXEC_HI : (sdst_zero ? EXEC_HI : $sdst)
    (DefVal $new_exec_hi_inner,
      (LLVMSelect (GetVal $sdst_zero),
                  (GetVal $exec_hi),
                  (GetVal $sdst_old))),
    (DefVal $new_exec_hi,
      (LLVMSelect (GetVal $exec_hi_nonzero),
                  (GetVal $exec_hi),
                  (GetVal $new_exec_hi_inner))),

    (DefVal $new_exec_lo64, (LLVMZExt (GetVal $new_exec_lo), si_llvm_i64_ty)),
    (DefVal $new_exec_hi64,
      (LLVMShl (LLVMZExt (GetVal $new_exec_hi), si_llvm_i64_ty),
               (ConstantInt si_llvm_i64_ty, 32))),
    (ImplicitDef EXEC,
      (LLVMOr (GetVal $new_exec_lo64), (GetVal $new_exec_hi64))),

    // new $sdst:
    //   exec_hi_nonzero ? old_sdst : (sdst_zero ? old_sdst : EXEC_LO)
    (DefVal $new_sdst_inner,
      (LLVMSelect (GetVal $sdst_zero),
                  (GetVal $sdst_old),
                  (GetVal $exec_lo))),
    (SetNamedOperand $sdst,
      (LLVMSelect (GetVal $exec_hi_nonzero),
                  (GetVal $sdst_old),
                  (GetVal $new_sdst_inner)))
  ];
}
