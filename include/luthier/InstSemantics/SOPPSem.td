//===- SOPPSem.td -------------------------------------------------*- tblgen -*-===//
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
// SOPP Instructions
// Most SOPP instructions are either control flow (branches, endpgm) or
// hardware synchronization/configuration operations mapped to intrinsics.
// Control flow instructions (S_BRANCH, S_CBRANCH_*, S_ENDPGM) are not
// translatable to LLVM IR data operations.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// S_NOP
//------------------------------------------------------------------------------

// No operation. Wait for SIMM16[2:0]+1 cycles.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 148.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 118.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 107.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 138.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 150.
//   rdna-shader-instruction-set-architecture.pdf, page 132.
//   rdna2-shader-instruction-set-architecture.pdf, page 133.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 242.
//   rdna4-instruction-set-architecture.pdf, page 276.
def S_NOP_sem : InstSISemantic<S_NOP> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_nop, [(GetNamedOperand $simm16, si_llvm_i16_ty)])
  ];
}

//------------------------------------------------------------------------------
// S_BARRIER
//------------------------------------------------------------------------------

// Synchronize waves within a workgroup.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 144.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 119.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 109.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 108.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 141.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 153.
//   rdna-shader-instruction-set-architecture.pdf, page 134.
//   rdna2-shader-instruction-set-architecture.pdf, page 134.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 254.
//   (Removed in RDNA4; replaced by S_BARRIER_SIGNAL + S_BARRIER_WAIT.)
def S_BARRIER_sem : InstSISemantic<S_BARRIER> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_barrier, []<dag>)
  ];
}

//------------------------------------------------------------------------------
// S_WAITCNT
//------------------------------------------------------------------------------

// Wait for outstanding memory operations to complete.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 152.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 120.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 109.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 108.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 141.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 153.
//   rdna-shader-instruction-set-architecture.pdf, page 133.
//   rdna2-shader-instruction-set-architecture.pdf, page 135.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 247.
//   (Removed in RDNA4; replaced by S_WAIT_KMCNT/LOADCNT/STORECNT/etc.)
def S_WAITCNT_sem : InstSISemantic<S_WAITCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}

//------------------------------------------------------------------------------
// S_SENDMSG / S_SENDMSGHALT
//------------------------------------------------------------------------------

// Send a message to fixed-function hardware.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 149.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 120.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 109.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 108.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 142.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 154.
//   rdna-shader-instruction-set-architecture.pdf, page 135.
//   rdna2-shader-instruction-set-architecture.pdf, page 135.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 254.
//   rdna4-instruction-set-architecture.pdf, page 286.
def S_SENDMSG_sem : InstSISemantic<S_SENDMSG> {
  let Semantic = [
    // `int_amdgcn_s_sendmsg` is (i32 immediate, i32 value-in-M0).
    // The instruction takes the message in $simm16 and implicitly
    // reads M0 as the data payload.
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_sendmsg,
      [(GetNamedOperand $simm16), (ImplicitUse M0)])
  ];
}

// Send a message then enter halt state.
// Sources: same pages as S_SENDMSG.
def S_SENDMSGHALT_sem : InstSISemantic<S_SENDMSGHALT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_sendmsghalt,
      [(GetNamedOperand $simm16), (ImplicitUse M0)])
  ];
}

//------------------------------------------------------------------------------
// S_SLEEP
//------------------------------------------------------------------------------

// Cause wave to sleep for (1 << SIMM16[2:0]) - 1 clocks.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 151.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 120.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 109.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 108.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 142.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 154.
//   rdna-shader-instruction-set-architecture.pdf, page 134.
//   rdna2-shader-instruction-set-architecture.pdf, page 135.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 243.
//   rdna4-instruction-set-architecture.pdf, page 277.
def S_SLEEP_sem : InstSISemantic<S_SLEEP> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_sleep, [(GetNamedOperand $simm16)])
  ];
}

//------------------------------------------------------------------------------
// S_SETHALT
//------------------------------------------------------------------------------

// Set HALT bit to value of SIMM16[0]. When set, wave idles until bit is cleared.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 150.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 120.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 109.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 108.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 141.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 153.
//   rdna-shader-instruction-set-architecture.pdf, page 133.
//   rdna2-shader-instruction-set-architecture.pdf, page 135.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 242.
//   rdna4-instruction-set-architecture.pdf, page 276.
def S_SETHALT_sem : InstSISemantic<S_SETHALT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_sethalt, [(GetNamedOperand $simm16)])
  ];
}

//------------------------------------------------------------------------------
// S_SETPRIO
//------------------------------------------------------------------------------

// Set wave priority to SIMM16[1:0].
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 151.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 120.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 109.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 108.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 142.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 154.
//   rdna-shader-instruction-set-architecture.pdf, page 134.
//   rdna2-shader-instruction-set-architecture.pdf, page 135.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 253.
//   rdna4-instruction-set-architecture.pdf, page 286.
def S_SETPRIO_sem : InstSISemantic<S_SETPRIO> {
  let Semantic = [
    // \c int_amdgcn_s_setprio is declared with \c [llvm_i16_ty]
    // (unlike s_sethalt / s_sleep / s_setprio_inc_wg which all take i32).
    // The default \c GetNamedOperand width comes back as i32, so request
    // i16 explicitly to match the intrinsic's parameter type.
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_setprio,
      [(GetNamedOperand $simm16, si_llvm_i16_ty)])
  ];
}

//------------------------------------------------------------------------------
// S_TRAP
//------------------------------------------------------------------------------

// Enter trap handler.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 151.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 120.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 110.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 109.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 142.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 154.
//   rdna-shader-instruction-set-architecture.pdf, page 134.
//   rdna2-shader-instruction-set-architecture.pdf, page 136.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 248.
//   rdna4-instruction-set-architecture.pdf, page 281.
def S_TRAP_sem : InstSISemantic<S_TRAP> {
  let Semantic = [
    /// TODO: Maybe handle trap manually, to make s_trap 3 point to HSA debug trap
    (LLVMIntrinsic int_trap, []<dag>)
  ];
}

//------------------------------------------------------------------------------
// S_ICACHE_INV
//------------------------------------------------------------------------------

// Invalidate entire instruction cache.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 147.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 120.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 110.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 109.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 143.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 155.
//   rdna-shader-instruction-set-architecture.pdf, page 134.
//   rdna2-shader-instruction-set-architecture.pdf, page 136.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 254.
//   rdna4-instruction-set-architecture.pdf, page 287.
def S_ICACHE_INV_sem : InstSISemantic<S_ICACHE_INV> {
  // No IR analog for I-cache invalidate. Modeled as a barrier.
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

//------------------------------------------------------------------------------
// S_TTRACEDATA / S_TTRACEDATA_IMM
//------------------------------------------------------------------------------

// Write M0 to thread trace data.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 151.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 121.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 110.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 143.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 155.
//   rdna-shader-instruction-set-architecture.pdf, page 136.
//   rdna2-shader-instruction-set-architecture.pdf, page 136.
//   (Not present in RDNA3, RDNA4.)
def S_TTRACEDATA_sem : InstSISemantic<S_TTRACEDATA> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_ttracedata, []<dag>)
  ];
}

// Write immediate value to thread trace data (GFX10+).
// Sources:
//   rdna-shader-instruction-set-architecture.pdf, page 136.
//   rdna2-shader-instruction-set-architecture.pdf, page 136.
//   (Not present in RDNA3, RDNA4.)
def S_TTRACEDATA_IMM_sem : InstSISemantic<S_TTRACEDATA_IMM> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_ttracedata_imm, [(GetNamedOperand $simm16)])
  ];
}

//------------------------------------------------------------------------------
// S_INCPERFLEVEL / S_DECPERFLEVEL
//------------------------------------------------------------------------------

// Increment performance counter level.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 148.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 121.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 110.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 109.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 143.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 155.
//   rdna-shader-instruction-set-architecture.pdf, page 134.
//   rdna2-shader-instruction-set-architecture.pdf, page 136.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 254.
//   rdna4-instruction-set-architecture.pdf, page 286.
def S_INCPERFLEVEL_sem : InstSISemantic<S_INCPERFLEVEL> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_incperflevel, [(GetNamedOperand $simm16)])
  ];
}

// Decrement performance counter level.
// Sources: same as S_INCPERFLEVEL.
def S_DECPERFLEVEL_sem : InstSISemantic<S_DECPERFLEVEL> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_decperflevel, [(GetNamedOperand $simm16)])
  ];
}

//------------------------------------------------------------------------------
// S_WAKEUP
//------------------------------------------------------------------------------

// Allow a sleeping wave to wake up (GFX9+).
// Sources:
//   vega-7nm-shader-instruction-set-architecture.pdf, page 119.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 108.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 107.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 139.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 151.
//   rdna-shader-instruction-set-architecture.pdf, page 132.
//   rdna2-shader-instruction-set-architecture.pdf, page 134.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 253.
//   rdna4-instruction-set-architecture.pdf, page 286.
//   (Not present in GCN3.)
def S_WAKEUP_sem : InstSISemantic<S_WAKEUP> {
  // Wake neighboring waves. HW-only side effect; no IR analog. Barrier.
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

//------------------------------------------------------------------------------
// GFX10+ SOPP Instructions
//------------------------------------------------------------------------------

// S_DENORM_MODE: Set denorm mode register.
// Sources:
//   rdna-shader-instruction-set-architecture.pdf, page 136.
//   rdna2-shader-instruction-set-architecture.pdf, page 136.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 248.
//   rdna4-instruction-set-architecture.pdf, page 282.
def S_DENORM_MODE_sem : InstSISemantic<S_DENORM_MODE> {
  // FP denormal mode register write. HW-only effect on MODE; LLVM has no
  // direct intrinsic. Modeled as barrier so the write is sequenced.
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

// S_ROUND_MODE: Set float rounding mode register.
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 248.
//   rdna4-instruction-set-architecture.pdf, page 282.
//   (Introduced in GFX11/RDNA3.)
def S_ROUND_MODE_sem : InstSISemantic<S_ROUND_MODE> {
  // FP rounding-mode register write. HW-only effect; LLVM's
  // float-control intrinsics don't map cleanly. Barrier.
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

//------------------------------------------------------------------------------
// GFX12+ Wait Count Instructions
//------------------------------------------------------------------------------

// S_WAIT_KMCNT: Wait for scalar memory operations.
// S_WAIT_LOADCNT: Wait for vector load operations.
// S_WAIT_STORECNT: Wait for vector store operations.
// S_WAIT_EXPCNT: Wait for export operations.
// S_WAIT_DSCNT: Wait for LDS/GDS operations.
// S_WAIT_SAMPLECNT: Wait for texture sample operations.
// S_WAIT_BVHCNT: Wait for BVH ray intersection operations.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 287.
//   (Introduced in GFX12, replacing S_WAITCNT.)
def S_WAIT_KMCNT_sem : InstSISemantic<S_WAIT_KMCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_kmcnt, [(GetNamedOperand $simm16, si_llvm_i16_ty)])
  ];
}
def S_WAIT_LOADCNT_sem : InstSISemantic<S_WAIT_LOADCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_loadcnt, [(GetNamedOperand $simm16, si_llvm_i16_ty)])
  ];
}
def S_WAIT_STORECNT_sem : InstSISemantic<S_WAIT_STORECNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_storecnt, [(GetNamedOperand $simm16, si_llvm_i16_ty)])
  ];
}
def S_WAIT_EXPCNT_sem : InstSISemantic<S_WAIT_EXPCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_expcnt, [(GetNamedOperand $simm16, si_llvm_i16_ty)])
  ];
}
def S_WAIT_DSCNT_sem : InstSISemantic<S_WAIT_DSCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_dscnt, [(GetNamedOperand $simm16, si_llvm_i16_ty)])
  ];
}
def S_WAIT_SAMPLECNT_sem : InstSISemantic<S_WAIT_SAMPLECNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_samplecnt, [(GetNamedOperand $simm16, si_llvm_i16_ty)])
  ];
}
def S_WAIT_BVHCNT_sem : InstSISemantic<S_WAIT_BVHCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_bvhcnt, [(GetNamedOperand $simm16, si_llvm_i16_ty)])
  ];
}

// S_BARRIER_WAIT: Wait on a barrier (GFX12+).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 282.
def S_BARRIER_WAIT_sem : InstSISemantic<S_BARRIER_WAIT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_barrier_wait, [(GetNamedOperand $simm16, si_llvm_i16_ty)])
  ];
}

// S_BARRIER_LEAVE: Leave a named barrier (GFX12+).
// Sources:
//   (Not present in RDNA4; present in CDNA3+.)
def S_BARRIER_LEAVE_sem : InstSISemantic<S_BARRIER_LEAVE> {
  let Semantic = [
    SetSCCImplicit<(LLVMIntrinsic si_llvm_i1_ty, int_amdgcn_s_barrier_leave, []<dag>)>.ret
  ];
}

//------------------------------------------------------------------------------
// Branch Instructions
//------------------------------------------------------------------------------

// S_BRANCH: Unconditional branch to target.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 112.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 117.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 132.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 144.
//   rdna-shader-instruction-set-architecture.pdf, page 131.
//   rdna2-shader-instruction-set-architecture.pdf, page 131.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 238.
//   rdna4-instruction-set-architecture.pdf, page 267.
def S_BRANCH_sem : InstSISemantic<S_BRANCH> {
  let Semantic = [
    (LLVMBranch (GetNamedOperandAsBB $simm16))
  ];
}

def S_BRANCH_pad_s_nop_sem : InstSISemantic<S_BRANCH_pad_s_nop> {
  let Semantic = [
    (LLVMBranch (GetNamedOperandAsBB $simm16))
  ];
}

// S_CBRANCH_SCC0: Branch if SCC == 0.
// Sources: same as S_BRANCH.
def S_CBRANCH_SCC0_sem : InstSISemantic<S_CBRANCH_SCC0> {
  let Semantic = [
    (LLVMCondBranch GetSCCImplicit<>.ret,
                   (GetNamedOperandAsBB $simm16),
                   (GetNextBB))
  ];
}

def S_CBRANCH_SCC0_pad_s_nop_sem : InstSISemantic<S_CBRANCH_SCC0_pad_s_nop> {
  let Semantic = [
    (LLVMCondBranch GetSCCImplicit<>.ret,
                   (GetNamedOperandAsBB $simm16),
                   (GetNextBB))
  ];
}

// S_CBRANCH_SCC1: Branch if SCC == 1.
// Sources: same as S_BRANCH.
def S_CBRANCH_SCC1_sem : InstSISemantic<S_CBRANCH_SCC1> {
  let Semantic = [
    (LLVMCondBranch GetSCCImplicit<>.ret,
                   (GetNamedOperandAsBB $simm16),
                   (GetNextBB))
  ];
}

def S_CBRANCH_SCC1_pad_s_nop_sem : InstSISemantic<S_CBRANCH_SCC1_pad_s_nop> {
  let Semantic = [
    (LLVMCondBranch GetSCCImplicit<>.ret,
                   (GetNamedOperandAsBB $simm16),
                   (GetNextBB))
  ];
}

// S_CBRANCH_EXECZ: Branch if EXEC == 0.
// Sources: same as S_BRANCH.
def S_CBRANCH_EXECZ_sem : InstSISemantic<S_CBRANCH_EXECZ> {
  let Semantic = [
    (LLVMCondBranch (LLVMCmp LLVMICmpEQ, (ImplicitUse GetExecReg),
                                     (ConstantInt (LLVMIntNTy (GetWavefrontSize)), 0)),
                   (GetNextBB),
                   (GetNamedOperandAsBB $simm16))
  ];
}

def S_CBRANCH_EXECZ_pad_s_nop_sem : InstSISemantic<S_CBRANCH_EXECZ_pad_s_nop> {
  let Semantic = [
    (LLVMCondBranch (LLVMCmp LLVMICmpEQ, (ImplicitUse GetExecReg),
                                     (ConstantInt (LLVMIntNTy (GetWavefrontSize)), 0)),
                   (GetNextBB),
                   (GetNamedOperandAsBB $simm16))
  ];
}

// S_CBRANCH_EXECNZ: Branch if EXEC != 0.
// Sources: same as S_BRANCH.
def S_CBRANCH_EXECNZ_sem : InstSISemantic<S_CBRANCH_EXECNZ> {
  let Semantic = [
    (LLVMCondBranch (LLVMCmp LLVMICmpNE, (ImplicitUse GetExecReg),
                                     (ConstantInt (LLVMIntNTy (GetWavefrontSize)), 0)),
                   (GetNamedOperandAsBB $simm16),
                   (GetNextBB))
  ];
}

def S_CBRANCH_EXECNZ_pad_s_nop_sem : InstSISemantic<S_CBRANCH_EXECNZ_pad_s_nop> {
  let Semantic = [
    (LLVMCondBranch (LLVMCmp LLVMICmpNE, (ImplicitUse GetExecReg),
                                     (ConstantInt (LLVMIntNTy (GetWavefrontSize)), 0)),
                   (GetNamedOperandAsBB $simm16),
                   (GetNextBB))
  ];
}

// S_CBRANCH_VCCZ: Branch if VCC == 0.
// Sources: same as S_BRANCH.
def S_CBRANCH_VCCZ_sem : InstSISemantic<S_CBRANCH_VCCZ> {
  let Semantic = [
    (LLVMCondBranch (LLVMCmp LLVMICmpEQ, (ImplicitUse GetVCCReg),
                                     (ConstantInt (LLVMIntNTy (GetWavefrontSize)), 0)),
                   (GetNextBB),
                   (GetNamedOperandAsBB $simm16))
  ];
}

def S_CBRANCH_VCCZ_pad_s_nop_sem : InstSISemantic<S_CBRANCH_VCCZ_pad_s_nop> {
  let Semantic = [
    (LLVMCondBranch (LLVMCmp LLVMICmpEQ, (ImplicitUse GetVCCReg),
                                     (ConstantInt (LLVMIntNTy (GetWavefrontSize)), 0)),
                   (GetNextBB),
                   (GetNamedOperandAsBB $simm16))
  ];
}

// S_CBRANCH_VCCNZ: Branch if VCC != 0.
// Sources: same as S_BRANCH.
def S_CBRANCH_VCCNZ_sem : InstSISemantic<S_CBRANCH_VCCNZ> {
  let Semantic = [
    (LLVMCondBranch (LLVMCmp LLVMICmpNE, (ImplicitUse GetVCCReg),
                                     (ConstantInt (LLVMIntNTy (GetWavefrontSize)), 0)),
                   (GetNamedOperandAsBB $simm16),
                   (GetNextBB))
  ];
}

def S_CBRANCH_VCCNZ_pad_s_nop_sem : InstSISemantic<S_CBRANCH_VCCNZ_pad_s_nop> {
  let Semantic = [
    (LLVMCondBranch (LLVMCmp LLVMICmpNE, (ImplicitUse GetVCCReg),
                                     (ConstantInt (LLVMIntNTy (GetWavefrontSize)), 0)),
                   (GetNamedOperandAsBB $simm16),
                   (GetNextBB))
  ];
}

// S_CBRANCH_CDBGSYS: Conditional branch for debug (HSA system debug).
// Sources: same as S_BRANCH.
def S_CBRANCH_CDBGSYS_sem : InstSISemantic<S_CBRANCH_CDBGSYS> {
  let Semantic = [
    (LLVMBranch (GetNamedOperandAsBB $simm16))
  ];
}

def S_CBRANCH_CDBGSYS_pad_s_nop_sem : InstSISemantic<S_CBRANCH_CDBGSYS_pad_s_nop> {
  let Semantic = [
    (LLVMBranch (GetNamedOperandAsBB $simm16))
  ];
}

def S_CBRANCH_CDBGSYS_AND_USER_sem : InstSISemantic<S_CBRANCH_CDBGSYS_AND_USER> {
  let Semantic = [
    (LLVMBranch (GetNamedOperandAsBB $simm16))
  ];
}

def S_CBRANCH_CDBGSYS_AND_USER_pad_s_nop_sem : InstSISemantic<S_CBRANCH_CDBGSYS_AND_USER_pad_s_nop> {
  let Semantic = [
    (LLVMBranch (GetNamedOperandAsBB $simm16))
  ];
}

def S_CBRANCH_CDBGSYS_OR_USER_sem : InstSISemantic<S_CBRANCH_CDBGSYS_OR_USER> {
  let Semantic = [
    (LLVMBranch (GetNamedOperandAsBB $simm16))
  ];
}

def S_CBRANCH_CDBGSYS_OR_USER_pad_s_nop_sem : InstSISemantic<S_CBRANCH_CDBGSYS_OR_USER_pad_s_nop> {
  let Semantic = [
    (LLVMBranch (GetNamedOperandAsBB $simm16))
  ];
}

def S_CBRANCH_CDBGUSER_sem : InstSISemantic<S_CBRANCH_CDBGUSER> {
  let Semantic = [
    (LLVMBranch (GetNamedOperandAsBB $simm16))
  ];
}

def S_CBRANCH_CDBGUSER_pad_s_nop_sem : InstSISemantic<S_CBRANCH_CDBGUSER_pad_s_nop> {
  let Semantic = [
    (LLVMBranch (GetNamedOperandAsBB $simm16))
  ];
}

//------------------------------------------------------------------------------
// End Program Instructions
//------------------------------------------------------------------------------

// S_ENDPGM: End program.
// Sources:
//   gcn3-instruction-set-architecture.pdf, page 113.
//   vega-7nm-shader-instruction-set-architecture.pdf, page 117.
//   instinct-mi100-cdna1-shader-instruction-set-architecture.pdf, page 107.
//   instinct-mi200-cdna2-instruction-set-architecture.pdf, page 106.
//   amd-instinct-mi300-cdna3-instruction-set-architecture.pdf, page 132.
//   amd-instinct-cdna4-instruction-set-architecture.pdf, page 144.
//   rdna-shader-instruction-set-architecture.pdf, page 131.
//   rdna2-shader-instruction-set-architecture.pdf, page 131.
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 238.
//   rdna4-instruction-set-architecture.pdf, page 268.
def S_ENDPGM_sem : InstSISemantic<S_ENDPGM> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_endpgm, []<dag>),
    (LLVMUnreachable)
  ];
}

def S_ENDPGM_ORDERED_PS_DONE_sem : InstSISemantic<S_ENDPGM_ORDERED_PS_DONE> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_endpgm, []<dag>),
    (LLVMUnreachable)
  ];
}

def S_ENDPGM_SAVED_sem : InstSISemantic<S_ENDPGM_SAVED> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_endpgm, []<dag>),
    (LLVMUnreachable)
  ];
}

//==============================================================================
// SOPP Wait Variants (Phase 5)
//==============================================================================
// Split wait-counter ops introduced in GFX10+. All map to the generic
// `int_amdgcn_s_waitcnt` intrinsic with the immediate field (loses split
// distinction but preserves memory-fence semantics). Soft variants are
// scheduler hints with the same observable IR effect.

// GFX10+: waits with sdst input + simm16 mask
foreach op = ["S_WAITCNT_VMCNT", "S_WAITCNT_LGKMCNT", "S_WAITCNT_EXPCNT",
              "S_WAITCNT_VSCNT", "S_WAITCNT_VSCNT_soft"] in {
  def !cast<InstSI>(op)#_sem : InstSISemantic<!cast<InstSI>(op)> {
    let Semantic = [
      (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt,
                     [(GetNamedOperand $simm16, si_llvm_i32_ty)])
    ];
  }
}

// GFX10+ soft generic waitcnt + waitcnt_depctr
foreach op = ["S_WAITCNT_soft", "S_WAITCNT_DEPCTR"] in {
  def !cast<InstSI>(op)#_sem : InstSISemantic<!cast<InstSI>(op)> {
    let Semantic = [
      (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt,
                     [(GetNamedOperand $simm16, si_llvm_i32_ty)])
    ];
  }
}

// RDNA4+: split per-event waits. Each maps to its own intrinsic.
def S_WAIT_LOADCNT_soft_sem : InstSISemantic<S_WAIT_LOADCNT_soft> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_loadcnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}
def S_WAIT_STORECNT_soft_sem : InstSISemantic<S_WAIT_STORECNT_soft> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_storecnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}
def S_WAIT_SAMPLECNT_soft_sem : InstSISemantic<S_WAIT_SAMPLECNT_soft> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_samplecnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}
def S_WAIT_BVHCNT_soft_sem : InstSISemantic<S_WAIT_BVHCNT_soft> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_bvhcnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}
def S_WAIT_DSCNT_soft_sem : InstSISemantic<S_WAIT_DSCNT_soft> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_dscnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}
def S_WAIT_KMCNT_soft_sem : InstSISemantic<S_WAIT_KMCNT_soft> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_kmcnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}

// Combined waits: wait on both event types. Emit two intrinsic calls.
def S_WAIT_LOADCNT_DSCNT_sem : InstSISemantic<S_WAIT_LOADCNT_DSCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_loadcnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)]),
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_dscnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}
def S_WAIT_STORECNT_DSCNT_sem : InstSISemantic<S_WAIT_STORECNT_DSCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_storecnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)]),
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_dscnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}

// Async/tensor/xcnt/event/idle — emit a generic-waitcnt-style intrinsic call.
def S_WAIT_ASYNCCNT_sem : InstSISemantic<S_WAIT_ASYNCCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_asynccnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}
def S_WAIT_TENSORCNT_sem : InstSISemantic<S_WAIT_TENSORCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_wait_tensorcnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}

// S_WAIT_XCNT / S_WAIT_XCNT_soft, S_WAIT_EVENT, S_WAIT_IDLE — no dedicated
// intrinsic; fall back to a generic s_waitcnt with the immediate (or 0 for
// S_WAIT_IDLE which has no operand). These are observable as memory fences.
def S_WAIT_XCNT_sem : InstSISemantic<S_WAIT_XCNT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}
def S_WAIT_XCNT_soft_sem : InstSISemantic<S_WAIT_XCNT_soft> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}
def S_WAIT_EVENT_sem : InstSISemantic<S_WAIT_EVENT> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt,
                   [(GetNamedOperand $simm16, si_llvm_i32_ty)])
  ];
}
def S_WAIT_IDLE_sem : InstSISemantic<S_WAIT_IDLE> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt,
                   [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

//==============================================================================
// EXP — Export instruction (Phase 14)
//==============================================================================
// Export src0..src3 (selected by $en mask) to the export target $tgt.
// The $done bit signals last-export; $vm bit is the validmask bit.
// Maps to int_amdgcn_exp (regular) or int_amdgcn_exp_row (gfx11+ row export).

// EXP: regular export, done=0, vm=0 (the encoding fields are absorbed below)
def EXP_sem : InstSISemantic<EXP> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_exp,
      [(GetNamedOperand $tgt, si_llvm_i32_ty),
       (GetNamedOperand $en, si_llvm_i32_ty),
       (GetNamedOperand $src0, si_llvm_float_ty),
       (GetNamedOperand $src1, si_llvm_float_ty),
       (GetNamedOperand $src2, si_llvm_float_ty),
       (GetNamedOperand $src3, si_llvm_float_ty),
       (ConstantInt si_llvm_i1_ty, 0),
       (GetNamedOperand $vm, si_llvm_i1_ty)])
  ];
}

// EXP_DONE: same intrinsic with done=1
def EXP_DONE_sem : InstSISemantic<EXP_DONE> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_exp,
      [(GetNamedOperand $tgt, si_llvm_i32_ty),
       (GetNamedOperand $en, si_llvm_i32_ty),
       (GetNamedOperand $src0, si_llvm_float_ty),
       (GetNamedOperand $src1, si_llvm_float_ty),
       (GetNamedOperand $src2, si_llvm_float_ty),
       (GetNamedOperand $src3, si_llvm_float_ty),
       (ConstantInt si_llvm_i1_ty, 1),
       (GetNamedOperand $vm, si_llvm_i1_ty)])
  ];
}

// EXP_ROW: gfx11+ row export; maps to int_amdgcn_exp_row with done=0
def EXP_ROW_sem : InstSISemantic<EXP_ROW> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_exp_row,
      [(GetNamedOperand $tgt, si_llvm_i32_ty),
       (GetNamedOperand $en, si_llvm_i32_ty),
       (GetNamedOperand $src0, si_llvm_float_ty),
       (GetNamedOperand $src1, si_llvm_float_ty),
       (GetNamedOperand $src2, si_llvm_float_ty),
       (GetNamedOperand $src3, si_llvm_float_ty),
       (ConstantInt si_llvm_i1_ty, 0),
       (GetNamedOperand $vm, si_llvm_i32_ty)])
  ];
}

// EXP_ROW_DONE: int_amdgcn_exp_row with done=1
def EXP_ROW_DONE_sem : InstSISemantic<EXP_ROW_DONE> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_exp_row,
      [(GetNamedOperand $tgt, si_llvm_i32_ty),
       (GetNamedOperand $en, si_llvm_i32_ty),
       (GetNamedOperand $src0, si_llvm_float_ty),
       (GetNamedOperand $src1, si_llvm_float_ty),
       (GetNamedOperand $src2, si_llvm_float_ty),
       (GetNamedOperand $src3, si_llvm_float_ty),
       (ConstantInt si_llvm_i1_ty, 1),
       (GetNamedOperand $vm, si_llvm_i32_ty)])
  ];
}

//==============================================================================
// S_* system / scheduler / mode-config ops (Phase 14) — fence placeholders
//==============================================================================

// S_ALLOC_VGPR (gfx12) — dynamic VGPR allocation. Per LLVM `Defs = [SCC]`:
// SCC indicates allocation success. Conservatively model as a barrier
// + SCC=0 (always-succeed approximation; HW resource state has no IR analog).
def S_ALLOC_VGPR_sem : InstSISemantic<S_ALLOC_VGPR> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)]),
    SetSCCImplicit<(ConstantInt si_llvm_i1_ty, 0)>.ret
  ];
}

def S_CLAUSE_sem : InstSISemantic<S_CLAUSE> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_CODE_END_sem : InstSISemantic<S_CODE_END> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_DELAY_ALU_sem : InstSISemantic<S_DELAY_ALU> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_GETREG_B32_const_sem : InstSISemantic<S_GETREG_B32_const> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_INST_PREFETCH_sem : InstSISemantic<S_INST_PREFETCH> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_MONITOR_SLEEP_sem : InstSISemantic<S_MONITOR_SLEEP> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_SETKILL_sem : InstSISemantic<S_SETKILL> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_SETPRIO_INC_WG_sem : InstSISemantic<S_SETPRIO_INC_WG> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_SETREG_B32_mode_sem : InstSISemantic<S_SETREG_B32_mode> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_SETREG_IMM32_B32_sem : InstSISemantic<S_SETREG_IMM32_B32> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_SETREG_IMM32_B32_mode_sem : InstSISemantic<S_SETREG_IMM32_B32_mode> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_SETVSKIP_sem : InstSISemantic<S_SETVSKIP> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_SET_VGPR_MSB_sem : InstSISemantic<S_SET_VGPR_MSB> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

def S_VERSION_sem : InstSISemantic<S_VERSION> {
  let Semantic = [
    (LLVMIntrinsic si_llvm_void_ty, int_amdgcn_s_waitcnt, [(ConstantInt si_llvm_i32_ty, 0)])
  ];
}

//------------------------------------------------------------------------------
// S_SET_GPR_IDX_OFF (GFX9) — disable VGPR indirect indexing.
//   MODE.GPR_IDX_EN := 0     // bit 27
// Other MODE bits are preserved.
//------------------------------------------------------------------------------
def S_SET_GPR_IDX_OFF_sem : InstSISemantic<S_SET_GPR_IDX_OFF> {
  let Semantic = [
    (ImplicitDef MODE,
      (LLVMAnd (ImplicitUse MODE),
               (ConstantInt si_llvm_i32_ty, 0xF7FFFFFF)))  // ~(1 << 27)
  ];
}

//------------------------------------------------------------------------------
// S_SET_GPR_IDX_MODE $simm16 (GFX9) — set the per-operand enable mask
// without touching MODE.GPR_IDX_EN.
//   M0[14:11] := $simm16 & 0xF
// `$simm16` is a `GPRIdxMode` immediate operand.
//------------------------------------------------------------------------------
def S_SET_GPR_IDX_MODE_sem : InstSISemantic<S_SET_GPR_IDX_MODE> {
  let Semantic = [
    (DefVal $m0_cleared,
      (LLVMAnd (ImplicitUse M0),
               (ConstantInt si_llvm_i32_ty, 0xFFFF87FF))),  // ~(0xF << 11)
    (DefVal $mask_shifted,
      (LLVMShl (LLVMAnd (GetNamedOperand $simm16),
                        (ConstantInt si_llvm_i32_ty, 0xF)),
               (ConstantInt si_llvm_i32_ty, 11))),
    (ImplicitDef M0, (LLVMOr (GetVal $m0_cleared), (GetVal $mask_shifted)))
  ];
}