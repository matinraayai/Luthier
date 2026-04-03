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
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_nop, [(GetNamedOperand $simm16)])
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
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_barrier, []<dag>)
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
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_waitcnt, [(GetNamedOperand $simm16)])
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
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_sendmsg, [(GetNamedOperand $simm16)])
  ];
}

// Send a message then enter halt state.
// Sources: same pages as S_SENDMSG.
def S_SENDMSGHALT_sem : InstSISemantic<S_SENDMSGHALT> {
  let Semantic = [
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_sendmsghalt, [(GetNamedOperand $simm16)])
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
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_sleep, [(GetNamedOperand $simm16)])
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
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_sethalt, [(GetNamedOperand $simm16)])
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
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_setprio, [(GetNamedOperand $simm16)])
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
    (LLVMUnaryIntrinsic int_trap)
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
  /// TODO: Define a Luthier intrinsic for this
  let Semantic = [
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
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_ttracedata, []<dag>)
  ];
}

// Write immediate value to thread trace data (GFX10+).
// Sources:
//   rdna-shader-instruction-set-architecture.pdf, page 136.
//   rdna2-shader-instruction-set-architecture.pdf, page 136.
//   (Not present in RDNA3, RDNA4.)
def S_TTRACEDATA_IMM_sem : InstSISemantic<S_TTRACEDATA_IMM> {
  let Semantic = [
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_ttracedata_imm, [(GetNamedOperand $simm16)])
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
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_incperflevel, [(GetNamedOperand $simm16)])
  ];
}

// Decrement performance counter level.
// Sources: same as S_INCPERFLEVEL.
def S_DECPERFLEVEL_sem : InstSISemantic<S_DECPERFLEVEL> {
  let Semantic = [
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_decperflevel, [(GetNamedOperand $simm16)])
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
  /// TODO: Model with Luthier intrinsic
  let Semantic = [];
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
  /// TODO: Model with Luthier intrinsic
  let Semantic = [];
}

// S_ROUND_MODE: Set float rounding mode register.
// Sources:
//   rdna3-shader-instruction-set-architecture-feb-2023_0.pdf, page 248.
//   rdna4-instruction-set-architecture.pdf, page 282.
//   (Introduced in GFX11/RDNA3.)
def S_ROUND_MODE_sem : InstSISemantic<S_ROUND_MODE> {
  let Semantic = [
    /// TODO use LLVM's floating point control instructions
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
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_wait_kmcnt, [(GetNamedOperand $simm16)])
  ];
}
def S_WAIT_LOADCNT_sem : InstSISemantic<S_WAIT_LOADCNT> {
  let Semantic = [
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_wait_loadcnt, [(GetNamedOperand $simm16)])
  ];
}
def S_WAIT_STORECNT_sem : InstSISemantic<S_WAIT_STORECNT> {
  let Semantic = [
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_wait_storecnt, [(GetNamedOperand $simm16)])
  ];
}
def S_WAIT_EXPCNT_sem : InstSISemantic<S_WAIT_EXPCNT> {
  let Semantic = [
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_wait_expcnt, [(GetNamedOperand $simm16)])
  ];
}
def S_WAIT_DSCNT_sem : InstSISemantic<S_WAIT_DSCNT> {
  let Semantic = [
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_wait_dscnt, [(GetNamedOperand $simm16)])
  ];
}
def S_WAIT_SAMPLECNT_sem : InstSISemantic<S_WAIT_SAMPLECNT> {
  let Semantic = [
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_wait_samplecnt, [(GetNamedOperand $simm16)])
  ];
}
def S_WAIT_BVHCNT_sem : InstSISemantic<S_WAIT_BVHCNT> {
  let Semantic = [
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_wait_bvhcnt, [(GetNamedOperand $simm16)])
  ];
}

// S_WAIT_IDLE: Wait for all outstanding operations.
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 281.
def S_WAIT_IDLE_sem : InstSISemantic<S_WAIT_IDLE> {
  let Semantic = [
    /// TODO: possibly model with Luthier intrinsic
  ];
}

// S_BARRIER_WAIT: Wait on a barrier (GFX12+).
// Sources:
//   rdna4-instruction-set-architecture.pdf, page 282.
def S_BARRIER_WAIT_sem : InstSISemantic<S_BARRIER_WAIT> {
  let Semantic = [
    (LLVMIntrinsic llvm_void_ty, int_amdgcn_s_barrier_wait, [(GetNamedOperand $simm16)])
  ];
}

// S_BARRIER_LEAVE: Leave a named barrier (GFX12+).
// Sources:
//   (Not present in RDNA4; present in CDNA3+.)
def S_BARRIER_LEAVE_sem : InstSISemantic<S_BARRIER_LEAVE> {
  let Semantic = [
    (ImplicitDef SCC, (LLVMIntrinsic llvm_i1_ty, int_amdgcn_s_barrier_leave, []<dag>))
  ];
}
