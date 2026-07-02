; Exercises the SVASpillCallback path in LuthierRegScavenger.
;
; Custom target stub (_target_stub_branch_relax.s.txt) defines s4..s101
; at function entry and uses them again after a forward branch. With
; --amdgpu-s-branch-bits=4 the branch is over-range and
; LuthierBranchRelaxation::fixupConditionalBranch runs. Every SReg_64
; pair in the allocation order has at least one half live across the
; branch insertion point, so the LuthierRegScavenger's
; findSurvivorBackwards loop cannot return Reg != 0 with SpillBefore =
; end() — it falls through to the spill path. Stock LLVM would then
; emit a FrameIndex-based spill, but our patcher installed an
; SVASpillCallback, so V_WRITELANE_B32 + V_READLANE_B32 against the
; SVA storage VGPR fire instead.
;
; This is the validation test deferred from task #30. It pins down
; that the entire chain — LuthierBranchRelaxation calling
; emitLuthierLongBranch calling LuthierRegScavenger.spill calling
; SVASpillCallback — actually fires when the natural scavenger fails.

; RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %S/_target_stub_branch_relax.s.txt -o %t.o && \
; RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
; RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
; RUN:    -amdgpu-s-branch-bits=4 \
; RUN:    -load-pass-plugin=%luthier_tool_code_gen_plugin \
; RUN:    %luthier_mock_injection_plugin \
; RUN:    -passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-apply-instrumentation,luthier-asm-printer \
; RUN:    -code-object-paths=%t \
; RUN:    -initial-entrypoint=0:stub_kernel.kd \
; RUN:    -initial-execution-point=0:stub_kernel.kd \
; RUN:    -imodule-path=%s \
; RUN:    -imodule-ir-passes=luthier-mock-inject-at-mbb-entry \
; RUN:    -o /dev/null > %t.asm 2>&1 && \
; RUN: FileCheck %s < %t.asm

; Sanity: AsmPrinter ran (header marker).
; CHECK: .amdgcn_target

; Long-jump trampoline emitted by emitLuthierLongBranch (lower-cased
; AMDGPU asm mnemonics).
; CHECK: s_getpc_b64
; CHECK: s_setpc_b64

; SVA-lane spill+reload from the callback. Match loosely — the lane
; numbers depend on findLowestFreeLanes, and the SGPR pair the
; scavenger picks isn't pinned. The presence of both mnemonics in the
; emitted assembly proves the callback fired.
; CHECK: v_writelane_b32
; CHECK: v_readlane_b32

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Minimal payload that touches the SVA so the patcher sets up an SVA
; storage register and an SVA spec — required for ReservedForSVA + the
; SVASpillCallback to be meaningful.
define void @luthier.payload.tiny() #0 {
  %sva = call i32 @"luthier::readSVA.i32"(i8 4)
  ret void
}

declare i32 @"luthier::readSVA.i32"(i8) #1

attributes #0 = { naked noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readSVA" "target-cpu"="gfx908" }
