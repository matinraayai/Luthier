; Smoke test of TargetModulePatcherPass's branch-relaxation step.
;
; Sets the AMDGPU backend's hidden debug knob `--amdgpu-s-branch-bits=4`
; (defined in SIInstrInfo.cpp) to shrink the s_branch range to ±7 dwords.
; Verifies that TargetModulePatcherPass's invocation of LLVM's stock
; BranchRelaxationPass runs cleanly: the pipeline does not hit the
; "branches remain over-range after BranchRelaxationPass" hard error
; that our post-relax detector would otherwise raise.
;
; This is the gfx908 wave64 path. The stub kernel for this test has no
; branches of its own, so BranchRelaxationPass has nothing to actually
; relax — but it must still run without crashing or leaving leftover
; over-range branches behind. Exercising the actual long-jump emission
; (s_getpc_b64 + s_add_u32 + s_addc_u32 + s_setpc_b64) needs a target
; stub with real branches and is left to a follow-up test.
;
; CodeGenerator::printAssemblyFile invokes TM.addAsmPrinter directly with
; no intervening machine-pass chain, so without our patcher running stock
; BranchRelaxationPass over the target MMI, far-jumps would silently emit
; with truncated displacements. This test pins down that the relaxer
; gets called.

; RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %intrinsic_mir_lowering_target_stub -o %t.o && \
; RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
; RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
; RUN:    -amdgpu-s-branch-bits=4 \
; RUN:    -load-pass-plugin=%luthier_tool_code_gen_plugin \
; RUN:    -passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-apply-instrumentation \
; RUN:    -code-object-paths=%t \
; RUN:    -initial-entrypoint=0:stub_kernel.kd \
; RUN:    -initial-execution-point=0:stub_kernel.kd \
; RUN:    -imodule-path=%s \
; RUN:    -target-module-output=%t.tgt.mir \
; RUN:    -imodule-ir-passes=luthier-process-intrinsics-at-ir-level \
; RUN:    '-imodule-mir-passes=isel,mir-lowering,injected-payload-accessed-regs,imodule-ip-pred-liveness,payload-preserve-live-regs,lr-sv-storage-load-locs,injected-payload-pei,target-module-patcher' \
; RUN:    -o /dev/null && \
; RUN: FileCheck %s < %t.tgt.mir

; Sanity: the target module dump fired (i.e., the patcher reached the
; end of runOnModule without early-returning on a hard error).
; CHECK: Luthier target-module-output

; The post-relax detector's hard-error message must NOT appear anywhere
; in the dump — its presence would indicate a relaxer failure.
; CHECK-NOT: remain over-range after BranchRelaxationPass

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @luthier.payload.tiny_sva() #0 {
  ; Minimal payload — even one extra instruction is enough to push the
  ; stub kernel's existing branches outside the ±7-dword window when
  ; --amdgpu-s-branch-bits=4 is in effect.
  %sva = call i32 @"luthier::readSVA.i32"(i8 4)
  ret void
}

declare i32 @"luthier::readSVA.i32"(i8) #1

attributes #0 = { naked noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readSVA" "target-cpu"="gfx908" }
