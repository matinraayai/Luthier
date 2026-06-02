; threadIdx work-item-id entry capture on gfx1200 (packed-TID): exercises
; TargetModulePatcherPass::emitCaptureWorkitemIdLane0 through the full default
; instrumentation pipeline + AsmPrinter.
;
; The use-site recompute (readSVA(WORKITEM_ID_PACKED_LANE0) + mbcnt + blockDim →
; urem/udiv) is produced by SubstituteAMDGCNIntrinsicsPass and verified at the IR
; level by ir_compilation/lower-hip-device-intrinsics.ll; that substitution runs
; during IModule embedding (CreateAndEmbedIModulePass) in production and is not
; reachable from the mock-inject harness. This test instead pre-substitutes the
; one piece the harness cannot otherwise reach — a direct read of the
; WORKITEM_ID_PACKED_LANE0 SVA (arg 13) — to drive the entry capture and assert
; its ordering. The full recompute + the work-item linearization formula are
; validated on hardware (see the blockIdx/threadIdx memory note).
;
; gfx1200 sets FeaturePackedTID, so the capture takes the packed path: it reads
; the wave's lane-0 packed work-item id from v0 (v_readlane v0, 0) and writes it
; into the WORKITEM_ID_PACKED_LANE0 SVA lane. Critically, that read must happen
; before the stack-pointer spill overwrites v0's lane 0 (the SVA storage register
; aliases v0 on a kernel that does not itself use the work-item id).

; RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj %S/_target_stub_blockidx_gfx1200.s.txt -o %t.o && \
; RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
; RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 \
; RUN:    %luthier_tool_code_gen_plugin \
; RUN:    %luthier_mock_injection_plugin \
; RUN:    -passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-apply-instrumentation,luthier-asm-printer \
; RUN:    -code-object-paths=%t \
; RUN:    -initial-entrypoint=0:stub_kernel.kd \
; RUN:    -initial-execution-point=0:stub_kernel.kd \
; RUN:    -imodule-path=%s \
; RUN:    -luthier-mock-hook-name=readWorkitemPacked \
; RUN:    -imodule-ir-passes=luthier-mock-inject-at-function-entry,always-inline,luthier-process-intrinsics-at-ir-level \
; RUN:    -o /dev/null > %t.asm 2>&1 && \
; RUN: FileCheck %s < %t.asm

; The capture reads v0 lane 0 and writes it into the WORKITEM_ID_PACKED_LANE0 SVA
; lane, BEFORE the stack-pointer spill writes v0's lane 0 (sequential CHECKs
; enforce that order).
; CHECK: stub_kernel:
; CHECK: v_readlane_b32 [[ACC:s[0-9]+]], v0, 0
; The work-item lane is 3 (past the SGPR0/SGPR1/StackPointerStore frame lanes
; 0/1/2), so it is not aliased/clobbered by the lane-2 stack-pointer store.
; CHECK: v_writelane_b32 v0, [[ACC]], 3
; CHECK: v_writelane_b32 v0, s{{[0-9]+}}, 0

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@sink = addrspace(1) global i32 0

define internal void @readWorkitemPacked() #0 {
entry:
  %packed = call i32 @"luthier::readSVA.i32"(i8 13) #1
  store i32 %packed, ptr addrspace(1) @sink, align 4
  ret void
}

declare i32 @"luthier::readSVA.i32"(i8) #1

attributes #0 = { alwaysinline "luthier.function.hook" "target-cpu"="gfx1200" "target-features"="+wavefrontsize32" }
attributes #1 = { "luthier.intrinsic"="luthier::readSVA" "target-cpu"="gfx1200" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
