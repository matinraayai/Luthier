; blockIdx (workgroup id) on a NON-architected-SGPRs target (gfx908): end-to-end
; through the full default instrumentation pipeline + AsmPrinter.
;
; This file doubles as the IModule (-imodule-path=%s): the @readBlockIdx hook
; reads luthier::workgroupIdX/Y/Z(), which declare ReadSVAs{WORKGROUP_ID_X/Y/Z}.
; The hook is mock-injected at the entry of the gfx908 stub kernel; the SVA
; requests make StateValueArraySpecs allocate the WORKGROUP_ID_* lanes.
;
; gfx908 does not set FeatureArchitectedSGPRs, so the workgroup IDs are preloaded
; into system SGPRs (enabled in the stub's kernel descriptor). The generic
; entry-prologue spill loop in TargetModulePatcherPass therefore saves each
; preloaded workgroup-id SGPR into its SVA lane with v_writelane, and the payload
; reads them back via v_readlane (lowered from the workgroupId MIR processor's
; SVA read) — so blockIdx is valid at the arbitrary injection point even though
; the application may have clobbered the preloaded SGPRs.

; RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %S/_target_stub_blockidx_gfx908.s.txt -o %t.o && \
; RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
; RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
; RUN:    %luthier_tool_code_gen_plugin \
; RUN:    %luthier_mock_injection_plugin \
; RUN:    -passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-apply-instrumentation,luthier-asm-printer \
; RUN:    -code-object-paths=%t \
; RUN:    -initial-entrypoint=0:stub_kernel.kd \
; RUN:    -initial-execution-point=0:stub_kernel.kd \
; RUN:    -imodule-path=%s \
; RUN:    -luthier-mock-hook-name=readBlockIdx \
; RUN:    -imodule-ir-passes=luthier-mock-inject-at-function-entry,always-inline,luthier-process-intrinsics-at-ir-level \
; RUN:    -o /dev/null > %t.asm 2>&1 && \
; RUN: FileCheck %s < %t.asm

; The non-architected blockIdx capture: the three preloaded workgroup-id system
; SGPRs are spilled into their SVA lanes (6/7/8) by the generic entry prologue,
; then read back in the payload. No TTMP reads on this target.
; CHECK: stub_kernel:
; CHECK-DAG: v_writelane_b32 v0, s{{[0-9]+}}, 6
; CHECK-DAG: v_writelane_b32 v0, s{{[0-9]+}}, 7
; CHECK-DAG: v_writelane_b32 v0, s{{[0-9]+}}, 8
; CHECK-NOT: ttmp

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@sink = addrspace(1) global i32 0

; Reads all three workgroup-id dims (so all three preloaded SGPRs are spilled
; into SVA lanes), but stores a single combined value to keep the payload's VGPR
; footprint small — the 3-element-store form would exceed this minimal stub's
; (pipeline-expanded) VGPR budget, which is incidental to the blockIdx routing
; under test here.
define internal void @readBlockIdx() #0 {
entry:
  %x = call i32 @"luthier::workgroupIdX.i32"() #1
  %y = call i32 @"luthier::workgroupIdY.i32"() #2
  %z = call i32 @"luthier::workgroupIdZ.i32"() #3
  %xy = add i32 %x, %y
  %xyz = add i32 %xy, %z
  store i32 %xyz, ptr addrspace(1) @sink, align 4
  ret void
}

declare i32 @"luthier::workgroupIdX.i32"() #1
declare i32 @"luthier::workgroupIdY.i32"() #2
declare i32 @"luthier::workgroupIdZ.i32"() #3

attributes #0 = { alwaysinline "luthier.function.hook" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::workgroupIdX" "target-cpu"="gfx908" }
attributes #2 = { "luthier.intrinsic"="luthier::workgroupIdY" "target-cpu"="gfx908" }
attributes #3 = { "luthier.intrinsic"="luthier::workgroupIdZ" "target-cpu"="gfx908" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
