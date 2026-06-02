; blockIdx (workgroup id) on an architected-SGPRs target (gfx1200): end-to-end
; through the full default instrumentation pipeline + AsmPrinter.
;
; This file doubles as the IModule (-imodule-path=%s): the @readBlockIdx hook
; reads luthier::workgroupIdX/Y/Z(), which declare ReadSVAs{WORKGROUP_ID_X/Y/Z}.
; The hook is mock-injected at the entry of the gfx1200 stub kernel; the SVA
; requests make StateValueArraySpecs allocate the WORKGROUP_ID_* lanes.
;
; Because gfx1200 sets FeatureArchitectedSGPRs, TargetModulePatcherPass skips the
; generic preloaded-system-SGPR spill for the workgroup-id lanes and instead
; calls emitCaptureArchitectedWorkgroupIds, which reads the HW-maintained TTMP
; registers (ttmp9 = blockIdx.x, ttmp7[15:0] = blockIdx.y, ttmp7[31:16] =
; blockIdx.z) and writes them into the SVA lanes. The payload then reads the
; lanes back via v_readlane (lowered from the workgroupId MIR processor's SVA
; read), so blockIdx is valid at the arbitrary injection point.

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
; RUN:    -luthier-mock-hook-name=readBlockIdx \
; RUN:    -imodule-ir-passes=luthier-mock-inject-at-function-entry,always-inline,luthier-process-intrinsics-at-ir-level \
; RUN:    -o /dev/null > %t.asm 2>&1 && \
; RUN: FileCheck %s < %t.asm

; The architected blockIdx capture: the workgroup IDs are read out of the TTMP
; registers and written into their SVA lanes by the entry prologue.
; CHECK: stub_kernel:
; CHECK-DAG: ttmp9
; CHECK-DAG: ttmp7
; CHECK-DAG: v_writelane_b32

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@sink = addrspace(1) global [3 x i32] zeroinitializer

define internal void @readBlockIdx() #0 {
entry:
  %x = call i32 @"luthier::workgroupIdX.i32"() #1
  %y = call i32 @"luthier::workgroupIdY.i32"() #2
  %z = call i32 @"luthier::workgroupIdZ.i32"() #3
  %px = getelementptr [3 x i32], ptr addrspace(1) @sink, i32 0, i32 0
  %py = getelementptr [3 x i32], ptr addrspace(1) @sink, i32 0, i32 1
  %pz = getelementptr [3 x i32], ptr addrspace(1) @sink, i32 0, i32 2
  store i32 %x, ptr addrspace(1) %px, align 4
  store i32 %y, ptr addrspace(1) %py, align 4
  store i32 %z, ptr addrspace(1) %pz, align 4
  ret void
}

declare i32 @"luthier::workgroupIdX.i32"() #1
declare i32 @"luthier::workgroupIdY.i32"() #2
declare i32 @"luthier::workgroupIdZ.i32"() #3

attributes #0 = { alwaysinline "luthier.function.hook" "target-cpu"="gfx1200" "target-features"="+wavefrontsize32" }
attributes #1 = { "luthier.intrinsic"="luthier::workgroupIdX" "target-cpu"="gfx1200" }
attributes #2 = { "luthier.intrinsic"="luthier::workgroupIdY" "target-cpu"="gfx1200" }
attributes #3 = { "luthier.intrinsic"="luthier::workgroupIdZ" "target-cpu"="gfx1200" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
