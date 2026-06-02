; Custom kernarg-buffer mechanism: end-to-end through the full default
; instrumentation pipeline + AsmPrinter.
;
; This file doubles as the IModule (-imodule-path=%s): the @readArg hook reads
; luthier::implicitArgPtr(), which (after the USER_ARG_PTR repoint) requests
; USER_ARG_PTR / IMPLICIT_ARG_OFFSET. The hook is mock-injected at the entry of a
; kernarg-enabled stub kernel; the USER_ARG_PTR request makes
; StateValueArraySpecs allocate those lanes and marks the kernel as using a
; Luthier-managed custom kernarg buffer. TargetModulePatcherPass then (a) emits
; the custom kernarg prologue — writing the implicit-args offset into the
; IMPLICIT_ARG_OFFSET SVA lane (an inline-constant v_writelane), saving the
; preloaded KERNARG_SEGMENT_PTR (the custom-buffer base) into the USER_ARG_PTR
; lane, and reloading the application's original kernarg pointer over the
; physical KERNARG_SEGMENT_PTR with an in-place s_load_dwordx2 + s_waitcnt — and
; (b) emits the .luthier.kernarg_layout ELF section the loader consumes.
;
; gfx942 (architected flat scratch) is used so emitCodeToSetupScratch takes its
; architected path and skips the FLAT_SCRATCH_INIT / private-segment-wave-byte-
; offset preload sequence, which a separate pre-existing CodeDiscovery gap does
; not seed on non-architected targets (tracked in the custom-kernarg memory).

; RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %S/_target_stub_kernarg.s.txt -o %t.o && \
; RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
; RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
; RUN:    %luthier_tool_code_gen_plugin \
; RUN:    %luthier_mock_injection_plugin \
; RUN:    -passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-apply-instrumentation,luthier-asm-printer \
; RUN:    -code-object-paths=%t \
; RUN:    -initial-entrypoint=0:stub_kernel.kd \
; RUN:    -initial-execution-point=0:stub_kernel.kd \
; RUN:    -imodule-path=%s \
; RUN:    -luthier-mock-hook-name=readArg \
; RUN:    -imodule-ir-passes=luthier-mock-inject-at-function-entry,always-inline,luthier-process-intrinsics-at-ir-level \
; RUN:    -o /dev/null > %t.asm 2>&1 && \
; RUN: FileCheck %s < %t.asm

; The custom kernarg prologue: the implicit-args region offset is written into
; its SVA lane as an inline constant, then the application's original kernarg
; pointer is reloaded in place over KERNARG_SEGMENT_PTR and waited on before the
; body runs.
; CHECK: stub_kernel:
; CHECK: v_writelane_b32 v{{[0-9]+}}, 8,
; CHECK: s_load_dwordx2 [[KARG:s\[[0-9]+:[0-9]+\]]], [[KARG]], 0x0
; CHECK: s_waitcnt lgkmcnt(0)
; The custom kernarg layout is published as a dedicated ELF section + symbol.
; CHECK-DAG: .luthier.kernarg_layout
; CHECK-DAG: __luthier_kernarg_layout

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@sink = addrspace(1) global i32 0

define internal void @readArg() #0 {
entry:
  %p = call ptr addrspace(4) @"luthier::implicitArgPtr.p4"()
  %v = load i32, ptr addrspace(4) %p, align 4
  store i32 %v, ptr addrspace(1) @sink, align 4
  ret void
}

declare ptr addrspace(4) @"luthier::implicitArgPtr.p4"() #1

attributes #0 = { alwaysinline "luthier.function.hook" "target-cpu"="gfx942" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::implicitArgPtr" "target-cpu"="gfx942" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
