; Custom kernarg-buffer mechanism on a NON-architected-flat-scratch target
; (gfx908), exercising the full default pipeline + AsmPrinter.
;
; Companion to custom-kernarg-prologue.ll (gfx942). gfx908 routes
; TargetModulePatcherPass::emitCodeToSetupScratch through its NON-architected
; path, which reads the preloaded FLAT_SCRATCH_INIT + private-segment
; wavefront-offset (PSWO) SGPRs. CodeDiscoveryPass must seed those into the
; lifted kernel's ArgInfo; this test is the regression guard for the
; FLAT_SCRATCH_INIT seeding (it was previously gated on the inverted
; flatScratchIsArchitected() polarity, so getPreloadedReg(FLAT_SCRATCH_INIT)
; returned NoRegister and the scratch prologue asserted). The custom kernarg
; prologue (IMPLICIT_ARG_OFFSET write + in-place original-kernarg reload) and the
; .luthier.kernarg_layout section are emitted on top of the scratch setup.

; RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %S/_target_stub_kernarg_gfx908.s.txt -o %t.o && \
; RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
; RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
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

; The non-architected scratch setup runs (proving FLAT_SCRATCH_INIT + PSWO were
; seeded): the per-wave-adjusted flat-scratch value is stored into the SVA.
; CHECK: stub_kernel:
; CHECK: flat_scratch_lo
; The custom kernarg prologue: implicit-args offset written as an inline
; constant, then the original kernarg pointer reloaded in place and waited on.
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

attributes #0 = { alwaysinline "luthier.function.hook" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::implicitArgPtr" "target-cpu"="gfx908" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
