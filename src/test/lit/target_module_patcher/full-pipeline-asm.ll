; End-to-end Luthier instrumentation pipeline test.
;
; Drives the FULL default pipeline (no --imodule-mir-passes override, so
; InstrumentationPMDriver's default codegen chain fires:
;   isel → IntrinsicMIRLoweringPass → InjectedPayloadAccessedRegsAnalysis
;   → IModuleIPPredicatedLivenessAnalysis → InjectedPayloadPreserveLiveRegsPass
;   → LRStateValueStorageAndLoadLocationsAnalysis → InjectedPayloadPEIPass
;   → TargetModulePatcherPass)
; and then chains Luthier's new-PM AsmPrinter
; (luthier::NewPMAsmPrinter, exposed via the pass plugin as the
; pipeline-parseable name "luthier-asm-printer") to emit the final
; assembly to stdout.
;
; NewPMAsmPrinter is documented as a "work-around for printing the
; assembly using the new PM" — it borrows the post-pipeline
; MachineModuleInfo out of MachineModuleAnalysis and runs a tiny legacy
; pass manager (TargetLibraryInfo + DummyCGSCC + TargetPassConfig +
; DummyMachineModuleInfoWrapperPass + AMDGPUResourceUsageAnalysis +
; addAsmPrinter) to drive the AMDGPU AsmPrinter against the borrowed
; MMI. This test pins down that the full chain runs to AsmPrinter
; without crashing or producing empty output.

; RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %intrinsic_mir_lowering_target_stub -o %t.o && \
; RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
; RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
; RUN:    -load-pass-plugin=%luthier_tool_code_gen_plugin \
; RUN:    -passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-apply-instrumentation,luthier-asm-printer \
; RUN:    -code-object-paths=%t \
; RUN:    -initial-entrypoint=0:stub_kernel.kd \
; RUN:    -initial-execution-point=0:stub_kernel.kd \
; RUN:    -imodule-path=%s \
; RUN:    -imodule-ir-passes=luthier-process-intrinsics-at-ir-level \
; RUN:    -o /dev/null > %t.asm 2>&1 && \
; RUN: FileCheck %s < %t.asm

; Sanity: the AsmPrinter produced AMDGPU assembly headers and the stub
; kernel's symbol appears in the output. These prove the full chain
; (default codegen pipeline + AsmPrinter) ran end-to-end against the
; patched target module.
; CHECK: .amdgcn_target
; CHECK: stub_kernel
; CHECK: .amdhsa_kernel stub_kernel

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @luthier.payload.trivial() #0 {
  ; Minimal payload — exercises the SVA preload setup for one scalar
  ; value argument (PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = enum 4, single
  ; 32-bit lane) so the patcher's Phase A initial-entry-kernel setup
  ; gets exercised even though the stub kernel itself is empty.
  %sva = call i32 @"luthier::readSVA.i32"(i8 4)
  ret void
}

declare i32 @"luthier::readSVA.i32"(i8) #1

attributes #0 = { naked noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readSVA" "target-cpu"="gfx908" }
