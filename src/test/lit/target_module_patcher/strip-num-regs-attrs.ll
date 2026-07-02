; End-to-end smoke test for TargetModulePatcherPass.
;
; Drives the IModule pipeline through `target-module-patcher` and dumps the
; resulting target module via `--target-module-output`. Verifies the
; Phase B step 2 effect (CodeDiscoveryPass-supplied `amdgpu-num-vgpr` /
; `amdgpu-num-sgpr` function attributes are stripped from every target
; function), which is the most stable, MMI-independent signal that the
; patcher actually ran over the cloned target module.
;
; What this test specifically verifies:
;
;   ✓ The patcher's analysis chain (IModuleMAMWrapperPass +
;     LRStateValueStorageAndLoadLocationsAnalysis +
;     IModuleIPPredicatedLivenessAnalysis + MachineModuleInfoWrapperPass)
;     resolves and runs to completion without crashing.
;   ✓ `--target-module-output` produces a readable dump of the post-patch
;     target module IR + MFs.
;   ✓ stripStaleNumRegsAttrs removed both attributes from every target
;     function — CHECK-NOT lines below would fire if either remained.
;
; Phase A effects (SVS-switch emission + initial-entry-kernel setup) and
; Phase B step 3 effects (payload inlining) are NOT FileChecked here
; because their visibility depends on which target opcodes / SVA storage
; schemes happen to land for this kernel + payload combination, which is
; codegen-noisy. A follow-up test pinned to a specific gfx target +
; payload pattern is the right place for those assertions.

; RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %intrinsic_mir_lowering_target_stub -o %t.o && \
; RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
; RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
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

; Sanity: the dump header is present (proves --target-module-output fired).
; CHECK: Luthier target-module-output

; The patcher must strip both amdgpu-num-* attrs from every target Function.
; CHECK-NOT: "amdgpu-num-vgpr"
; CHECK-NOT: "amdgpu-num-sgpr"

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @luthier.payload.minimal_sva() #0 {
  ; Minimal payload that touches the SVA so the patcher's Phase A
  ; initial-entry-kernel setup gets exercised. Enum 4 =
  ; PRIVATE_SEGMENT_WAVE_BYTE_OFFSET (1-lane SA, always available on gfx908).
  %sva = call i32 @"luthier::readSVA.i32"(i8 4)
  ret void
}

declare i32 @"luthier::readSVA.i32"(i8) #1

attributes #0 = { naked noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readSVA" "target-cpu"="gfx908" }
