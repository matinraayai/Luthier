; Verifies the physical-register access path: luthier::readReg lowers the
; target's physreg into a vreg via the entry-block COPY-from-physreg dance,
; and InjectedPayloadAccessedRegsAnalysis classifies that physreg as
; Read-only (pure preserve — payload reads it but writes back the
; unchanged value).
;
; AMDGPU::SGPR4 is MCRegister enum value 328, a single 32-bit SGPR — the
; simplest allocatable phys-reg we can read.
;
; RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %intrinsic_mir_lowering_target_stub -o %t.o && \
; RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
; RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
; RUN:    -load-pass-plugin=%luthier_tool_code_gen_plugin \
; RUN:    -passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,luthier-apply-instrumentation \
; RUN:    -code-object-paths=%t \
; RUN:    -initial-entrypoint=0:stub_kernel.kd \
; RUN:    -initial-execution-point=0:stub_kernel.kd \
; RUN:    -imodule-path=%s \
; RUN:    -imodule-output=/dev/null \
; RUN:    -imodule-ir-passes=luthier-process-intrinsics-at-ir-level \
; RUN:    -imodule-mir-passes=isel,mir-lowering,injected-payload-accessed-regs-print \
; RUN:    -o /dev/null 2>&1 | %tee_out FileCheck %s

; Analysis output: SGPR4 appears in Reads only; Writes is empty.
; CHECK:      Payload luthier.payload.physreg_read:
; CHECK-NEXT:     Reads: SGPR4
; CHECK-NEXT:     Writes:
; CHECK-NOT:      SGPR4

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @luthier.payload.physreg_read() #0 {
  %v = call i32 @"luthier::readReg.i32"(i32 328)
  ret void
}

declare i32 @"luthier::readReg.i32"(i32) #1

attributes #0 = { noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readReg" "target-cpu"="gfx908" }
