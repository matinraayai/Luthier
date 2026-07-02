; Verifies that a 2-lane scalar-value argument (KERNEL_ARG_PTR, SA enum 1)
; lowers to two V_READLANE_B32s merged via a REG_SEQUENCE into an SGPR_64.
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
; RUN:    -imodule-output=%t.mir \
; RUN:    -imodule-ir-passes=luthier-process-intrinsics-at-ir-level \
; RUN:    -imodule-mir-passes=isel,mir-lowering \
; RUN:    -o /dev/null && \
; RUN: FileCheck %s < %t.mir

; CHECK:     !luthier.intrinsic.placeholders = !{{{.+}}}

; Each SA sub-lane gets its own SGPRSpill-stack-id FI;
; SI_SPILL_S32_RESTORE loads them and a REG_SEQUENCE merges the two
; SGPR_32 vregs into an SGPR_64.
; CHECK:         stack-id: sgpr-spill
; CHECK:         stack-id: sgpr-spill
; CHECK:         %{{[0-9]+}}:sgpr_32 = SI_SPILL_S32_RESTORE %stack.{{[0-9]+}}
; CHECK:         %{{[0-9]+}}:sgpr_32 = SI_SPILL_S32_RESTORE %stack.{{[0-9]+}}
; CHECK:         %{{[0-9]+}}:sgpr_64 = REG_SEQUENCE %{{[0-9]+}}, %subreg.sub0, %{{[0-9]+}}, %subreg.sub1

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @luthier.payload.sva2() #0 {
  %v = call i64 @"luthier::readSVA.i64"(i8 1)
  ret void
}

declare i64 @"luthier::readSVA.i64"(i8) #1

attributes #0 = { noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readSVA" "target-cpu"="gfx908" }
