; Verifies that a 4-lane scalar-value argument
; (WAVEFRONT_PRIVATE_SEGMENT_BUFFER, SA enum 0) lowers to four V_READLANE_B32s
; merged via a REG_SEQUENCE into an SGPR_128.
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

; CHECK: !luthier.intrinsic.placeholders = !{{{.+}}}

; Four distinct SGPRSpill stack slots (one per SA sub-lane), four
; SI_SPILL_S32_RESTORE loads, followed by a REG_SEQUENCE merge into
; SGPR_128.
; CHECK-COUNT-4: stack-id: sgpr-spill
; CHECK-COUNT-4: %{{[0-9]+}}:sgpr_32 = SI_SPILL_S32_RESTORE %stack.{{[0-9]+}}
; CHECK:         REG_SEQUENCE %{{[0-9]+}}, %subreg.sub0, %{{[0-9]+}}, %subreg.sub1, %{{[0-9]+}}, %subreg.sub2, %{{[0-9]+}}, %subreg.sub3

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @luthier.payload.sva4() #0 {
  %v = call i128 @"luthier::readSVA.i128"(i8 0)
  ret void
}

declare i128 @"luthier::readSVA.i128"(i8) #1

attributes #0 = { noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readSVA" "target-cpu"="gfx908" }
