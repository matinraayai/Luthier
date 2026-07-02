; Stress-tests every ScalarValueArgument enum value (0..9) in one payload.
; Each unique SA produces a distinct named-MD entry (one per unique key);
; lane counts are 4, 2, 2, 2, 1, 1, 2, 1, 2, 1, summing to 18 readlane sites.
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

; Ten distinct placeholder keys, one per SA enum value 0..9.
; CHECK-COUNT-10: !"luthier.placeholder.{{[0-9]+}}", !"luthier::readSVA"

; Total SA sub-lane count: 4+2+2+2+1+1+2+1+2+1 = 18. Each becomes its own
; SGPRSpill FI loaded via SI_SPILL_S32_RESTORE.
; CHECK-COUNT-18: stack-id: sgpr-spill
; CHECK-COUNT-18: SI_SPILL_S32_RESTORE %stack.{{[0-9]+}}

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @luthier.payload.svaAll() #0 {
  %sva0 = call i128 @"luthier::readSVA.i128"(i8 0)
  %sva1 = call i64  @"luthier::readSVA.i64"(i8 1)
  %sva2 = call i64  @"luthier::readSVA.i64"(i8 2)
  %sva3 = call i64  @"luthier::readSVA.i64"(i8 3)
  %sva4 = call i32  @"luthier::readSVA.i32"(i8 4)
  %sva5 = call i32  @"luthier::readSVA.i32"(i8 5)
  %sva6 = call i64  @"luthier::readSVA.i64"(i8 6)
  %sva7 = call i32  @"luthier::readSVA.i32"(i8 7)
  %sva8 = call i64  @"luthier::readSVA.i64"(i8 8)
  %sva9 = call i32  @"luthier::readSVA.i32"(i8 9)
  ret void
}

declare i32  @"luthier::readSVA.i32"(i8)  #1
declare i64  @"luthier::readSVA.i64"(i8)  #1
declare i128 @"luthier::readSVA.i128"(i8) #1

attributes #0 = { noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readSVA" "target-cpu"="gfx908" }
