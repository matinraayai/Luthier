; Verifies that when an injected payload calls a non-inlined helper that
; reads a scalar-value argument (KERNEL_ARG_PTR, 2 lanes), the
; ForwardISAStateToCalleesPass:
;   - extends the helper's signature with two extra i32 lane parameters,
;   - rewrites the helper body's readSVA placeholder to reassemble the
;     i64 from those two i32 params (via zext+shl+or),
;   - rewrites the payload's call site to pass the two i32 lanes through
;     a single new caller-side readSVA placeholder.
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
; RUN:    -imodule-output=%t.luthier \
; RUN:    -imodule-mir-passes= \
; RUN:    -imodule-ir-passes=luthier-process-intrinsics-at-ir-level,luthier-forward-isa-state-to-callees \
; RUN:    -o /dev/null && \
; RUN: FileCheck %s < %t.luthier

; The payload's call site is rewritten: a new caller-side readSVA placeholder
; is emitted (lanes split via trunc/lshr) and the helper is called with the
; two i32 lane args.
; CHECK:      define void @luthier.payload.sva_callee()
; CHECK:      call i64 asm sideeffect "luthier.placeholder.{{[0-9]+}}", "=s"()
; CHECK:      call void @helper_reads_sva(i32 {{.*}}, i32 {{.*}})

; The helper now takes two extra i32 lane parameters.
; CHECK: define void @helper_reads_sva(i32 %luthier.fwd.sva.1.0, i32 %luthier.fwd.sva.1.1)

; Inside the helper body the lanes are reassembled into the i64 the original
; readSVA returned (lower lane in bits 0..31, upper in 32..63).
; CHECK-DAG: zext i32 %luthier.fwd.sva.1.0 to i64
; CHECK-DAG: zext i32 %luthier.fwd.sva.1.1 to i64
; CHECK-DAG: shl{{.*}} 32

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @helper_reads_sva() #2 {
  %v = call i64 @"luthier::readSVA.i64"(i8 1)
  ret void
}

define void @luthier.payload.sva_callee() #0 {
  call void @helper_reads_sva()
  ret void
}

declare i64 @"luthier::readSVA.i64"(i8) #1

attributes #0 = { noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readSVA" "target-cpu"="gfx908" }
attributes #2 = { noinline "target-cpu"="gfx908" }
