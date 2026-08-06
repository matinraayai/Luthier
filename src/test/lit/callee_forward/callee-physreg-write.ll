; Verifies that when an injected payload calls a non-inlined helper that
; writes to a 32-bit physical register (SGPR4, MCRegister enum 328),
; ForwardISAStateToCalleesPass:
;   - extends the helper's signature with an i32 input param for the
;     channel (preserve semantics) and changes its return to {} → {i32},
;   - replaces the helper body's writeReg placeholder with a struct return
;     carrying the written value,
;   - at the call site, extractvalues the channel and emits a fresh
;     writeReg placeholder so the caller's MIR lowering handles the
;     post-call write the same as a user-written one.
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

; Payload calls the rewritten helper and extracts the written channel back
; into a fresh writeReg placeholder.
; CHECK:      define void @luthier.payload.physreg_write()
; CHECK:      %{{[0-9]+}} = call { i32 } @helper_writes_sgpr(i32 %{{[0-9]+}})
; CHECK:      extractvalue { i32 } %{{[0-9]+}}, 0
; CHECK:      call i32 asm sideeffect "luthier.placeholder.{{[0-9]+}}", "=s,s"(i32 %{{[0-9]+}})

; Helper signature: takes the channel as input (preserve) and returns
; { i32 } (the written channel).
; CHECK: define { i32 } @helper_writes_sgpr(i32 %luthier.fwd.chan.328)

; Helper body returns a struct with the written value as slot 0. Constant
; insertvalue chains get folded by IRBuilder; accept either form.
; CHECK: ret { i32 }

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @helper_writes_sgpr() #2 {
  call i32 @"luthier::writeReg.i32"(i32 328, i32 42)
  ret void
}

define void @luthier.payload.physreg_write() #0 {
  call void @helper_writes_sgpr()
  ret void
}

declare i32 @"luthier::writeReg.i32"(i32, i32) #1

attributes #0 = { noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::writeReg" "target-cpu"="gfx908" }
attributes #2 = { noinline "target-cpu"="gfx908" }
