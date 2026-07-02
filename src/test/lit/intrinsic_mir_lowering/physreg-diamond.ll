; DAG (diamond) correctness test: only one of two parallel branches
; writes the phys-reg, then both paths re-merge and the join block reads
; it again. The post-merge read must source a PHI taking the entry COPY
; on the unwritten path and the body's write vreg on the other.
;
; Same two-phase placeholder fix the back-edge test exercises — for a
; DAG without a back edge, single-phase would also have worked, but
; encoding both correctness scenarios in lit guards future regressions.
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
; RUN:    -imodule-mir-passes=isel,mir-lowering,injected-payload-accessed-regs-print \
; RUN:    -o /dev/null 2>&1 | %tee_out FileCheck %s --check-prefix=ANALYSIS && \
; RUN: FileCheck %s --check-prefix=MIR < %t.mir

; ANALYSIS:      Payload luthier.payload.diamond:
; ANALYSIS-NEXT:     Reads: SGPR4
; ANALYSIS-NEXT:     Writes: SGPR4

; Entry COPY-from-physreg present.
; MIR:     bb.0.entry:
; MIR:       liveins: $sgpr4
; MIR:       [[ENTRY_VREG:%[0-9]+]]:sreg_32 = COPY $sgpr4

; The taken branch defines the back-half vreg via a COPY of the writeReg
; input chain.
; MIR:     bb.{{[0-9]+}}.taken:
; MIR:       [[WRITE_VREG:%[0-9]+]]:sreg_32 = COPY %{{[0-9]+}}

; The join block carries a PHI of (entry-vreg, write-vreg).
; MIR:     bb.{{[0-9]+}}.join:
; MIR:       %{{[0-9]+}}:sreg_32 = PHI [[ENTRY_VREG]], %bb.{{[0-9]+}}, [[WRITE_VREG]], %bb.{{[0-9]+}}

; Phase 2 left no IMPLICIT_DEF placeholders behind.
; MIR-NOT:   IMPLICIT_DEF

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @luthier.payload.diamond() #0 {
entry:
  %v_entry = call i32 @"luthier::readReg.i32"(i32 328)
  %cond = icmp eq i32 %v_entry, 0
  br i1 %cond, label %taken, label %ntaken

taken:
  %newval = add i32 %v_entry, 7
  %_dead = call i32 @"luthier::writeReg.i32"(i32 328, i32 %newval)
  br label %join

ntaken:
  br label %join

join:
  %final = call i32 @"luthier::readReg.i32"(i32 328)
  ret void
}

declare i32 @"luthier::readReg.i32"(i32)  #1
declare i32 @"luthier::writeReg.i32"(i32, i32) #2

attributes #0 = { noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readReg" "target-cpu"="gfx908" }
attributes #2 = { "luthier.intrinsic"="luthier::writeReg" "target-cpu"="gfx908" }
