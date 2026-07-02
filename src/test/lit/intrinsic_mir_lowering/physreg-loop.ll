; Back-edge correctness test: a payload with a loop where the header reads
; a phys-reg via luthier::readReg and the body writes the same phys-reg via
; luthier::writeReg. Without the two-phase placeholder resolution, the
; SSAUpdater would only see the entry-block COPY when the loop header is
; processed (the back-edge predecessor hasn't been visited yet) and would
; synthesize an IMPLICIT_DEF for that back-edge PHI source.
;
; The MIR after lowering must contain:
;   * A PHI at the loop header taking the entry-block COPY as the
;     preheader source AND the back-edge write's vreg as the body source.
;   * No IMPLICIT_DEF feeding the readReg's COPY result.
;   * No IMPLICIT_DEF feeding the return-block restore COPY.
;
; InjectedPayloadAccessedRegsAnalysis must classify SGPR4 as Read+Write
; (not pure-preserve), since the body genuinely writes it.
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

; The analysis must report SGPR4 as both Read and Written (loop body
; writes the new value back via luthier::writeReg).
; ANALYSIS:      Payload luthier.payload.loop_test:
; ANALYSIS-NEXT:     Reads: SGPR4
; ANALYSIS-NEXT:     Writes: SGPR4

; The loop header MBB carries a PHI for the phys-reg's vreg with two
; sources — one from the entry/preheader, one from the back edge. The
; exact MBB numbers and vreg ids depend on ISel, so use captures.
; MIR:     bb.0.entry:
; MIR:       liveins: $sgpr4
; MIR:       [[ENTRY_VREG:%[0-9]+]]:sreg_32 = COPY $sgpr4
; MIR:     bb.1.loop:
; MIR:       %{{[0-9]+}}:sreg_32 = PHI [[ENTRY_VREG]], %bb.0, [[BACKEDGE_VREG:%[0-9]+]], %bb.1

; The PHI's back-edge source must be defined inside the loop block (it
; flows from the writeReg's input chain).
; MIR:       [[BACKEDGE_VREG]]:sreg_32 = COPY %{{[0-9]+}}

; Phase 2 must have erased the IMPLICIT_DEF placeholders. The only
; remaining IMPLICIT_DEFs in the MIR should be for the SVA placeholder
; (none in this test, since the payload doesn't use any scalar args).
; MIR-NOT:   IMPLICIT_DEF

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @luthier.payload.loop_test() #0 {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %cur = call i32 @"luthier::readReg.i32"(i32 328)
  %newval = add i32 %cur, 1
  %_dead = call i32 @"luthier::writeReg.i32"(i32 328, i32 %newval)
  %iv.next = add i32 %iv, 1
  %cond = icmp ult i32 %iv.next, 10
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}

declare i32 @"luthier::readReg.i32"(i32)  #1
declare i32 @"luthier::writeReg.i32"(i32, i32) #2

attributes #0 = { noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readReg" "target-cpu"="gfx908" }
attributes #2 = { "luthier.intrinsic"="luthier::writeReg" "target-cpu"="gfx908" }
