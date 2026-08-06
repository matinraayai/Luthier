; Verifies that a single-lane scalar-value argument (PRIVATE_SEGMENT_WAVE_BYTE_OFFSET,
; SA enum value 4) is lowered to one V_READLANE_B32 plus the per-MF SVA VGPR
; IMPLICIT_DEF placeholder, with no REG_SEQUENCE merge step.
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

; The placeholder gets an opaque key as the InlineAsm template string. The
; named-MD lookup table is emitted alongside the lowered MIR.
; CHECK:     !luthier.intrinsic.placeholders = !{[[ENTRY:![0-9]+]]}
; CHECK-DAG: [[ENTRY]] = !{!"luthier.placeholder.0", !"luthier::readSVA", {{![0-9]+}}, {{![0-9]+}}}

; Lowered MIR after mir-lowering: exactly one SI_SPILL_S32_RESTORE for the
; 1-lane SA, loading from a TargetStackID::SGPRSpill stack slot. The
; downstream pipeline (SILowerSGPRSpills, SIPreAllocateWWMRegs, the WWM
; greedy regalloc + AMDGPUReserveWWMRegs) converts this to a physreg-
; operand SI_RESTORE_S32_FROM_VGPR and reserves the LaneVGPR's physreg
; in MFInfo->WWMReservedRegs. The SVAPhysVGPRPinPass hints the LaneVGPR
; to the load-plan physreg so cross-MF consistency holds.
; CHECK:     stack-id: sgpr-spill
; CHECK:     %{{[0-9]+}}:sgpr_32 = SI_SPILL_S32_RESTORE %stack.{{[0-9]+}}
; CHECK-NOT: SI_SPILL_S32_RESTORE %stack.{{[0-9]+}}

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define void @luthier.payload.sva1() #0 {
  %v = call i32 @"luthier::readSVA.i32"(i8 4)
  ret void
}

declare i32 @"luthier::readSVA.i32"(i8) #1

attributes #0 = { noinline "luthier.function.injected_payload" "target-cpu"="gfx908" "target-features"="+wavefrontsize64" }
attributes #1 = { "luthier.intrinsic"="luthier::readSVA" "target-cpu"="gfx908" }
