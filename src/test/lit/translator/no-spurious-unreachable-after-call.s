// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc --disable-verify -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z6kernelv.kd \
// RUN:   -initial-execution-point=0:_Z6kernelv.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// --disable-verify: after CodeDiscoveryPass rewrites the $simm16 operand
// on S_CALL_B64 from MO_Immediate (raw 16-bit branch displacement) to
// MO_GlobalAddress(@callee), stock LLVM's SIInstrInfo::verifyInstruction
// (SIInstrInfo.cpp SOPK verification path) unconditionally calls
// Op->getImm() on the simm16 operand and aborts the assertion
// "isImm() && Wrong MachineOperand accessor". S_CALL_B64 has isCall=1
// but not isBranch=1, so the verifier's isBranch check does NOT skip
// the immediate path. This is a pre-existing LLVM limitation
// unrelated to the synthetic-endpgm / spurious-unreachable behavior
// under test — the sibling translator/s-call-b64-direct-call.s test
// skips the verifier for the same reason.

// TraceFunctionTranslator's tail-call emitter (emitDirectTailCall /
// emitIndirectTailCall) must NOT emit an \c unreachable after the raised
// call. Prior behavior did — a leftover from when call MIs shared their
// MBB with post-call instructions in the lifted MIR. After the discovery
// pass began appending a synthetic \c S_ENDPGM after every call MI
// (which raises to `call @llvm.amdgcn.endpgm` + `unreachable` via
// S_ENDPGM_sem), the tail-call emitter's own \c CreateUnreachable
// double-terminated the BB, leaving trailing `call llvm.amdgcn.endpgm`
// and a second `unreachable` sitting AFTER the terminator — invalid IR.
//
// This test lifts a kernel that calls one device function and checks
// the translated IR of the kernel's call BB: the tail call to
// @_Z6calleev is immediately followed by the fall-through branch to
// the synthetic-endpgm BB (\c br \c label — the terminator the
// translateMBBBody fall-through path emits when the call MBB has a
// successor edge to the synthetic-endpgm MBB), with NO stray
// \c unreachable between the tail-call and the branch. The synthetic-
// endpgm BB itself then holds the \c call \c @llvm.amdgcn.endpgm() +
// \c unreachable pair raised from S_ENDPGM_sem.

// CHECK-LABEL: define {{.*}} @_Z6kernelv(
// The tail call to the resolved callee is terminated by the fall-
// through \c br \c label to the synthetic-endpgm BB. If
// TraceFunctionTranslator regresses and re-emits its own explicit
// \c unreachable via \c Builder.CreateUnreachable after the call,
// that \c unreachable becomes the BB's terminator and CHECK-NEXT
// below sees it instead of the expected \c br, failing the test.
// The synthetic \c S_ENDPGM CodeDiscoveryPass places in the separate
// endpgm MBB raises to \c call \c @llvm.amdgcn.endpgm + \c unreachable
// in a distinct IR BB whose only predecessor is the call BB.
// CHECK:      tail call void @_Z6calleev
// CHECK-NEXT: br label %
// CHECK:      ; preds =
// CHECK-NEXT: call void @llvm.amdgcn.endpgm()
// CHECK-NEXT: unreachable

  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .amdhsa_code_object_version 6
  .text

  ; callee — returns via s[30:31] (no work)
  .p2align 2
  .hidden _Z6calleev
  .globl  _Z6calleev
  .type   _Z6calleev,@function
_Z6calleev:
  s_setpc_b64 s[30:31]
.Lfunc_end_callee:
  .size _Z6calleev, .Lfunc_end_callee-_Z6calleev
  .set _Z6calleev.num_vgpr, 0
  .set _Z6calleev.num_agpr, 0
  .set _Z6calleev.numbered_sgpr, 32
  .set _Z6calleev.private_seg_size, 0
  .set _Z6calleev.uses_vcc, 0
  .set _Z6calleev.uses_flat_scratch, 0
  .set _Z6calleev.has_dyn_sized_stack, 0
  .set _Z6calleev.has_recursion, 0
  .set _Z6calleev.has_indirect_call, 0

  ; kernel — calls callee via S_CALL_B64 (direct call, 16-bit signed disp)
  .protected _Z6kernelv
  .globl  _Z6kernelv
  .p2align 8
  .type   _Z6kernelv,@function
_Z6kernelv:
  s_mov_b32 s32, 0
  s_call_b64 s[30:31], _Z6calleev
  s_endpgm
.Lfunc_end_kernel:
  .size _Z6kernelv, .Lfunc_end_kernel-_Z6kernelv
  .set _Z6kernelv.num_vgpr, max(1, _Z6calleev.num_vgpr)
  .set _Z6kernelv.num_agpr, 0
  .set _Z6kernelv.numbered_sgpr, 33
  .set _Z6kernelv.private_seg_size, 0
  .set _Z6kernelv.uses_vcc, 1
  .set _Z6kernelv.uses_flat_scratch, 0
  .set _Z6kernelv.has_dyn_sized_stack, 0
  .set _Z6kernelv.has_recursion, 0
  .set _Z6kernelv.has_indirect_call, 0

  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel _Z6kernelv
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 0
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_queue_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 0
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_user_sgpr_private_segment_size 0
    .amdhsa_uses_dynamic_stack 0
    .amdhsa_enable_private_segment 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 33
    .amdhsa_reserve_vcc 1
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_fp16_overflow 0
    .amdhsa_wavefront_size32 1
    .amdhsa_workgroup_processor_mode 1
    .amdhsa_memory_ordered 1
    .amdhsa_forward_progress 0
    .amdhsa_exception_fp_ieee_invalid_op 0
    .amdhsa_exception_fp_denorm_src 0
    .amdhsa_exception_fp_ieee_div_zero 0
    .amdhsa_exception_fp_ieee_overflow 0
    .amdhsa_exception_fp_ieee_underflow 0
    .amdhsa_exception_fp_ieee_inexact 0
    .amdhsa_exception_int_div_zero 0
  .end_amdhsa_kernel
  .text
  .p2alignl 6, 3215226880
  .fill 256, 4, 3215226880

  .amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z6kernelv
    .private_segment_fixed_size: 0
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         _Z6kernelv.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     1
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1100
amdhsa.version:
  - 1
  - 2
...
  .end_amdgpu_metadata
