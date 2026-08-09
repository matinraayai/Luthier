// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc --disable-verify -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z6kernelv.kd \
// RUN:   -initial-execution-point=0:_Z6kernelv.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// --disable-verify: after CodeDiscoveryPass rewrites the $simm16 operand
// from MO_Immediate to MO_GlobalAddress, SIInstrInfo::verifyInstruction's
// SOPK branch unconditionally calls Op->getImm() and aborts.
//
// Smoke test for the S_CALL_B64 semantic (Phase 26):
//   - CodeDiscoveryPass rewrites the direct-call $simm16 from a 16-bit
//     displacement to MO_GlobalAddress(callee Function) after the
//     callgraph resolves the target.
//   - MIRToIRTranslator's S_CALL_B64_sem emits:
//       (a) a call to llvm.amdgcn.s.getpc that writes the return address
//           (post-PC) into the $sdst register file (s[30:31] in this kernel);
//       (b) a direct call to the resolved callee Function.

// CHECK: define {{.*}} @_Z6kernelv
// CHECK-DAG: call i64 @llvm.amdgcn.s.getpc
// CHECK-DAG: tail call void @_Z6calleev

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
