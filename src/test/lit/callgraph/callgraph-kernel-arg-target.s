// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   -passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,trace-callgraph-printer \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z6kernelv.kd \
// RUN:   -initial-execution-point=0:_Z6kernelv.kd \
// RUN:   -o - 2>&1 | %tee_out FileCheck %s

// CHECK: TraceCallGraph (fully_recovered=no)
// CHECK: Incomplete call sites (1):
// CHECK: _Z6kernelv

// HIP source (compiled with --offload-device-only for gfx1201):
//
//   __global__ void kernel(void (*fn)()) { fn(); }
//
// The function pointer comes from kernarg memory (s_load_dwordx2 from the
// kernarg segment pointer).  The loaded value is a runtime quantity that
// cannot be constant-folded, so the call site cannot be resolved and the
// analysis must report fully_recovered=no.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text

	; kernel(void (*fn)()) — loads fn ptr from kernarg and calls it
	.protected _Z6kernelv
	.globl  _Z6kernelv
	.p2align 8
	.type   _Z6kernelv,@function
_Z6kernelv:
	; s[0:1] = kernarg segment ptr (from .amdhsa_user_sgpr_kernarg_segment_ptr 1)
	s_load_dword s2, s[0:1], 0x0   ; load lo 32 bits of fn ptr from kernarg
	s_load_dword s3, s[0:1], 0x4   ; load hi 32 bits of fn ptr from kernarg
	s_waitcnt lgkmcnt(0)
	s_swappc_b64 s[30:31], s[2:3]  ; call fn — target is runtime, unresolvable
	s_endpgm
	.section .rodata,"a",@progbits
	.p2align 6, 0x0
	.amdhsa_kernel _Z6kernelv
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
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
		.amdhsa_forward_progress 1
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end_kernel:
	.size _Z6kernelv, .Lfunc_end_kernel-_Z6kernelv
	.set _Z6kernelv.num_vgpr, 1
	.set _Z6kernelv.num_agpr, 0
	.set _Z6kernelv.numbered_sgpr, 33
	.set _Z6kernelv.private_seg_size, 0
	.set _Z6kernelv.uses_vcc, 1
	.set _Z6kernelv.uses_flat_scratch, 0
	.set _Z6kernelv.has_dyn_sized_stack, 0
	.set _Z6kernelv.has_recursion, 0
	.set _Z6kernelv.has_indirect_call, 1
	.p2alignl 6, 3215226880
	.fill 256, 4, 3215226880
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
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
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
