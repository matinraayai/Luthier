// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,trace-callgraph-printer' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z6kernelv.kd \
// RUN:   -initial-execution-point=0:_Z6kernelv.kd \
// RUN:   -o - 2>&1 | %tee_out FileCheck %s

// CHECK: TraceCallGraph (fully_recovered=yes)
// CHECK: _Z6calleev{{x0x[0-9a-f]+}} -> [_Z3barvx0x0]
// CHECK: Incomplete call sites (0):

// HIP source (compiled with --offload-device-only for gfx1100):
//
//   __device__ void bar() {}
//   // callee receives fn ptr in s[6:7] and tail-calls it
//   __device__ void callee(/* fn ptr in s[6:7] */) { /* call via s_setpc_b64 s[6:7] */ }
//   __global__ void kernel() { callee(&bar); }
//
// The kernel calls callee with bar's address in s[6:7].  The inter-procedural
// analysis of the callgraph should trace the function pointer through the
// argument and resolve callee's indirect call to bar.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
	.amdhsa_code_object_version 6
	.text

	; bar() — called via the function pointer argument; does not return
	.p2align 2
	.hidden _Z3barv
	.globl  _Z3barv
	.type   _Z3barv,@function
_Z3barv:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_endpgm
.Lfunc_end_bar:
	.size _Z3barv, .Lfunc_end_bar-_Z3barv
	.set _Z3barv.num_vgpr, 0
	.set _Z3barv.num_agpr, 0
	.set _Z3barv.numbered_sgpr, 32
	.set _Z3barv.private_seg_size, 0
	.set _Z3barv.uses_vcc, 0
	.set _Z3barv.uses_flat_scratch, 0
	.set _Z3barv.has_dyn_sized_stack, 0
	.set _Z3barv.has_recursion, 0
	.set _Z3barv.has_indirect_call, 0

	; callee() — receives fn ptr in s[6:7], tail-calls it
	.p2align 2
	.hidden _Z6calleev
	.globl  _Z6calleev
	.type   _Z6calleev,@function
_Z6calleev:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[6:7]
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
	.set _Z6calleev.has_indirect_call, 1

	; kernel() — computes &bar into s[6:7], then calls callee via SWAPPC
	.protected _Z6kernelv
	.globl  _Z6kernelv
	.p2align 8
	.type   _Z6kernelv,@function
_Z6kernelv:
	s_mov_b32 s32, 0
	; Compute &bar into s[6:7] (fn ptr argument to callee)
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, _Z3barv@rel32@lo+4
	s_addc_u32 s7, s7, _Z3barv@rel32@hi+12
	; Compute &callee into s[0:1]
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z6calleev@rel32@lo+4
	s_addc_u32 s1, s1, _Z6calleev@rel32@hi+12
	; Call callee — it will tail-call bar and end
	s_swappc_b64 s[30:31], s[0:1]
	s_endpgm
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
.Lfunc_end_kernel:
	.size _Z6kernelv, .Lfunc_end_kernel-_Z6kernelv
	.set _Z6kernelv.num_vgpr, max(1, max(_Z3barv.num_vgpr, _Z6calleev.num_vgpr))
	.set _Z6kernelv.num_agpr, 0
	.set _Z6kernelv.numbered_sgpr, 33
	.set _Z6kernelv.private_seg_size, 0
	.set _Z6kernelv.uses_vcc, 1
	.set _Z6kernelv.uses_flat_scratch, 0
	.set _Z6kernelv.has_dyn_sized_stack, 0
	.set _Z6kernelv.has_recursion, 0
	.set _Z6kernelv.has_indirect_call, 0
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
