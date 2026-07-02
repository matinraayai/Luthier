// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   -passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,trace-callgraph-printer \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z6kernelv.kd \
// RUN:   -initial-execution-point=0:_Z6kernelv.kd \
// RUN:   -o - 2>&1 | %tee_out FileCheck %s

// CHECK: TraceCallGraph (fully_recovered=yes)
// CHECK: _Z8dispatchi{{x0x[0-9a-f]+}} -> [_Z2t1vx0x0, _Z2t2vx0x0, _Z2t3vx0x0]
// CHECK: Incomplete call sites (0):

// HIP source (compiled with --offload-device-only for gfx1030):
//
//   __device__ void t1() {}
//   __device__ void t2() {}
//   __device__ void t3() {}
//   // dispatch receives fn ptr in s[6:7] and tail-calls it
//   __device__ void dispatch() { /* fn ptr in s[6:7] */ }
//   __global__ void kernel() { dispatch(); dispatch(); dispatch(); }
//
// dispatch is called three times from the kernel, each time with a different
// function pointer in s[6:7].  The inter-procedural analysis of the callgraph
// should collect all three targets and produce a single call site with three
// possible targets, keeping fully_recovered=yes.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.amdhsa_code_object_version 6
	.text

	; t1(), t2(), t3() — trivial device functions
	.p2align 2
	.hidden _Z2t1v
	.globl  _Z2t1v
	.type   _Z2t1v,@function
_Z2t1v:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_endpgm
.Lfunc_end_t1:
	.size _Z2t1v, .Lfunc_end_t1-_Z2t1v
	.set _Z2t1v.num_vgpr, 0
	.set _Z2t1v.num_agpr, 0
	.set _Z2t1v.numbered_sgpr, 32
	.set _Z2t1v.private_seg_size, 0
	.set _Z2t1v.uses_vcc, 0
	.set _Z2t1v.uses_flat_scratch, 0
	.set _Z2t1v.has_dyn_sized_stack, 0
	.set _Z2t1v.has_recursion, 0
	.set _Z2t1v.has_indirect_call, 0

	.p2align 2
	.hidden _Z2t2v
	.globl  _Z2t2v
	.type   _Z2t2v,@function
_Z2t2v:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_endpgm
.Lfunc_end_t2:
	.size _Z2t2v, .Lfunc_end_t2-_Z2t2v
	.set _Z2t2v.num_vgpr, 0
	.set _Z2t2v.num_agpr, 0
	.set _Z2t2v.numbered_sgpr, 32
	.set _Z2t2v.private_seg_size, 0
	.set _Z2t2v.uses_vcc, 0
	.set _Z2t2v.uses_flat_scratch, 0
	.set _Z2t2v.has_dyn_sized_stack, 0
	.set _Z2t2v.has_recursion, 0
	.set _Z2t2v.has_indirect_call, 0

	.p2align 2
	.hidden _Z2t3v
	.globl  _Z2t3v
	.type   _Z2t3v,@function
_Z2t3v:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_endpgm
.Lfunc_end_t3:
	.size _Z2t3v, .Lfunc_end_t3-_Z2t3v
	.set _Z2t3v.num_vgpr, 0
	.set _Z2t3v.num_agpr, 0
	.set _Z2t3v.numbered_sgpr, 32
	.set _Z2t3v.private_seg_size, 0
	.set _Z2t3v.uses_vcc, 0
	.set _Z2t3v.uses_flat_scratch, 0
	.set _Z2t3v.has_dyn_sized_stack, 0
	.set _Z2t3v.has_recursion, 0
	.set _Z2t3v.has_indirect_call, 0

	; dispatch() — receives fn ptr in s[6:7], tail-calls it
	.p2align 2
	.hidden _Z8dispatchi
	.globl  _Z8dispatchi
	.type   _Z8dispatchi,@function
_Z8dispatchi:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[6:7]
.Lfunc_end_dispatch:
	.size _Z8dispatchi, .Lfunc_end_dispatch-_Z8dispatchi
	.set _Z8dispatchi.num_vgpr, 0
	.set _Z8dispatchi.num_agpr, 0
	.set _Z8dispatchi.numbered_sgpr, 32
	.set _Z8dispatchi.private_seg_size, 0
	.set _Z8dispatchi.uses_vcc, 0
	.set _Z8dispatchi.uses_flat_scratch, 0
	.set _Z8dispatchi.has_dyn_sized_stack, 0
	.set _Z8dispatchi.has_recursion, 0
	.set _Z8dispatchi.has_indirect_call, 1

	; kernel() — calls dispatch(&t1), dispatch(&t2), dispatch(&t3) in sequence
	.protected _Z6kernelv
	.globl  _Z6kernelv
	.p2align 8
	.type   _Z6kernelv,@function
_Z6kernelv:
	s_mov_b32 s32, 0
	; --- dispatch(&t1) ---
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, _Z2t1v@rel32@lo+4
	s_addc_u32 s7, s7, _Z2t1v@rel32@hi+12
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z8dispatchi@rel32@lo+4
	s_addc_u32 s1, s1, _Z8dispatchi@rel32@hi+12
	s_swappc_b64 s[30:31], s[0:1]
	; --- dispatch(&t2) ---
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, _Z2t2v@rel32@lo+4
	s_addc_u32 s7, s7, _Z2t2v@rel32@hi+12
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z8dispatchi@rel32@lo+4
	s_addc_u32 s1, s1, _Z8dispatchi@rel32@hi+12
	s_swappc_b64 s[30:31], s[0:1]
	; --- dispatch(&t3) ---
	s_getpc_b64 s[6:7]
	s_add_u32 s6, s6, _Z2t3v@rel32@lo+4
	s_addc_u32 s7, s7, _Z2t3v@rel32@hi+12
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z8dispatchi@rel32@lo+4
	s_addc_u32 s1, s1, _Z8dispatchi@rel32@hi+12
	s_swappc_b64 s[30:31], s[0:1]
	s_endpgm
	.section .rodata,"a",@progbits
	.p2align 6, 0x0
	.amdhsa_kernel _Z6kernelv
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_user_sgpr_private_segment_buffer 0
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 0
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
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
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
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
	.set _Z6kernelv.num_vgpr, max(1, max(_Z2t1v.num_vgpr, max(_Z2t2v.num_vgpr, max(_Z2t3v.num_vgpr, _Z8dispatchi.num_vgpr))))
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
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
