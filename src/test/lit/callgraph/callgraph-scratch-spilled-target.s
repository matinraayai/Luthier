// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,trace-callgraph-printer' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z6kernelv.kd \
// RUN:   -initial-execution-point=0:_Z6kernelv.kd \
// RUN:   -o - 2>&1 | %tee_out FileCheck %s

// CHECK: TraceCallGraph (fully_recovered=yes)
// CHECK: Resolved call sites ({{[0-9]+}}):
// CHECK: _Z6kernelv -> [_Z3barvx0x0]
// CHECK: Incomplete call sites (0):

// HIP-ish source (hand-written for gfx908):
//
//   __device__ void bar() {}
//   __global__ void kernel() { bar(); }
//
// The callee address is a compile-time constant (getpc + rel32), but the
// scalar register pair holding it is spilled to scratch (private memory) via
// s_scratch_store_dwordx2 and reloaded via s_scratch_load_dwordx2 before the
// call -- modelling a function pointer spilled under register pressure.  Both
// scratch accesses use the same SGPR base + immediate offset, so they lift to
// an addrspace(5) store/load pair against a common pointer.  The callgraph
// analysis recovers the target by tracing the load back to its must-aliasing
// clobbering store via MemorySSA, then folding the stored value.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.amdhsa_code_object_version 6
	.text
	.p2align 2
	.hidden _Z3barv
	.globl  _Z3barv
	.type   _Z3barv,@function
_Z3barv:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size _Z3barv, .Lfunc_end0-_Z3barv
	.set _Z3barv.num_vgpr, 0
	.set _Z3barv.num_agpr, 0
	.set _Z3barv.numbered_sgpr, 32
	.set _Z3barv.private_seg_size, 0
	.set _Z3barv.uses_vcc, 0
	.set _Z3barv.uses_flat_scratch, 0
	.set _Z3barv.has_dyn_sized_stack, 0
	.set _Z3barv.has_recursion, 0
	.set _Z3barv.has_indirect_call, 0

	.protected _Z6kernelv
	.globl  _Z6kernelv
	.p2align 8
	.type   _Z6kernelv,@function
_Z6kernelv:
	s_mov_b32 s32, 0
	; scalar scratch base (value is irrelevant to the analysis)
	s_mov_b32 s2, 0
	s_mov_b32 s3, 0
	; compute &bar into s[0:1]
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z3barv@rel32@lo+4
	s_addc_u32 s1, s1, _Z3barv@rel32@hi+12
	; spill the pointer to scratch slot 0, then reload it into s[4:5]
	s_scratch_store_dwordx2 s[0:1], s[2:3], 0x0
	s_scratch_load_dwordx2 s[4:5], s[2:3], 0x0
	s_waitcnt lgkmcnt(0)
	s_swappc_b64 s[30:31], s[4:5]
	s_endpgm
	.section .rodata,"a",@progbits
	.p2align 6, 0x0
	.amdhsa_kernel _Z6kernelv
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 8
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
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size _Z6kernelv, .Lfunc_end1-_Z6kernelv
	.set _Z6kernelv.num_vgpr, max(1, _Z3barv.num_vgpr)
	.set _Z6kernelv.num_agpr, max(0, _Z3barv.num_agpr)
	.set _Z6kernelv.numbered_sgpr, max(33, _Z3barv.numbered_sgpr)
	.set _Z6kernelv.private_seg_size, 8+max(_Z3barv.private_seg_size)
	.set _Z6kernelv.uses_vcc, or(1, _Z3barv.uses_vcc)
	.set _Z6kernelv.uses_flat_scratch, or(0, _Z3barv.uses_flat_scratch)
	.set _Z6kernelv.has_dyn_sized_stack, or(0, _Z3barv.has_dyn_sized_stack)
	.set _Z6kernelv.has_recursion, or(0, _Z3barv.has_recursion)
	.set _Z6kernelv.has_indirect_call, or(0, _Z3barv.has_indirect_call)
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
    .private_segment_fixed_size: 8
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         _Z6kernelv.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     1
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
