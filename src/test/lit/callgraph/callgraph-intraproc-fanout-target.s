// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,prototype-callgraph-printer' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z6kernelv.kd \
// RUN:   -initial-execution-point=0:_Z6kernelv.kd \
// RUN:   -o - 2>&1 | %tee_out FileCheck %s

// CHECK: PrototypeCallGraph (fully_recovered=yes)
// CHECK: _Z8dispatchv{{x0x[0-9a-f]+}} -> [_Z3barvx0x0, _Z3bazvx0x0]
// CHECK: Incomplete call sites (0):

// Intra-procedural fan-out: a SINGLE indirect call site resolves to MULTIPLE
// constant targets, all discovered within one function (no inter-procedural
// argument threading).
//
// dispatch() picks one of two compile-time callee addresses (getpc + rel32) on
// divergent control flow, spills the chosen pointer to the same scratch slot in
// each branch, then reloads it after the merge and tail-calls it.  The reload
// lifts to a load whose clobber is a MemoryPhi over the two branch stores;
// evalConstTargets unions the stored values, so the dispatch call site resolves
// to BOTH _Z3barv and _Z3bazv.  (A pointer phi'd as two separate 32-bit halves
// is deliberately NOT recombined -- that would risk spurious mixed targets --
// so the fan-out is exercised through the whole-pointer memory form.)
//
// kernel() simply calls dispatch() once.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.amdhsa_code_object_version 6
	.text
	.p2align 2
	.hidden _Z3barv
	.globl  _Z3barv
	.type   _Z3barv,@function
_Z3barv:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_endpgm
.Le0:
	.size _Z3barv, .Le0-_Z3barv
	.set _Z3barv.num_vgpr, 0
	.set _Z3barv.num_agpr, 0
	.set _Z3barv.numbered_sgpr, 32
	.set _Z3barv.private_seg_size, 0
	.set _Z3barv.uses_vcc, 0
	.set _Z3barv.uses_flat_scratch, 0
	.set _Z3barv.has_dyn_sized_stack, 0
	.set _Z3barv.has_recursion, 0
	.set _Z3barv.has_indirect_call, 0

	.p2align 2
	.hidden _Z3bazv
	.globl  _Z3bazv
	.type   _Z3bazv,@function
_Z3bazv:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_endpgm
.Le1:
	.size _Z3bazv, .Le1-_Z3bazv
	.set _Z3bazv.num_vgpr, 0
	.set _Z3bazv.num_agpr, 0
	.set _Z3bazv.numbered_sgpr, 32
	.set _Z3bazv.private_seg_size, 0
	.set _Z3bazv.uses_vcc, 0
	.set _Z3bazv.uses_flat_scratch, 0
	.set _Z3bazv.has_dyn_sized_stack, 0
	.set _Z3bazv.has_recursion, 0
	.set _Z3bazv.has_indirect_call, 0

	; dispatch() — stores &bar or &baz to a scratch slot on divergent paths,
	; reloads, and tail-calls the result (a single indirect site, two targets)
	.p2align 2
	.hidden _Z8dispatchv
	.globl  _Z8dispatchv
	.type   _Z8dispatchv,@function
_Z8dispatchv:
	s_mov_b32 s8, 0
	s_mov_b32 s9, 0
	s_cmp_eq_u32 s6, s7
	s_cbranch_scc1 .Lbar
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z3bazv@rel32@lo+4
	s_addc_u32 s1, s1, _Z3bazv@rel32@hi+12
	s_scratch_store_dwordx2 s[0:1], s[8:9], 0x0
	s_branch .Lmerge
.Lbar:
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z3barv@rel32@lo+4
	s_addc_u32 s1, s1, _Z3barv@rel32@hi+12
	s_scratch_store_dwordx2 s[0:1], s[8:9], 0x0
.Lmerge:
	s_scratch_load_dwordx2 s[6:7], s[8:9], 0x0
	s_waitcnt lgkmcnt(0)
	s_setpc_b64 s[6:7]
.Le2:
	.size _Z8dispatchv, .Le2-_Z8dispatchv
	.set _Z8dispatchv.num_vgpr, 1
	.set _Z8dispatchv.num_agpr, 0
	.set _Z8dispatchv.numbered_sgpr, 32
	.set _Z8dispatchv.private_seg_size, 8
	.set _Z8dispatchv.uses_vcc, 1
	.set _Z8dispatchv.uses_flat_scratch, 0
	.set _Z8dispatchv.has_dyn_sized_stack, 0
	.set _Z8dispatchv.has_recursion, 0
	.set _Z8dispatchv.has_indirect_call, 1

	.protected _Z6kernelv
	.globl  _Z6kernelv
	.p2align 8
	.type   _Z6kernelv,@function
_Z6kernelv:
	s_mov_b32 s32, 0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z8dispatchv@rel32@lo+4
	s_addc_u32 s1, s1, _Z8dispatchv@rel32@hi+12
	s_swappc_b64 s[30:31], s[0:1]
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
.Le3:
	.size _Z6kernelv, .Le3-_Z6kernelv
	.set _Z6kernelv.num_vgpr, max(1, max(_Z3barv.num_vgpr, max(_Z3bazv.num_vgpr, _Z8dispatchv.num_vgpr)))
	.set _Z6kernelv.num_agpr, 0
	.set _Z6kernelv.numbered_sgpr, 33
	.set _Z6kernelv.private_seg_size, 8
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
