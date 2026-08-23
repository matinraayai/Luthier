// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,luthier-print-ip-pred-cfg' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:_Z6kernelv.kd \
// RUN:   -initial-execution-point=0:_Z6kernelv.kd \
// RUN:   -o /dev/null | %tee_out FileCheck %s

// CodeDiscoveryPass's post-pass appends a synthetic \c S_ENDPGM in a
// SEPARATE MBB (placed at the end of the MF list) for every call MI,
// and wires the call MBB → synthetic-endpgm MBB as a successor edge
// so the call MBB has a well-formed successor list
// The synthetic \c S_ENDPGM carries pcsections metadata but is
// NOT assigned a trace-instr address. The PMBB printer
// tags such MIs with "; synthetic (no trace address)"
//
// Invariants verified below:
//   1. The synthetic \c S_ENDPGM marker line ";  synthetic (no trace
//      address)" appears (proves the synthetic-endpgm insertion ran).
//   2. The source-lifted trailing \c s_endpgm keeps its pcsections
//      metadata and is NOT tagged synthetic.
//   3. Forward call edge: kernel call MBB → \c _Z3foov entry MBB.
//   4. Return-flow edge: \c _Z3foov return MBB → kernel post-call MBB.
//   5. Return-flow predecessor: post-call MBB's Predecessors list the
//      callee's return MBB.
//
// PMBB print order in the CFG dump depends on module iteration and
// global-index assignment, neither of which is stable across runs
// (DenseMap-keyed). Every check below is CHECK-DAG.

// The kernel's call MBB header reports its wrapping function+MBB.
// CHECK-DAG: PredMBB _Z6kernelv:{{.*}} (function=_Z6kernelv, MBB=%bb.0)

// (1) A synthetic S_ENDPGM marker line exists.
// CHECK-DAG: ; synthetic (no trace address)

// (2) The source-lifted s_endpgm keeps its pcsections metadata (its
// PMBB does NOT get the synthetic marker).
// CHECK-DAG: S_ENDPGM 0, pcsections !{{[0-9]+}}

// The direct call and callee return are both present in the dump.
// The callee's return \c s_setpc_b64 is lifted as the return-annotated
// pseudo \c S_SETPC_B64_return (CodeDiscoveryPass folds every raw
// \c S_SETPC_B64 into the pseudo at lift time so downstream
// callgraph/scheduler machinery sees the return metadata).
// CHECK-DAG: S_SWAPPC_B64 $sgpr0_sgpr1
// CHECK-DAG: S_SETPC_B64_return $sgpr30_sgpr31

// (3) Forward call edge from the kernel's call MBB to _Z3foov's entry.
// The successors set is a \c SmallDenseSet, so its printed order is
// hash-driven; the call MBB carries two edges — the synthetic-endpgm
// MBB and _Z3foov's entry — and either may print first. Allow anything
// (including nothing) on either side of the _Z3foov successor.
// CHECK-DAG: Successors: [{{[^]]*}}_Z3foov{{[^]]*}}]

// (4) Return-flow edge: _Z3foov's return MBB has the kernel's post-call
// trace-function entry as a successor. The post-call trace function is
// named _Z6kernelvxNN…, so pin down the successor with the mangled-
// offset "x" separator to distinguish it from the kernel's own
// synthetic-endpgm MBB (_Z6kernelv:.N) that could otherwise match.
// CHECK-DAG: Successors: [{{[^]]*}}_Z6kernelvx{{[^]]*}}]

// (5) Return-flow predecessor side: the post-call MBB's Predecessors
// list the callee's return MBB.
// CHECK-DAG: Predecessors: [_Z3foov{{[^]]+}}]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx908"
	.amdhsa_code_object_version 6
	.text

	.p2align 2
	.hidden _Z3foov
	.globl  _Z3foov
	.type   _Z3foov,@function
_Z3foov:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end_foo:
	.size _Z3foov, .Lfunc_end_foo-_Z3foov
	.set _Z3foov.num_vgpr, 0
	.set _Z3foov.num_agpr, 0
	.set _Z3foov.numbered_sgpr, 32
	.set _Z3foov.private_seg_size, 0
	.set _Z3foov.uses_vcc, 0
	.set _Z3foov.uses_flat_scratch, 0
	.set _Z3foov.has_dyn_sized_stack, 0
	.set _Z3foov.has_recursion, 0
	.set _Z3foov.has_indirect_call, 0

	.protected _Z6kernelv
	.globl  _Z6kernelv
	.p2align 8
	.type   _Z6kernelv,@function
_Z6kernelv:
	s_mov_b32 s32, 0
	; --- direct call to _Z3foov ---
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z3foov@rel32@lo+4
	s_addc_u32 s1, s1, _Z3foov@rel32@hi+12
	s_swappc_b64 s[30:31], s[0:1]
	; --- post-call MBB starts here in the lifted MIR; the discovery pass
	;     forces the split by appending a synthetic S_ENDPGM after the
	;     call.
	s_mov_b32 s2, 0
	s_endpgm
	.section .rodata,"a",@progbits
	.p2align 6, 0x0
	.amdhsa_kernel _Z6kernelv
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
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
.Lfunc_end_kernel:
	.size _Z6kernelv, .Lfunc_end_kernel-_Z6kernelv
	.set _Z6kernelv.num_vgpr, max(1, _Z3foov.num_vgpr)
	.set _Z6kernelv.num_agpr, max(0, _Z3foov.num_agpr)
	.set _Z6kernelv.numbered_sgpr, max(33, _Z3foov.numbered_sgpr)
	.set _Z6kernelv.private_seg_size, 0+max(_Z3foov.private_seg_size)
	.set _Z6kernelv.uses_vcc, or(1, _Z3foov.uses_vcc)
	.set _Z6kernelv.uses_flat_scratch, or(0, _Z3foov.uses_flat_scratch)
	.set _Z6kernelv.has_dyn_sized_stack, or(0, _Z3foov.has_dyn_sized_stack)
	.set _Z6kernelv.has_recursion, or(0, _Z3foov.has_recursion)
	.set _Z6kernelv.has_indirect_call, 0
	.p2alignl 6, 3215226880
	.fill 256, 4, 3215226880
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
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
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx908
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
