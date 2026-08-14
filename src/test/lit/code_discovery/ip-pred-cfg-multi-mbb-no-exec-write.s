// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,luthier-print-ip-pred-cfg' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:multi_mbb_no_exec_kern.kd \
// RUN:   -initial-execution-point=0:multi_mbb_no_exec_kern.kd \
// RUN:   -o /dev/null | %tee_out FileCheck %s

// Multi-MBB kernel with no EXEC writes. Because EXEC is the kernel-entry
// all-ones value at every vector MBB, foldTriviallyActiveExecChecks folds
// every Check/Skip scaffold, and Phase 2's IR-successor walk sees straight
// edges between MBBs. The kernel shape:
//
//   entry (scalar) --cbranch--> alt
//                    |            |
//                    v            v
//                  fall_thru    alt (both vector, both end in s_endpgm)
//
// The conditional S_CBRANCH_SCC1 splits entry into two successors; the two
// leaves each have zero successors.

// Entry MBB: scalar, no predecessors, ends in the conditional branch that
// picks between the fall-through vector MBB and the branch-target vector
// MBB. Both arms must appear in the successor list.
// CHECK: Predecessors: []
// CHECK: S_CMP_LT_U32
// CHECK: S_CBRANCH_SCC1
// CHECK: Successors: [multi_mbb_no_exec_kern:{{[a-zA-Z0-9._]+}}, multi_mbb_no_exec_kern:{{[a-zA-Z0-9._]+}}]

// Each of the two leaf MBBs: entry (globalidx 0) as sole predecessor,
// a VALU op, and s_endpgm with no successors.
// CHECK: Predecessors: [multi_mbb_no_exec_kern:.0]
// CHECK: V_MOV_B32
// CHECK: S_ENDPGM
// CHECK: Successors: []

// CHECK: Predecessors: [multi_mbb_no_exec_kern:.0]
// CHECK: V_MOV_B32
// CHECK: S_ENDPGM
// CHECK: Successors: []

// After the three MBBs no fourth block appears: scaffold BBs must not
// surface as separate PredicatedMachineBasicBlocks.
// CHECK-NOT: Predecessors:
// CHECK-NOT: Successors:

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  multi_mbb_no_exec_kern
  .p2align  8
  .type   multi_mbb_no_exec_kern,@function
multi_mbb_no_exec_kern:
  s_add_u32 s0, s0, 1
  s_cmp_lt_u32 s0, 8
  s_cbranch_scc1 alt
  v_mov_b32_e32 v0, 7
  s_endpgm
alt:
  v_mov_b32_e32 v1, 11
  s_endpgm

multi_mbb_no_exec_kern_end:
  .size   multi_mbb_no_exec_kern, multi_mbb_no_exec_kern_end-multi_mbb_no_exec_kern

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel multi_mbb_no_exec_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_queue_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_user_sgpr_flat_scratch_init 0
    .amdhsa_user_sgpr_private_segment_size 0
    .amdhsa_uses_dynamic_stack 0
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
