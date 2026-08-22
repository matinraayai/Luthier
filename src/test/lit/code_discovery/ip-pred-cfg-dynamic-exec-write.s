// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print),luthier-print-ip-pred-cfg' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:dyn_exec_kern.kd \
// RUN:   -initial-execution-point=0:dyn_exec_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// EXEC written to a statically non-deterministic value (loaded from
// memory). CodeDiscoveryPass splits both *before* and *after* every EXEC
// write, so we end up with:
//
//   .0 (scalar):  s_load_dwordx2 + s_waitcnt
//   .1 (scalar):  $exec = S_MOV_B64 $sgpr0_sgpr1  (EXEC-writing MBB, alone)
//   .2 (scalar):  s_branch
//   .3 (vector):  v_mov
//   .4 (scalar):  s_endpgm
//
// The vector MBB's translated IR must carry a Check/Skip scaffold whose
// ExecVal is the loaded 64-bit value: with no compile-time proof of
// all-ones, foldTriviallyActiveExecChecks must not collapse the check.
// Phase 2's walk still resolves the scaffold-transparent edges.

// .0: entry, S_LOAD + S_WAITCNT.
// CHECK: Predecessors: []
// CHECK: S_LOAD_DWORDX2_IMM
// CHECK: S_WAITCNT
// CHECK: Successors: [dyn_exec_kern:.1]

// .1: the EXEC-writing MBB, alone (split before AND after the EXEC write).
// CHECK: Predecessors: [dyn_exec_kern:.0]
// CHECK: $exec = S_MOV_B64 $sgpr0_sgpr1
// CHECK: Successors: [dyn_exec_kern:.2]

// .2: scalar branch alone.
// CHECK: Predecessors: [dyn_exec_kern:.1]
// CHECK: S_BRANCH
// CHECK: Successors: [dyn_exec_kern:{{[^]]*}}]

// .3: vector tail — v_mov under the runtime-loaded EXEC.
// CHECK: Predecessors: [dyn_exec_kern:.2]
// CHECK: V_MOV_B32
// CHECK: Successors: [dyn_exec_kern:.4]

// .4: s_endpgm; Phase 2's scaffold-transparent walk gives over-
// approximated predecessors here.
// CHECK: Predecessors: [dyn_exec_kern:{{[^]]*}}]
// CHECK: S_ENDPGM
// CHECK: Successors: []

// -- IR translator: after the PredCFG dump, target(print) writes the
// translated IR. The exec-check scaffold must survive because EXEC is a
// runtime-loaded value — the mbcnt lane-id computation and the per-lane
// conditional branch to a skip block must all remain.
// CHECK: call i32 @llvm.amdgcn.mbcnt.lo
// CHECK: call i32 @llvm.amdgcn.mbcnt.hi
// CHECK: br i1 %{{[a-zA-Z0-9._]+}}, label %{{[a-zA-Z0-9._]+}}, label %{{[a-zA-Z0-9._]*skip[a-zA-Z0-9._]*}}

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  dyn_exec_kern
  .p2align  8
  .type   dyn_exec_kern,@function
dyn_exec_kern:
  s_load_dwordx2 s[0:1], s[4:5], 0x0
  s_waitcnt lgkmcnt(0)
  s_mov_b64 exec, s[0:1]
  s_branch label1
label1:
  v_mov_b32_e32 v0, 7
  s_endpgm

dyn_exec_kern_end:
  .size   dyn_exec_kern, dyn_exec_kern_end-dyn_exec_kern

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel dyn_exec_kern
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
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
