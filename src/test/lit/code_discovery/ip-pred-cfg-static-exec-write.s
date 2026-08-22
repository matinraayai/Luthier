// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,luthier-print-ip-pred-cfg' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:static_exec_kern.kd \
// RUN:   -initial-execution-point=0:static_exec_kern.kd \
// RUN:   -o /dev/null | %tee_out FileCheck %s

// EXEC-mask flipping through a compile-time constant. The kernel writes
// EXEC = 0x3 (only two lanes active) before entering a vector MBB.
// CodeDiscoveryPass splits the MF on every EXEC write and also *after*
// every EXEC write (so subsequent MIs see a consistent EXEC state at
// MBB entry, not a mid-MBB EXEC transition). That gives four MBBs
// here:
//
//   .0 (scalar entry): the S_MOV that writes EXEC.
//   .1 (scalar): S_BRANCH to the vector tail.
//   .2 (vector): v_mov.
//   .3 (scalar): s_endpgm.
//
// The IR translator wraps .2 in a Check/Skip scaffold whose ExecVal is
// a PHI collapsing to the (constant, non-all-ones) 3; Phase 2 threads
// the walk through whatever scaffold survives to produce scaffold-
// transparent edges, so scaffold BBs never surface as PredMBBs.

// .0: entry MBB, EXEC-writing S_MOV alone.
// CHECK: Predecessors: []
// CHECK: $exec = S_MOV_B64 3
// CHECK: Successors: [static_exec_kern:.1]

// .1: scalar MBB with just the unconditional branch.
// CHECK: Predecessors: [static_exec_kern:.0]
// CHECK: S_BRANCH
// CHECK: Successors: [static_exec_kern:{{[^]]*}}]

// .2: vector tail — v_mov under the narrow EXEC.
// CHECK: Predecessors: [static_exec_kern:.1]
// CHECK: V_MOV_B32
// CHECK: Successors: [static_exec_kern:.3]

// .3: s_endpgm tail; Phase 2 threads the diamond skip edge through so
// this block ends up with an over-approximated predecessor list.
// CHECK: Predecessors: [static_exec_kern:{{[^]]*}}]
// CHECK: S_ENDPGM
// CHECK: Successors: []

// No fifth block: scaffold BBs still never surface as PredMBBs.
// CHECK-NOT: Predecessors:
// CHECK-NOT: Successors:

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  static_exec_kern
  .p2align  8
  .type   static_exec_kern,@function
static_exec_kern:
  s_mov_b64 exec, 3
  s_branch label1
label1:
  v_mov_b32_e32 v0, 7
  s_endpgm

static_exec_kern_end:
  .size   static_exec_kern, static_exec_kern_end-static_exec_kern

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel static_exec_kern
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
