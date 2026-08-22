// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print),luthier-print-ip-pred-cfg' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:dyn_exec_kern_gfx1030.kd \
// RUN:   -initial-execution-point=0:dyn_exec_kern_gfx1030.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Same shape as ip-pred-cfg-dynamic-exec-write.s but for gfx1030, where
// ExecBaseReg is SGPR_NULL instead of M0. Both bases share the "generic
// enum carries HWEncoding=0" pattern, so this test guards the audit-
// broadened \c getRegFileKey fix (subtarget-specific base resolution via
// \c getPhysReg) against a regression on GFX10+. The scaffold must
// survive here for the same reason as gfx908: EXEC is a runtime-loaded
// value, so the Trace function translator must leave the per-lane
// predicate check intact.

// PredCFG: CodeDiscoveryPass splits both before AND after every EXEC
// write, so the EXEC-writing MBB lands alone between the loader and the
// scalar branch. Layout: .0 loader / .1 EXEC-write / .2 branch / .3
// vector v_mov / .4 s_endpgm. Phase 2 threads the scaffold-transparent
// diamond edges through to over-approximate the s_endpgm predecessors.
// CHECK: Predecessors: []
// CHECK: S_LOAD_DWORD
// CHECK: S_WAITCNT
// CHECK: Successors: [dyn_exec_kern_gfx1030:.1]
// CHECK: Predecessors: [dyn_exec_kern_gfx1030:.0]
// CHECK: $exec_lo = S_MOV_B32 $sgpr0
// CHECK: Successors: [dyn_exec_kern_gfx1030:.2]
// CHECK: Predecessors: [dyn_exec_kern_gfx1030:.1]
// CHECK: S_BRANCH
// CHECK: Successors: [dyn_exec_kern_gfx1030:{{[^]]*}}]
// CHECK: Predecessors: [dyn_exec_kern_gfx1030:.2]
// CHECK: V_MOV_B32
// CHECK: Successors: [dyn_exec_kern_gfx1030:.4]
// CHECK: Predecessors: [dyn_exec_kern_gfx1030:{{[^]]*}}]
// CHECK: S_ENDPGM
// CHECK: Successors: []

// IR translator: exec-check scaffold must survive. Wave32 target, so only
// mbcnt.lo is emitted (mbcnt.hi must NOT appear).
// CHECK: call i32 @llvm.amdgcn.mbcnt.lo
// CHECK-NOT: call i32 @llvm.amdgcn.mbcnt.hi
// CHECK: br i1 %{{[a-zA-Z0-9._]+}}, label %{{[a-zA-Z0-9._]+}}, label %{{[a-zA-Z0-9._]*skip[a-zA-Z0-9._]*}}

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
  .globl  dyn_exec_kern_gfx1030
  .p2align  8
  .type   dyn_exec_kern_gfx1030,@function
dyn_exec_kern_gfx1030:
  s_load_dword s0, s[4:5], 0x0
  s_waitcnt lgkmcnt(0)
  s_mov_b32 exec_lo, s0
  s_branch label1
label1:
  v_mov_b32_e32 v0, 7
  s_endpgm

dyn_exec_kern_gfx1030_end:
  .size   dyn_exec_kern_gfx1030, dyn_exec_kern_gfx1030_end-dyn_exec_kern_gfx1030

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel dyn_exec_kern_gfx1030
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 4
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
    .amdhsa_next_free_sgpr 4
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
  .end_amdhsa_kernel
