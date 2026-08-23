// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,luthier-print-ip-pred-cfg' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ip_pred_cfg_kern.kd \
// RUN:   -initial-execution-point=0:ip_pred_cfg_kern.kd \
// RUN:   -o /dev/null | %tee_out FileCheck %s

// Tests the simple non-exec writing case for predicated CFG.

// Entry: scalar block, no predecessors, unconditional S_BRANCH to the tail.
// CHECK: Predecessors: []
// CHECK: S_ADD_U32
// CHECK: S_BRANCH
// The entry MBB has exactly one successor — that's the tail MBB, entered
// through its CheckBB scaffold in the translated IR.
// CHECK: Successors: [ip_pred_cfg_kern:{{[a-zA-Z0-9._]+}}]

// Tail: predecessor is the entry MBB (ip_pred_cfg_kern:.0); the tail is a
// vector MBB (V_MOV_B32) that ends in S_ENDPGM and has no CFG successors.
// CHECK: Predecessors: [ip_pred_cfg_kern:.0]
// CHECK: V_MOV_B32
// CHECK: S_ENDPGM
// CHECK: Successors: []

// After the two MBBs there must be no third block: scaffold BBs never
// surface as PredicatedMachineBasicBlocks.
// CHECK-NOT: Predecessors:
// CHECK-NOT: Successors:

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ip_pred_cfg_kern
  .p2align  8
  .type   ip_pred_cfg_kern,@function
ip_pred_cfg_kern:
  s_add_u32 s0, s0, 1
  s_branch tail
tail:
  v_mov_b32_e32 v0, 7
  s_endpgm

ip_pred_cfg_kern_end:
  .size   ip_pred_cfg_kern, ip_pred_cfg_kern_end-ip_pred_cfg_kern

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ip_pred_cfg_kern
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
