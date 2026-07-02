// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:entry_loop.kd \
// RUN:   -initial-execution-point=0:entry_loop.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Tests CodeDiscoveryPass's synthetic-preheader insertion when the entry MBB
// has predecessors (loop entry is the function entry).
//
// The kernel below jumps back to its own entry label, so the first decoded
// MBB has a back-edge from itself. LLVM IR/MIR conventions require the
// entry block to have no predecessors, so CodeDiscoveryPass must splice in
// a new MBB at the front that unconditionally branches to the original
// entry. After lifting, the IR should have a synthetic entry BB whose only
// instruction is an unconditional `br` to the loop-header BB.

// The synthetic entry BB has just the unconditional branch.
// CHECK: define {{.*}} @entry_loop
// CHECK: br label %[[LOOP:[0-9]+]]
// The original entry (loop header) is reachable via the synthetic preheader
// and has a back-edge to itself.
// CHECK: [[LOOP]]:
// CHECK: br {{.*}}label %[[LOOP]]

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  entry_loop
  .p2align  8
  .type   entry_loop,@function
entry_loop:
  s_add_u32 s0, s0, 1
  s_cmp_lt_u32 s0, 10
  s_cbranch_scc1 entry_loop
  s_endpgm
entry_loop_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel entry_loop
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
