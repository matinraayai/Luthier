// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:entry_mid.kd \
// RUN:   -initial-execution-point=0:entry_mid.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Tests CodeDiscoveryPass's "entry in middle" synthesis: when the lowest
// address in the MF's trace group is strictly less than the entry-point
// address, the pass splices a synthetic preheader MBB at the front that
// unconditionally branches to the MBB containing the entry instruction.
//
// Construction: the .text section starts with a small region (preamble_target)
// of code laid out BEFORE the kernel symbol's first byte. The kernel itself
// has a backward branch into that earlier region. The trace decoder seeds a
// new trace at the backward-branch target, whose start address is below the
// kernel entry. After merge, FirstTraceStartAddr < EntryPointAddress, which
// fires the entry-in-middle synthesis.

// CHECK: define {{.*}} @entry_mid
// Synthetic entry BB: a single unconditional branch to the actual entry
// instruction's BB.
// CHECK: br label %[[ENTRY_BODY:[0-9]+]]
// CHECK: [[ENTRY_BODY]]:

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .p2align  8
preamble_target:
  s_add_u32 s0, s0, 1
  s_branch entry_mid
  .globl  entry_mid
  .type   entry_mid,@function
entry_mid:
  s_add_u32 s0, s0, 2
  s_cmp_lt_u32 s0, 8
  s_cbranch_scc1 preamble_target
  s_endpgm
entry_mid_end:
  .size   entry_mid, entry_mid_end-entry_mid

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel entry_mid
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
