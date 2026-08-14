// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:bb_exit_kern.kd \
// RUN:   -initial-execution-point=0:bb_exit_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// CHECK: define {{.*}} @bb_exit_kern{{.*}} !luthier.bb_exit_reg_map ![[#EXIT:]]
// CHECK: ![[#EXIT]] = !{
// The vector kernel body materializes a check BB (with its own entry) and
// a body BB (with the VM snapshot).
// CHECK-DAG: !{ptr blockaddress(@bb_exit_kern, %{{[a-zA-Z0-9._]*}}check{{[a-zA-Z0-9._]*}}), !{{[0-9]+}}}
// SkipBBs are intentionally omitted from the exit map.
// CHECK-NOT: !{ptr blockaddress(@bb_exit_kern, %{{[a-zA-Z0-9._]*}}skip{{[a-zA-Z0-9._]*}})
// The per-slice tuples reuse the entry_reg_map shape.
// CHECK-DAG: !{i32 %{{[a-zA-Z0-9_.]+}}, !"vgpr0", i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 2}
// CHECK-DAG: !{i32 %{{[a-zA-Z0-9_.]+}}, !"vgpr1", i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 2}

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  bb_exit_kern
  .p2align  8
  .type   bb_exit_kern,@function
bb_exit_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 1
  v_add_u32_e32 v2, v0, v1
  s_endpgm

bb_exit_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel bb_exit_kern
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
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
