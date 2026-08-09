// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:wave32_vec.kd \
// RUN:   -initial-execution-point=0:wave32_vec.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// On wave32 targets (gfx10+), the EXEC register is 32 bits wide and only
// mbcnt.lo is needed to compute the lane id; mbcnt.hi must NOT be emitted.

// CHECK: define {{.*}} @wave32_vec
// CHECK-DAG: call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
// CHECK-DAG: lshr i32 {{.*}}, %{{.*}}
// CHECK-DAG: trunc i32 %{{.*}} to i1
// CHECK-DAG: br i1 %{{.*}}, label %{{.*}}, label %skip
// CHECK-NOT: call i32 @llvm.amdgcn.mbcnt.hi

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .globl  wave32_vec
  .p2align  8
  .type   wave32_vec,@function
wave32_vec:
  v_add_u32_e32 v0, 1, v0
  s_endpgm

wave32_vec_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel wave32_vec
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 0
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_queue_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 0
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_user_sgpr_private_segment_size 0
    .amdhsa_uses_dynamic_stack 0
    .amdhsa_enable_private_segment 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 1
    .amdhsa_reserve_vcc 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_fp16_overflow 0
    .amdhsa_wavefront_size32 1
    .amdhsa_workgroup_processor_mode 1
    .amdhsa_memory_ordered 1
    .amdhsa_forward_progress 0
  .end_amdhsa_kernel
