// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:scalar_only.kd \
// RUN:   -initial-execution-point=0:scalar_only.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// A kernel whose only MBB is purely scalar (SALU + s_endpgm) must NOT
// receive an EXEC mask predicate check.

// CHECK: define {{.*}} @scalar_only
// CHECK-NOT: call i32 @llvm.amdgcn.mbcnt.lo
// CHECK-NOT: call i32 @llvm.amdgcn.mbcnt.hi
// CHECK-NOT: exec.is.active

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  scalar_only
  .p2align  8
  .type   scalar_only,@function
scalar_only:
  s_load_dwordx2 s[0:1], s[4:5], 0x0
  s_waitcnt lgkmcnt(0)
  s_add_u32 s0, s0, 1
  s_endpgm

scalar_only_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel scalar_only
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
