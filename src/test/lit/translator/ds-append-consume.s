// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_appcon_kern.kd \
// RUN:   -initial-execution-point=0:ds_appcon_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// DSSem fix #2: DS_APPEND / DS_CONSUME lower to llvm.amdgcn.ds.{append,consume}
// with a ptr addrspace(3) constructed from the offset (NOT from a nonexistent
// $addr operand).

// CHECK: define {{.*}} @ds_appcon_kern
// CHECK-DAG: call i32 @llvm.amdgcn.ds.append.p3(ptr addrspace(3)
// CHECK-DAG: call i32 @llvm.amdgcn.ds.consume.p3(ptr addrspace(3)

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_appcon_kern
  .p2align  8
  .type   ds_appcon_kern,@function
ds_appcon_kern:
  ds_append v0 offset:0
  s_waitcnt lgkmcnt(0)
  ds_consume v1 offset:0
  s_waitcnt lgkmcnt(0)
  s_endpgm

ds_appcon_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_appcon_kern
    .amdhsa_group_segment_fixed_size 16
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
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
