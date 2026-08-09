// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_wrap_kern.kd \
// RUN:   -initial-execution-point=0:ds_wrap_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// DSSem fix #5: DS_WRAP_RTN_B32 should implement the ring-buffer wrap formula:
//   MEM = (old >= data0) ? old - data0 : old + data1
// Previously had empty `Semantic = []`.

// CHECK: define {{.*}} @ds_wrap_kern
// CHECK-DAG: icmp uge i32
// CHECK-DAG: sub i32
// CHECK-DAG: add i32
// CHECK-DAG: select i1

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_wrap_kern
  .p2align  8
  .type   ds_wrap_kern,@function
ds_wrap_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 100
  v_mov_b32_e32 v2, 7
  ds_wrap_rtn_b32 v3, v0, v1, v2
  s_waitcnt lgkmcnt(0)
  s_endpgm

ds_wrap_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_wrap_kern
    .amdhsa_group_segment_fixed_size 4
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
