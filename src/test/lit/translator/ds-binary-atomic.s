// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_binary_kern.kd \
// RUN:   -initial-execution-point=0:ds_binary_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Group B coverage: DS_ADD_U32 / DS_SUB_U32 / DS_RSUB_U32 / DS_AND_B32 /
// DS_OR_B32 / DS_XOR_B32 / DS_ADD_RTN_U32 must each lower to load+binop+store
// against ptr addrspace(3). RSUB is the outlier with reversed operand order
// (data - tmp instead of tmp - data).

// CHECK: define {{.*}} @ds_binary_kern
// One load+store pair per non-RTN; ADD_RTN_U32 sets vdst (insertelement
// into the reg map). Use *-DAG so order doesn't matter.
// CHECK-DAG: add i32
// CHECK-DAG: sub i32
// CHECK-DAG: and i32
// CHECK-DAG: or i32
// CHECK-DAG: xor i32
// CHECK-DAG: load i32, ptr addrspace(3)
// CHECK-DAG: store i32 {{.*}} ptr addrspace(3)

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_binary_kern
  .p2align  8
  .type   ds_binary_kern,@function
ds_binary_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 42
  ds_add_u32 v0, v1
  ds_sub_u32 v0, v1
  ds_rsub_u32 v0, v1
  ds_and_b32 v0, v1
  ds_or_b32 v0, v1
  ds_xor_b32 v0, v1
  ds_add_rtn_u32 v2, v0, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm

ds_binary_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_binary_kern
    .amdhsa_group_segment_fixed_size 4
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
