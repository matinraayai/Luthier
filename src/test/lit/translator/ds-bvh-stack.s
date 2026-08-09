// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_bvh_kern.kd \
// RUN:   -initial-execution-point=0:ds_bvh_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// BVH-stack coverage: each DS_BVH_STACK_* pseudo lowers to its dedicated
// LLVM intrinsic returning a {vdst, i32-addr} struct, split into the
// in-out $addr register and the output $vdst register via extractvalue.
//   DS_BVH_STACK_RTN_B32             -> int_amdgcn_ds_bvh_stack_rtn
//   DS_BVH_STACK_PUSH8_POP1_RTN_B32  -> int_amdgcn_ds_bvh_stack_push8_pop1_rtn
//   DS_BVH_STACK_PUSH8_POP2_RTN_B64  -> int_amdgcn_ds_bvh_stack_push8_pop2_rtn

// CHECK: define {{.*}} @ds_bvh_kern
// CHECK-DAG: call { i32, i32 } @llvm.amdgcn.ds.bvh.stack.rtn(
// CHECK-DAG: call { i32, i32 } @llvm.amdgcn.ds.bvh.stack.push8.pop1.rtn(
// CHECK-DAG: call { i64, i32 } @llvm.amdgcn.ds.bvh.stack.push8.pop2.rtn(
// Each struct result is split:
// CHECK-DAG: extractvalue { i32, i32 } %{{[0-9]+}}, 0
// CHECK-DAG: extractvalue { i32, i32 } %{{[0-9]+}}, 1
// CHECK-DAG: extractvalue { i64, i32 } %{{[0-9]+}}, 0
// CHECK-DAG: extractvalue { i64, i32 } %{{[0-9]+}}, 1

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
  .globl  ds_bvh_kern
  .p2align  8
  .type   ds_bvh_kern,@function
ds_bvh_kern:
  // v0 = addr (in-out), v1 = data0 (last-node-ptr), v[2:5] = data1 v4i32 for STACK_RTN
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0xffffffff
  v_mov_b32_e32 v2, 0x1
  v_mov_b32_e32 v3, 0x2
  v_mov_b32_e32 v4, 0x3
  v_mov_b32_e32 v5, 0x4
  ds_bvh_stack_rtn_b32 v10, v0, v1, v[2:5]
  s_waitcnt lgkmcnt(0)
  // v[2:9] = data1 v8i32 for PUSH8_POP1
  v_mov_b32_e32 v6, 0x5
  v_mov_b32_e32 v7, 0x6
  v_mov_b32_e32 v8, 0x7
  v_mov_b32_e32 v9, 0x8
  ds_bvh_stack_push8_pop1_rtn_b32 v11, v0, v1, v[2:9]
  s_waitcnt lgkmcnt(0)
  // PUSH8_POP2 returns 64-bit vdst (v[12:13])
  ds_bvh_stack_push8_pop2_rtn_b64 v[12:13], v0, v1, v[2:9]
  s_waitcnt lgkmcnt(0)
  s_endpgm

ds_bvh_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_bvh_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 14
    .amdhsa_next_free_sgpr 1
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
