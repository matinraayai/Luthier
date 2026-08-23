// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+lds-barrier-arrive-atomic -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+lds-barrier-arrive-atomic \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_barrier_kern.kd \
// RUN:   -initial-execution-point=0:ds_barrier_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// LDS atomic barrier-arrive helpers (gfx1250+ HasLdsBarrierArriveAtomic):
//   DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64
//     -> int_amdgcn_ds_atomic_async_barrier_arrive_b64(ptr addrspace(3))  void
//   DS_ATOMIC_BARRIER_ARRIVE_RTN_B64
//     -> int_amdgcn_ds_atomic_barrier_arrive_rtn_b64(ptr addrspace(3), i64)  i64

// CHECK: define {{.*}} @ds_barrier_kern
// CHECK-DAG: call void @llvm.amdgcn.ds.atomic.async.barrier.arrive.b64(ptr addrspace(3)
// CHECK-DAG: call i64 @llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64(ptr addrspace(3)

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
  .globl  ds_barrier_kern
  .p2align  8
  .type   ds_barrier_kern,@function
ds_barrier_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v2, 1
  v_mov_b32_e32 v3, 2
  // Async barrier arrive: addr in v0; no return value.
  ds_atomic_async_barrier_arrive_b64 v0
  s_wait_kmcnt 0
  // RTN form: addr in v0, data in even-aligned v[2:3], result in v[4:5].
  ds_atomic_barrier_arrive_rtn_b64 v[4:5], v0, v[2:3]
  s_wait_kmcnt 0
  s_endpgm
ds_barrier_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_barrier_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 6
    .amdhsa_next_free_sgpr 1
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
