// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: (luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 \
// RUN:    -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:    '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:    -code-object-paths=%t \
// RUN:    -initial-entrypoint=0:global_fp_kern.kd \
// RUN:    -initial-execution-point=0:global_fp_kern.kd \
// RUN:    -o /dev/null 2>&1 || true) > %t.out && \
// RUN: FileCheck %s < %t.out

// FP / CMPSWAP / specialty GLOBAL atomics: exercises foreach-compressed
// records to verify they emit correct IR. The kernel is single-MBB (no flow
// control) so the FP-atomic load/fadd/store and FCmp paths are exercised
// uniformly.

// CHECK: define {{.*}} @global_fp_kern

// GLOBAL_ATOMIC_ADD_F32: bitcast → load float → fadd → bitcast → store i32.
// CHECK-DAG: load float, ptr addrspace(1)
// CHECK-DAG: bitcast i32 {{.*}} to float
// CHECK-DAG: fadd float
// CHECK-DAG: bitcast float {{.*}} to i32

// GLOBAL_ATOMIC_FMIN: select on FCmp OLT.
// CHECK-DAG: fcmp olt float

// GLOBAL_ATOMIC_CMPSWAP: extract v2i32 lanes, ICmp EQ, select, store.
// CHECK-DAG: extractelement <2 x i32>
// CHECK-DAG: icmp eq i32

// GLOBAL_ATOMIC_COND_SUB_U32: ICmp UGE, select, sub.
// CHECK-DAG: icmp uge i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
  .globl  global_fp_kern
  .p2align  8
  .type   global_fp_kern,@function
global_fp_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v2, 0x40400000  // 3.0f
  v_mov_b32_e32 v3, 7           // u32 RHS
  v_mov_b32_e32 v4, 0x3f800000  // 1.0f (src for cmpswap)
  v_mov_b32_e32 v5, 0x40000000  // 2.0f (cmp for cmpswap)
  // FP add
  global_atomic_add_f32 v[0:1], v2, off
  s_waitcnt vmcnt(0) lgkmcnt(0)
  // FP min (FMIN_F32)
  global_atomic_fmin v[0:1], v2, off
  s_waitcnt vmcnt(0) lgkmcnt(0)
  // CMPSWAP — v[4:5] holds {src=v4, cmp=v5}
  global_atomic_cmpswap v[0:1], v[4:5], off
  s_waitcnt vmcnt(0) lgkmcnt(0)
  // COND_SUB_U32
  global_atomic_cond_sub_u32 v[0:1], v3, off
  s_waitcnt vmcnt(0) lgkmcnt(0)
  s_endpgm
global_fp_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel global_fp_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 4
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
