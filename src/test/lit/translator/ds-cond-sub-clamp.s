// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: (luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 \
// RUN:    '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:    -code-object-paths=%t \
// RUN:    -initial-entrypoint=0:ds_csubclamp_kern.kd \
// RUN:    -initial-execution-point=0:ds_csubclamp_kern.kd \
// RUN:    -o /dev/null 2>&1 || true) > %t.out && \
// RUN: FileCheck %s < %t.out

// Group L coverage: RDNA4 DS_COND_SUB_U32 and DS_SUB_CLAMP_U32.
//   COND_SUB:  MEM = (tmp >= data) ? tmp - data : tmp
//   SUB_CLAMP: MEM = (tmp <  data) ? 0          : tmp - data

// CHECK: define {{.*}} @ds_csubclamp_kern
// CHECK-DAG: icmp uge i32
// CHECK-DAG: icmp ult i32
// CHECK-DAG: sub i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
  .globl  ds_csubclamp_kern
  .p2align  8
  .type   ds_csubclamp_kern,@function
ds_csubclamp_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 10
  ds_cond_sub_u32 v0, v1
  ds_sub_clamp_u32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm
ds_csubclamp_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_csubclamp_kern
    .amdhsa_group_segment_fixed_size 4
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 1
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
