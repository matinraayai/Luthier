// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:smem_bcmpswap.kd \
// RUN:   -initial-execution-point=0:smem_bcmpswap.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// S_BUFFER_ATOMIC_CMPSWAP — scalar buffer compare-and-swap. $sdata holds
// (src, cmp) in an SReg_64 pair; intrinsic returns i32; $sdst is SReg_64
// so we ZExt to i64.

// CHECK: define {{.*}} @smem_bcmpswap
// CHECK-DAG: call i32 @llvm.amdgcn.struct.buffer.atomic.cmpswap.i32
// CHECK-DAG: zext i32 {{.*}} to i64

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  smem_bcmpswap
  .p2align  8
  .type   smem_bcmpswap,@function
smem_bcmpswap:
  s_buffer_atomic_cmpswap s[4:5], s[0:3], 0x10 glc
  s_waitcnt lgkmcnt(0)
  s_endpgm

smem_bcmpswap_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel smem_bcmpswap
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
