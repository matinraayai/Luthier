// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:sopp_icache.kd \
// RUN:   -initial-execution-point=0:sopp_icache.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// S_ICACHE_INV — instruction-cache invalidate. Was empty Semantic = [];
// now models as s_waitcnt(0) barrier (HW-only effect; no IR analog for
// I-cache invalidate, but the sequencing-barrier conveys the boundary).

// CHECK: define {{.*}} @sopp_icache
// CHECK: call void @llvm.amdgcn.s.waitcnt

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  sopp_icache
  .p2align  8
  .type   sopp_icache,@function
sopp_icache:
  s_icache_inv
  s_endpgm

sopp_icache_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel sopp_icache
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 4
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
