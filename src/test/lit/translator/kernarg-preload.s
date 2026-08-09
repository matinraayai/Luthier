// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:preload_kern.kd \
// RUN:   -initial-execution-point=0:preload_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Kernarg-preload coverage (gfx942 has FeatureKernargPreload).
//
// The kernel enables a 2-SGPR kernarg-segment pointer (s0-s1), then preloads 2
// dwords of the kernarg segment — starting at dword OFFSET=1 — into the SGPRs
// immediately after the user SGPRs (s2, s3). The body reads s2/s3 directly (no
// s_load), so the translator must seed those SGPRs with the equivalent
// kernarg-segment loads.
//
// With OFFSET=1 the two preload slots map to kernarg dwords 1 and 2, i.e.
// `kernarg.segment.ptr + i64 1` and `+ i64 2`, each an invariant i32 load.
// (Neither folds to the base pointer, so both GEPs are observable.)

// CHECK: define {{.*}} @preload_kern
// CHECK-DAG: call ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr()
// CHECK-DAG: getelementptr inbounds i32, ptr addrspace(4) %{{[0-9a-zA-Z._]+}}, i64 1
// CHECK-DAG: getelementptr inbounds i32, ptr addrspace(4) %{{[0-9a-zA-Z._]+}}, i64 2
// CHECK-DAG: load i32, ptr addrspace(4) %{{[0-9a-zA-Z._]+}}, align 4, !invariant.load

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  preload_kern
  .p2align  8
  .type   preload_kern,@function
preload_kern:
  // s2, s3 are the preloaded kernarg dwords (offset 1 and 2). Consume both so
  // the seeded loads are kept live. s6 avoids the workgroup-id SGPR (s4).
  s_add_u32 s6, s2, s3
  s_endpgm
preload_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel preload_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 4
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_kernarg_preload_length 2
    .amdhsa_user_sgpr_kernarg_preload_offset 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
