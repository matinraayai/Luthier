// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:smem_cmpswap.kd \
// RUN:   -initial-execution-point=0:smem_cmpswap.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// S_ATOMIC_CMPSWAP_IMM_RTN — compare-and-swap. $sdata holds (src, cmp) as
// two i32 halves; cmpxchg returns {value, i1} struct, RTN takes element 0
// (the loaded pre-swap value).
//
// Previously buggy: used AtomicRMWXchg (unconditional swap, dropped cmp).
// Now uses new LLVMAtomicCmpXchg DSL op → IR `cmpxchg` instruction.

// CHECK: define {{.*}} @smem_cmpswap
// Lane extraction: src from element 0, cmp from element 1.
// CHECK-DAG: extractelement {{.*}}, i32 0
// CHECK-DAG: extractelement {{.*}}, i32 1
// CHECK-DAG: cmpxchg ptr addrspace(1) {{.*}} syncscope({{.*}}) monotonic monotonic
// CHECK-DAG: extractvalue { i32, i1 }

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  smem_cmpswap
  .p2align  8
  .type   smem_cmpswap,@function
smem_cmpswap:
  s_atomic_cmpswap s[4:5], s[0:1], 0x10 glc
  s_waitcnt lgkmcnt(0)
  s_endpgm

smem_cmpswap_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel smem_cmpswap
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
