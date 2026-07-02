// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:wo_kern.kd \
// RUN:   -initial-execution-point=0:wo_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Private-segment wave byte offset on a NON-architected-flat-scratch target
// (gfx908). The backend allocates this system SGPR for every entry function
// when !flatScratchIsArchitected() (unconditionally — NOT gated on the
// architected-only ENABLE_PRIVATE_SEGMENT bit), as the last system SGPR. So
// CodeDiscoveryPass must allocate + the translator must seed it here.
//
// Layout: user SGPRs s0-s3 (private-seg buffer) + s4-s5 (kernarg ptr); system
// SGPRs: workgroup id X at s6, wave byte offset at s7. The body reads both.
//
// s6 (wg id X) lifts to the workgroup.id.x intrinsic. s7 (wave byte offset)
// has no intrinsic, so it is seeded as a frozen-poison PLACEHOLDER carrying
// register provenance (!luthier.reg + an "sgpr7" reg-value node). Without the
// allocation/seed, reading s7 would produce a bare `freeze i32 poison` with no
// provenance — so the provenance is the signal that the seed fired.

// CHECK: define {{.*}} @wo_kern
// CHECK-DAG: call i32 @llvm.amdgcn.workgroup.id.x()
// CHECK-DAG: freeze i32 poison, !luthier.reg
// CHECK-DAG: !{!"sgpr7",

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  wo_kern
  .p2align  8
  .type   wo_kern,@function
wo_kern:
  // s7 = wave byte offset (placeholder), s6 = workgroup id X. Consume both.
  s_add_u32 s8, s7, s6
  s_endpgm
wo_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel wo_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 9
  .end_amdhsa_kernel
