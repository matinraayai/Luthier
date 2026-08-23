// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:wg_kern.kd \
// RUN:   -initial-execution-point=0:wg_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Architected-SGPRs workgroup IDs (gfx12+ has the architected-sgprs feature).
//
// On these targets the backend does NOT allocate system SGPRs for the
// workgroup IDs; the hardware delivers them in fixed architected TTMP
// registers and the code reads those directly:
//   TTMP9        = workgroup id X
//   TTMP7[15:0]  = workgroup id Y, TTMP7[31:16] = workgroup id Z
// CodeDiscoveryPass therefore skips the system-SGPR allocation here (gated on
// hasArchitectedSGPRs()), and the translator seeds the TTMPs instead. The body
// below reads ttmp9 and ttmp7, so the seeds must materialize.
//
// The distinctive signature of the TTMP7 packing — and i32 .., 65535 (Y) plus
// shl i32 .., 16 (Z) OR'd together — is unique to this architected path; the
// regular system-SGPR workgroup-id seeds emit no such packing.

// CHECK: define {{.*}} @wg_kern
// CHECK-DAG: call i32 @llvm.amdgcn.workgroup.id.x()
// CHECK-DAG: call i32 @llvm.amdgcn.workgroup.id.y()
// CHECK-DAG: call i32 @llvm.amdgcn.workgroup.id.z()
// CHECK-DAG: and i32 %{{[0-9]+}}, 65535
// CHECK-DAG: shl i32 %{{[0-9]+}}, 16

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
  .globl  wg_kern
  .p2align  8
  .type   wg_kern,@function
wg_kern:
  // Reads workgroup id X from ttmp9 and the packed Y/Z from ttmp7, consuming
  // both so the seeded values are kept live.
  s_add_co_u32 s0, ttmp9, ttmp7
  s_endpgm
wg_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel wg_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 2
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
  .end_amdhsa_kernel
