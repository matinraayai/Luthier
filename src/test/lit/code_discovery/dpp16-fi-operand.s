// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1030 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t.so %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print-mir-prepare,function(machine-function(print)))' \
// RUN:   -code-object-paths=%t.so \
// RUN:   -initial-entrypoint=0:dpp16_fi_operand.kd \
// RUN:   -initial-execution-point=0:dpp16_fi_operand.kd \
// RUN:   -o /dev/null 2>&1 | %tee_out FileCheck %s

// The gfx10+ DPP16 encodings carry a fetch-inactive ($fi) operand that the
// V_MOV_B32_dpp pseudo has no slot for, so discovery has to drop it when it
// maps the real opcode onto the pseudo.

// CHECK: name: dpp16_fi_operand

// CHECK: $vgpr1 = V_MOV_B32_dpp $vgpr1, $vgpr2, 177, 15, 15, 0, implicit $exec

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
  .globl  dpp16_fi_operand
  .p2align  8
  .type   dpp16_fi_operand,@function
dpp16_fi_operand:
  v_mov_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
  s_endpgm

dpp16_fi_operand_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel dpp16_fi_operand
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 4
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
