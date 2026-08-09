// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mtbuf_d16.kd \
// RUN:   -initial-execution-point=0:mtbuf_d16.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// MTBUF D16: TBUFFER_LOAD_FORMAT_D16_XY_OFFEN. D16 widths produce
// <N x half> result types (XY → <2 x half>).

// CHECK: define {{.*}} @mtbuf_d16
// CHECK: call <2 x half> @llvm.amdgcn.struct.tbuffer.load.v2f16

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mtbuf_d16
  .p2align  8
  .type   mtbuf_d16,@function
mtbuf_d16:
  v_mov_b32_e32 v1, 0
  tbuffer_load_format_d16_xy v0, v1, s[0:3], 0 format:[BUF_DATA_FORMAT_16_16,BUF_NUM_FORMAT_FLOAT] offen
  s_waitcnt vmcnt(0)
  s_endpgm

mtbuf_d16_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mtbuf_d16
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 4
  .end_amdhsa_kernel
