// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mtbuf_idxen.kd \
// RUN:   -initial-execution-point=0:mtbuf_idxen.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// MTBUF: TBUFFER_LOAD_FORMAT_XYZW_IDXEN + TBUFFER_STORE_FORMAT_XYZW_IDXEN.
// XYZW width → <4 x float>; IDXEN passes $vaddr as the struct vindex.

// CHECK: define {{.*}} @mtbuf_idxen
// CHECK-DAG: call {{.*}} @llvm.amdgcn.struct.tbuffer.load
// CHECK-DAG: call void @llvm.amdgcn.struct.tbuffer.store

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mtbuf_idxen
  .p2align  8
  .type   mtbuf_idxen,@function
mtbuf_idxen:
  v_mov_b32_e32 v4, 0
  tbuffer_load_format_xyzw v[0:3], v4, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_FLOAT] idxen
  s_waitcnt vmcnt(0)
  tbuffer_store_format_xyzw v[0:3], v4, s[0:3], 0 format:[BUF_DATA_FORMAT_32_32_32_32,BUF_NUM_FORMAT_FLOAT] idxen
  s_endpgm

mtbuf_idxen_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mtbuf_idxen
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 5
    .amdhsa_next_free_sgpr 4
  .end_amdhsa_kernel
