// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mtbuf_offset.kd \
// RUN:   -initial-execution-point=0:mtbuf_offset.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// MTBUF semantics smoke test: TBUFFER_LOAD_FORMAT_X_OFFSET +
// TBUFFER_STORE_FORMAT_X_OFFSET. OFFSET addressing has no $vaddr — the
// semantics pass vindex=0 to int_amdgcn_struct_tbuffer_load/store.

// CHECK: define {{.*}} @mtbuf_offset
// CHECK-DAG: call {{.*}} @llvm.amdgcn.struct.tbuffer.load
// CHECK-DAG: call {{.*}} @llvm.amdgcn.struct.tbuffer.store

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mtbuf_offset
  .p2align  8
  .type   mtbuf_offset,@function
mtbuf_offset:
  tbuffer_load_format_x v0, off, s[0:3], 0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] offset:4
  s_waitcnt vmcnt(0)
  v_add_f32_e32 v0, 1.0, v0
  tbuffer_store_format_x v0, off, s[0:3], 0 format:[BUF_DATA_FORMAT_32,BUF_NUM_FORMAT_FLOAT] offset:4
  s_endpgm

mtbuf_offset_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mtbuf_offset
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 4
  .end_amdhsa_kernel
