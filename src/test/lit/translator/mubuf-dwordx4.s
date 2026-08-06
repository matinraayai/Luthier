// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mubuf_dwx4.kd \
// RUN:   -initial-execution-point=0:mubuf_dwx4.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// MUBUF wide load/store: BUFFER_LOAD/STORE_DWORDX4_OFFEN. Exercises the
// XYZW (4-dword) width path through the same struct.buffer.{load,store}
// modeling.

// CHECK: define {{.*}} @mubuf_dwx4
// CHECK-DAG: call {{.*}} @llvm.amdgcn.struct.buffer.load
// CHECK-DAG: call {{.*}} @llvm.amdgcn.struct.buffer.store

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mubuf_dwx4
  .p2align  8
  .type   mubuf_dwx4,@function
mubuf_dwx4:
  v_mov_b32_e32 v4, 0
  buffer_load_dwordx4 v[0:3], v4, s[0:3], 0 offen
  s_waitcnt vmcnt(0)
  buffer_store_dwordx4 v[0:3], v4, s[0:3], 0 offen
  s_endpgm

mubuf_dwx4_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mubuf_dwx4
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
