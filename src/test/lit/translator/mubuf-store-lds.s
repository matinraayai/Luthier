// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mubuf_store_lds.kd \
// RUN:   -initial-execution-point=0:mubuf_store_lds.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// BUFFER_STORE_LDS_DWORD — per MI200 ISA p.184: "Store one DWORD from LDS
// memory to system memory without utilizing VGPRs." LDS source address
// is in M0 (per MI200 ISA p.~55 M0-write requirements). Modeled as a
// 2-step sequence: load i32 from addrspace(3) ptr-derived-from-M0, then
// struct_buffer_store the value.

// CHECK: define {{.*}} @mubuf_store_lds
// CHECK-DAG: load i32, ptr addrspace(3)
// CHECK-DAG: call void @llvm.amdgcn.struct.buffer.store.i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mubuf_store_lds
  .p2align  8
  .type   mubuf_store_lds,@function
mubuf_store_lds:
  s_mov_b32 m0, 0
  buffer_store_lds_dword s[0:3], 0 offset:0 lds
  s_waitcnt vmcnt(0)
  s_endpgm

mubuf_store_lds_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mubuf_store_lds
    .amdhsa_group_segment_fixed_size 16
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 4
  .end_amdhsa_kernel
