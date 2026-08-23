// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mubuf_lds.kd \
// RUN:   -initial-execution-point=0:mubuf_lds.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Class B: MUBUF LDS-target buffer load. `lds` modifier directs the
// loaded data into LDS via M0 (no $vdata output). Must lift via
// int_amdgcn_struct_buffer_load_lds, not the regular struct_buffer_load.

// CHECK: define {{.*}} @mubuf_lds
// CHECK-DAG: call void @llvm.amdgcn.struct.buffer.load.lds
// CHECK-NOT: call i32 @llvm.amdgcn.struct.buffer.load.i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mubuf_lds
  .p2align  8
  .type   mubuf_lds,@function
mubuf_lds:
  s_mov_b32 m0, 0
  buffer_load_dword v1, s[0:3], 0 offen lds
  s_waitcnt vmcnt(0)
  buffer_load_ubyte v1, s[0:3], 0 offen lds
  s_waitcnt vmcnt(0)
  s_endpgm

mubuf_lds_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mubuf_lds
    .amdhsa_group_segment_fixed_size 16
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 4
  .end_amdhsa_kernel
