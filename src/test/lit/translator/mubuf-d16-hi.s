// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mubuf_d16_hi.kd \
// RUN:   -initial-execution-point=0:mubuf_d16_hi.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Class C: D16_HI partial-write merge / extract.
//   - LOAD_SHORT_D16_HI must read i16, ZExt to i32, shl 16, OR with low half
//     (preserves bits [15:0] of $vdata) — confirmed by `or i32 ...` in IR.
//   - STORE_BYTE_D16_HI must lshr by 16 then trunc to i8 (extract bits [23:16]).

// CHECK: define {{.*}} @mubuf_d16_hi
// LOAD merge: tmp i16 → zext → shl 16 → or with low half
// CHECK-DAG: call i16 @llvm.amdgcn.struct.buffer.load
// CHECK-DAG: shl i32 {{.*}}, 16
// CHECK-DAG: or i32
// STORE extract: lshr 16 then trunc to i8
// CHECK-DAG: lshr i32 {{.*}}, 16
// CHECK-DAG: call void @llvm.amdgcn.struct.buffer.store.i8

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mubuf_d16_hi
  .p2align  8
  .type   mubuf_d16_hi,@function
mubuf_d16_hi:
  v_mov_b32_e32 v1, 0
  buffer_load_short_d16_hi v0, v1, s[0:3], 0 offen
  s_waitcnt vmcnt(0)
  buffer_store_byte_d16_hi v0, v1, s[0:3], 0 offen
  s_endpgm

mubuf_d16_hi_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mubuf_d16_hi
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
