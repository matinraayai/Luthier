// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:flat_load_d16_kern.kd \
// RUN:   -initial-execution-point=0:flat_load_d16_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// FLAT D16 / D16_HI loads — merge with tied $vdst_in.
//   Low form : vdst = (vdst_in & 0xFFFF0000) | zext_i32(ext_i16(load))
//   High form: vdst = (vdst_in & 0x0000FFFF) | (zext_i32(ext_i16(load)) << 16)

// CHECK: define {{.*}} @flat_load_d16_kern

// Half-preserve masks must appear once Low and once High form is exercised.
// CHECK-DAG: and i32 {{.*}}, -65536
// CHECK-DAG: and i32 {{.*}}, 65535
// CHECK-DAG: shl i32 {{.*}}, 16
// CHECK-DAG: load i8
// CHECK-DAG: load i16

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  flat_load_d16_kern
  .p2align  8
  .type   flat_load_d16_kern,@function
flat_load_d16_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v6, 0
  flat_load_ubyte_d16    v6, v[0:1]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_load_ubyte_d16_hi v6, v[0:1]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  flat_load_short_d16    v6, v[0:1]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  s_endpgm
flat_load_d16_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel flat_load_d16_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
