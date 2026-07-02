// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_incdec_kern.kd \
// RUN:   -initial-execution-point=0:ds_incdec_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Group E coverage: DS_INC_U32 and DS_DEC_U32 implement bounded
// increment/decrement with wraparound:
//   INC: MEM = (tmp >= data) ? 0      : tmp + 1
//   DEC: MEM = (tmp == 0 || tmp > data) ? data : tmp - 1

// CHECK: define {{.*}} @ds_incdec_kern
// CHECK-DAG: icmp uge i32
// CHECK-DAG: icmp eq i32
// CHECK-DAG: icmp ugt i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_incdec_kern
  .p2align  8
  .type   ds_incdec_kern,@function
ds_incdec_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 100
  ds_inc_u32 v0, v1
  ds_dec_u32 v0, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm

ds_incdec_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_incdec_kern
    .amdhsa_group_segment_fixed_size 4
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
