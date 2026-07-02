// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx906 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc --disable-verify -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:s_movrel.kd \
// RUN:   -initial-execution-point=0:s_movrel.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// S_MOVREL{S,D}_B{32,64}: relative scalar moves indexed by M0.
//
//   S_MOVRELS_B32 sdst, src0:  sdst = SGPR[src0 + M0]
//   S_MOVRELD_B32 sdst, src0:  SGPR[sdst + M0] = src0
//
// Modelled with the modified getRegisterFile/setRegisterFile that slices
// from the base register's offset within its file (not from SGPR0), and
// the new GetRegisterFile/SetRegisterFile DSL primitives that take a
// named-operand argname (e.g. `$src0`) to identify the base register at
// MI translation time.
//
// IR-level expectations:
//   - S_MOVRELS_B32: extractelement of a <K x i32> register-file slice at M0.
//   - S_MOVRELD_B32: insertelement into the slice at M0, then write back.

// CHECK: define {{.*}} @s_movrel

// (1) S_MOVRELS_B32 — read indexed source.
// CHECK-DAG: extractelement <{{[0-9]+}} x i32>

// (2) S_MOVRELD_B32 — write indexed destination.
// CHECK-DAG: insertelement <{{[0-9]+}} x i32>

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx906"
  .globl  s_movrel
  .p2align  8
  .type   s_movrel,@function
s_movrel:
  // Set up an index in M0.
  s_mov_b32 m0, 2
  // S_MOVRELS_B32: s10 = SGPR[s4 + M0] (with M0=2 this reads s6 into s10).
  s_movrels_b32 s10, s4
  // S_MOVRELD_B32: SGPR[s12 + M0] = s5 (with M0=2 this writes s5 into s14).
  s_movreld_b32 s12, s5
  s_endpgm

s_movrel_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel s_movrel
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_queue_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_dispatch_id 0
    .amdhsa_user_sgpr_flat_scratch_init 0
    .amdhsa_user_sgpr_private_segment_size 0
    .amdhsa_uses_dynamic_stack 0
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 32
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
