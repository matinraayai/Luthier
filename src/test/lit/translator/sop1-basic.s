// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:sop1_basic.kd \
// RUN:   -initial-execution-point=0:sop1_basic.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// SOP1 basic semantics:
//   S_NOT_B32         — XOR with -1; SCC updated
//   S_BREV_B32        — int_bitreverse; SCC not updated
//   S_QUADMASK_B32    — int_amdgcn_s_quadmask; SCC update (Class-A fix)
//   S_AND_SAVEEXEC_B64 — old EXEC saved to sdst, new EXEC = src0 & old; SCC update

// CHECK: define {{.*}} @sop1_basic
// S_NOT — XOR with -1 (true sext'd)
// CHECK-DAG: xor i32 {{.*}}, -1
// S_BREV — bitreverse
// CHECK-DAG: call i32 @llvm.bitreverse.i32
// S_QUADMASK — quadmask intrinsic
// CHECK-DAG: call i32 @llvm.amdgcn.s.quadmask.i32
// S_AND_SAVEEXEC — implicit EXEC read/write surfaces in @luthier.reg metadata.
// The body's `and` of EXEC with src0 is observable as `and i64`.
// CHECK-DAG: and i64

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  sop1_basic
  .p2align  8
  .type   sop1_basic,@function
sop1_basic:
  s_not_b32 s4, s0
  s_brev_b32 s5, s0
  s_quadmask_b32 s6, s0
  s_and_saveexec_b64 s[8:9], s[0:1]
  s_endpgm

sop1_basic_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel sop1_basic
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 10
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
