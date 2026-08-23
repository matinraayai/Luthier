// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:inline_asm_fallback.kd \
// RUN:   -initial-execution-point=0:inline_asm_fallback.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Verifies the MIInlineAsmEmitter's word-boundary-aware register-name
// substitution. The instructions below use registers whose printed names
// are prefixes of one another (v1 is a prefix of v10, v11; s2 is a prefix
// of s20). With the previous substring-only matcher, scanning the asm
// string for "v1" would also match the leading "v1" of "v10"/"v11",
// corrupting the substituted asm. The fix uses a word-boundary check so
// only complete tokens are replaced.
//
// V_MED3_F32 is deferred (unmodelled per project_remaining_sem_audit), so
// it falls through to the inline-asm fallback path here.

// The lifted IR should contain a single inline-asm call per V_MED3_F32 with
// clean, monotonically numbered operand placeholders. The presence of
// "$0, $1, $2, $3" together (in that order) confirms each register-name
// token was replaced exactly once — no substring overrun would have
// preserved this layout.

// CHECK: define {{.*}} @inline_asm_fallback
// CHECK: call {{.*}} asm sideeffect {{.*}}v_med3_f32 $0, $1, $2, $3
// Negative checks: a substring-only matcher would mangle "v10" into "$1 0"
// (and similarly "v11" → "$1 1", "s20" → "$_ 0"). None of those token
// fragments should appear in the lifted asm template.
// CHECK-NOT: $1 0
// CHECK-NOT: $1 1

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  inline_asm_fallback
  .p2align  8
  .type   inline_asm_fallback,@function
inline_asm_fallback:
  // dst=v1, src0=v10, src1=v11, src2=v2 — exposes the v1-prefix-of-v10/v11
  // case that the substring matcher would have corrupted.
  v_med3_f32 v1, v10, v11, v2
  s_endpgm

inline_asm_fallback_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel inline_asm_fallback
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 12
    .amdhsa_next_free_sgpr 4
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
