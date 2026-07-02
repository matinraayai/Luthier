// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:mubuf_atomic_noret.kd \
// RUN:   -initial-execution-point=0:mubuf_atomic_noret.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Non-RTN atomic (no `glc`): $vdata is input-only — the body must NOT
// `SetNamedOperand $vdata` (would clobber the input register). The
// intrinsic call's result is discarded.

// CHECK: define {{.*}} @mubuf_atomic_noret
// Non-RTN: call exists but result is unused — must not assign back to $vdata.
// CHECK: call i32 @llvm.amdgcn.struct.buffer.atomic.add.i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  mubuf_atomic_noret
  .p2align  8
  .type   mubuf_atomic_noret,@function
mubuf_atomic_noret:
  v_mov_b32_e32 v1, 0
  v_mov_b32_e32 v0, 7
  // No glc → non-RTN form. $vdata=v0 is input-only.
  buffer_atomic_add v0, v1, s[0:3], 0 offen
  s_waitcnt vmcnt(0)
  s_endpgm

mubuf_atomic_noret_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel mubuf_atomic_noret
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
