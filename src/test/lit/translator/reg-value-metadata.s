// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:reg_md.kd \
// RUN:   -initial-execution-point=0:reg_md.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// TraceFunctionTranslator: every IR value produced for a (slice of a) physical
// register is tagged with !luthier.reg metadata; constant/argument seeds
// at kernel entry (and seeds whose tags survived simplifyInstruction folds
// onto a Constant) land in the function-level !luthier.entry_reg_map.

// CHECK: define {{.*}} @reg_md{{.*}} !luthier.entry_reg_map ![[#MAP:]]
// The map lists at least the EXEC, SCC, MODE seeds.
// CHECK: ![[#MAP]] = !{
// CHECK-DAG: !{i64 -1, !"exec{{[^"]*}}", i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 4}
// CHECK-DAG: !{i32 0, !"src_scc{{[^"]*}}", i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 2}
// CHECK-DAG: !{i32 0, !"mode{{[^"]*}}", i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 2}
// V_MOV_B32 semantics wrap their result in `llvm.ssa.copy` so the
// move materializes an explicit IR CallInst that the per-MI builder's
// callback can tag with `!pcsections`. That CallInst — not the source
// Constant — is what lands in the register-value map, so the reg-tag
// emitter routes vgpr0/1 through the per-instruction `!luthier.reg`
// path (4-element tuple) rather than the entry-reg-map's 5-element
// "constant-survival" tuple. The ssa.copy CallInst persists in IR with
// no IR uses, kept alive by optimizeNonTraceInsts' trace check.
// CHECK-DAG: !{!"vgpr0", i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 2}
// CHECK-DAG: !{!"vgpr1", i32 {{[0-9]+}}, i32 {{[0-9]+}}, i32 2}

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  reg_md
  .p2align  8
  .type   reg_md,@function
reg_md:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 1
  v_add_u32_e32 v2, v0, v1
  s_endpgm

reg_md_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel reg_md
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
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
