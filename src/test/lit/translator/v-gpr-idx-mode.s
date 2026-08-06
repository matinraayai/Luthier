// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc --disable-verify -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:gpr_idx_kernel.kd \
// RUN:   -initial-execution-point=0:gpr_idx_kernel.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Phase B: GFX9 GPR_IDX_EN-driven indexed VGPR access.
//
// At kernel entry MODE.GPR_IDX_EN is 0 (cleared by buildInitialModeValue),
// so every VALU's source-side indexing wrapper folds to the direct read.
// To force the indexed branch to survive into the lifted IR we set
// MODE.GPR_IDX_EN := 1 via an explicit s_setreg, set M0[7:0] to an index,
// then read v0 from a v_mov — the source-side wrapper should now select
// the indexed `extractelement <K x i32> %vgpr_slice` path rather than the
// direct read of v0.
//
// hwreg encoding for MODE.GPR_IDX_EN (bit 27, width 1):
//   ID_MODE(1) | (27 << 6) | ((1-1) << 11) = 1 | 1728 | 0 = 1729 = 0x6C1.
//
// The fold pass rewrites the s_setreg to a direct write of the MODE
// tracker, and `optimizeNonTraceInsts` const-folds the GPR_IDX_EN bit
// to 1 inside the wrapper — so `select(true, indexed, direct) → indexed`
// and we should see `extractelement <K x i32>` left in the IR.

// CHECK: define {{.*}} @gpr_idx_kernel
// A store consuming the indexed-read chain survives DCE. The exact
// folded value depends on tracker initialization of v2 (uninit poison),
// but the lifted IR must contain a `store i32` whose value is sourced
// by the per-slot select / read chain emitted by `emitIndexedVGPRSrc`.
// CHECK-DAG: store i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  gpr_idx_kernel
  .p2align  8
  .type   gpr_idx_kernel,@function
gpr_idx_kernel:
  // M0[7:0] = 2 (index into VGPR file from the named base).
  s_mov_b32 m0, 2
  // Force MODE.GPR_IDX_EN := 1 via setreg(MODE, bit 27, width 1).
  s_mov_b32 s0, 1
  s_setreg_b32 0x6C1, s0
  // Indexed read of v0: with GPR_IDX_EN=1 and M0=2 this reads v2.
  v_mov_b32_e32 v3, v0
  // Store v3 to memory so the indexed-read chain survives DCE.
  flat_store_dword v[4:5], v3
  s_endpgm

gpr_idx_kernel_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel gpr_idx_kernel
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
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 8
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
