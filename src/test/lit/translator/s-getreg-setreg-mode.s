// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx906 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc --disable-verify -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:hwreg_kernel.kd \
// RUN:   -initial-execution-point=0:hwreg_kernel.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// S_GETREG_B32 / S_SETREG_B32 — MODE-hwreg fold (Phase A).
//
// The translator's `foldHwregIntrinsics` step rewrites every
// `@llvm.amdgcn.s.getreg.i32` / `s.setreg.i32` whose hwreg encoding
// is a constant naming a translator-tracked register (MODE) into
// direct read/write of the tracked SSA value with explicit bitfield
// extract/insert. Combined with `optimizeNonTraceInsts` and the
// kernel-entry MODE constant assembled by `buildInitialModeValue`,
// this lets the optimizer constant-fold every read of a MODE field
// that the function never writes.
//
// The kernel below reads two MODE fields. Neither field is written, so
// both reads should fold to constants and the `@llvm.amdgcn.s.getreg`
// intrinsic should disappear from the lifted IR entirely.

// CHECK: define {{.*}} @hwreg_kernel
// CHECK-NOT: call i32 @llvm.amdgcn.s.getreg
// CHECK-NOT: call void @llvm.amdgcn.s.setreg

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx906"
  .globl  hwreg_kernel
  .p2align  8
  .type   hwreg_kernel,@function
hwreg_kernel:
  // Read FP_DENORM_F32 (MODE bits [4:5], i.e. ID=1, offset=4, width=2).
  //   encoding = ID_MODE(1) | (offset<<6) | ((width-1)<<11)
  //            = 1 | (4<<6) | (1<<11) = 1 | 256 | 2048 = 2305 = 0x901.
  s_getreg_b32 s0, 0x901
  // Read IEEE-mode bit (MODE bit 9, width 1).
  //   encoding = 1 | (9<<6) | (0<<11) = 1 | 576 = 577 = 0x241.
  s_getreg_b32 s1, 0x241
  s_endpgm

hwreg_kernel_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel hwreg_kernel
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
    .amdhsa_next_free_sgpr 8
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
