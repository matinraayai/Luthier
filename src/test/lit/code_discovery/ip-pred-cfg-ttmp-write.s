// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ttmp_write_kern.kd \
// RUN:   -initial-execution-point=0:ttmp_write_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// Audit companion for the getRegFileKey subtarget-resolution fix (see
// ip-pred-cfg-dynamic-exec-write.s). The bug's root cause — reading the
// HW encoding off the generic \c AMDGPU::M0 / \c TTMP0 / \c SGPR_NULL
// MCRegister instead of the per-subtarget alias — dropped every write
// into the register files rooted at those special bases. EXEC (rooted at
// M0 pre-GFX11) was the visible symptom because the exec-check fold read
// the seed value. TTMP shares the same code path on GFX9+ (TTMPBaseReg =
// AMDGPU::TTMP0, which is defined with HWEncoding=0 in
// llvm/lib/Target/AMDGPU/SIRegisterInfo.td:318; the real 108 lives on
// the TTMP0_gfx9plus alias at line 317).
//
// The kernel loads a value from memory and moves it into TTMP0. The
// \c luthier.bb_exit_reg_map metadata for the kernel's IR block must
// contain a TTMP0 slice pointing at the loaded SSA value — if the write
// were being dropped, TTMP0 would either be absent from the exit map or
// carry the trap-handler entry value instead.

// The kernel function must carry an exit-map metadata reference.
// CHECK: define {{.*}} @ttmp_write_kern{{.*}} !luthier.bb_exit_reg_map ![[#EXIT:]]

// The exit map for the kernel's body block must list a TTMP0 slice whose
// value is an SSA reference (indicating the written value flowed through
// the register tracker, not the entry seed).
// CHECK-DAG: !{i32 %{{[a-zA-Z0-9_.]+}}, !"ttmp0{{[^"]*}}", {{.*}}}

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ttmp_write_kern
  .p2align  8
  .type   ttmp_write_kern,@function
ttmp_write_kern:
  s_load_dword s0, s[4:5], 0x0
  s_waitcnt lgkmcnt(0)
  s_mov_b32 ttmp0, s0
  s_endpgm

ttmp_write_kern_end:
  .size   ttmp_write_kern, ttmp_write_kern_end-ttmp_write_kern

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ttmp_write_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 4
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
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
