// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_gws_kern.kd \
// RUN:   -initial-execution-point=0:ds_gws_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// GWS (Global Wave Sync) coverage: 1:1 pseudo-to-intrinsic mapping per
// AMDGPUInstructionSelector::gwsIntrinToOpcode. 1D ops take (data0, offset)
// → 2-arg intrinsic; 0D ops take just offset → 1-arg intrinsic.

// CHECK: define {{.*}} @ds_gws_kern
// CHECK-DAG: call void @llvm.amdgcn.ds.gws.init(
// CHECK-DAG: call void @llvm.amdgcn.ds.gws.barrier(
// CHECK-DAG: call void @llvm.amdgcn.ds.gws.sema.br(
// CHECK-DAG: call void @llvm.amdgcn.ds.gws.sema.v(
// CHECK-DAG: call void @llvm.amdgcn.ds.gws.sema.p(
// CHECK-DAG: call void @llvm.amdgcn.ds.gws.sema.release.all(

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_gws_kern
  .p2align  8
  .type   ds_gws_kern,@function
ds_gws_kern:
  v_mov_b32_e32 v0, 7
  ds_gws_init v0 offset:0 gds
  s_waitcnt lgkmcnt(0)
  ds_gws_barrier v0 offset:0 gds
  s_waitcnt lgkmcnt(0)
  ds_gws_sema_br v0 offset:0 gds
  s_waitcnt lgkmcnt(0)
  ds_gws_sema_v offset:0 gds
  s_waitcnt lgkmcnt(0)
  ds_gws_sema_p offset:0 gds
  s_waitcnt lgkmcnt(0)
  ds_gws_sema_release_all offset:0 gds
  s_waitcnt lgkmcnt(0)
  s_endpgm
ds_gws_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_gws_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
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
