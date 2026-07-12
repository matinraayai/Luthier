// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:tr_kern.kd \
// RUN:   -initial-execution-point=0:tr_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// GLOBAL_LOAD_TR_B{64,128} (+ SADDR/w32/w64) — WMMA transpose loads.
// Returns differ by wave size: w32 packs 2 values, w64 packs 1 (per the
// AMDGPULoadIntrinsic overloaded-return signature).

// CHECK: define {{.*}} @tr_kern
// CHECK-DAG: call <2 x i32> @llvm.amdgcn.global.load.tr.b64.v2i32(
// CHECK-DAG: call <4 x i32> @llvm.amdgcn.global.load.tr.b128.v4i32(

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
  .globl  tr_kern
  .p2align  8
  .type   tr_kern,@function
tr_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0
  global_load_tr_b64 v[2:3], v[0:1], off
  s_waitcnt vmcnt(0)
  global_load_tr_b128 v[4:7], v[0:1], off
  s_waitcnt vmcnt(0)
  s_endpgm
tr_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel tr_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 4
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
