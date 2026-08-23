// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_gsreg_kern.kd \
// RUN:   -initial-execution-point=0:ds_gsreg_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// DS_{ADD,SUB}_GS_REG_RTN — RDNA3-only atomic update of a GDS_STRMOUT_*
// register bank entry. (i32 data, i32 imm_offset_idx) -> i64.

// CHECK: define {{.*}} @ds_gsreg_kern
// CHECK-DAG: call i64 @llvm.amdgcn.ds.add.gs.reg.rtn.i64(
// CHECK-DAG: call i64 @llvm.amdgcn.ds.sub.gs.reg.rtn.i64(

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .globl  ds_gsreg_kern
  .p2align  8
  .type   ds_gsreg_kern,@function
ds_gsreg_kern:
  v_mov_b32_e32 v0, 1
  // offset:32 selects a 64-bit GS register slot (offset[5:2] >= 8).
  ds_add_gs_reg_rtn v[2:3], v0 offset:32 gds
  s_waitcnt lgkmcnt(0)
  ds_sub_gs_reg_rtn v[4:5], v0 offset:32 gds
  s_waitcnt lgkmcnt(0)
  s_endpgm
ds_gsreg_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_gsreg_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 6
    .amdhsa_next_free_sgpr 1
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
