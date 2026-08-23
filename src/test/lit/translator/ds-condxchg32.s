// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:ds_condxchg_kern.kd \
// RUN:   -initial-execution-point=0:ds_condxchg_kern.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// DSSem fix #7: DS_CONDXCHG32_RTN_B64 — pair-aligned conditional swap:
//   addr0 = (addr + offset) & ~7
//   if data[31]: MEM[addr0] = data[30:0]
//   if data[63]: MEM[addr0+4] = data[62:32]
// Previously had empty `Semantic = []`.

// CHECK: define {{.*}} @ds_condxchg_kern
// CHECK-DAG: call i32 @llvm.ssa.copy.i32(i32 0)
// CHECK-DAG: call i32 @llvm.ssa.copy.i32(i32 -2147483643)
// CHECK-DAG: call i32 @llvm.ssa.copy.i32(i32 -2147483641)
// CHECK-DAG: and i32 %{{[0-9]+}}, -8
// CHECK-DAG: inttoptr i32 %{{[0-9]+}} to ptr addrspace(3)
// CHECK-DAG: and i32 %{{[0-9]+}}, 2147483647
// CHECK-DAG: load i32, ptr addrspace(3) %{{[0-9]+}}
// CHECK-DAG: store i32 %{{[0-9]+}}, ptr addrspace(3) %{{[0-9]+}}
// CHECK-DAG: insertelement <2 x i32>

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  ds_condxchg_kern
  .p2align  8
  .type   ds_condxchg_kern,@function
ds_condxchg_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0x80000005
  v_mov_b32_e32 v2, 0x80000007
  ds_condxchg32_rtn_b64 v[3:4], v0, v[1:2]
  s_waitcnt lgkmcnt(0)
  s_endpgm

ds_condxchg_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_condxchg_kern
    .amdhsa_group_segment_fixed_size 16
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 5
    .amdhsa_next_free_sgpr 4
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
