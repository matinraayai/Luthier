// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   -load-pass-plugin=%luthier_tool_code_gen_plugin \
// RUN:   '-passes=luthier-mock-load-amdgpu-code-objects,luthier-code-discovery,print' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:smem_dcache.kd \
// RUN:   -initial-execution-point=0:smem_dcache.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// S_DCACHE_INV / S_DCACHE_WB / S_MEMTIME — were previously empty bodies.
// Now S_DCACHE_INV/WB use their dedicated intrinsics; S_MEMTIME returns i64.

// CHECK: define {{.*}} @smem_dcache
// CHECK-DAG: call void @llvm.amdgcn.s.dcache.inv()
// CHECK-DAG: call void @llvm.amdgcn.s.dcache.wb()
// CHECK-DAG: call i64 @llvm.amdgcn.s.memtime()

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  smem_dcache
  .p2align  8
  .type   smem_dcache,@function
smem_dcache:
  s_dcache_inv
  s_dcache_wb
  s_memtime s[4:5]
  s_endpgm

smem_dcache_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel smem_dcache
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_private_segment_buffer 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 8
  .end_amdhsa_kernel
