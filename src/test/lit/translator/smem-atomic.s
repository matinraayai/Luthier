// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:smem_atomic.kd \
// RUN:   -initial-execution-point=0:smem_atomic.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// S_ATOMIC_ADD: scalar atomic add to global memory. Body uses LLVMAtomicRMW
// with syncscope decoded from $cpol (was previously hardcoded "system").
// S_BUFFER_ATOMIC_ADD: uses int_amdgcn_struct_buffer_atomic_add (sgpr-source
// data + sgpr rsrc) with $cpol forwarded.

// CHECK: define {{.*}} @smem_atomic
// CHECK-DAG: atomicrmw add ptr addrspace(1)
// CHECK-DAG: call i32 @llvm.amdgcn.struct.buffer.atomic.add.i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  smem_atomic
  .p2align  8
  .type   smem_atomic,@function
smem_atomic:
  s_atomic_add s4, s[0:1], 0x10
  s_buffer_atomic_add s5, s[0:3], 0x10
  s_waitcnt lgkmcnt(0)
  s_endpgm

smem_atomic_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel smem_atomic
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
