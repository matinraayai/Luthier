// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1100 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: (luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 \
// RUN:    '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:    -code-object-paths=%t \
// RUN:    -initial-entrypoint=0:scratch_kern.kd \
// RUN:    -initial-execution-point=0:scratch_kern.kd \
// RUN:    -o /dev/null 2>&1 || true) > %t.out && \
// RUN: FileCheck %s < %t.out

// SCRATCH loads/stores — exercises the 4 address forms (base, SADDR, ST, SVS)
// + widths. All scratch pointers must be addrspace(5).

// CHECK: define {{.*}} @scratch_kern

// SCRATCH addrspace(5) load/store.
// CHECK-DAG: load i32, ptr addrspace(5)
// CHECK-DAG: store i32 {{.*}} ptr addrspace(5)
// CHECK-DAG: load i8,  ptr addrspace(5)
// CHECK-DAG: zext i8 {{.*}} to i32

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx1100"
  .globl  scratch_kern
  .p2align  8
  .type   scratch_kern,@function
scratch_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 42
  // Standard form (vaddr + offset)
  scratch_load_dword v2, v0, off
  s_waitcnt vmcnt(0)
  // SADDR form
  scratch_load_dword v3, off, s0
  s_waitcnt vmcnt(0)
  // ST (stack-only — no vaddr, just inst offset)
  scratch_load_dword v4, off, off offset:16
  s_waitcnt vmcnt(0)
  // SVS (vaddr + saddr) — exercises CalculateScratchSVSAddr (i32 + i32 + offset)
  scratch_load_dword v7, v0, s0 offset:32
  s_waitcnt vmcnt(0)
  // Store: standard
  scratch_store_dword v0, v1, off
  s_waitcnt vmcnt(0)
  // Sub-DWORD: ubyte (ext to i32)
  scratch_load_ubyte v5, v0, off
  s_waitcnt vmcnt(0)
  s_endpgm
scratch_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel scratch_kern
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 1024
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 4
    .amdhsa_wavefront_size32 1
  .end_amdhsa_kernel
