// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
// RUN:    '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:    -code-object-paths=%t \
// RUN:    -initial-entrypoint=0:ds_pkadd_kern.kd \
// RUN:    -initial-execution-point=0:ds_pkadd_kern.kd \
// RUN:    -o /dev/null > %t.out 2>&1 && \
// RUN: FileCheck %s < %t.out

// This used to abort in the post-translation machine verifier on gfx942, and
// the test worked around it with `|| true` and checked only the dump produced
// before the abort. The cause was the AMDGPU disassembler emitting a duplicate
// gds operand for DS instructions on chips without GDS, which convertAndAdd-
// MCOperandsToMI then added to the MachineInstr, giving it more explicit
// operands than its opcode allows. The surplus is now dropped, so this runs to
// completion — and the absence of `|| true` is what makes this a regression test
// for that rather than a record of a known failure.

// Group M coverage: DS_PK_ADD_F16 / DS_PK_ADD_BF16 lower to packed fadd on
// <2 x half> / <2 x bfloat>. gfx942 is the first arch with both
// (HasAtomicDsPkAdd16Insts is enabled via FeatureISAVersion9_4_Common).

// CHECK: define {{.*}} @ds_pkadd_kern
// CHECK-DAG: bitcast i32 {{.*}} to <2 x half>
// CHECK-DAG: bitcast i32 {{.*}} to <2 x bfloat>
// CHECK-DAG: fadd <2 x half>
// CHECK-DAG: fadd <2 x bfloat>

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .globl  ds_pkadd_kern
  .p2align  8
  .type   ds_pkadd_kern,@function
ds_pkadd_kern:
  v_mov_b32_e32 v0, 0
  v_mov_b32_e32 v1, 0x3c003c00
  ds_pk_add_f16 v0, v1
  ds_pk_add_bf16 v0, v1
  s_waitcnt lgkmcnt(0)
  s_endpgm
ds_pkadd_kern_end:

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel ds_pkadd_kern
    .amdhsa_group_segment_fixed_size 4
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 0
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 2
    .amdhsa_next_free_sgpr 4
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
