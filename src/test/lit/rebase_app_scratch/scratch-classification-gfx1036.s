// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx1036 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1036 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,luthier-rebase-app-scratch-accesses,target(print-mir-prepare,function(machine-function(print)))' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:scratch_stub.kd \
// RUN:   -initial-execution-point=0:scratch_stub.kd \
// RUN:   -o /dev/null 2>&1 | %tee_out FileCheck %s

// RebaseAppScratchAccessesPass replaces every application instruction that can
// reach the wavefront's private segment with a payload call that redoes the
// access at the displaced address, and erases the original. The stub below
// covers all the cases the classifier distinguishes:
//
//   scratch_load_dword / scratch_store_dword
//       always private, always rewritten
//   flat_load_dword
//       may be private, rewritten; the payload decides at run time with
//       __builtin_amdgcn_is_private
//   global_load_dword
//       cannot be private, left alone
//   buffer_load_dword ... idxen
//       cannot be a scratch buffer (scratch requires idxen == 0), left alone

// CHECK: name: scratch_stub

// Every rewritten access is gone, replaced by a PATCHPOINT marker naming its
// payload. The implicit operands are the register operands the pass forwarded:
// the address it reads, and for a load the destination it writes back.
//
//   scratch_load_dword v2, v0, off      -> reads v0,     defines v2
//   scratch_store_dword v1, v2, off:16  -> reads v2 (data) and v1 (address)
//   flat_load_dword v3, v[0:1]          -> reads v[0:1],  defines v3
//
// CHECK: PATCHPOINT 0, 0, @luthier.payload, 0, 0, implicit $vgpr0, implicit-def $vgpr2
// CHECK: PATCHPOINT 0, 0, @luthier.payload{{[.0-9]*}}, 0, 0, implicit $vgpr2, implicit $vgpr1
// CHECK: PATCHPOINT 0, 0, @luthier.payload{{[.0-9]*}}, 0, 0, implicit $vgpr0_vgpr1, implicit-def $vgpr3

// CHECK-NOT: SCRATCH_LOAD_DWORD
// CHECK-NOT: SCRATCH_STORE_DWORD
// CHECK-NOT: FLAT_LOAD_DWORD

// The two forms that cannot reach scratch survive untouched.
// CHECK-DAG: GLOBAL_LOAD_DWORD
// CHECK-DAG: BUFFER_LOAD_DWORD_IDXEN

  .amdgcn_target "amdgcn-amd-amdhsa--gfx1036"
  .amdhsa_code_object_version 6
  .text
  .protected scratch_stub
  .globl scratch_stub
  .p2align 8
  .type scratch_stub,@function
scratch_stub:
; %bb.0:
  scratch_load_dword v2, v0, off
  s_waitcnt vmcnt(0)
  scratch_store_dword v1, v2, off offset:16
  flat_load_dword v3, v[0:1]
  global_load_dword v4, v[0:1], off
  buffer_load_dword v5, v0, s[8:11], 0 idxen
  s_waitcnt vmcnt(0)
  s_endpgm

  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel scratch_stub
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 64
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
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_reserve_vcc 1
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_dx10_clamp 1
    .amdhsa_ieee_mode 1
    .amdhsa_fp16_overflow 0
    .amdhsa_wavefront_size32 1
    .amdhsa_workgroup_processor_mode 1
    .amdhsa_memory_ordered 1
    .amdhsa_forward_progress 0
  .end_amdhsa_kernel
  .text
.Lfunc_end0:
  .size scratch_stub, .Lfunc_end0-scratch_stub
  .set scratch_stub.num_vgpr, 8
  .set scratch_stub.num_agpr, 0
  .set scratch_stub.numbered_sgpr, 16
  .set scratch_stub.private_seg_size, 64
  .set scratch_stub.uses_vcc, 1
  .set scratch_stub.uses_flat_scratch, 1
