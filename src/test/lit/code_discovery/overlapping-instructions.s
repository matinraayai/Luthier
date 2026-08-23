// RUN: llvm-mc --triple amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=obj %s -o %t.o && \
// RUN: ld.lld -shared --unresolved-symbols=ignore-all -o %t %t.o && \
// RUN: luthier-llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:   '-passes=target(luthier-mock-load-amdgpu-code-objects),luthier-code-discovery,target(print)' \
// RUN:   -code-object-paths=%t \
// RUN:   -initial-entrypoint=0:overlap_kernel.kd \
// RUN:   -initial-execution-point=0:overlap_kernel.kd \
// RUN:   -o - 2>/dev/null | %tee_out FileCheck %s

// AMDGPU analog of the "overlapping instructions" construct from
// Meng & Miller, "Binary Code Is Not Easy" (ISSTA 2016) §4 / Fig. 3.
//
// Construction:
//   * `main_path` contains `s_mov_b32 s3, 0xbf810000` — 8 bytes total. The
//     literal value 0xbf810000 IS the binary encoding of `s_endpgm`.
//   * scc=0 (fall-through) path runs main_path's full 8-byte s_mov as one
//     instruction.
//   * scc=1 path computes `lit_addr := main_path + 4` via s_getpc + add and
//     does an indirect call (`s_swappc_b64`) to it. CodeDiscoveryPass enqueues
//     lit_addr as a new entry-point whose body contains the same 4
//     bytes as a standalone `s_endpgm`.
//

// CHECK: define {{.*}} @overlap_kernel()
// CHECK: define {{.*}} @overlap_kernelx0x28(
// CHECK-NEXT: call void @llvm.amdgcn.endpgm()
// CHECK-NEXT: unreachable

  .text
  .amdgcn_target "amdgcn-amd-amdhsa--gfx908"
  .globl  overlap_kernel
  .p2align  8
  .type   overlap_kernel,@function
overlap_kernel:
  s_cmp_lt_u32 s0, 5
  s_cbranch_scc0 main_path
  // scc=1: indirect-call to main_path+4 via s_getpc + s_add + s_swappc
  s_getpc_b64 s[10:11]
.Lpc_pos:
  s_add_u32 s10, s10, (lit_addr - .Lpc_pos)
  s_addc_u32 s11, s11, (lit_addr - .Lpc_pos) >> 32
  s_swappc_b64 s[30:31], s[10:11]
  s_endpgm
main_path:
  s_mov_b32 s3, 0xbf810000
  s_setpc_b64 s[30:31]
overlap_kernel_end:
  .size   overlap_kernel, overlap_kernel_end - overlap_kernel
  .set    lit_addr, main_path + 4

  .section .rodata,"a",@progbits
  .p2align  6, 0x0
  .amdhsa_kernel overlap_kernel
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
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
    .amdhsa_system_sgpr_private_segment_wavefront_offset 0
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 0
    .amdhsa_system_sgpr_workgroup_id_z 0
    .amdhsa_system_sgpr_workgroup_info 0
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 1
    .amdhsa_next_free_sgpr 32
    .amdhsa_reserve_flat_scratch 0
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 0
    .amdhsa_float_denorm_mode_16_64 0
    .amdhsa_dx10_clamp 0
    .amdhsa_ieee_mode 0
  .end_amdhsa_kernel
