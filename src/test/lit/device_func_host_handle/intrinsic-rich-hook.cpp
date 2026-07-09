// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies that a hook whose body uses many \c __device__-only AMDGCN
/// intrinsics still compiles cleanly: the original stays pure \c __device__
/// (so its body never enters host CodeGen), the synthesized \c __host__ handle
/// is the empty host-side shadow, and no AMDGCN intrinsic leaks into host
/// emission.

#include <hip/hip_runtime.h>

__attribute__((device)) void intrinsicRich() {
  unsigned long long Exec = __builtin_amdgcn_read_exec();
  unsigned TidX = __builtin_amdgcn_workitem_id_x();
  unsigned TidY = __builtin_amdgcn_workitem_id_y();
  unsigned TidZ = __builtin_amdgcn_workitem_id_z();
  unsigned WgX = __builtin_amdgcn_workgroup_id_x();
  unsigned WgY = __builtin_amdgcn_workgroup_id_y();
  unsigned WgZ = __builtin_amdgcn_workgroup_id_z();
  unsigned long long Mt = __builtin_amdgcn_s_memrealtime();
  unsigned long long Ballot = __builtin_amdgcn_ballot_w64(true);
  (void)Exec;
  (void)TidX;
  (void)TidY;
  (void)TidZ;
  (void)WgX;
  (void)WgY;
  (void)WgZ;
  (void)Mt;
  (void)Ballot;
}

void hostFunction(const void **out) {
  out[0] = reinterpret_cast<const void *>(&intrinsicRich);
}

// clang-format off
/// The handle is emitted with an empty body and carries the export-handle
/// annotation. The original __device__ function (with the intrinsics) is not
/// host-emitted.
/// CHECK: define dso_local void @_Z13intrinsicRichv()
/// CHECK-NEXT: entry:
/// CHECK-NEXT: ret void

/// Host code's address-take resolves to the handle.
/// CHECK: store ptr @_Z13intrinsicRichv

/// Most importantly: none of the AMDGCN intrinsics survive into the host
/// module.
/// CHECK-NOT: llvm.amdgcn.
/// CHECK-NOT: __builtin_amdgcn
// clang-format on
