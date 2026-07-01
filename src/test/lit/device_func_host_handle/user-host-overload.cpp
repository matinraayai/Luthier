// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies that when the tool already provides its own \c __host__ overload of
/// a \c __device__ function, the plugin does NOT synthesize a second handle and
/// reuses the already existing handle.

#include <hip/hip_runtime.h>

__attribute__((device)) void hook() {
  unsigned long long Exec = __builtin_amdgcn_read_exec();
  (void)Exec;
}
__attribute__((host)) void hook() {}

void hostFunction(const void **out) {
  out[0] = reinterpret_cast<const void *>(&hook);
}

// clang-format off
/// Exactly one host-side `hook` (the user overload), annotated for harvesting.
/// CHECK: @llvm.global.annotations {{.*}}@_Z4hookv
/// CHECK-COUNT-1: define dso_local void @_Z4hookv()
/// CHECK-NOT: define dso_local void @_Z4hookv()

/// Host address-take resolves to it; no AMDGCN intrinsic leaks to host.
/// CHECK: store ptr @_Z4hookv
/// CHECK-NOT: llvm.amdgcn.
// clang-format on
