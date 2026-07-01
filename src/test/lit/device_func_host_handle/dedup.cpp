// clang-format off
/// RUN: %clangxx -x hip --offload-arch=gfx908 \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies that multiple host references to the same \c __device__ function
/// all resolve to a single synthesized \c __host__ handle (the consumer dedups
/// via the location-keyed Synthesized set, and overload resolution does the
/// rest).


#include <hip/hip_runtime.h>

__attribute__((device)) void myHook() {}

void useThrice(const void **out) {
  out[0] = reinterpret_cast<const void *>(&myHook);
  out[1] = reinterpret_cast<const void *>(&myHook);
  out[2] = reinterpret_cast<const void *>(&myHook);
}

// clang-format off
/// Exactly one host handle for myHook.
/// CHECK-COUNT-1: define dso_local void @_Z6myHookv()
/// CHECK-NOT: define dso_local void @_Z6myHookv()
// clang-format on
