/// RUN: not %clangxx -x hip --offload-arch=gfx908 -std=c++20 \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -fsyntax-only %s 2>&1 \
/// RUN:   | %tee_out FileCheck %s
/// Verifies the plugin handles a consteval __device__ function gracefully.

#include <hip/hip_runtime.h>

__attribute__((device)) consteval int myConstevalHook(int x) { return x + 1; }

void hostFunction(const void **out) {
  out[0] = reinterpret_cast<const void *>(&myConstevalHook);
}

// clang-format off
/// CHECK: error: cannot take address of consteval function 'myConstevalHook' outside of an immediate invocation
// clang-format on
