// clang-format off
/// RUN: not %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -fsyntax-only %s 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies that *calling* a __device__ function from host code is still an
/// error — only taking its address yields a handle.

#include <hip/hip_runtime.h>

__attribute__((device)) void myHook() {}

void hostFunction() {
  myHook(); // direct call from host context — must error
}

// clang-format off
/// CHECK: error: no matching function for call to 'myHook'
/// CHECK: note: candidate function not viable: call to __device__ function from __host__ function
// clang-format on
