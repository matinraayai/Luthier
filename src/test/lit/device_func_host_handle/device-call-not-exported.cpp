// clang-format off
/// RUN: %clangxx -x hip -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies that a \c __device__ function used only from device code (called by
/// a \c __global__ kernel) is NOT exported: only host references warrant a
/// handle.

#include <hip/hip_runtime.h>

__attribute__((device)) void helper() {}

__global__ void kernel() { helper(); }

void hostFunction() {}

// clang-format off
/// Host code is emitted normally
/// CHECK: define dso_local void @_Z12hostFunctionv()

/// Nothing is exported and `helper` is not host-emitted.
/// CHECK-NOT: @llvm.global.annotations
/// CHECK-NOT: @_Z6helperv
// clang-format on
