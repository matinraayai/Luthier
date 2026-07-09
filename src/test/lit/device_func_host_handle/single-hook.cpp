// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies: a \c __device__-only function whose address is taken
/// from host code gets an empty-bodied \c __host__ handle synthesized under the
/// device function's own Itanium mangling, tagged with the export-handle
/// annotation.

#include <hip/hip_runtime.h>

__attribute__((device)) void myHook() {}

void hostFunction(const void **out) {
  out[0] = reinterpret_cast<const void *>(&myHook);
}

// clang-format off
/// The handle is registered in @llvm.global.annotations.
/// CHECK: @llvm.global.annotations {{.*}}@_Z6myHookv

/// It is emitted host-side with an empty body, under the device mangling.
/// CHECK: define dso_local void @_Z6myHookv()
/// CHECK-NEXT: entry:
/// CHECK-NEXT: ret void

/// Host code's address-take resolves to the handle.
/// CHECK: define dso_local void @_Z12hostFunctionPPKv
/// CHECK: store ptr @_Z6myHookv
// clang-format on
