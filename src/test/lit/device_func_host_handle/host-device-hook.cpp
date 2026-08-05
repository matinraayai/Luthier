// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies that a <tt>__host__ __device__</tt> function is its own host
/// counterpart: no separate handle is synthesized; instead the function itself
/// is tagged with the export annotation in place (here it is `used`, so it is
/// exported even without a host reference).

#include <hip/hip_runtime.h>

__attribute__((host, device, used)) void hdHook() {}

// clang-format off
/// The function itself is annotated and emitted host-side (one definition).
/// CHECK: @llvm.global.annotations {{.*}}@_Z6hdHookv
/// CHECK-COUNT-1: define dso_local void @_Z6hdHookv()
/// CHECK-NOT: define dso_local void @_Z6hdHookv()
// clang-format on
