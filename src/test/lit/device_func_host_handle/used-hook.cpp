// clang-format off
/// RUN: %clangxx -x hip -O2 \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies that a \c __device__ function carrying \c __attribute__((used)) is
/// exported even when nothing references it from host code: the synthesized
/// \c __host__ handle inherits `used`, so it lands in \c @llvm.compiler.used
/// and its export annotation reaches \c @llvm.global.annotations.

#include <hip/hip_runtime.h>

__attribute__((device, used)) void usedHook() {
  unsigned long long Exec = __builtin_amdgcn_read_exec();
  (void)Exec;
}

// clang-format off
/// CHECK: @llvm.global.annotations {{.*}}@_Z8usedHookv
/// The handle inherited `used`, so it appears in the compiler-used list.
/// CHECK: @llvm.compiler.used {{.*}}@_Z8usedHookv
/// CHECK: define dso_local void @_Z8usedHookv()
/// CHECK-NEXT: entry:
/// CHECK-NEXT: ret void
/// CHECK-NOT: llvm.amdgcn.
// clang-format on
