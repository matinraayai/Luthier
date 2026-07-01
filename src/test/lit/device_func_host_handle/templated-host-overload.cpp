// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies that when a user-provided \c __host__ overload of a \c __device__
/// function template already exists, the plugin uses (annotates) that host
/// overload instead of synthesizing a new empty handle. The exported symbol
/// carries the user's host body — not an empty stub and not the device body —
/// and no AMDGCN intrinsic leaks into host emission.

#include <hip/hip_runtime.h>

template <typename T> __attribute__((device)) T tmplHook(T x) {
  unsigned long long Exec = __builtin_amdgcn_read_exec();
  (void)Exec;
  return x;
}

/// The user's own __host__ overload: its body is `x + x`.
template <typename T> __attribute__((host)) T tmplHook(T x) { return x + x; }

void hostFunction(const void **out) {
  int (*pi)(int) = tmplHook<int>;
  out[0] = reinterpret_cast<const void *>(pi);
}

// clang-format off
/// The host overload's specialization is exported and its use-site resolves to
/// it.
/// CHECK: @.str = {{.*}}"luthier.export_function_handle
/// CHECK: @llvm.global.annotations {{.*}}@_Z8tmplHookIiET_S0_
/// CHECK: store ptr @_Z8tmplHookIiET_S0_
/// The emitted definition is the user's __host__ body (x + x), proving the
/// existing host overload was used rather than a synthesized empty stub.
/// CHECK: define {{.*}}i32 @_Z8tmplHookIiET_S0_(i32
/// CHECK: add nsw i32
/// CHECK: ret i32
/// CHECK-NOT: llvm.amdgcn.
// clang-format on
