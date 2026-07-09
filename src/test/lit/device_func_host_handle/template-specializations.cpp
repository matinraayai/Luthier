// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies per-specialization export when a <tt>__host__ __device__</tt>
/// function template is declared without a body and the user provides distinct
/// explicit specializations that are each address-taken from host code. Because
/// the template is already host-callable, each addressed specialization is
/// annotated in place (no synthesized stub) and keeps its own body.

#include <hip/hip_runtime.h>

/// __host__ __device__ primary template, declared without a body.
template <typename T> __attribute__((host, device)) T tmplHook(T x);

/// User-defined explicit specializations with distinct bodies.
template <> __attribute__((host, device)) int tmplHook<int>(int x) {
  return x + 1;
}
template <> __attribute__((host, device)) float tmplHook<float>(float x) {
  return x * 2.0f;
}

void hostFunction(const void **out) {
  int (*pi)(int) = tmplHook<int>;
  float (*pf)(float) = tmplHook<float>;
  out[0] = reinterpret_cast<const void *>(pi);
  out[1] = reinterpret_cast<const void *>(pf);
}

// clang-format off
/// Both explicit specializations are exported under their own manglings.
/// CHECK: @.str = {{.*}}"luthier.function.export_device_handle
/// CHECK: @llvm.global.annotations = {{.*}}[2 x
/// CHECK-SAME: @_Z8tmplHookIiET_S0_
/// CHECK-SAME: @_Z8tmplHookIfET_S0_

/// Each specialization is emitted with its own distinct body — proving they are
/// independently exported, not collapsed to a single stub.
/// CHECK: define {{.*}}i32 @_Z8tmplHookIiET_S0_(i32
/// CHECK: add nsw i32 {{.*}}, 1
/// CHECK: define {{.*}}float @_Z8tmplHookIfET_S0_(float
/// CHECK: fmul contract float {{.*}}, 2.000000e+00

/// Host use-sites resolve to the per-specialization handles.
/// CHECK: store ptr @_Z8tmplHookIiET_S0_
/// CHECK: store ptr @_Z8tmplHookIfET_S0_
/// CHECK-NOT: llvm.amdgcn.
// clang-format on
