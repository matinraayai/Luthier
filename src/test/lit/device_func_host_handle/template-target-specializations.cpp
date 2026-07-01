// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s --check-prefix=HOST
/// RUN: %clangxx -x hip --offload-arch=gfx908 \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-device-only -nogpulib -emit-llvm \
/// RUN:   -S %s -o - 2>&1 | %tee_out FileCheck %s --check-prefix=DEVICE
// clang-format on
/// Verifies separate \c __device__ and \c __host__ explicit specializations of
/// the same template for the same type. The host specialization is exported
/// (annotated in place) on the host side, while the device specialization stays
/// device-only; both carry distinct bodies and share a mangling.

#include <hip/hip_runtime.h>

/// Two overloaded primary templates, distinguished by target.
template <typename T> __attribute__((device)) T tmplHook(T x);
template <typename T> __attribute__((host)) T tmplHook(T x);

/// Separate \c __device__ and \c __host__ explicit specializations for double.
template <> __attribute__((device)) double tmplHook<double>(double x) {
  return x + 3.0;
}
template <> __attribute__((host)) double tmplHook<double>(double x) {
  return x + 4.0;
}

void hostFunction(const void **out) {
  double (*pd)(double) = tmplHook<double>;
  out[0] = reinterpret_cast<const void *>(pd);
}

// clang-format off
/// Host side: the \c __host__ specialization is exported; its body is `x + 4.0`,
/// and the host use-site resolves to it. Only the host specialization exists in
/// the host module (the \c __device__ one never enters host emission).
/// HOST: @.str = {{.*}}"luthier.export_function_handle
/// HOST: @llvm.global.annotations = {{.*}}[1 x
/// HOST-SAME: @_Z8tmplHookIdET_S0_
/// HOST: define {{.*}}double @_Z8tmplHookIdET_S0_(double
/// HOST: fadd contract double {{.*}}, 4.000000e+00
/// HOST: store ptr @_Z8tmplHookIdET_S0_

/// Device side: the \c __device__ specialization is emitted under the same
/// mangling, carrying its own distinct body `x + 3.0`.
/// DEVICE: define {{.*}}double @_Z8tmplHookIdET_S0_(double
/// DEVICE: fadd contract double {{.*}}, 3.000000e+00
// clang-format on
