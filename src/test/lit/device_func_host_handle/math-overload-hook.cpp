// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies that a standard math function with a CUDA/HIP \c __device__
/// overload, when its host counterpart is address-taken, has that host handle
/// exported (annotated in place). Covers both linkage forms of the host math
/// symbol:
///   - \c sqrtf      (\c ::sqrtf): an external libm function, emitted as a
///                   declaration-only external symbol;
///   - \c std::sqrt  (float overload): an inline in-header definition, emitted
///                   as a linkonce_odr definition.
/// Both are annotated even though they are only referenced (not called) from
/// host code.

#include <hip/hip_runtime.h>
#include <cmath>

void hostFunction(const void **out) {
  // External libm math function (declaration-only, external linkage).
  out[0] =
      reinterpret_cast<const void *>(static_cast<float (*)(float)>(&sqrtf));
  // Inline std math overload (defined in-header, linkonce_odr linkage).
  out[1] =
      reinterpret_cast<const void *>(static_cast<float (*)(float)>(&std::sqrt));
}

// clang-format off
/// Both host math handles are exported.
/// CHECK: @.str = {{.*}}"luthier.export_function_handle
/// CHECK: @llvm.global.annotations
/// CHECK-SAME: @sqrtf
/// CHECK-SAME: @_ZSt4sqrtf

/// Host address-takes resolve to the exported handles.
/// CHECK-DAG: store ptr @sqrtf
/// CHECK-DAG: store ptr @_ZSt4sqrtf

/// The external libm symbol stays a declaration; the inline std overload keeps
/// its in-header linkonce_odr definition.
/// CHECK-DAG: declare dso_local float @sqrtf(
/// CHECK-DAG: define linkonce_odr {{.*}}float @_ZSt4sqrtf(
// clang-format on
