// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Verifies a \c __device__ function template handle synthesis: the consumer
/// clones it into a \c __host__ template, so each specialization addressed from
/// host gets an empty-bodied handle under its own natural Itanium mangling,
/// tagged with the export-handle annotation.

#include <hip/hip_runtime.h>

template <typename T> __attribute__((device)) void myHook(T) {}

void hostFunction(const void **out) {
  void (*pi)(int) = myHook<int>;
  void (*pf)(float) = myHook<float>;
  out[0] = reinterpret_cast<const void *>(pi);
  out[1] = reinterpret_cast<const void *>(pf);
}

// clang-format off
/// One handle per concrete specialization, each annotated for harvesting.
/// CHECK-DAG: @_Z6myHookIiEvT_
/// CHECK-DAG: @_Z6myHookIfEvT_
/// CHECK-DAG: define {{.*}}void @_Z6myHookIiEvT_(i32
/// CHECK-DAG: define {{.*}}void @_Z6myHookIfEvT_(float

/// Host use-sites resolve to the per-specialization handles.
/// CHECK-DAG: store ptr @_Z6myHookIiEvT_
/// CHECK-DAG: store ptr @_Z6myHookIfEvT_
// clang-format on
