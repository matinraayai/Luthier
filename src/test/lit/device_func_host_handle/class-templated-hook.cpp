/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
/// Verifies per-specialization handles when the hook is a function template
/// that is also a static member of an anonymous-namespace class. Each
/// instantiation addressed from host gets an empty-bodied handle under its own
/// natural mangling, annotated for harvesting, and no AMDGCN intrinsic leaks
/// into host emission. Covers the three export paths at template scope:
///   - hook  (__device__ only): a __host__ handle is synthesized per
///            instantiation (hook<int>, hook<float>);
///   - hook2 (explicit __host__ __device__): the host stub is annotated in
///            place (hook2<int>);
///   - hook3 (__device__ + separate __host__ template overload): the host
///            overload is annotated in place (hook3<int>).

#include <hip/hip_runtime.h>

namespace {

struct Tool {
  template <typename T> __attribute__((device)) static void hook(T x) {
    unsigned long long Exec = __builtin_amdgcn_read_exec();
    (void)Exec;
    (void)x;
  }

  template <typename T> __attribute__((device, host)) static void hook2(T x) {
    (void)x;
  }

  template <typename T> __attribute__((device)) static void hook3(T x) {
    unsigned long long Exec = __builtin_amdgcn_read_exec();
    (void)Exec;
    (void)x;
  }

  template <typename T> __attribute__((host)) static void hook3(T x) {
    (void)x;
  }
};

} // namespace

void hostFunction(const void **out) {
  void (*pi)(int) = Tool::hook<int>;
  void (*pf)(float) = Tool::hook<float>;
  void (*p2)(int) = Tool::hook2<int>;
  void (*p3)(int) = Tool::hook3<int>;
  out[0] = reinterpret_cast<const void *>(pi);
  out[1] = reinterpret_cast<const void *>(pf);
  out[2] = reinterpret_cast<const void *>(p2);
  out[3] = reinterpret_cast<const void *>(p3);
}

// clang-format off
/// Each addressed instantiation is exported (annotated for harvesting).
/// CHECK: @.str = {{.*}}"luthier.function.{{.*}}export_device_handle
/// CHECK: @llvm.global.annotations = {{.*}}[4 x
/// CHECK-DAG: @_ZN12_GLOBAL__N_14Tool4hookIiEEvT_
/// CHECK-DAG: @_ZN12_GLOBAL__N_14Tool4hookIfEEvT_
/// CHECK-DAG: @_ZN12_GLOBAL__N_14Tool5hook2IiEEvT_
/// CHECK-DAG: @_ZN12_GLOBAL__N_14Tool5hook3IiEEvT_

/// One per-specialization handle per instantiation, under natural manglings.
/// CHECK-DAG: define {{.*}}void @_ZN12_GLOBAL__N_14Tool4hookIiEEvT_(i32
/// CHECK-DAG: define {{.*}}void @_ZN12_GLOBAL__N_14Tool4hookIfEEvT_(float
/// CHECK-DAG: define {{.*}}void @_ZN12_GLOBAL__N_14Tool5hook2IiEEvT_(i32
/// CHECK-DAG: define {{.*}}void @_ZN12_GLOBAL__N_14Tool5hook3IiEEvT_(i32

/// Host use-sites resolve to the per-specialization handles.
/// CHECK-DAG: store ptr @_ZN12_GLOBAL__N_14Tool4hookIiEEvT_
/// CHECK-DAG: store ptr @_ZN12_GLOBAL__N_14Tool4hookIfEEvT_
/// CHECK-DAG: store ptr @_ZN12_GLOBAL__N_14Tool5hook2IiEEvT_
/// CHECK-DAG: store ptr @_ZN12_GLOBAL__N_14Tool5hook3IiEEvT_

/// No AMDGCN intrinsic leaks to host emission.
/// CHECK-NOT: llvm.amdgcn.
// clang-format on
