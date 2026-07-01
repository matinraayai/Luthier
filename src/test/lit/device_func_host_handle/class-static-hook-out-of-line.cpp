// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s --implicit-check-not=llvm.amdgcn.
// clang-format on
/// Verifies export fires exactly once per hook when static members have a
/// separate in-class declaration and out-of-line definition. The decl and def
/// are seen in different decl groups, but the location-keyed Synthesized set
/// suppresses duplicates; exactly one host-side handle is emitted per hook with
/// an empty body, across the three export paths:
///   - hook  (__device__ only, intrinsic body): synthesized empty __host__
///            handle; the intrinsic-bearing device body never enters host
///            emission;
///   - hook2 (explicit __host__ __device__): host stub annotated in place;
///   - hook3 (__device__ + user-provided __host__ overload): host overload
///            annotated in place.

#include <hip/hip_runtime.h>

namespace {

struct Tool {
  __attribute__((device)) static void hook();

  __attribute__((device, host)) static void hook2();

  __attribute__((device)) static void hook3();

  __attribute__((host)) static void hook3();
};

__attribute__((device)) void Tool::hook() {
  unsigned long long Exec = __builtin_amdgcn_read_exec();
  (void)Exec;
}

__attribute__((device, host)) void Tool::hook2() {}

__attribute__((device)) void Tool::hook3() {
  unsigned long long Exec = __builtin_amdgcn_read_exec();
  (void)Exec;
}

__attribute__((host)) void Tool::hook3() {}

} // namespace

void hostFunction(const void **out) {
  out[0] = reinterpret_cast<const void *>(&Tool::hook);
  out[1] = reinterpret_cast<const void *>(&Tool::hook2);
  out[2] = reinterpret_cast<const void *>(&Tool::hook3);
}

// clang-format off
/// Exactly three annotation entries — one per hook, proving no duplicate
/// export from the separate declaration/definition decl groups.
/// CHECK: @.str = {{.*}}"luthier.export_function_handle
/// CHECK: @llvm.global.annotations = {{.*}}[3 x
/// CHECK-SAME: @_ZN12_GLOBAL__N_14Tool4hookEv
/// CHECK-SAME: @_ZN12_GLOBAL__N_14Tool5hook2Ev
/// CHECK-SAME: @_ZN12_GLOBAL__N_14Tool5hook3Ev
/// The host address-takes resolve to the handles.
/// CHECK: store ptr @_ZN12_GLOBAL__N_14Tool4hookEv
/// CHECK: store ptr @_ZN12_GLOBAL__N_14Tool5hook2Ev
/// CHECK: store ptr @_ZN12_GLOBAL__N_14Tool5hook3Ev
/// Each hook is defined exactly once, host-side, with an empty body.
/// CHECK-COUNT-1: define internal void @_ZN12_GLOBAL__N_14Tool4hookEv()
/// CHECK-NEXT: entry:
/// CHECK-NEXT: ret void
/// CHECK-COUNT-1: define internal void @_ZN12_GLOBAL__N_14Tool5hook2Ev()
/// CHECK-NEXT: entry:
/// CHECK-NEXT: ret void
/// CHECK-COUNT-1: define internal void @_ZN12_GLOBAL__N_14Tool5hook3Ev()
/// CHECK-NEXT: entry:
/// CHECK-NEXT: ret void
/// CHECK-NOT: define internal void @_ZN12_GLOBAL__N_14Tool4hookEv()
/// CHECK-NOT: define internal void @_ZN12_GLOBAL__N_14Tool5hook2Ev()
/// CHECK-NOT: define internal void @_ZN12_GLOBAL__N_14Tool5hook3Ev()
// clang-format on
