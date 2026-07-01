// clang-format off
/// RUN: %clangxx -x hip -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s --implicit-check-not=llvm.amdgcn.
// clang-format on
/// Verifies handle export for static members of an anonymous-namespace class
/// across the three export paths, each address-taken from host code:
///   - hook  (__device__ only, intrinsic body): a __host__ handle is
///            synthesized as an empty sibling member; the intrinsic-bearing
///            __device__ body never enters host emission (nothing leaks to
///            host);
///   - hook2 (explicit __host__ __device__): the host stub is annotated in
///            place, no synthesized handle;
///   - hook3 (__device__ + user-provided __host__ overload): the host overload
///            is annotated in place.

#include <hip/hip_runtime.h>

namespace {

struct Tool {
  __attribute__((device)) static void hook() {
    unsigned long long Exec = __builtin_amdgcn_read_exec();
    (void)Exec;
  }

  __attribute__((device, host)) static void hook2() {}

  __attribute__((device)) static void hook3() {
    unsigned long long Exec = __builtin_amdgcn_read_exec();
    (void)Exec;
  }

  __attribute__((host)) static void hook3() {}
};

} // namespace

void hostFunction(const void **out) {
  out[0] = reinterpret_cast<const void *>(&Tool::hook);
  out[1] = reinterpret_cast<const void *>(&Tool::hook2);
  out[2] = reinterpret_cast<const void *>(&Tool::hook3);
}

// clang-format off
/// All three handles carry the export annotation and the host address-takes
/// resolve to them. Internal-linkage members are emitted lazily, after their
/// referencing function, so the stores precede the definitions. Each handle is
/// emitted with an empty body — the original intrinsic-bearing __device__ body
/// is not in the host module (enforced by --implicit-check-not above).
/// CHECK: @.str = {{.*}}"luthier.export_function_handle
/// CHECK: @llvm.global.annotations
/// CHECK-SAME: @_ZN12_GLOBAL__N_14Tool4hookEv
/// CHECK-SAME: @_ZN12_GLOBAL__N_14Tool5hook2Ev
/// CHECK-SAME: @_ZN12_GLOBAL__N_14Tool5hook3Ev
/// CHECK: store ptr @_ZN12_GLOBAL__N_14Tool4hookEv
/// CHECK: store ptr @_ZN12_GLOBAL__N_14Tool5hook2Ev
/// CHECK: store ptr @_ZN12_GLOBAL__N_14Tool5hook3Ev
/// CHECK: define internal void @_ZN12_GLOBAL__N_14Tool4hookEv()
/// CHECK-NEXT: entry:
/// CHECK-NEXT: ret void
/// CHECK: define internal void @_ZN12_GLOBAL__N_14Tool5hook2Ev()
/// CHECK-NEXT: entry:
/// CHECK-NEXT: ret void
/// CHECK: define internal void @_ZN12_GLOBAL__N_14Tool5hook3Ev()
/// CHECK-NEXT: entry:
/// CHECK-NEXT: ret void
// clang-format on
