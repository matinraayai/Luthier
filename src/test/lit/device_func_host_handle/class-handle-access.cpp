/// Private RUN line: addressing the handle from outside the class is rejected.
/// The synthesized \c __host__ handle inherits the \c __device__ member's
/// private access, so the out-of-class address-take is an access violation.
// clang-format off
/// RUN: not %clangxx -x hip -DACCESS=private \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -fsyntax-only %s 2>&1 \
/// RUN:   | %tee_out FileCheck %s --check-prefix=PRIVATE
// clang-format on
/// Public RUN line: the handle is public, so the same out-of-class address-take
/// is well-formed; the handle is emitted and annotated.
// clang-format off
/// RUN: %clangxx -x hip -DACCESS=public \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s --check-prefix=PUBLIC
// clang-format on
/// Confirms a synthesized \c __host__ handle carries the same access specifier
/// as its associated \c __device__ member: the identical out-of-class
/// address-take is well-formed for a public member but an access violation for
/// a private one. The two RUN lines flip the member's access via the ACCESS
/// macro.

#include <hip/hip_runtime.h>

class Tool {
  ACCESS : __attribute__((device)) static void hook() {}
};

void hostFunction(const void **out) {
  out[0] = reinterpret_cast<const void *>(&Tool::hook);
}

// clang-format off
/// PRIVATE: error: 'hook' is a private member of 'Tool'
/// PRIVATE: note: declared private here

/// PUBLIC: @.str = {{.*}}"luthier.function.synthesized_export_device_handle
/// PUBLIC: @llvm.global.annotations {{.*}}@_ZN4Tool4hookEv
/// PUBLIC: define {{.*}}void @_ZN4Tool4hookEv()
/// PUBLIC-NEXT: entry:
/// PUBLIC-NEXT: ret void
/// PUBLIC: store ptr @_ZN4Tool4hookEv
// clang-format on
