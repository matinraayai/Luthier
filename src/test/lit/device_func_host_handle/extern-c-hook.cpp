// clang-format off
/// RUN: %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format off
/// Verifies handle synthesis for a \c __device__ function declared inside an
/// \c `extern "C"` block. The handle inherits C linkage (it's added to the same
/// DeclContext as the original), so its IR symbol is the source identifier
/// verbatim — no Itanium mangling. This exercises the IR-pass path that uses
/// the symbol directly when demangling a non-Itanium name fails.

#include <hip/hip_runtime.h>

extern "C" {

__attribute__((device)) void myCHook() {}

} // extern "C"

void hostFunction(const void **out) {
  out[0] = reinterpret_cast<const void *>(&myCHook);
}

// clang-format off
/// The handle has C linkage; its host symbol is the source identifier verbatim,
/// and it is annotated.
/// CHECK: @llvm.global.annotations {{.*}}@myCHook
/// CHECK: define dso_local void @myCHook()
/// CHECK-NEXT: entry:
/// CHECK-NEXT: ret void

/// Host code's address-take resolves to the C-linkage handle.
/// CHECK: store ptr @myCHook
// clang-format on
