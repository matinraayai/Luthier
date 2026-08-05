// clang-format off
/// RUN: not %clangxx -x hip \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -fsyntax-only %s 2>&1 \
/// RUN:   | %tee_out FileCheck %s
// clang-format on
/// Same as host-call-forbidden.cpp but for a templated hook: a direct host
/// call to an explicit instantiation must error.

#include <hip/hip_runtime.h>

template <typename T> __attribute__((device)) void myHook(T) {}

void hostFunction() { myHook<int>(0); }

// clang-format off
/// CHECK: error: no matching function for call to 'myHook'
// clang-format on
