// clang-format off
/// RUN: %clangxx -x hip --offload-arch=gfx908 \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-device-only -nogpulib -emit-llvm \
/// RUN:   -S %s -o - 2>&1 | %tee_out FileCheck %s
// clang-format on
/// Verifies that on the device-side compile the plugin is inert (it only acts
/// on host compiles): the original \c __device__ function is emitted under its
/// natural Itanium mangling for the IModule extraction path, with no host
/// handle or export annotation added.

__attribute__((device, used)) void myHook() {}

// clang-format off
/// CHECK: define {{.*}}void @_Z6myHookv()
/// CHECK-NOT: @llvm.global.annotations
// clang-format on
