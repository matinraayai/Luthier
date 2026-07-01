// clang-format off
/// RUN: %clangxx -x hip --offload-arch=gfx90a \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-host-only -emit-llvm -S %s -o - 2>&1 \
/// RUN:   | %tee_out FileCheck %s --check-prefix=HOST \
/// RUN:     --implicit-check-not=llvm.amdgcn.
/// RUN: %clangxx -x hip --offload-arch=gfx90a \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include --cuda-device-only -nogpulib -emit-llvm \
/// RUN:   -S %s -o - 2>&1 | %tee_out FileCheck %s --check-prefix=DEVICE
// clang-format on
/// Verifies correct handle creation of constexpr \c __device__ functions.

#include <hip/hip_runtime.h>

__attribute__((device)) constexpr int myAdd(int a) { return a + 1; }

__attribute__((device, host)) constexpr int myAdd2(int a) { return a + 2; }

__attribute__((device)) constexpr int myAdd3(int a) { return a * 2; }

__attribute__((host)) constexpr int myAdd3(int a) { return a + 5; }

__global__ void square(int *arr, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n)
    arr[tid] = arr[tid] * arr[tid] + myAdd(2) + myAdd2(3) + myAdd3(4);
}

void hostFunction(const void **out) {
  out[0] = reinterpret_cast<const void *>(&myAdd);
  out[1] = reinterpret_cast<const void *>(&myAdd2);
  out[2] = reinterpret_cast<const void *>(&myAdd3);
}

// clang-format off
/// Host: each constexpr function is exported — tagged with the export-handle
/// marker (@.str), emitted host-side, and reached by the host address-take:
///   - myAdd  (__device__ constexpr): CUDA treats it as __host__ __device__,
///            so its host stub is annotated in place (no synthesized handle);
///   - myAdd2 (explicit __host__ __device__ constexpr): annotated in place;
///   - myAdd3 (__device__ constexpr + __host__ constexpr overload):
///            the host overload is annotated.
/// --implicit-check-not on the RUN line proves no AMDGCN intrinsic leaks host.
/// HOST-DAG: @.str = {{.*}}"luthier.export_function_handle
/// HOST-DAG: @_Z5myAddi, ptr @.str
/// HOST-DAG: @_Z6myAdd2i, ptr @.str
/// HOST-DAG: @_Z6myAdd3i, ptr @.str
/// HOST-DAG: define {{.*}}i32 @_Z5myAddi(i32
/// HOST-DAG: define {{.*}}i32 @_Z6myAdd2i(i32
/// HOST-DAG: define {{.*}}i32 @_Z6myAdd3i(i32
/// HOST-DAG: store ptr @_Z5myAddi
/// HOST-DAG: store ptr @_Z6myAdd2i
/// HOST-DAG: store ptr @_Z6myAdd3i

/// Device: the constexpr functions handles are emitted under their mangled
/// names.
/// DEVICE-DAG: define {{.*}}i32 @_Z5myAddi(i32
/// DEVICE-DAG: define {{.*}}i32 @_Z6myAdd2i(i32
/// DEVICE-DAG: define {{.*}}i32 @_Z6myAdd3i(i32
// clang-format on
