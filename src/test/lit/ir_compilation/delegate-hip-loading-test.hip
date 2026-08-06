/// A minimal HIP tool source to confirm correct generation of handles and
/// section markers by the \c ToolDeviceCodeOffloadParserPass, when compiled
/// in one-shot (device + host in a single invocation) and
/// separable (device compiled first into a single-code-object fat binary that is
/// then included into the host compile via -fcuda-include-gpubinary) manners.
/// The test confirms that not only the IR transformation works, but the final
/// linked object's boundaries are also correctly populated (defined, non-zero),
/// regardless of the compilation process used.

/// ===========================================================================
/// Device + host in a single invocation.
/// ===========================================================================
// clang-format off
/// RUN: %clangxx -x hip --offload-arch=gfx908 -nogpulib -fuse-cuid=none -fPIC \
/// RUN:   -fpass-plugin=%luthier_tool_ir_compilation_plugin_path \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include -emit-llvm -S %s -o %t.oneshot.ll
/// RUN: FileCheck %s --check-prefix=CHECK --implicit-check-not=__hipRegister \
/// RUN:   --implicit-check-not=__hip_module_ctor \
/// RUN:   --implicit-check-not=__hip_register_globals < %t.oneshot.ll
/// RUN: %clangxx -fPIC -c %t.oneshot.ll -o %t.oneshot.o
/// RUN: %clangxx -shared -fPIC %t.oneshot.o -o %t.oneshot.so
/// RUN: llvm-readelf -S -s %t.oneshot.so | %tee_out FileCheck %s --check-prefix=ELF
// clang-format on

/// ===========================================================================
/// Device-only compile -> single-code-object fat binary -> include
/// into the host compile.
/// ===========================================================================

// clang-format off
/// RUN: %clangxx -x hip --cuda-device-only --offload-arch=gfx908 -nogpulib \
/// RUN:   -emit-llvm --no-gpu-bundle-output -I/opt/rocm/include -c %s -o %t.dev.bc
/// RUN: clang-offload-bundler --type=o \
/// RUN:   --targets=hipv4-amdgcn-amd-amdhsa--gfx908 --input=%t.dev.bc \
/// RUN:   --output=%t.dev.hipfb --bundle-align=8
/// RUN: %clangxx -x hip --cuda-host-only -fuse-cuid=none -fPIC \
/// RUN:   -Xclang -fcuda-include-gpubinary -Xclang %t.dev.hipfb \
/// RUN:   -fpass-plugin=%luthier_tool_ir_compilation_plugin_path \
/// RUN:   -fplugin=%luthier_tool_cxx_compilation_plugin_path \
/// RUN:   -Xclang -add-plugin -Xclang luthier-emit-device-function-host-handle \
/// RUN:   -I/opt/rocm/include -emit-llvm -S %s -o %t.sep.ll
/// RUN: FileCheck %s --check-prefix=CHECK --implicit-check-not=__hipRegister \
/// RUN:   --implicit-check-not=__hip_module_ctor \
/// RUN:   --implicit-check-not=__hip_register_globals < %t.sep.ll
/// RUN: %clangxx -fPIC -c %t.sep.ll -o %t.sep.o
/// RUN: %clangxx -shared -fPIC %t.sep.o -o %t.sep.so
/// RUN: llvm-readelf -S -s %t.sep.so | %tee_out FileCheck %s --check-prefix=ELF
// clang-format on

#include <hip/hip_runtime.h>

namespace luthier {
struct HipHandleInfo {
  void *HostHandle{nullptr};
  const char *DeviceName{nullptr};
};
} // namespace luthier

__global__ void luthierKernel() {}
__device__ int luthierDeviceVar = 0;
__managed__ int luthierManagedVar = 0;
texture<float, 1, hipReadModeElementType> luthierTex;

template <typename T, int Dim>
struct __attribute__((device_builtin_surface_type)) luthierSurface {};
__device__ luthierSurface<void, 2> luthierSurf;

__device__ void luthierDeviceFunc() {}
void hostTakeAddr(const void **out) {
  out[0] = reinterpret_cast<const void *>(&luthierDeviceFunc);
}

__attribute__((used, annotate("luthier_clang_offload_section_begin")))
const char *FatBegin;
__attribute__((used, annotate("luthier_clang_offload_section_end")))
const char *FatEnd;
__attribute__((used, annotate("luthier_hip_handle_section_begin")))
luthier::HipHandleInfo *HipBegin;
__attribute__((used, annotate("luthier_hip_handle_section_end")))
luthier::HipHandleInfo *HipEnd;

// ===========================================================================
// (1) IR transformation (identical for both compilation paths).
// ===========================================================================

// CHECK-DAG: %"struct.luthier::HipHandleInfo" = type { ptr, ptr }

// Every registered kind plus the exported device function collapse into one
// six-element array in the luthier_hip_handles section, each pairing the host
// handle with its device-name string.
// CHECK-DAG: @[[HDATA:[._a-zA-Z0-9]+]] = private constant [6 x %"struct.luthier::HipHandleInfo"] [
// CHECK-DAG: %"struct.luthier::HipHandleInfo" { ptr @_Z13luthierKernelv, ptr @{{[0-9]+}} }
// CHECK-DAG: %"struct.luthier::HipHandleInfo" { ptr @luthierDeviceVar, ptr @{{[0-9]+}} }
// CHECK-DAG: %"struct.luthier::HipHandleInfo" { ptr @luthierManagedVar, ptr @{{[0-9]+}} }
// CHECK-DAG: %"struct.luthier::HipHandleInfo" { ptr @luthierTex, ptr @{{[0-9]+}} }
// CHECK-DAG: %"struct.luthier::HipHandleInfo" { ptr @luthierSurf, ptr @{{[0-9]+}} }
// CHECK-DAG: %"struct.luthier::HipHandleInfo" { ptr @_Z17luthierDeviceFuncv, ptr @.luthier.device_fn_name
// CHECK-DAG: section "luthier_hip_handles"

// The fat binary is moved into luthier_fatbin, and both it and the handle array
// are retained via llvm.used.
// CHECK-DAG: @__luthier_fatbin = {{.*}} section "luthier_fatbin"
// CHECK-DAG: @llvm.used = {{.*}}@__luthier_fatbin{{.*}}@[[HDATA]]

// At the IR level the boundary symbols are external declarations (the linker
// defines them); the four slots point at them.
// CHECK-DAG: @__start_luthier_fatbin = external constant i8
// CHECK-DAG: @__stop_luthier_fatbin = external constant i8
// CHECK-DAG: @__start_luthier_hip_handles = external constant i8
// CHECK-DAG: @__stop_luthier_hip_handles = external constant i8
// CHECK-DAG: @FatBegin = {{.*}}ptr @__start_luthier_fatbin
// CHECK-DAG: @FatEnd = {{.*}}ptr @__stop_luthier_fatbin
// CHECK-DAG: @HipBegin = {{.*}}ptr @__start_luthier_hip_handles
// CHECK-DAG: @HipEnd = {{.*}}ptr @__stop_luthier_hip_handles

// ===========================================================================
// (2) After linking, the sections exist and the boundary symbols are resolved
// by the linker. The fat-binary size varies by compilation path, so only its
// section + retain flag is checked; the handle section is exactly six 16-byte
// { ptr, ptr } records (0x60). Both sections are retained (R = SHF_GNU_RETAIN).
// ===========================================================================

// ELF-DAG: luthier_fatbin {{.*}} AR
// ELF-DAG: luthier_hip_handles {{.*}} 000060 00 WAR

// Each boundary symbol is DEFINED (a numeric section index, not UND) and holds
// a non-zero address — i.e. the __start/__stop trick is populated, not left as
// external/undefined.
// ELF-DAG: : {{0*[1-9a-f][0-9a-f]*}} {{.*}} {{[0-9]+}} __start_luthier_fatbin
// ELF-DAG: : {{0*[1-9a-f][0-9a-f]*}} {{.*}} {{[0-9]+}} __stop_luthier_fatbin
// ELF-DAG: : {{0*[1-9a-f][0-9a-f]*}} {{.*}} {{[0-9]+}} __start_luthier_hip_handles
// ELF-DAG: : {{0*[1-9a-f][0-9a-f]*}} {{.*}} {{[0-9]+}} __stop_luthier_hip_handles
