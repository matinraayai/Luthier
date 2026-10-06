//===-- InitFiniKernels.hip -----------------------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file
/// Device source for \c InstrumentedKernelLoaderE2ETest. Compiled on its own
/// to an AMDGCN relocatable, which is the shape
/// \c InstrumentedKernelLoaderAndLauncher::loadInstrumented takes.
///
/// It is built to force the AMDGPU backend to emit *both* global
/// constructor/destructor kernels, so the loader's init/fini paths are
/// exercised:
///
/// \li \b init: two class-type \c __device__ globals with non-trivial,
///     non-constexpr constructors. Each gets a \c __cxx_global_var_init, both
///     reached from \c _GLOBAL__sub_I_..., which lands in
///     \c llvm.global_ctors and lowers to \c amdgcn.device.init.
/// \li \b fini: HIP routes ordinary C++ destructors through \c __cxa_atexit
///     registered *inside* the init kernel, so a class destructor does not
///     produce a fini kernel. A function marked \c __attribute__((destructor))
///     lands directly in \c llvm.global_dtors, which lowers to
///     \c amdgcn.device.fini.
///
/// The driver needs \c -fgpu-allow-device-init, or clang drops dynamic
/// initializers for device globals and no constructor is emitted at all.
///
/// The constructor also calls \c printf, which is what drags the hostcall
/// machinery in: the kernel blocks on a \c SERVICE_PRINTF hostcall until the
/// loader's listener answers it, so the test fails by hanging (and then by
/// timing out) if the hostcall buffer is not stood up and serviced.
//===----------------------------------------------------------------------===//
#include <hip/hip_runtime.h>

extern "C" {

/// Set by the constructor, cleared by the destructor. The test reads this back
/// to prove each kernel ran, and when.
__device__ int LuthierTestCtorRan = 0;
__device__ int LuthierTestDtorRan = 0;

/// Written by the two variable constructors so the test can tell they ran in
/// the expected order and both took effect.
__device__ int LuthierTestCtorSum = 0;

/// The value the constructor's printf renders. Statically initialized on
/// purpose: clang does not define the order of equal-priority entries in
/// llvm.global_ctors, and in practice it puts luthierTestDeviceInit ahead of
/// the _GLOBAL__sub_I_ that runs the variable initializers, so anything a
/// variable constructor writes is still zero when the printf reads it.
__device__ int LuthierTestPrintfArg = 62;
}

namespace {

/// Non-trivial and non-constexpr, so the front end has to emit a dynamic
/// initializer for every global of this type. The destructor is left trivial
/// on purpose — a non-trivial one would be registered through
/// \c __cxa_atexit inside the init kernel rather than producing a fini kernel.
struct Accumulator {
  int Seed;
  int Doubled;

  __device__ explicit Accumulator(int S) {
    Seed = S;
    // A body the front end cannot constant-fold away, which is what keeps the
    // initializer dynamic.
    Doubled = S * 2 + 1;
    LuthierTestCtorSum += Doubled;
  }
};

__device__ Accumulator AccumulatorA(10);
__device__ Accumulator AccumulatorB(20);

} // namespace

extern "C" {

/// Runs inside \c amdgcn.device.init. The printf is a hostcall: the wave
/// blocks here until the host answers it.
__attribute__((constructor)) __device__ void luthierTestDeviceInit() {
  LuthierTestCtorRan = 1;
  printf("luthier-test: ctor ran, arg=%d\n", LuthierTestPrintfArg);
}

/// Runs inside \c amdgcn.device.fini, dispatched by the loader just before it
/// tears the executable down.
__attribute__((destructor)) __device__ void luthierTestDeviceFini() {
  LuthierTestDtorRan = 1;
  printf("luthier-test: dtor ran\n");
}
}

/// The one and only user-visible kernel in this relocatable — the loader
/// requires exactly one, excluding the constructor/destructor kernels. It
/// reads every global so nothing above is dead-stripped.
extern "C" __global__ void luthierTestKernel(int *Out) {
  *Out = AccumulatorA.Seed + AccumulatorA.Doubled + AccumulatorB.Seed +
         AccumulatorB.Doubled + LuthierTestCtorRan + LuthierTestDtorRan +
         LuthierTestCtorSum + LuthierTestPrintfArg;
}
