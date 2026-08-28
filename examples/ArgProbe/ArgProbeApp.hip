//===-- ArgProbeApp.hip - Simple driver for the ArgProbe tool ---*- C++ -*-===//
// Copyright @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License").
//===----------------------------------------------------------------------===//
///
/// \file
/// Tiny HIP program used as a target for the \c ArgProbe Luthier tool.
///
/// The kernel is intentionally minimal: a single global load per thread. The
/// tool instruments THAT load, so we want exactly one load per lane per
/// dispatch — anything fancier would multiply the payload's printf traffic
/// without making the arg-probe test any more revealing. Grid geometry is
/// picked so several distinct workgroup / workitem IDs show up in the
/// payload's readSVA output.
///
/// Usage:
///   LD_PRELOAD=<path-to-liblauncher>.so \
///   HSA_TOOLS_LIB=path/to/libLuthierArgProbe.so \
///   ./LuthierArgProbeApp
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

//===----------------------------------------------------------------------===//
// The target kernel.
//===----------------------------------------------------------------------===//

/// One global load per lane. Deliberately kept as a plain assignment so
/// CodeGen produces a single \c global_load_dword: no address arithmetic,
/// no reductions, no branching. The tool instruments the load, and the
/// injected payload is what actually does the interesting reads (dispatch
/// geometry, hidden args, SGPRs, VGPRs, hostcall printf, etc.).
__global__ void probeTarget(const uint32_t *In, uint32_t *Out) {
  unsigned Tid = blockIdx.x * blockDim.x + threadIdx.x;
  Out[Tid] = In[Tid];
}

//===----------------------------------------------------------------------===//
// Driver.
//===----------------------------------------------------------------------===//

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      std::fprintf(stderr, "%s:%d %s failed: %s\n", __FILE__, __LINE__,        \
                   #expr, hipGetErrorString(_e));                              \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

int main() {
  // Small workload: two workgroups of four threads. Enough distinct IDs
  // for the readSVA payloads to show non-zero values without flooding
  // the printf hostcall.
  constexpr unsigned BlockDim = 4;
  constexpr unsigned GridDim = 2;
  constexpr unsigned N = BlockDim * GridDim;

  uint32_t *DevIn = nullptr;
  uint32_t *DevOut = nullptr;
  HIP_CHECK(hipMalloc(&DevIn, N * sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&DevOut, N * sizeof(uint32_t)));

  uint32_t HostIn[N];
  for (unsigned I = 0; I < N; ++I)
    HostIn[I] = 0xA5000000u | I;
  HIP_CHECK(hipMemcpy(DevIn, HostIn, N * sizeof(uint32_t),
                      hipMemcpyHostToDevice));

  std::printf("[app] launching probeTarget grid=%u block=%u\n", GridDim,
              BlockDim);
  hipLaunchKernelGGL(probeTarget, dim3(GridDim), dim3(BlockDim), 0, 0, DevIn,
                     DevOut);
  HIP_CHECK(hipDeviceSynchronize());
  std::printf("[app] probeTarget completed\n");

  uint32_t HostOut[N] = {};
  HIP_CHECK(hipMemcpy(HostOut, DevOut, N * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  bool OK = true;
  for (unsigned I = 0; I < N; ++I)
    OK &= (HostOut[I] == HostIn[I]);
  std::printf("[app] result %s\n", OK ? "OK" : "MISMATCH");

  HIP_CHECK(hipFree(DevIn));
  HIP_CHECK(hipFree(DevOut));
  return OK ? 0 : 1;
}
