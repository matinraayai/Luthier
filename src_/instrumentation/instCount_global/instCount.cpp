#include "hip/hip_runtime.h"
#include <stdio.h>
#define CHECK(cmd)                                                             \
  {                                                                            \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),  \
              error, __FILE__, __LINE__);                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

__managed__ __device__ unsigned long counter = 0;

extern "C" __global__ __noinline__ void incr_counter() {
  atomicAdd(&counter, 1);
}

int main() {
  static int device = 0;
  CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
  printf("info: running on device %s\n", props.name);
  const unsigned blocks = 512;
  const unsigned threadsPerBlock = 256;
  hipLaunchKernelGGL(incr_counter, dim3(blocks), dim3(threadsPerBlock), 0, 0);
  hipDeviceSynchronize();
  printf("counter = %lu\n", counter);
  return 0;
}
