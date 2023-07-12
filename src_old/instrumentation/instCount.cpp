#include "hip/hip_runtime.h"

__managed__ __device__ unsigned long counter = 0;

// extern "C" __device__ __noinline__ void incr_counter() {
//   atomicAdd(&counter, 1);
// }
extern "C" __global__ void incr_counter() { atomicAdd(&counter, 1); }