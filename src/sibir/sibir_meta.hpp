#ifndef SRC_SIBIR_SRC_SIBIR_SIBIR_META_HPP_
#define SRC_SIBIR_SRC_SIBIR_SIBIR_META_HPP_
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

hipError_t __hipPopCallConfiguration(dim3 *gridDim,
                                     dim3 *blockDim,
                                     size_t *sharedMem,
                                     hipStream_t *stream);


hipError_t hipMalloc(void** ptr, size_t size);

#endif
