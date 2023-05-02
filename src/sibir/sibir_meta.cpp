#include "sibir_meta.hpp"
#include "sibir.h"
#include <sibir_impl.hpp>

hipError_t __hipPopCallConfiguration(dim3 *gridDim,
                                     dim3 *blockDim,
                                     size_t *sharedMem,
                                     hipStream_t *stream) {
    auto sibir = Sibir::getInstance();
    sibir.intercept();
    auto original_function = reinterpret_cast<hipError_t(*)(dim3*, dim3*, size_t*, hipStream_t*)>(sibir.getOriginalHipSymbol("__hipPopCallConfiguration"));
    return original_function(gridDim, blockDim, sharedMem, stream);
}


hipError_t hipMalloc(void** ptr, size_t size) {
    auto sibir = Sibir::getInstance();
    sibir.intercept();
    auto original_hipMalloc = reinterpret_cast<hipError_t(*)(void**, size_t)>(Sibir::getInstance().getOriginalHipSymbol("hipMalloc"));
    return original_hipMalloc(ptr, size);
}