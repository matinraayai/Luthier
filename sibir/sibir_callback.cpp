#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <dlfcn.h>
#include "sibir.h"

void __attribute__((constructor)) SibirConstructor() {
    sibir_at_init();
}

void __attribute__((destructor)) SibirDestructor() {
    sibir_at_term();
}


static hipError_t (*original_hipMalloc)(void**, size_t)  = nullptr;


hipError_t hipMalloc(void** ptr, size_t size) {
    if (original_hipMalloc == nullptr)
        original_hipMalloc = reinterpret_cast<hipError_t(*)(void**, size_t)>(dlsym(RTLD_NEXT, "hipMalloc"));
    if (original_hipMalloc == nullptr)
        std::cerr << "Failed to transfer." << std::endl;
    sibir_at_hipMalloc(ptr, size);
    return original_hipMalloc(ptr, size);
}
