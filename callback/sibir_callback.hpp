#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <dlfcn.h>


static hipError_t (*original_hipMalloc)(void**, size_t)  = nullptr;

void __attribute__((constructor)) run_me_first() {
    std::cout << "I was run first!" << std::endl;
}

hipError_t hipMalloc(void** ptr, size_t size);