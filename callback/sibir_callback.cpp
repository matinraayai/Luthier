#include "sibir_callback.hpp"

hipError_t hipMalloc(void** ptr, size_t size) {
    if (original_hipMalloc == nullptr)
        original_hipMalloc = reinterpret_cast<hipError_t(*)(void**, size_t)>(dlsym(RTLD_NEXT, "hipMalloc"));
    if (original_hipMalloc == nullptr)
        std::cerr << "Failed to transfer." << std::endl;
    std::cout << "Overridden with values: " << ptr <<", and " << size << "." << std::endl;
    auto out = original_hipMalloc(ptr, size);
    std::cout << "hipMalloc returned: " << out << std::endl;
    return out;
}