#ifndef SIBIR_IMPL
#define SIBIR_IMPL

namespace Sibir {
    static void init();
    static void destroy();
    static void hip_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
    static void hsa_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
};


#endif
