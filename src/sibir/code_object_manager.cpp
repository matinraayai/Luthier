#include "code_object_manager.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include "hsa_intercept.h"


struct __CudaFatBinaryWrapper {
    unsigned int magic;
    unsigned int version;
    void* binary;
    void* dummy1;
};

constexpr unsigned __hipFatMAGIC2 = 0x48495046;  // "HIPF"


constexpr char const* CLANG_OFFLOAD_BUNDLER_MAGIC_STR = "__CLANG_OFFLOAD_BUNDLE__";
constexpr char const* OFFLOAD_KIND_HIP = "hip";
constexpr char const* OFFLOAD_KIND_HIPV4 = "hipv4";
constexpr char const* OFFLOAD_KIND_HCC = "hcc";
constexpr char const* AMDGCN_TARGET_TRIPLE = "amdgcn-amd-amdhsa-";


void SibirCodeObjectManager::registerFatBinary(const void* data) {
    assert(data != nullptr);
    auto fb_wrapper = reinterpret_cast<const __CudaFatBinaryWrapper*>(data);
    assert(fb_wrapper->magic == __hipFatMAGIC2 && fb_wrapper->version == 1);
    if (!fatBinaries_.contains(fb_wrapper->binary)) {
        amd_comgr_data_t fb_data;
        SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &fb_data));
        SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(fb_data, 4096, reinterpret_cast<const char*>(fb_wrapper->binary)));
        fatBinaries_.insert({fb_wrapper->binary, fb_data});

        std::unordered_map<std::string, std::pair<size_t, size_t>> unique_isa_names{{
            "amdgcn-amd-amdhsa--gfx908", {0, 0}}
        };

        // Create a query list using COMGR info for unique ISAs.
        auto query_list_array = new amd_comgr_code_object_info_t[1];
        auto isa_it = unique_isa_names.begin();
        for (size_t isa_idx = 0; isa_idx < unique_isa_names.size(); ++isa_idx) {
            std::advance(isa_it, isa_idx);
            query_list_array[isa_idx].isa = isa_it->first.c_str();
            query_list_array[isa_idx].size = 0;
            query_list_array[isa_idx].offset = 0;
        }


        SIBIR_AMD_COMGR_CHECK(amd_comgr_lookup_code_object(fb_data, query_list_array, unique_isa_names.size()));
        std::cout << "Isa is: " << query_list_array[0].isa << std::endl;
        std::cout << "size: " << query_list_array[0].size << std::endl;
        std::cout << "offset: " << query_list_array[0].offset << std::endl;
        auto co_start = reinterpret_cast<void*>(reinterpret_cast<unsigned char*>(fb_wrapper->binary)
                                                 + query_list_array[0].offset);
        std::cout << "Where it should be located: " << co_start << std::endl;

        delete[] query_list_array;
    }
}

void SibirCodeObjectManager::registerFunction(const void* fatBinary,
                                              const char *funcName, const void *hostFunction, const char *deviceName) {
    if (!fatBinaries_.contains(fatBinary))
        registerFatBinary(fatBinary);
    if (!functions_.contains(funcName))
        functions_.insert({funcName, {funcName, hostFunction, deviceName, fatBinary}});
}

hsa_executable_t SibirCodeObjectManager::getInstrumentationFunction(const char *functionName) {
//    amd_comgr_data_t co
    return hsa_executable_t();
}
