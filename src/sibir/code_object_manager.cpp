#include "code_object_manager.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include "hsa_intercept.h"


inline void check_comgr_error(amd_comgr_status_t err, const char* call_name) {
    if (err != AMD_COMGR_STATUS_SUCCESS) {
        const char* err_msg = "Unknown Error";
        amd_comgr_status_string(err, &err_msg);
        std::cerr << "HSA call " << call_name << "failed! Error code: " <<
            err_msg << std::endl << std::flush; \
    }
}

#define COMGR_CHECK(call) check_comgr_error(call, #call)


inline void check_hsa_error(hsa_status_t err, const char* call_name) {
    if (err != HSA_STATUS_SUCCESS) {
        const char* err_msg = "Unknown Error";
        hsa_status_string(err, &err_msg);
        std::cerr << "HSA call " << call_name << "failed! Error code: " <<
            err_msg << std::endl << std::flush; \
    }
}

#define CHECK_HSA_CALL(call) check_hsa_error(call, #call)


struct __CudaFatBinaryWrapper {
    unsigned int magic;
    unsigned int version;
    void* binary;
    void* dummy1;
};

constexpr unsigned __hipFatMAGIC2 = 0x48495046;  // "HIPF"


size_t constexpr strLiteralLength(char const* str) {
    return *str ? 1 + strLiteralLength(str + 1) : 0;
}
constexpr char const* CLANG_OFFLOAD_BUNDLER_MAGIC_STR = "__CLANG_OFFLOAD_BUNDLE__";
constexpr char const* OFFLOAD_KIND_HIP = "hip";
constexpr char const* OFFLOAD_KIND_HIPV4 = "hipv4";
constexpr char const* OFFLOAD_KIND_HCC = "hcc";
constexpr char const* AMDGCN_TARGET_TRIPLE = "amdgcn-amd-amdhsa-";

// ClangOFFLOADBundle info.
static constexpr size_t bundle_magic_string_size =
    strLiteralLength(CLANG_OFFLOAD_BUNDLER_MAGIC_STR);

// Clang Offload bundler description & Header.
struct __ClangOffloadBundleInfo {
    uint64_t offset;
    uint64_t size;
    uint64_t bundleEntryIdSize;
    const char bundleEntryId[1];
};

struct __ClangOffloadBundleHeader {
    const char magic[bundle_magic_string_size - 1];
    uint64_t numOfCodeObjects;
    __ClangOffloadBundleInfo desc[1];
};




// This will be moved to COMGR eventually
hipError_t extractCodeObjectFromFatBinary(
    const void* data) {
    std::string magic((const char*)data, bundle_magic_string_size);
    if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC_STR)) {
        return hipErrorInvalidKernelFile;
    }

    const auto obheader = reinterpret_cast<const __ClangOffloadBundleHeader*>(data);
    const auto* desc = &obheader->desc[0];
    std::cout << "Number of code objects: " << obheader->numOfCodeObjects << std::endl;
    int num_co = obheader->numOfCodeObjects;
    for (uint64_t i = 0; i < num_co; ++i) {
        const void* image =
            reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(obheader) + desc->offset);
        const size_t image_size = desc->size;

        std::string bundleEntryId{desc->bundleEntryId, desc->bundleEntryIdSize};

        std::cout << "Bundle entry ID: " << bundleEntryId << std::endl;
        std::cout << "Where it should be: " << image << std::endl;
        std::cout << "Offset: " << desc->offset << std::endl;
        std::cout << "Size: " << image_size << std::endl;
        desc = reinterpret_cast<const __ClangOffloadBundleInfo*>(
            reinterpret_cast<uintptr_t>(&desc->bundleEntryId[0]) +
            desc->bundleEntryIdSize);
    }
    return hipSuccess;
}

void SibirCodeObjectManager::setLastFatBinary(const void *data) {
    lastFatBinary_ = data;

}
void SibirCodeObjectManager::saveLastFatBinary() {
    assert(lastFatBinary_ != nullptr);
    auto fb_wrapper = reinterpret_cast<const __CudaFatBinaryWrapper*>(lastFatBinary_);
    assert(fb_wrapper->magic == __hipFatMAGIC2 && fb_wrapper->version == 1);
    if (!fatBinaries_.contains(fb_wrapper->binary)) {
        amd_comgr_data_t fb_data;
        COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &fb_data));
        COMGR_CHECK(amd_comgr_set_data(fb_data, 4096, reinterpret_cast<const char*>(fb_wrapper->binary)));
        fatBinaries_.insert({fb_wrapper->binary, fb_data});

        extractCodeObjectFromFatBinary(fb_wrapper->binary);

        std::unordered_map<std::string, std::pair<size_t, size_t>> unique_isa_names{{
            "amdgcn-amd-amdhsa--gfx908", {0, 0}}
//              "gfx908", {0, 0}}
        };
//        for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
//            std::string device_name = devices[dev_idx]->devices()[0]->isa().isaName();
//            if (unique_isa_names.cend() == unique_isa_names.find(device_name)) {
//                unique_isa_names.insert({device_name, std::make_pair<size_t, size_t>(0,0)});
//            }
//        }

        // Create a query list using COMGR info for unique ISAs.
        auto query_list_array = new amd_comgr_code_object_info_t[1];
        auto isa_it = unique_isa_names.begin();
        for (size_t isa_idx = 0; isa_idx < unique_isa_names.size(); ++isa_idx) {
            std::advance(isa_it, isa_idx);
            query_list_array[isa_idx].isa = isa_it->first.c_str();
            query_list_array[isa_idx].size = 0;
            query_list_array[isa_idx].offset = 0;
        }


        COMGR_CHECK(amd_comgr_lookup_code_object(fb_data, query_list_array, unique_isa_names.size()));
        std::cout << "Isa is: " << query_list_array[0].isa << std::endl;
        std::cout << "size: " << query_list_array[0].size << std::endl;
        std::cout << "offset: " << query_list_array[0].offset << std::endl;
        auto co_start = reinterpret_cast<void*>(reinterpret_cast<unsigned char*>(fb_wrapper->binary)
                                                 + query_list_array[0].offset);
        std::cout << "Where it should be located: " << co_start << std::endl;
        hsa_code_object_reader_t co_reader;
        hsa_executable_t executable;
        const auto& hsaTables = SibirHsaInterceptor::Instance().getSavedHsaTables();

        CHECK_HSA_CALL(hsaTables.core.hsa_code_object_reader_create_from_memory_fn(
            reinterpret_cast<void*>(reinterpret_cast<unsigned char*>(fb_wrapper->binary)
                                     + query_list_array[0].offset),
            query_list_array[0].size,
            &co_reader
            ));
        CHECK_HSA_CALL(hsaTables.core.hsa_executable_create_alt_fn(HSA_PROFILE_FULL,
                                                                   HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                                                   nullptr,
                                                                   &executable));
//        hsaTables.core.hsa_executable_load_agent_code_object_fn()
        delete[] query_list_array;
    }
}

void SibirCodeObjectManager::registerFunction(const char *funcName, const void *hostFunction, const char *deviceName) {
    if (!functions_.contains(funcName)) {
        functions_.insert({funcName, {funcName, hostFunction, deviceName, lastFatBinary_}});
    }
}
