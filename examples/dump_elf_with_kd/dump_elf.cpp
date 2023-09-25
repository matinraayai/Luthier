#include <fstream>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <iostream>
#include <luthier.h>
#include <map>
#include <mutex>
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_hsa.h>
#include <string>

int number_of_dumped_elfs = 0;
int number_of_dumped_kernels = 0;

void luthier_at_init() {
    std::cout << "Staring the ELF dump tool." << std::endl;
}

void luthier_at_term() {
    std::cout << "ELF dumping done." << std::endl;
}

std::pair<luthier_address_t, size_t> getCodeObject(hsa_executable_t executable) {
    auto amdTable = luthier_get_hsa_ven_amd_loader();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void *data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t> *>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable->hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, &loadedCodeObjects);

    // Can be removed
    assert(loadedCodeObjects.size() == 1);

    uint64_t lcoBaseAddr;
    amdTable->hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0], HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE, &lcoBaseAddr);
    // Query the size of the loaded code object
    uint64_t lcoSize;
    amdTable->hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0], HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE, &lcoSize);
    return {reinterpret_cast<luthier_address_t>(lcoBaseAddr), static_cast<size_t>(lcoSize)};
}
//
//hsa_status_t iteration_callback(hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data) {
//    std::cout << "HSA Executable address: " << reinterpret_cast<void*>(executable.handle) << std::endl;
//    size_t symbol_name_size = 0;
//    CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &symbol_name_size));
//    std::cout << "Symbol name size: " << symbol_name_size << std::endl;
//    char* symbol_name = static_cast<char *>(malloc(sizeof(char) * symbol_name_size));
//    CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, symbol_name));
//    std::cout << "Symbol name: " << symbol_name << std::endl;
//
//    hsa_symbol_kind_t symbol_type;
//    CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbol_type));
//    std::cout << "Symbol Type: " << symbol_type << std::endl;
//    if (symbol_type == HSA_SYMBOL_KIND_KERNEL) {
//        hsa_code_object_t code_object;
//        size_t kernel_size = 0;
//        CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &code_object.handle));
//        CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, static_cast<hsa_executable_symbol_info_t>(100), &kernel_size));
//        std::cout << "Code object handle:" << reinterpret_cast<void*>(code_object.handle) << std::endl;
//        hipPointerAttribute_t packet_attrs;
//        hipPointerGetAttributes(&packet_attrs, reinterpret_cast<void*>(code_object.handle));
//        std::cout << "Device type: " << packet_attrs.memoryType << std::endl;
//        std::cout << "Size: " << kernel_size << std::endl;
//        std::cout << "Content: " << std::endl;
//        auto handle = reinterpret_cast<char*>(code_object.handle);
//        auto f = std::fstream("./" + std::string(symbol_name, symbol_name_size) + ".kd", std::ios::out);
//        for (int i = 0; i < max(kernel_size, 64); i++)
//            f << handle[i];
//        number_of_dumped_kernels++;
//        free(symbol_name);
//
////         << std::hex << std::string(reinterpret_cast<char*>(code_object.handle), 100) << std::endl;
////        size_t co_size;
////        void * ser_co;
////        hsa_callback_data_t d;
////
////        CHECK_HSA_CALL(hsa_code_object_serialize(code_object, allocation_callback, d,
////                                                 nullptr, &ser_co, &co_size));
//    }
////    if (out == HSA_STATUS_ERROR_INVALID_CODE_OBJECT)
////        std::cout << "Failed" << std::endl;
////    else {
////        std::string content(reinterpret_cast<char*>(&ser_co), co_size);
////        std::cout << "code size: " << co_size << std::endl;
////        std::cout << content << std::endl;
////    }
////    free(ser_co);
//    return HSA_STATUS_SUCCESS;
//}

void luthier_at_hsa_event(hsa_api_args_t *args, luthier_api_phase_t phase, hsa_api_id_t api_id) {
    if (phase == SIBIR_API_PHASE_EXIT) {
        if (api_id == HSA_API_ID_hsa_executable_freeze) {
            auto executable = args->hsa_executable_freeze.executable;
            std::pair<luthier_address_t, size_t> execCodeObject = getCodeObject(executable);
            auto f = std::fstream("./executable-" + std::to_string(number_of_dumped_elfs) + ".elf", std::ios::out);
            f << std::string(reinterpret_cast<const char *>(execCodeObject.first), execCodeObject.second);
            number_of_dumped_elfs++;
            f.close();
        }
    }
}