#include <functional>
#include <hsa/hsa.h>
#include <iostream>
#include <fstream>
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_hsa.h>
#include <sibir.h>
#include <string>

int number_of_dumped_elfs = 0;
int number_of_dumped_kernels = 0;

void sibir_at_init() {
    std::cout << "Hi from Sibir!" << std::endl;
}


void sibir_at_term() {
    std::cout << "Bye from Sibir!" << std::endl;
}

inline void check_error(hsa_status_t err, const char* call_name) {
    if (err != HSA_STATUS_SUCCESS) {
        const char* err_msg = "Unknown Error";
        hsa_status_string(err, &err_msg);
        std::cerr << "HSA call " << call_name << "failed! Error code: " <<
            err_msg << std::endl << std::flush; \
    }
}

#define CHECK_HSA_CALL(call) check_error(call, #call)

hsa_status_t iteration_callback(hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data) {
    std::cout << "HSA Executable address: " << reinterpret_cast<void*>(executable.handle) << std::endl;
    size_t symbol_name_size = 0;
    CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &symbol_name_size));
    std::cout << "Symbol name size: " << symbol_name_size << std::endl;
    char* symbol_name = static_cast<char *>(malloc(sizeof(char) * symbol_name_size));
    CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, symbol_name));
    std::cout << "Symbol name: " << symbol_name << std::endl;

    hsa_symbol_kind_t symbol_type;
    CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbol_type));
    std::cout << "Symbol Type: " << symbol_type << std::endl;
    if (symbol_type == HSA_SYMBOL_KIND_KERNEL) {
        hsa_code_object_t code_object;
        size_t kernel_size = 0;
        CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &code_object.handle));
        CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, static_cast<hsa_executable_symbol_info_t>(100), &kernel_size));
        std::cout << "Code object handle:" << reinterpret_cast<void*>(code_object.handle) << std::endl;
        std::cout << "Size: " << kernel_size << std::endl;
        std::cout << "Content: " << std::endl;
        auto handle = reinterpret_cast<char*>(code_object.handle);
        auto f = std::fstream("./" + std::string(symbol_name, symbol_name_size) + ".kd", std::ios::out);
        for (int i = 0; i < max(kernel_size, 64); i++)
            f << handle[i];
        number_of_dumped_kernels++;
        free(symbol_name);

//         << std::hex << std::string(reinterpret_cast<char*>(code_object.handle), 100) << std::endl;
//        size_t co_size;
//        void * ser_co;
//        hsa_callback_data_t d;
//
//        CHECK_HSA_CALL(hsa_code_object_serialize(code_object, allocation_callback, d,
//                                                 nullptr, &ser_co, &co_size));
    }
//    if (out == HSA_STATUS_ERROR_INVALID_CODE_OBJECT)
//        std::cout << "Failed" << std::endl;
//    else {
//        std::string content(reinterpret_cast<char*>(&ser_co), co_size);
//        std::cout << "code size: " << co_size << std::endl;
//        std::cout << content << std::endl;
//    }
//    free(ser_co);
    return HSA_STATUS_SUCCESS;
}

hsa_status_t get_agent(hsa_agent_t& agent) {
    int i = 0;
    std::vector<hsa_agent_t> agent_list;
    auto iterateAgentCallback = [](hsa_agent_t agent, void* data) {
        auto agent_list = reinterpret_cast<std::vector<hsa_agent_t>*>(data);
        hsa_status_t stat = HSA_STATUS_ERROR;
        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

        stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

        if (stat != HSA_STATUS_SUCCESS) {
            return stat;
        }
        if (dev_type == HSA_DEVICE_TYPE_GPU) {
            agent_list->push_back(agent);
        }
        return stat;
    };
    hsa_status_t err = hsa_iterate_agents(iterateAgentCallback, &agent_list);
    agent = agent_list[0];
    return err;
}

void sibir_at_hsa_event(hsa_api_args_t* args, sibir_api_phase_t phase, hsa_api_id_t api_id) {
    if (phase == SIBIR_API_PHASE_EXIT) {
        if (api_id == HSA_API_ID_hsa_executable_freeze) {
            auto executable = args->hsa_executable_freeze.executable;
            std::cout << "HSA Freeze callback" << std::endl;
            hsa_executable_state_t e_state;
            CHECK_HSA_CALL(hsa_executable_get_info(executable, HSA_EXECUTABLE_INFO_STATE, &e_state));
            std::cout << "State of the executable: " << e_state << std::endl;
            hsa_agent_t current_device;
            get_agent(current_device);
            CHECK_HSA_CALL(hsa_executable_iterate_agent_symbols(executable,
                                                                current_device,
                                                                iteration_callback, nullptr));
            fprintf(stdout, "");
        }
    }
    else if (phase == SIBIR_API_PHASE_ENTER) {
        if (api_id == HSA_API_ID_hsa_code_object_reader_create_from_memory) {
            auto binary = reinterpret_cast<const char*>(args->hsa_code_object_reader_create_from_memory.code_object);
            size_t size = args->hsa_code_object_reader_create_from_memory.size;
            std::cout << "Dumping binary at " << args->hsa_code_object_reader_create_from_memory.code_object << std::endl;
            std::string content = std::string(binary, size);
            auto f = std::fstream("./" + std::to_string(number_of_dumped_elfs) + ".elf",
                                  std::ios_base::out);
            f << content << std::endl;
            number_of_dumped_elfs++;
        }
    }
}

void sibir_at_hip_event(hip_api_args_t* args, sibir_api_phase_t phase, hip_api_id_t api_id) {
    fprintf(stdout, "<call to (%s)\t on %s> ",
            hip_api_name(api_id),
            phase == SIBIR_API_PHASE_ENTER ? "entry" : "exit"
            );
    switch (api_id) {
        case HIP_API_ID_hipMemcpy:
            fprintf(stdout, "dst(%p) src(%p) size(0x%x) kind(%u)",
                    args->hipMemcpy.dst,
                    args->hipMemcpy.src,
                    (uint32_t) (args->hipMemcpy.sizeBytes),
                    (uint32_t) (args->hipMemcpy.kind));
            break;
        case HIP_API_ID_hipMalloc:
            fprintf(stdout, "ptr(%p) size(0x%x)",
                    args->hipMalloc.ptr,
                    (uint32_t) (args->hipMalloc.size));
            break;
        case HIP_API_ID_hipFree:
            fprintf(stdout, "ptr(%p)", args->hipFree.ptr);
            break;
        case HIP_API_ID_hipLaunchKernel:
            fprintf(stdout, "kernel(\"%s\") stream(%p)",
                    hipKernelNameRefByPtr(args->hipLaunchKernel.function_address,
                                          args->hipLaunchKernel.stream),
                    args->hipModuleLaunchKernel.stream);
            break;
        default:
            break;
    }
    fprintf(stdout, "\n"); fflush(stdout);
}