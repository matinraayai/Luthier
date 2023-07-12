//#include "src/sibir/sibir.h"
//#include <fstream>
//#include <functional>
//#include <hip/hip_runtime_api.h>
//#include <hsa/hsa.h>
//#include <hsa/hsa_ext_amd.h>
//#include <hsa/hsa_ven_amd_loader.h>
//#include <iostream>
//#include <map>
//#include <mutex>
//#include <roctracer/roctracer.h>
//#include <roctracer/roctracer_hip.h>
//#include <roctracer/roctracer_hsa.h>
//#include <string>
//
//int number_of_dumped_elfs = 0;
//int number_of_dumped_kernels = 0;
//
//
//template <typename Integral = uint64_t> constexpr Integral bit_mask(int first, int last) {
//    assert(last >= first && "Error: hsa_support::bit_mask -> invalid argument");
//    size_t num_bits = last - first + 1;
//    return ((num_bits >= sizeof(Integral) * 8) ? ~Integral{0}
//                                               /* num_bits exceed the size of Integral */
//                                               : ((Integral{1} << num_bits) - 1))
//        << first;
//}
//
//template <typename Integral> constexpr Integral bit_extract(Integral x, int first, int last) {
//    return (x >> first) & bit_mask<Integral>(0, last - first);
//}
//
//static std::map<decltype(hsa_signal_t::handle), hsa_queue_t*> queue_map;
//std::mutex mutex;
//
//void sibir_at_init() {
//    std::cout << "Hi from Sibir!" << std::endl;
//}
//
//
//void sibir_at_term() {
//    std::cout << "Bye from Sibir!" << std::endl;
//}
//
//struct kernel_descriptor_t {
//    uint8_t reserved0[16];
//    int64_t kernel_code_entry_byte_offset;
//    uint8_t reserved1[20];
//    uint32_t compute_pgm_rsrc3;
//    uint32_t compute_pgm_rsrc1;
//    uint32_t compute_pgm_rsrc2;
//    uint16_t kernel_code_properties;
//    uint8_t reserved2[6];
//};
//
//inline void check_error(hsa_status_t err, const char* call_name) {
//    if (err != HSA_STATUS_SUCCESS) {
//        const char* err_msg = "Unknown Error";
//        hsa_status_string(err, &err_msg);
//        std::cerr << "HSA call " << call_name << "failed! Error code: " <<
//            err_msg << std::endl << std::flush; \
//    }
//}
//
//#define CHECK_HSA_CALL(call) check_error(call, #call)
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
//
//static unsigned extractAqlBits(unsigned v, unsigned pos, unsigned width) {
//    return (v >> pos) & ((1 << width) - 1);
//};
//
//hsa_status_t getGpuAgents(hsa_agent_t& agent) {
//    int i = 0;
//    std::vector<hsa_agent_t> agent_list;
//    auto iterateAgentCallback = [](hsa_agent_t agent, void* data) {
//        auto agent_list = reinterpret_cast<std::vector<hsa_agent_t>*>(data);
//        hsa_status_t stat = HSA_STATUS_ERROR;
//        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;
//
//        stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);
//
//        if (stat != HSA_STATUS_SUCCESS) {
//            return stat;
//        }
//        if (dev_type == HSA_DEVICE_TYPE_GPU) {
//            agent_list->push_back(agent);
//        }
//        return stat;
//    };
//    hsa_status_t err = hsa_iterate_agents(iterateAgentCallback, &agent_list);
//    agent = agent_list[0];
//    return err;
//}
//
//void sibir_at_hsa_event(hsa_api_args_t* args, sibir_api_phase_t phase, hsa_api_id_t api_id) {
//    if (phase == SIBIR_API_PHASE_EXIT) {
//        if (api_id == HSA_API_ID_hsa_executable_freeze) {
//            auto executable = args->hsa_executable_freeze.executable;
//            std::cout << "HSA Freeze callback" << std::endl;
//            hsa_executable_state_t e_state;
//            CHECK_HSA_CALL(hsa_executable_get_info(executable, HSA_EXECUTABLE_INFO_STATE, &e_state));
//            std::cout << "State of the executable: " << e_state << std::endl;
//            hsa_agent_t current_device;
//            get_agent(current_device);
//            CHECK_HSA_CALL(hsa_executable_iterate_agent_symbols(executable,
//                                                                current_device,
//                                                                iteration_callback, nullptr));
//            fprintf(stdout, "");
//        }
//        else if (api_id == HSA_API_ID_hsa_queue_create) {
//            auto queue = args->hsa_queue_create.queue;
//            queue_map[(*queue)->doorbell_signal.handle] = *queue;
//        }
//        else if (api_id == HSA_API_ID_hsa_signal_store_relaxed) {
//            hsa_signal_t signal = args->hsa_signal_store_relaxed.signal;
//            hsa_signal_value_t value = args->hsa_signal_store_relaxed.value;
//            auto it = queue_map.find(signal.handle);
//            if (it != queue_map.end()) {
//                hsa_queue_t *queue = it->second;
//                mutex.lock();
//                const uint64_t begin = hsa_queue_load_read_index_relaxed(queue);
//                const uint64_t end = value + 1;
//                mutex.unlock();
//                for (uint64_t j = begin; j < end; ++j) {
//                    const unsigned int idx = j & (queue->size - 1);
//                    hsa_barrier_or_packet_t *packet = reinterpret_cast<hsa_barrier_or_packet_t *>(queue->base_address) + idx;
//
//                    unsigned int packet_type = extractAqlBits(packet->header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
//                    fprintf(stdout, "Packet Header: %u\n", packet_type);
//                    if (packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
//                        auto *dispatch_packet = reinterpret_cast<hsa_kernel_dispatch_packet_t *>(packet);
//                        std::cout << dispatch_packet->grid_size_x << ", " << dispatch_packet->grid_size_y << ", " << dispatch_packet->grid_size_z << std::endl;
////                        dispatch_packet->grid_size_x = 4;
////                        dispatch_packet->grid_size_y = 256;
////                        dispatch_packet->grid_size_z = 24;
////                        dispatch_packet->workgroup_size_x = 0;
//                        const kernel_descriptor_t *kernel_code = nullptr;
//                        //                        hsa_ven_amd_loader_1_01_pfn_t* amd_table;
//                        //                        sibir_get_hsa_table()->core_->hsa_system_get_extension_table_fn(
//                        //                            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_01_pfn_t),
//                        //                            amd_table);
//                        //                        amd_table->hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void*>(dispatch_packet->kernel_object),
//                        //                                                                         reinterpret_cast<const void**>(&kernel_code));
//
//                        //                        std::cout << "here!!" << std::endl;
//                        //                        std::cout << kernel_code->kernel_code_entry_byte_offset
//                        //                        std::cout << kernel_code->kernel_code_entry_byte_offset << std::endl;
//                    }
//                }
//            }
//        }
//
//        else if (api_id == HSA_API_ID_hsa_signal_store_screlease) {
//
//            hsa_signal_t signal = args->hsa_signal_store_screlease.signal;
//            hsa_signal_value_t value = args->hsa_signal_store_screlease.value;
//            auto it = queue_map.find(signal.handle);
//            if (it != queue_map.end()) {
//                hsa_queue_t* queue = it->second;
//                mutex.lock();
//                const uint64_t begin = hsa_queue_load_read_index_relaxed(queue);
//                const uint64_t end = value + 1;
//                //                hsa_queue_store_read_index_relaxed(queue, end);
//                mutex.unlock();
//                for (uint64_t j = begin; j < end; ++j) {
//                    const unsigned int idx = j & (queue->size - 1);
//                    hsa_barrier_or_packet_t *packet = reinterpret_cast<hsa_barrier_or_packet_t*>(queue->base_address) + idx;
//
//                    unsigned int packet_type = extractAqlBits(packet->header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
//                    fprintf(stdout, "Packet Header: %u.\n", packet_type);
//                    //                    std::cout << "Packet Header: " << packet_type << std::endl;
//                    if (packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
//                        auto* dispatch_packet = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet);
//                        std::cout << dispatch_packet->grid_size_x << ", " << dispatch_packet->grid_size_y << ", " << dispatch_packet->grid_size_z << std::endl;
//                        const kernel_descriptor_t* kernel_code = nullptr;
//                        hsa_ven_amd_loader_1_01_pfn_t amd_table{};
//                        CHECK_HSA_CALL(sibir_get_hsa_table()->core_->hsa_system_get_major_extension_table_fn(
//                            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_01_pfn_t),
//                            &amd_table));
//                        //                        hipPointerAttribute_t packet_attrs;
//                        //                        hipPointerGetAttributes(&packet_attrs, reinterpret_cast<void*>(dispatch_packet->kernel_object));
//                        //                        std::cout << "Device type: " << packet_attrs.memoryType << std::endl;
////                        dispatch_packet->grid_size_x = 4;
////                        dispatch_packet->grid_size_y = 256;
////                        dispatch_packet->grid_size_z = 24;
////                        dispatch_packet->workgroup_size_x = 0;
//                        CHECK_HSA_CALL(amd_table.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void*>(dispatch_packet->kernel_object),
//                                                                                       reinterpret_cast<const void**>(&kernel_code)));
//                        std::cout << kernel_code->kernel_code_entry_byte_offset << std::endl;
//                        std::cout << dispatch_packet->kernel_object << std::endl;
//                        //                        CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &code_object.handle));
//                    }
//                }
//
//            }
//
//        }
//    }
//    else if (phase == SIBIR_API_PHASE_ENTER) {
//        if (api_id == HSA_API_ID_hsa_code_object_reader_create_from_memory) {
//            auto binary = reinterpret_cast<const char*>(args->hsa_code_object_reader_create_from_memory.code_object);
//            size_t size = args->hsa_code_object_reader_create_from_memory.size;
//            std::cout << "Dumping binary at " << args->hsa_code_object_reader_create_from_memory.code_object << std::endl;
//            std::string content = std::string(binary, size);
//            auto f = std::fstream("./" + std::to_string(number_of_dumped_elfs) + ".elf",
//                                  std::ios_base::out);
//            f << content << std::endl;
//            number_of_dumped_elfs++;
//        }
//        else if (api_id == HSA_API_ID_hsa_signal_store_screlease) {
//
//            hsa_signal_t signal = args->hsa_signal_store_screlease.signal;
//            hsa_signal_value_t value = args->hsa_signal_store_screlease.value;
//            auto it = queue_map.find(signal.handle);
//            if (it != queue_map.end()) {
//                hsa_queue_t* queue = it->second;
//                mutex.lock();
//                const uint64_t begin = hsa_queue_load_read_index_relaxed(queue);
//                const uint64_t end = value + 1;
//                mutex.unlock();
//                for (uint64_t j = begin; j < end; ++j) {
//                    const unsigned int idx = j & (queue->size - 1);
//                    hsa_barrier_or_packet_t *packet = reinterpret_cast<hsa_barrier_or_packet_t*>(queue->base_address) + idx;
//
//                    unsigned int packet_type = extractAqlBits(packet->header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
//                    fprintf(stdout, "Packet Header: %u.\n", packet_type);
////                    std::cout << "Packet Header: " << packet_type << std::endl;
//                    if (packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
//                        auto* dispatch_packet = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet);
//                        std::cout << dispatch_packet->grid_size_x << ", " << dispatch_packet->grid_size_y << ", " << dispatch_packet->grid_size_z << std::endl;
//                        const kernel_descriptor_t* kernel_code = nullptr;
//                        hsa_ven_amd_loader_1_01_pfn_t amd_table{};
//                        CHECK_HSA_CALL(sibir_get_hsa_table()->core_->hsa_system_get_major_extension_table_fn(
//                            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_01_pfn_t),
//                            &amd_table));
////                        hipPointerAttribute_t packet_attrs;
////                        hipPointerGetAttributes(&packet_attrs, reinterpret_cast<void*>(dispatch_packet->kernel_object));
////                        std::cout << "Device type: " << packet_attrs.memoryType << std::endl;
////                        dispatch_packet->grid_size_x = 300;
////                        dispatch_packet->grid_size_y = 300;
////                        dispatch_packet->grid_size_z = 300;
////                        dispatch_packet->workgroup_size_x = 1;
//                        CHECK_HSA_CALL(amd_table.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void*>(dispatch_packet->kernel_object),
//                                                              reinterpret_cast<const void**>(&kernel_code)));
//                        std::cout << kernel_code->kernel_code_entry_byte_offset << std::endl;
//                        std::cout << reinterpret_cast<void*>(dispatch_packet->kernel_object) << std::endl;
////                        CHECK_HSA_CALL(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &code_object.handle));
//                    }
//                }
//
//            }
//
//        }
//        else if (api_id == HSA_API_ID_hsa_signal_store_relaxed) {
//            hsa_signal_t signal = args->hsa_signal_store_relaxed.signal;
//            hsa_signal_value_t value = args->hsa_signal_store_relaxed.value;
//            auto it = queue_map.find(signal.handle);
//            if (it != queue_map.end()) {
//                hsa_queue_t* queue = it->second;
//                mutex.lock();
//                const uint64_t begin = hsa_queue_load_read_index_relaxed(queue);
//                const uint64_t end = value + 1;
//                mutex.unlock();
//                for (uint64_t j = begin; j < end; ++j) {
//                    const unsigned int idx = j & (queue->size - 1);
//                    hsa_barrier_or_packet_t *packet = reinterpret_cast<hsa_barrier_or_packet_t*>(queue->base_address) + idx;
//
//                    unsigned int packet_type = extractAqlBits(packet->header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
//                    fprintf(stdout, "Packet Header: %u\n", packet_type);
//                    if (packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
//                        auto* dispatch_packet = reinterpret_cast<hsa_kernel_dispatch_packet_t*>(packet);
//                        std::cout << dispatch_packet->grid_size_x << ", " << dispatch_packet->grid_size_y << ", " << dispatch_packet->grid_size_z << std::endl;
////                        dispatch_packet->grid_size_x = 1;
////                        dispatch_packet->grid_size_y = 1;
////                        dispatch_packet->grid_size_z = 1;
////                        dispatch_packet->workgroup_size_x = 1;
////                        const kernel_descriptor_t* kernel_code = nullptr;
////                        hsa_ven_amd_loader_1_01_pfn_t* amd_table;
////                        sibir_get_hsa_table()->core_->hsa_system_get_extension_table_fn(
////                            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_01_pfn_t),
////                            amd_table);
////                        amd_table->hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void*>(dispatch_packet->kernel_object),
////                                                                         reinterpret_cast<const void**>(&kernel_code));
//
////                        std::cout << "here!!" << std::endl;
////                        std::cout << kernel_code->kernel_code_entry_byte_offset
////                        std::cout << kernel_code->kernel_code_entry_byte_offset << std::endl;
//                    }
//                }
//            }
//        }
//    }
//}
//
//void sibir_at_hip_event(hip_api_args_t* args, sibir_api_phase_t phase, hip_api_id_t api_id) {
//    fprintf(stdout, "<call to (%s)\t on %s> ",
//            hip_api_name(api_id),
//            phase == SIBIR_API_PHASE_ENTER ? "entry" : "exit"
//            );
//    switch (api_id) {
//        case HIP_API_ID_hipMemcpy:
//            fprintf(stdout, "dst(%p) src(%p) size(0x%x) kind(%u)",
//                    args->hipMemcpy.dst,
//                    args->hipMemcpy.src,
//                    (uint32_t) (args->hipMemcpy.sizeBytes),
//                    (uint32_t) (args->hipMemcpy.kind));
//            break;
//        case HIP_API_ID_hipMalloc:
//            fprintf(stdout, "ptr(%p) size(0x%x)",
//                    args->hipMalloc.ptr,
//                    (uint32_t) (args->hipMalloc.size));
//            break;
//        case HIP_API_ID_hipFree:
//            fprintf(stdout, "ptr(%p)", args->hipFree.ptr);
//            break;
//        case HIP_API_ID_hipLaunchKernel:
//            fprintf(stdout, "kernel(\"%s\") stream(%p)",
//                    hipKernelNameRefByPtr(args->hipLaunchKernel.function_address,
//                                          args->hipLaunchKernel.stream),
//                    args->hipModuleLaunchKernel.stream);
//            break;
//        default:
//            break;
//    }
//    fprintf(stdout, "\n"); fflush(stdout);
//}