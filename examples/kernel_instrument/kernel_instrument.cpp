#include "sibir.h"
#include <fstream>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <iostream>
#include <map>
#include <mutex>
#include <roctracer/roctracer_hip.h>
#include <roctracer/roctracer_hsa.h>
#include <string>

// Since the kernel launch doesn't give us access to the queue the signal is associated with, we need to save the relationship in a map
static std::unordered_map<decltype(hsa_signal_t::handle), hsa_queue_t*> queue_map;
// This is for preventing the queue content to be modified during our interception. Might not be necessary.
// Might use the HSA variants that locks a mutex when reading the queue index.
std::mutex mutex;

static std::unordered_map<decltype(hsa_kernel_dispatch_packet_t::kernel_object), hsa_executable_symbol_t> ko_symbol_map;

static bool instrumented{true};

__managed__ int counter;

__device__ void instrumentation_kernel() {
    atomicAdd(&counter, 1);
}

SIBIR_EXPORT_FUNC(instrumentation_kernel)

/**
 * Returns the list of GPU HSA agents (devices) on the system
 * @param [out] agentList a vector of GPU agents
 * @return HSA_STATUS_SUCCESS
 */
hsa_status_t getGpuAgents(std::vector<hsa_agent_t>& agentList) {
    int i = 0;
    auto coreTable = sibir_get_hsa_table()->core_;
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
    return coreTable->hsa_iterate_agents_fn(iterateAgentCallback, &agentList);
//    for (auto agent: agentList) {
//        std::string vendorName, name;
//        vendorName.resize(64);
//        name.resize(64);
//        coreTable->hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_VENDOR_NAME, vendorName.data());
//        std::cout << "Agent Vendor Name: " << vendorName << std::endl;
//        coreTable->hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_NAME, name.data());
//        std::cout << "Agent Name: " << name << std::endl;
//    }
//    return HSA_STATUS_SUCCESS;
}

//hsa_status_t getAllExecutableSymbols(const hsa_executable_t& executable,
//                                     std::vector<hsa_executable_symbol_t>& symbols) {
//    hsa_status_t out = HSA_STATUS_ERROR;
//
//    amdTable.hsa_ven_amd_loader_query_executable
//    for (auto agent : agentList) {
//        auto iterCallback = [](hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data) {
//            auto symbolVec = reinterpret_cast<std::vector<hsa_executable_symbol_t>*>(data);
//            auto& coreTable = sibir_get_hsa_table()->core_;
//            int kernelObject;
//            CHECK_HSA_CALL(coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject));
//            std::cout << "Kernel Object: " << kernelObject << std::endl;
//            size_t nameSize;
//            CHECK_HSA_CALL(coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameSize));
//            std::cout << "Kernel name size: " << nameSize << std::endl;
//            auto name = new char[nameSize];
//            CHECK_HSA_CALL(coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &name));
//            std::cout << "Kernel Name: " << name << std::endl;
////            ko_symbol_map[kernelObject] = symbol;
//            delete[] name;
//
//            symbolVec->push_back(symbol);
//            return HSA_STATUS_SUCCESS;
//        };
//        out = hsa_executable_iterate_agent_symbols(executable,
//                                                   agent,
//                                                   iterCallback, &symbols);
//        if (out != HSA_STATUS_SUCCESS)
//            return HSA_STATUS_ERROR;
//    }
//    return HSA_STATUS_SUCCESS;
//}

std::ostream& operator<<(std::ostream& os, const kernel_descriptor_t& kd) {
    os << "Kernel Descriptor Content" << std::endl;
    os << "Reserved 0: ";
    for (uint8_t r: kd.reserved0)
        os << r;
    os << std::endl;
    os << "Code Entry Offset: " << kd.kernel_code_entry_byte_offset << std::endl;
    os << "Reserved 1: ";
    for (uint8_t r: kd.reserved1)
        os << r;
    os << std::endl;
    os << "PGM src 3: " << kd.compute_pgm_rsrc3 << std::endl;
    os << "PGM src 1: " << kd.compute_pgm_rsrc1 << std::endl;
    os << "PGM src 2: " << kd.compute_pgm_rsrc2 << std::endl;
    os << "Kernel code properties: " << kd.kernel_code_properties << std::endl;
    os << "Reserved 2: ";
    for (uint8_t r: kd.reserved2)
        os << r;
    return os;
}

std::ostream& operator<<(std::ostream& os, const hsa_packet_type_t& packetType) {
    switch (packetType) {
        case HSA_PACKET_TYPE_VENDOR_SPECIFIC:
            os << "HSA_PACKET_TYPE_VENDOR_SPECIFIC";
            break;
        case HSA_PACKET_TYPE_INVALID:
            os << "HSA_PACKET_TYPE_INVALID";
            break;
        case HSA_PACKET_TYPE_KERNEL_DISPATCH:
            os << "HSA_PACKET_TYPE_KERNEL_DISPATCH";
            break;
        case HSA_PACKET_TYPE_BARRIER_AND:
            os << "HSA_PACKET_TYPE_BARRIER_AND";
            break;
        case HSA_PACKET_TYPE_AGENT_DISPATCH:
            os << "HSA_PACKET_TYPE_AGENT_DISPATCH";
            break;
        case HSA_PACKET_TYPE_BARRIER_OR:
            os << "HSA_PACKET_TYPE_BARRIER_OR";
            break;
        default: throw std::invalid_argument("Failed to locate the HSA packet type enum from number " +
                                             std::to_string(static_cast<int>(packetType)) + ".");
    }
    return os;
}



static unsigned extractAqlBits(unsigned v, unsigned pos, unsigned width) {
    return (v >> pos) & ((1 << width) - 1);
};

void instrumentKernel(const amd_dbgapi_global_address_t entrypoint) {

    // For now assume gfx908
    // TODO: add the architecture code from the dbgapi headers
    amd_dbgapi_architecture_id_t arch;
    amd_dbgapi_get_architecture(0x030, &arch);


    amd_dbgapi_size_t maxInstrLen;
    amd_dbgapi_architecture_get_info(arch, AMD_DBGAPI_ARCHITECTURE_INFO_LARGEST_INSTRUCTION_SIZE,
                                     sizeof(amd_dbgapi_size_t),
                                     &maxInstrLen);

    bool is_end = false;
    // The decoded instruction will be malloced by ::amd_dbgapi_disassemble_instruction
    // It has to be copied and freed
    char*instChar{};
    auto curr_address = entrypoint;
    amd_dbgapi_size_t instrSize;

    while(!is_end) {
        instrSize = maxInstrLen;

        amd_dbgapi_disassemble_instruction(arch, curr_address, &instrSize,
                                           reinterpret_cast<void*>(curr_address),
                                           &instChar, nullptr, {});

        std::vector<std::byte> instBytes(instrSize);
        // Copy the instruction bytes
        for (amd_dbgapi_size_t i = 0; i < instrSize; i++) {
            instBytes[i] = reinterpret_cast<std::byte*>(curr_address)[i];
        }
        // Copy the decoded instruction string
        std::string instStr(instChar);
//        std::cout << instStr << ": ";
        if (instStr.find("s_add_i32") != std::string::npos) {
            std::cout << instStr << ": ";
            for (const auto& el: instBytes)
                std::cout << std::hex << std::setfill('0') << std::setw(2) << uint16_t(el) << " ";
            auto overwrite_address = reinterpret_cast<uint8_t*>(curr_address);

            overwrite_address[1] = 0x85;
//            curr_address
        }
        free(instChar);

        curr_address += instrSize;
        is_end = instStr.find("s_endpgm") != std::string::npos;
    }

}

void instrumentKernelLaunchCallback(hsa_signal_t signal, hsa_signal_value_t value) {
    auto amdTable = sibir_get_hsa_ven_amd_loader();
    std:: cout << "was found in map? " << queue_map.contains(signal.handle) << std::endl;
    if (queue_map.contains(signal.handle)) {
        hsa_queue_t *queue = queue_map[signal.handle];
        mutex.lock();
        const uint64_t begin = sibir_get_hsa_table()->core_->hsa_queue_load_read_index_relaxed_fn(queue);
        const uint64_t end = value + 1;
        mutex.unlock();
        uint32_t mask = queue->size - 1;
        for (uint64_t j = begin; j < end; ++j) {
            const unsigned int idx = j & (queue->size - 1);
            hsa_barrier_or_packet_t *packet = reinterpret_cast<hsa_barrier_or_packet_t *>(queue->base_address) + idx;

            unsigned int packetType = extractAqlBits(packet->header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
            if (packetType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
                auto *dispatchPacket = reinterpret_cast<hsa_kernel_dispatch_packet_t *>(packet);
                const kernel_descriptor_t *kernelDescriptor = nullptr;
                SIBIR_HSA_CHECK(amdTable->hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(dispatchPacket->kernel_object),
                                                                              reinterpret_cast<const void **>(&kernelDescriptor)));
                auto kernelEntryPoint =
                    reinterpret_cast<amd_dbgapi_global_address_t>(dispatchPacket->kernel_object) + kernelDescriptor->kernel_code_entry_byte_offset;


                // A way to backtrack from the kernel object to the symbol it belongs to (besides keeping track of a map)
                hsa_executable_t executable;

                SIBIR_HSA_CHECK(amdTable->hsa_ven_amd_loader_query_executable(
                    reinterpret_cast<void*>(dispatchPacket->kernel_object), &executable));

                fprintf(stdout, "The kernel launch belongs to executable with handle: %lX.\n", executable.handle);
                auto& coreApi = sibir_get_hsa_table()->core_;
                std::vector<hsa_agent_t> agents;
                SIBIR_HSA_CHECK(getGpuAgents(agents));

                auto callbackData = std::make_pair(hsa_executable_symbol_t{}, dispatchPacket->kernel_object);

                auto execIterate = [](hsa_executable_t e, hsa_agent_t a, hsa_executable_symbol_t s, void* data){
                    auto cbd = reinterpret_cast<std::pair<hsa_executable_symbol_t, uint64_t>*>(data);
                    uint64_t ko;
                    auto& coreApi = sibir_get_hsa_table()->core_;
                    SIBIR_HSA_CHECK(coreApi->hsa_executable_symbol_get_info_fn(s, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &ko));
                    if (ko == cbd->second)
                        cbd->first = s;
                    return HSA_STATUS_SUCCESS;
                };

                for (auto agent: agents)
                    SIBIR_HSA_CHECK(coreApi->hsa_executable_iterate_agent_symbols_fn(executable, agent, execIterate, &callbackData));

//                CHECK_HSA_CALL(coreApi->hsa_executable_symbol_get_info_fn(kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &agent));

                if (ko_symbol_map.contains(dispatchPacket->kernel_object)) {
                    fprintf(stdout, "Kernel Object %lX was intercepted before.\n", dispatchPacket->kernel_object);
                    hsa_executable_symbol_t kernel_symbol = ko_symbol_map[dispatchPacket->kernel_object];
                    fprintf(stdout, "Symbol mapped to the KO: %lX\n", kernel_symbol.handle);
                }
                fprintf(stdout, "Kernel Object found the traditional way: %lX\n", callbackData.first.handle);
//
//                std::vector<hsa_isa_t> supportedIsas;
//                auto agentIsaCallback = [](hsa_isa_t isa, void* data){
//                    std::cout << "Here!" << std::endl;
//                    reinterpret_cast<std::vector<hsa_isa_t>*>(data)->push_back(isa);
//                    size_t nameLen;
//                    auto& coreApi = sibir_get_hsa_table()->core_;
//                    CHECK_HSA_CALL(coreApi->hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME_LENGTH, &nameLen));
//                    auto isaName = new char[nameLen];
//                    CHECK_HSA_CALL(coreApi->hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME_LENGTH, &isaName));
//                    std::cout << "ISA supported by the executable: " << isaName << std::endl;
//                    delete[] isaName;
//                    return HSA_STATUS_SUCCESS;
//                };

//                coreApi->hsa_agent_iterate_isas_fn(agent, agentIsaCallback, &supportedIsas);
//                std::cout << "Length of supported Isas: " << supportedIsas.size() << std::endl;
//                for (auto isa: supportedIsas) {
//                    size_t nameLen;
//                    coreApi->hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME_LENGTH, &nameLen);
//                    auto isaName = new char[nameLen];
//                    coreApi->hsa_isa_get_info_alt_fn(isa, HSA_ISA_INFO_NAME_LENGTH, &isaName);
//                    std::cout << "ISA supported by the executable: " << isaName << std::endl;
//                    delete[] isaName;
//                }
//                instrumentKernel(kernelEntryPoint);
            }
        }
    }
//    else {
//        throw std::invalid_argument("Signal handle " + std::to_string(signal.handle) + "'s queue was not found in the map.");
//    }
}


void iterateLoadedCodeObjects(hsa_executable_t executable) {
    auto amdTable = sibir_get_hsa_ven_amd_loader();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void* data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t>*>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable->hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, & loadedCodeObjects);

    // Dump all the code objects into files
    for (int i = 0; i < loadedCodeObjects.size(); i++) {
        // Query the base address of the loaded code object on host
        uint64_t lcoBaseAddrHost;
        amdTable->hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
                                                                HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
                                                                &lcoBaseAddrHost);
        // Query the size of the loaded code object on host
        uint64_t lcoSizeHost;
        amdTable->hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
                                                                HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
                                                                &lcoSizeHost);



        uint64_t lcoBaseAddrDevice;
        amdTable->hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
                                                                HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                                                                &lcoBaseAddrDevice);
        // Query the size of the loaded code object
        uint64_t lcoSizeDevice;
        amdTable->hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
                                                                HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                                                                &lcoSizeDevice);


        std::cout << "Base address of the executable on the host: " << reinterpret_cast<void*>(lcoBaseAddrHost) << std::endl;
        std::cout << "Size of the executable on the host: " << lcoSizeHost << std::endl;
        std::cout << "Base address of the executable on the device: " << reinterpret_cast<void*>(lcoBaseAddrDevice) << std::endl;
        std::cout << "Size of the executable on the device: " << lcoSizeDevice << std::endl;


        std::string coContentHost(reinterpret_cast<char*>(lcoBaseAddrHost), lcoSizeHost);
        std::string coContentDevice(reinterpret_cast<char*>(lcoBaseAddrDevice), lcoSizeDevice);
        auto h = std::fstream("./" + std::to_string(executable.handle) + "_host_" + std::to_string(i) + ".elf",
                              std::ios_base::out);
        h << coContentHost << std::endl;
        h.close();
        auto d = std::fstream("./" + std::to_string(executable.handle) + "_device_" + std::to_string(i) + ".elf",
                              std::ios_base::out);
        d << coContentDevice << std::endl;
        d.close();
        std::cout << "=============================================================================================" << std::endl;
    }
}


void sibir_at_init() {
    std::cout << "Kernel Instrument Tool is launching." << std::endl;
}


void sibir_at_term() {
    int counterHost;
    reinterpret_cast<hipError_t(*)(void*, void*, size_t, hipMemcpyKind)>(sibir_get_hip_function("hipMemcpy"))(
        &counterHost, &counter, 4, hipMemcpyDeviceToHost
    );
    std::cout << "Counter Value: " << counterHost << std::endl;
    std::cout << "Kernel Launch Intercept Tool is terminating!" << std::endl;
}


void sibir_at_hsa_event(hsa_api_args_t* args, sibir_api_phase_t phase, hsa_api_id_t api_id) {
    if (phase == SIBIR_API_PHASE_EXIT) {
        // Save the doorbell signal handles in the map whenever a queue is created.
        if (api_id == HSA_API_ID_hsa_queue_create) {
            auto queue = args->hsa_queue_create.queue;
            queue_map[(*queue)->doorbell_signal.handle] = *queue;
        }

        else if (api_id == HSA_API_ID_hsa_executable_freeze) {
            auto executable = args->hsa_executable_freeze.executable;
            fprintf(stdout, "HSA Executable Freeze Callback\n");
            // Get the state of the executable (frozen or not frozen)
            hsa_executable_state_t e_state;
            SIBIR_HSA_CHECK(hsa_executable_get_info(executable, HSA_EXECUTABLE_INFO_STATE, &e_state));

            fprintf(stdout, "Is executable frozen: %s\n", (e_state == HSA_EXECUTABLE_STATE_FROZEN ? "yes" : "no"));
            auto& coreTable = sibir_get_hsa_table()->core_;
            fprintf(stdout, "Executable handle: %lX", executable.handle);

            std::vector<hsa_agent_t> agentList;
            getGpuAgents(agentList);


            auto symbolCallbackFunction = [](hsa_executable_t exec,
                                             hsa_agent_t agent,
                                             hsa_executable_symbol_t symbol,
                                             void *data) {
                auto out = reinterpret_cast<std::vector<hsa_executable_symbol_t>*>(data);
                out->push_back(symbol);
                return HSA_STATUS_SUCCESS;
            };

            for (size_t i = 0; i < agentList.size(); i++) {
                std::cout << "iterating over agent " << i << " with handle: " << agentList[i] << std::endl;
                std::vector<hsa_executable_symbol_t> symbols;
                SIBIR_HSA_CHECK(
                    coreTable->hsa_executable_iterate_agent_symbols_fn(executable, agentList[i],
                                                                       symbolCallbackFunction, &symbols)
                );
                std::cout << "Number of symbols found: " << symbols.size() << std::endl;
                for (auto symbol: symbols) {
                    uint32_t nameLength;
                    coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameLength);
                    std::string name;
                    name.resize(nameLength);
                    coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data());
                    std::cout << "Symbol Name: " << name << std::endl;
                    uint64_t kernelObject;
                    coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject);
                    std::cout << "Kernel Object is " << kernelObject << std::endl;
                    ko_symbol_map[kernelObject] = symbol;
                }
            }

//            printf("Test create input data set\n");
//
//            size_t Size1;
//            char *Buf1;
//            amd_comgr_data_t DataIn1;
//            amd_comgr_data_set_t DataSetIn, DataSetOut;
//            amd_comgr_action_info_t DataAction;
//            amd_comgr_status_t Status;
//
//            SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data_set(&DataSetIn));
//
//            // File 1
//            SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataIn1));
//
//            SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(DataIn1, Size1, Buf1));
//
//
//            SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data_name(DataIn1, "source1.s"));
//
//
//            SIBIR_AMD_COMGR_CHECK(amd_comgr_data_set_add(DataSetIn, DataIn1));
//
//
//            SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data_set(&DataSetOut));
//
//            {
//                printf("Test action assemble\n");
//                Status = amd_comgr_create_action_info(&DataAction);
//                checkError(Status, "amd_comgr_create_action_info");
//                amd_comgr_action_info_set_isa_name(DataAction, "amdgcn-amd-amdhsa--gfx803");
//                checkError(Status, "amd_comgr_action_info_set_language");
//                Status = amd_comgr_action_info_set_option_list(DataAction, NULL, 0);
//                checkError(Status, "amd_comgr_action_info_set_option_list");
//                Status =
//                    amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
//                                        DataAction, DataSetIn, DataSetOut);
//                checkError(Status, "amd_comgr_do_action");
//            }


            //
//            getAllExecutableSymbols(executable, symbols);
//            auto coreTable = sibir_get_hsa_table()->core_;
//            for (auto s: symbols) {
//                hsa_symbol_kind_t kind;
//                coreTable->hsa_executable_symbol_get_info_fn(s, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind);
//                if (kind == HSA_SYMBOL_KIND_KERNEL) {
//                    int kernelObject;
//                    CHECK_HSA_CALL(coreTable->hsa_executable_symbol_get_info_fn(s, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject));
//                    std::cout << "Kernel Object: " << kernelObject << std::endl;
//                    size_t nameSize;
//                    CHECK_HSA_CALL(coreTable->hsa_executable_symbol_get_info_fn(s, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameSize));
//                    std::cout << "Kernel name size: " << nameSize << std::endl;
//                    auto name = new char[nameSize];
//                    CHECK_HSA_CALL(coreTable->hsa_executable_symbol_get_info_fn(s, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &name));
//                    std::cout << "Kernel Name: " << name << std::endl;
//                    ko_symbol_map[kernelObject] = s;
//                    delete[] name;
//                }
//
//            }
//            iterateLoadedCodeObjects(executable);
            fprintf(stdout, "");
        }
    }
    else if (phase == SIBIR_API_PHASE_ENTER) {
        if (api_id == HSA_API_ID_hsa_signal_store_screlease) {
            fprintf(stdout, "<call to (%s)\t on %s> ",
                    "hsa_signal_store_screlease",
                    "entry");
            instrumentKernelLaunchCallback(args->hsa_signal_store_screlease.signal,
                                           args->hsa_signal_store_screlease.value);
            }
        else if (api_id == HSA_API_ID_hsa_signal_store_relaxed) {
            fprintf(stdout, "<call to (%s)\t on %s> ",
                    "hsa_signal_store_relaxed",
                    "entry");

            instrumentKernelLaunchCallback(args->hsa_signal_store_relaxed.signal,
                                           args->hsa_signal_store_relaxed.value);
        }
    }
}

void sibir_at_hip_event(void* args, sibir_api_phase_t phase, int hip_api_id) {
    fprintf(stdout, "<call to (%s)\t on %s> ",
            hip_api_name(hip_api_id),
            phase == SIBIR_API_PHASE_ENTER ? "entry" : "exit"
            );
    if (hip_api_id == HIP_API_ID_hipLaunchKernel) {
        auto kern_args = reinterpret_cast<hip_hipLaunchKernel_api_args_t*>(args);
        fprintf(stdout, "kernel(\"%s\") stream(%p)",
                hipKernelNameRefByPtr(kern_args->function_address,
                                      kern_args->stream),
                kern_args->stream);
    }
    fprintf(stdout, "\n"); fflush(stdout);
}

