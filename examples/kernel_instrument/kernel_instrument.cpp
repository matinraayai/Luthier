#include <fstream>
#include <functional>
#include <hip/hip_runtime_api.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <iostream>
#include <luthier.h>
#include <string>

static bool instrumented{false};

MARK_LUTHIER_DEVICE_MODULE

__managed__ int globalCounter = 20;

 __device__ __noinline__ extern "C" void instrumentation_kernel(int* counter) {
//    int i = 0;
//    i = i + 4;
//    i = i * 40;
//    return i;
//    return 1;
    *counter = *counter + 1;
//    atomicAdd(counter, 1);
//    printf("Hello from LUTHIER!\n");
}

LUTHIER_EXPORT_FUNC(instrumentation_kernel)



hsa_status_t getAllExecutableSymbols(const hsa_executable_t& executable,
                                     std::vector<hsa_executable_symbol_t>& symbols) {
    std::vector<hsa_agent_t> agents;

    auto& coreTable = luthier_get_hsa_table()->core_;

    auto queryAgentsCallback = [](hsa_agent_t agent, void* data) {
        auto agents = reinterpret_cast<std::vector<hsa_agent_t>*>(data);
        hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

        hsa_status_t stat = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

        if (stat != HSA_STATUS_SUCCESS)
            return stat;
        if (dev_type == HSA_DEVICE_TYPE_GPU)
            agents->push_back(agent);

        return stat;
    };

    LUTHIER_HSA_CHECK(coreTable->hsa_iterate_agents_fn(queryAgentsCallback, &agents));


    hsa_status_t out = HSA_STATUS_ERROR;
    auto iterCallback = [](hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data) {
        std::cout << "Agent handle: " << agent.handle << std::endl;
        auto& coreTable = luthier_get_hsa_table()->core_;
        hsa_symbol_kind_t symbolKind;
        LUTHIER_HSA_CHECK(coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbolKind));

        std::cout << "Symbol kind: " << symbolKind << std::endl;

        uint32_t nameSize;
        LUTHIER_HSA_CHECK(coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameSize));
        std::cout << "Symbol name size: " << nameSize << std::endl;
        std::string name;
        name.resize(nameSize);
        LUTHIER_HSA_CHECK(coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()));
        std::cout << "Symbol Name: " << name << std::endl;

        if (symbolKind == HSA_SYMBOL_KIND_VARIABLE) {
            luthier_address_t variableAddress;
            LUTHIER_HSA_CHECK(coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &variableAddress));
            std::cout << "Variable location: " << std::hex << variableAddress << std::dec << std::endl;
        }
        else {
            luthier_address_t kernelObject;
            LUTHIER_HSA_CHECK(coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject));
            std::cout << "Kernel location: " << std::hex << kernelObject << std::dec << std::endl;
//                            std::vector<luthier::Instr> instList = luthier_disassemble_kernel_object(kernelObject);
//                            std::cout << "Disassembly of the KO: " << std::endl;
//                            for (const auto& i : instList) {
//                                std::cout << i.getInstr() << std::endl;
//                            }
        }

        //            symbolVec->push_back(symbol);
        return HSA_STATUS_SUCCESS;
    };
    for (auto agent : agents) {

        out = hsa_executable_iterate_agent_symbols(executable,
                                                   agent,
                                                   iterCallback, &symbols);
        if (out != HSA_STATUS_SUCCESS)
            return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
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

//void getLoadedCodeObject(hsa_executable_t executable) {
//    auto amdTable = luthier_get_hsa_ven_amd_loader();
//    // Get a list of loaded code objects inside the executable
//    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
//    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void* data) -> hsa_status_t {
//        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t>*>(data);
//        out->push_back(lco);
//        return HSA_STATUS_SUCCESS;
//    };
//    amdTable->hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, & loadedCodeObjects);
//
//    // Dump all the code objects into files
//    for (int i = 0; i < loadedCodeObjects.size(); i++) {
//        // Query the base address of the loaded code object on host
//        uint64_t lcoBaseAddrHost;
//        amdTable->hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
//                                                                 HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
//                                                                 &lcoBaseAddrHost);
//        // Query the size of the loaded code object on host
//        uint64_t lcoSizeHost;
//        amdTable->hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
//                                                                 HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
//                                                                 &lcoSizeHost);
//
//
//
//        uint64_t lcoBaseAddrDevice;
//        amdTable->hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
//                                                                 HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
//                                                                 &lcoBaseAddrDevice);
//        // Query the size of the loaded code object
//        uint64_t lcoSizeDevice;
//        amdTable->hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
//                                                                 HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
//                                                                 &lcoSizeDevice);
//
//
//        std::cout << "Base address of the executable on the host: " << reinterpret_cast<void*>(lcoBaseAddrHost) << std::endl;
//        std::cout << "Size of the executable on the host: " << lcoSizeHost << std::endl;
//        std::cout << "Base address of the executable on the device: " << reinterpret_cast<void*>(lcoBaseAddrDevice) << std::endl;
//        std::cout << "Size of the executable on the device: " << lcoSizeDevice << std::endl;
//
//
//        std::string coContentHost(reinterpret_cast<char*>(lcoBaseAddrHost), lcoSizeHost);
//        std::string coContentDevice(reinterpret_cast<char*>(lcoBaseAddrDevice), lcoSizeDevice);
//        auto h = std::fstream("./" + std::to_string(executable.handle) + "_host_" + std::to_string(i) + ".elf",
//                              std::ios_base::out);
//        h << coContentHost << std::endl;
//        h.close();
//        auto d = std::fstream("./" + std::to_string(executable.handle) + "_device_" + std::to_string(i) + ".elf",
//                              std::ios_base::out);
//        d << coContentDevice << std::endl;
//        d.close();
//        std::cout << "=============================================================================================" << std::endl;
//    }
//}





void luthier_at_init() {
    std::cout << "Kernel Instrument Tool is launching." << std::endl;
}


void luthier_at_term() {
//    int counterHost;
//    reinterpret_cast<hipError_t(*)(void*, void*, size_t, hipMemcpyKind)>(luthier_get_hip_function("hipMemcpy"))(
//        &counterHost, &globalCounter, 4, hipMemcpyDeviceToHost
//    );
    std::cout << "Counter Value: " << globalCounter << std::endl;
    std::cout << "Kernel Launch Intercept Tool is terminating!" << std::endl;
}


void luthier_at_hsa_event(hsa_api_evt_args_t* args, luthier_api_evt_phase_t phase, hsa_api_evt_id_t api_id) {
    if (phase == LUTHIER_API_EVT_PHASE_EXIT) {
        if (api_id == HSA_API_ID_hsa_queue_create) {
            std::cout << "Queue created called!" << std::endl;
            std::cout << "Signal handle: " << (*(args->api_args.hsa_queue_create.queue))->doorbell_signal.handle << std::endl;
        }
        else if (api_id == HSA_API_ID_hsa_signal_store_screlease) {
            fprintf(stdout, "<call to (%s)\t on %s> ",
                    "hsa_signal_store_relaxed",
                    "entry");
            std::cout << "Signal handle" << args->api_args.hsa_signal_store_relaxed.signal.handle << std::endl;

        }
        else if (api_id == HSA_API_ID_hsa_executable_freeze) {
            auto executable = args->api_args.hsa_executable_freeze.executable;
            fprintf(stdout, "HSA Executable Freeze Callback\n");
            // Get the state of the executable (frozen or not frozen)
            hsa_executable_state_t e_state;
            LUTHIER_HSA_CHECK(hsa_executable_get_info(executable, HSA_EXECUTABLE_INFO_STATE, &e_state));

            fprintf(stdout, "Is executable frozen: %s\n", (e_state == HSA_EXECUTABLE_STATE_FROZEN ? "yes" : "no"));
            auto& coreTable = luthier_get_hsa_table()->core_;
            fprintf(stdout, "Executable handle: %lX\n", executable.handle);
//            std::vector<hsa_executable_symbol_t> symbols;
//            getAllExecutableSymbols(executable, symbols);
//
//            std::vector<hsa_agent_t> agentList;
////            getGpuAgents(agentList);
//
//
//            auto symbolCallbackFunction = [](hsa_executable_t exec,
//                                             hsa_agent_t agent,
//                                             hsa_executable_symbol_t symbol,
//                                             void *data) {
//                auto out = reinterpret_cast<std::vector<hsa_executable_symbol_t>*>(data);
//                out->push_back(symbol);
//                return HSA_STATUS_SUCCESS;
//            };
//
//            for (size_t i = 0; i < agentList.size(); i++) {
//                std::cout << "iterating over agent " << i << " with handle: " << agentList[i] << std::endl;
//                std::vector<hsa_executable_symbol_t> symbols;
//                LUTHIER_HSA_CHECK(
//                    coreTable->hsa_executable_iterate_agent_symbols_fn(executable, agentList[i],
//                                                                       symbolCallbackFunction, &symbols)
//                );
//                std::cout << "Number of symbols found: " << symbols.size() << std::endl;
//                for (auto symbol: symbols) {
//                    uint32_t nameLength;
//                    coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameLength);
//                    std::string name;
//                    name.resize(nameLength);
//                    coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data());
//                    std::cout << "Symbol Name: " << name << std::endl;
//                    uint64_t kernelObject;
//                    coreTable->hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject);
//                    std::cout << "Kernel Object is " << kernelObject << std::endl;
//                    ko_symbol_map[kernelObject] = symbol;
//                }
//            }

//            printf("Test create input data set\n");
//
//            size_t Size1;
//            char *Buf1;
//            amd_comgr_data_t DataIn1;
//            amd_comgr_data_set_t DataSetIn, DataSetOut;
//            amd_comgr_action_info_t DataAction;
//            amd_comgr_status_t Status;
//
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&DataSetIn));
//
//            // File 1
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &DataIn1));
//
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(DataIn1, Size1, Buf1));
//
//
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data_name(DataIn1, "source1.s"));
//
//
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_data_set_add(DataSetIn, DataIn1));
//
//
//            LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&DataSetOut));
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
//            auto coreTable = luthier_get_hsa_table()->core_;
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

    if (api_id == HSA_EVT_ID_hsa_queue_packet_submit) {
        std::cout << "In packet submission callback" << std::endl;
        auto packets = reinterpret_cast<const hsa_ext_amd_aql_pm4_packet_t*>(args->evt_args.hsa_queue_packet_submit.packets);
        for (unsigned int i = 0; i < args->evt_args.hsa_queue_packet_submit.pkt_count; i++) {
            unsigned int packetType = extractAqlBits(packets[i].header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
            if (packetType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
                std::cout << "Dispatch packet found!" << std::endl;
            }
        }
        std::cout << "End of callback" << std::endl;
    }
}

void luthier_at_hip_event(void* args, luthier_api_evt_phase_t phase, int hip_api_id) {
    fprintf(stdout, "<call to (%s)\t on %s> ",
            hip_api_name(hip_api_id),
            phase == LUTHIER_API_EVT_PHASE_ENTER ? "entry" : "exit"
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
