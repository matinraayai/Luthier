#include "luthier.h"
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
static std::map<decltype(hsa_signal_t::handle), hsa_queue_t *> queue_map;
// This is for preventing the queue content to be modified during our interception. Might not be necessary.
// Might use the HSA variants that locks a mutex when reading the queue index.
std::mutex mutex;
// Holds the AMD Vendor function pointer table.
static hsa_ven_amd_loader_1_03_pfn_t amdTable{};
static bool amdTableInitialized{false};
static bool initializeGlobalVar{false};

static bool instrumented{true};

__device__ __managed__ int counter;

__device__ void instrumentation_kernel() {
    atomicAdd(&counter, 1);
}

SIBIR_EXPORT_FUNC(instrumentation_kernel)

/**
 * Returns the list of GPU HSA agents (devices) on the system
 * @param [out] agentList a vector of GPU agents
 * @return HSA_STATUS_SUCCESS
 */
hsa_status_t getGpuAgents(std::vector<hsa_agent_t> &agentList) {
    int i = 0;
    auto iterateAgentCallback = [](hsa_agent_t agent, void *data) {
        auto agent_list = reinterpret_cast<std::vector<hsa_agent_t> *>(data);
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
    return hsa_iterate_agents(iterateAgentCallback, &agentList);
}

hsa_status_t getAllExecutableSymbols(const hsa_executable_t &executable,
                                     const std::vector<hsa_agent_t> &agentList,
                                     std::vector<hsa_executable_symbol_t> &symbols) {
    hsa_status_t out = HSA_STATUS_ERROR;
    for (auto agent: agentList) {
        auto iterCallback = [](hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void *data) {
            auto symbolVec = reinterpret_cast<std::vector<hsa_executable_symbol_t> *>(data);
            symbolVec->push_back(symbol);
            return HSA_STATUS_SUCCESS;
        };
        out = hsa_executable_iterate_agent_symbols(executable,
                                                   agent,
                                                   iterCallback, &symbols);
        if (out != HSA_STATUS_SUCCESS)
            return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
}

std::ostream &operator<<(std::ostream &os, const kernel_descriptor_t &kd) {
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

std::ostream &operator<<(std::ostream &os, const hsa_packet_type_t &packetType) {
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
        default: throw std::invalid_argument("Failed to locate the HSA packet type enum from number " + std::to_string(static_cast<int>(packetType)) + ".");
    }
    return os;
}

inline void check_error(hsa_status_t err, const char *call_name) {
    if (err != HSA_STATUS_SUCCESS) {
        const char *err_msg = "Unknown Error";
        hsa_status_string(err, &err_msg);
        std::cerr << "HSA call " << call_name << "failed! Error code: " << err_msg << std::endl
                  << std::flush;
    }
}

#define CHECK_HSA_CALL(call) check_error(call, #call)

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
    char *instChar{};
    auto curr_address = entrypoint;
    amd_dbgapi_size_t instrSize;

    while (!is_end) {
        instrSize = maxInstrLen;

        amd_dbgapi_disassemble_instruction(arch, curr_address, &instrSize,
                                           reinterpret_cast<void *>(curr_address),
                                           &instChar, nullptr, {});

        std::vector<std::byte> instBytes(instrSize);
        // Copy the instruction bytes
        for (amd_dbgapi_size_t i = 0; i < instrSize; i++) {
            instBytes[i] = reinterpret_cast<std::byte *>(curr_address)[i];
        }
        // Copy the decoded instruction string
        std::string instStr(instChar);
        //        std::cout << instStr << ": ";
        if (instStr.find("s_add_i32") != std::string::npos) {
            std::cout << instStr << ": ";
            for (const auto &el: instBytes)
                std::cout << std::hex << std::setfill('0') << std::setw(2) << uint16_t(el) << " ";
            auto overwrite_address = reinterpret_cast<uint8_t *>(curr_address);
            //            auto out =
            //            reinterpret_cast<hipError_t (*)(void* dst, int value, size_t sizeBytes)>(luthier_get_hip_function("hipMemset"))(overwrite_address + 1,
            //                                                                                                                          0xc5, 1);
            //            assert(out == HIP_SUCCESS);
            //            reinterpret_cast<hipError_t(*)(void*, void*, size_t, hipMemcpyKind)>(luthier_get_hip_function("hipMemcpy"))(
            //                overwrite_address + 1,
            //                )
            overwrite_address[1] = 0x85;
            //            curr_address
        }
        free(instChar);

        curr_address += instrSize;
        is_end = instStr.find("s_endpgm") != std::string::npos;
    }
    //    return instList;

    //    disassemble_kd(kd);
    //    auto kernel_entry = reinterpret_cast<const unsigned char*>(kd) + kd->kernel_code_entry_byte_offset;
    //    return {};
    // 02 c5 02 81
    //02 85 02 81
}

void instrumentKernelLaunchCallback(hsa_signal_t signal, hsa_signal_value_t value) {
    auto it = queue_map.find(signal.handle);
    if (it != queue_map.end()) {
        hsa_queue_t *queue = it->second;
        mutex.lock();
        const uint64_t begin = luthier_get_hsa_table()->core_->hsa_queue_load_read_index_relaxed_fn(queue);
        const uint64_t end = value + 1;
        mutex.unlock();
        uint32_t mask = queue->size - 1;
        for (uint64_t j = begin; j < end; ++j) {
            const unsigned int idx = j & (queue->size - 1);
            hsa_barrier_or_packet_t *packet = reinterpret_cast<hsa_barrier_or_packet_t *>(queue->base_address) + idx;

            unsigned int packetType = extractAqlBits(packet->header, HSA_PACKET_HEADER_TYPE, HSA_PACKET_HEADER_WIDTH_TYPE);
            std::cout << "Intercepted Packet Type: " << static_cast<hsa_packet_type_t>(packetType) << std::endl;
            if (packetType == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
                auto *dispatchPacket = reinterpret_cast<hsa_kernel_dispatch_packet_t *>(packet);

                // Print the dispatch packet content
                std::cout << "Dispatch Packet Content:" << *dispatchPacket << std::endl;

                // Print the Kernel Descriptor Content
                std::cout << "KD Location: " << dispatchPacket << std::endl;
                const kernel_descriptor_t *kernelDescriptor = nullptr;

                CHECK_HSA_CALL(amdTable.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(dispatchPacket->kernel_object),
                                                                              reinterpret_cast<const void **>(&kernelDescriptor)));
                std::cout << *kernelDescriptor << std::endl;
                // Print the address of the kernel Object
                std::cout << "Kernel Descriptor Location: " << reinterpret_cast<void *>(dispatchPacket->kernel_object) << std::endl;
                std::cout << "Kernel Entry: " << reinterpret_cast<const void *>(reinterpret_cast<const std::byte *>(kernelDescriptor) + kernelDescriptor->kernel_code_entry_byte_offset) << std::endl;
                std::cout << "Kernel content: " << std::endl;
                auto kernelEntryPoint =
                    reinterpret_cast<amd_dbgapi_global_address_t>(dispatchPacket->kernel_object) + kernelDescriptor->kernel_code_entry_byte_offset;
                instrumentKernel(kernelEntryPoint);
                //                auto instructions = luthier_disassemble_kd(kernelEntryPoint);
                //                hipPointerAttribute_t counterAttributes;
                //                reinterpret_cast<hipError_t (*)(hipPointerAttribute_t *, const void *)>(luthier_get_hip_function("hipPointerGetAttributes"))(
                //                    &counterAttributes, counter);
                ////                std::cout << "Counter address on host: " << counterAttributes.hostPointer << std::endl;
                ////                std::cout << "Counter address on device:" << (reinterpret_cast<uint64_t>(counterAttributes.devicePointer) & 0xFFFFFFFF) << std::endl;
                //                uint8_t counterDeviceAddress = (reinterpret_cast<uint64_t>(counterAttributes.devicePointer) & 0xFFFFFFFF);
                //                uint8_t firstByte = 0x00FF0000 & counterDeviceAddress;
                //                uint8_t secondByte = 0x0000FF0000 & counterDeviceAddress;
                //                uint8_t thirdByte = 0x0000FF00 & counterDeviceAddress;
                //                uint8_t fourthByte = 0x000000FF & counterDeviceAddress;
                //                std::vector<std::vector<uint8_t>> counterCode{
                //                    {0x00, 0xFF, 0x00, 0x80, fourthByte, thirdByte, secondByte, firstByte},
                //                                                              {0x01, 0xFF, 0x01, 0x82, 0x00, 0x00, 0x00, 0x00},
                //                                                              {0x80, 0x02, 0x00, 0x7E},
                //                                                              {0x81, 0x02, 0x02, 0x7E},
                //                                                              {0x00, 0x80, 0x08, 0xDD, 0x00, 0x01, 0x00, 0x00},
                //                                                              {0x00, 0x00, 0x81, 0xbf}};
                //
                ////                    Decoded by rocdbg-api: s_add_u32 s0, s0, 0x1ffc Instruction Size: 8 Address: 7fe86616d004 Bytes: 0 ff 0 80 fc 1f 0 0
                ////                    Decoded by rocdbg-api: s_addc_u32 s1, s1, 0 Instruction Size: 8 Address: 7fe86616d00c Bytes: 1 ff 1 82 0 0 0 0
                ////                    Decoded by rocdbg-api: v_mov_b32_e32 v0, 0 Instruction Size: 4 Address: 7fe86616d014 Bytes: 80 2 0 7e
                ////                    Decoded by rocdbg-api: v_mov_b32_e32 v1, 1 Instruction Size: 4 Address: 7fe86616d018 Bytes: 81 2 2 7e
                ////                    Decoded by rocdbg-api: global_store_dword v0, v1, s[0:1] Instruction Size: 8 Address: 7fe86616d01c Bytes: 0 80 70 dc 0 1 0 0
                //
                ////                hipPointerGetAttribute(&counterAttributes, )
                ////                std::vector<
                ////
                //                for (auto& instr: instructions) {
                //                    std::cout << instr.first << ": ";
                //                    for (std::byte &el: instr.second) {
                //                        std::cout << std::hex << std::setfill('0') << std::setw(2) << uint16_t(el) << " ";
                //                    }
                //                    std::cout << std::dec << std::endl;
                //                }
                //                if (!instrumnted) {
                //
                //
                //                    auto overwriteAddress = reinterpret_cast<uint8_t*>(dispatchPacket->kernel_object) + kernelDescriptor->kernel_code_entry_byte_offset;
                //                    std::cout << "overwrite Address:" << reinterpret_cast<void*>(overwriteAddress) << std::endl;
                //                    // instrument the kernel lol
                //
                //                    for (const auto &i: counterCode)
                //                        for (const auto &b: i) {
                //                            reinterpret_cast<hipError_t(*)(void*, void*, size_t, hipMemcpyKind)>(luthier_get_hip_function("hipMemcpy"))(
                //                                overwriteAddress, (void *) &b, 1, hipMemcpyHostToDevice);
                ////                            *overwriteAddress = b;
                //                            overwriteAddress++;
                //                        }
                ////                    for (const auto &i: instructions)
                ////                        for (const auto &b: i.second) {
                ////                            *overwriteAddress = static_cast<uint8_t>(b);
                ////                            overwriteAddress++;
                ////                        }
                //                    //                std::cout << isa.size() << std::endl;
                //                    //                print_instructions(isa);
                //                    instrumnted = true;
                //                }
            }
        }
    } else {
        throw std::invalid_argument("Signal handle " + std::to_string(signal.handle) + "'s queue was not found in the map.");
    }
}

void iterateLoadedCodeObjects(hsa_executable_t executable) {
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void *data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t> *>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, &loadedCodeObjects);

    // Dump all the code objects into files
    for (int i = 0; i < loadedCodeObjects.size(); i++) {
        // Query the base address of the loaded code object on host
        uint64_t lcoBaseAddrHost;
        amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
                                                                HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
                                                                &lcoBaseAddrHost);
        // Query the size of the loaded code object on host
        uint64_t lcoSizeHost;
        amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
                                                                HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
                                                                &lcoSizeHost);

        uint64_t lcoBaseAddrDevice;
        amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
                                                                HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                                                                &lcoBaseAddrDevice);
        // Query the size of the loaded code object
        uint64_t lcoSizeDevice;
        amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[i],
                                                                HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                                                                &lcoSizeDevice);

        std::cout << "Base address of the executable on the host: " << reinterpret_cast<void *>(lcoBaseAddrHost) << std::endl;
        std::cout << "Size of the executable on the host: " << lcoSizeHost << std::endl;
        std::cout << "Base address of the executable on the device: " << reinterpret_cast<void *>(lcoBaseAddrDevice) << std::endl;
        std::cout << "Size of the executable on the device: " << lcoSizeDevice << std::endl;

        std::string coContentHost(reinterpret_cast<char *>(lcoBaseAddrHost), lcoSizeHost);
        std::string coContentDevice(reinterpret_cast<char *>(lcoBaseAddrDevice), lcoSizeDevice);
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

void luthier_at_init() {
    std::cout << "Kernel Launch Intercept Tool is launching." << std::endl;
    //    reinterpret_cast<hipError_t(*)(void**,size_t)>(luthier_get_hip_function("hipMalloc"))(reinterpret_cast<void **>(&counter), sizeof(uint64_t));
    //    dummy<<<1, 1>>>();
}

void luthier_at_term() {
    memset(&amdTable, 0, sizeof(hsa_ven_amd_loader_1_01_pfn_t));
    int counterHost;
    reinterpret_cast<hipError_t (*)(void *, void *, size_t, hipMemcpyKind)>(luthier_get_hip_function("hipMemcpy"))(
        &counterHost, &counter, 4, hipMemcpyDeviceToHost);
    std::cout << "Counter Value: " << counterHost << std::endl;
    //    reinterpret_cast<hipError_t(*)(void*)>(luthier_get_hip_function("hipFree"))(counter);
    std::cout << "Kernel Launch Intercept Tool is terminating!" << std::endl;
}

void luthier_at_hsa_event(hsa_api_args_t *args, luthier_api_phase_t phase, hsa_api_id_t api_id) {
    if (!amdTableInitialized) {
        CHECK_HSA_CALL(luthier_get_hsa_table()->core_->hsa_system_get_major_extension_table_fn(
            HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_01_pfn_t),
            &amdTable));
        amdTableInitialized = true;
    }
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
            CHECK_HSA_CALL(hsa_executable_get_info(executable, HSA_EXECUTABLE_INFO_STATE, &e_state));

            fprintf(stdout, "Is executable frozen: %s\n", (e_state == HSA_EXECUTABLE_STATE_FROZEN ? "yes" : "no"));
            std::vector<hsa_agent_t> agentList;
            std::vector<hsa_executable_symbol_t> symbols;
            getGpuAgents(agentList);
            getAllExecutableSymbols(executable, agentList, symbols);
            iterateLoadedCodeObjects(executable);
            fprintf(stdout, "");
        }
    } else if (phase == SIBIR_API_PHASE_ENTER) {
        if (api_id == HSA_API_ID_hsa_signal_store_screlease) {
            fprintf(stdout, "<call to (%s)\t on %s> ",
                    "hsa_signal_store_screlease",
                    "entry");
            interceptKernelLaunchCallback(args->hsa_signal_store_screlease.signal,
                                          args->hsa_signal_store_screlease.value);
        } else if (api_id == HSA_API_ID_hsa_signal_store_relaxed) {
            fprintf(stdout, "<call to (%s)\t on %s> ",
                    "hsa_signal_store_relaxed",
                    "entry");

            interceptKernelLaunchCallback(args->hsa_signal_store_relaxed.signal,
                                          args->hsa_signal_store_relaxed.value);
        }
    }
}

void luthier_at_hip_event(void *args, luthier_api_phase_t phase, int hip_api_id) {
    if (!initializeGlobalVar) {
        reinterpret_cast<hipError_t (*)(void **, size_t)>(luthier_get_hip_function("hipMalloc"))(
            reinterpret_cast<void **>(&counter), sizeof(uint64_t));
        //        reinterpret_cast<hipError_t (*)(void* dst, int value, size_t sizeBytes)>(luthier_get_hip_function("hipMemset"))(counter, 0, 4);
        initializeGlobalVar = true;
    }
    std::string name;
    //    if (!has_launched) {
    //        auto kernelLaunchFunc =
    //        reinterpret_cast<hipError_t(*)(const void*,
    //                                          dim3, dim3, void**, size_t, hipStream_t)>(luthier_get_hip_function("hipLaunchKernel"));
    //        kernelLaunchFunc(reinterpret_cast<const void*>(dummy), 1, 1, nullptr, 0, nullptr);
    ////        dummy<<<1, 1>>>();
    //        has_launched = true;
    //    }
    fprintf(stdout, "<call to (%s)\t on %s> ",
            hip_api_name(hip_api_id),
            phase == SIBIR_API_PHASE_ENTER ? "entry" : "exit");
    if (hip_api_id == HIP_API_ID_hipLaunchKernel) {
        auto kern_args = reinterpret_cast<hip_hipLaunchKernel_api_args_t *>(args);
        fprintf(stdout, "kernel(\"%s\") stream(%p)",
                hipKernelNameRefByPtr(kern_args->function_address,
                                      kern_args->stream),
                kern_args->stream);
    }
    fprintf(stdout, "\n");
    fflush(stdout);
}