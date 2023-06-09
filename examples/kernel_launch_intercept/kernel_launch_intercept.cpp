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
static std::map<decltype(hsa_signal_t::handle), hsa_queue_t*> queue_map;
// This is for preventing the queue content to be modified during our interception. Might not be necessary.
// Might use the HSA variants that locks a mutex when reading the queue index.
std::mutex mutex;
// Holds the AMD Vendor function pointer table.
static hsa_ven_amd_loader_1_01_pfn_t amdTable{};
static bool amdTableInitialized{false};


struct kernel_descriptor_t {
    uint8_t reserved0[16];
    int64_t kernel_code_entry_byte_offset;
    uint8_t reserved1[20];
    uint32_t compute_pgm_rsrc3;
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;
    uint8_t reserved2[6];
};

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

inline void check_error(hsa_status_t err, const char* call_name) {
    if (err != HSA_STATUS_SUCCESS) {
        const char* err_msg = "Unknown Error";
        hsa_status_string(err, &err_msg);
        std::cerr << "HSA call " << call_name << "failed! Error code: " <<
            err_msg << std::endl << std::flush; \
    }
}

#define CHECK_HSA_CALL(call) check_error(call, #call)

static unsigned extractAqlBits(unsigned v, unsigned pos, unsigned width) {
    return (v >> pos) & ((1 << width) - 1);
};

void interceptKernelLaunchCallback(hsa_signal_t signal, hsa_signal_value_t value) {
    auto it = queue_map.find(signal.handle);
    if (it != queue_map.end()) {
        hsa_queue_t *queue = it->second;
        mutex.lock();
        const uint64_t begin = sibir_get_hsa_table()->core_->hsa_queue_load_read_index_relaxed_fn(queue);
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
                const kernel_descriptor_t *kernelDescriptor = nullptr;

                CHECK_HSA_CALL(amdTable.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(dispatchPacket->kernel_object),
                                                                              reinterpret_cast<const void **>(&kernelDescriptor)));
                std::cout << *kernelDescriptor << std::endl;
                // Print the address of the kernel Object
                std::cout << "Kernel Descriptor Location: " << reinterpret_cast<void *>(dispatchPacket->kernel_object) << std::endl;
                std::cout << "Kernel Entry: " << reinterpret_cast<const void*>(reinterpret_cast<const unsigned char*>(kernelDescriptor) +
                                                  kernelDescriptor->kernel_code_entry_byte_offset) << std::endl;
            }
        }
    } else {
        throw std::invalid_argument("Signal handle " + std::to_string(signal.handle) + "'s queue was not found in the map.");
    }
}


void sibir_at_init() {
    std::cout << "Kernel Launch Intercept Tool is launching." << std::endl;
}


void sibir_at_term() {
    memset(&amdTable, 0, sizeof(hsa_ven_amd_loader_1_01_pfn_t));
    std::cout << "Kernel Launch Intercept Tool is terminating!" << std::endl;
}


void sibir_at_hsa_event(hsa_api_args_t* args, sibir_api_phase_t phase, hsa_api_id_t api_id) {
    if (!amdTableInitialized) {
        CHECK_HSA_CALL(sibir_get_hsa_table()->core_->hsa_system_get_major_extension_table_fn(
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
    }
    else if (phase == SIBIR_API_PHASE_ENTER) {
        if (api_id == HSA_API_ID_hsa_signal_store_screlease) {
            interceptKernelLaunchCallback(args->hsa_signal_store_screlease.signal,
                                          args->hsa_signal_store_screlease.value);
            }
        else if (api_id == HSA_API_ID_hsa_signal_store_relaxed) {
            interceptKernelLaunchCallback(args->hsa_signal_store_relaxed.signal,
                                          args->hsa_signal_store_relaxed.value);
        }
    }
}

void sibir_at_hip_event(hip_api_args_t* args, sibir_api_phase_t phase, hip_api_id_t api_id) {
    fprintf(stdout, "<call to (%s)\t on %s> ",
            hip_api_name(api_id),
            phase == SIBIR_API_PHASE_ENTER ? "entry" : "exit"
            );
    switch (api_id) {
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