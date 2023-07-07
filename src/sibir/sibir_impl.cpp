#include <sibir_impl.hpp>
#include <sibir.h>
#include <roctracer/roctracer.h>
#include "hsa_intercept.h"
#include <amd-dbgapi/amd-dbgapi.h>
#include <unistd.h>
#include <iomanip>
#include "code_object_manager.h"

static amd_dbgapi_callbacks_t amd_dbgapi_callbacks{
    .allocate_memory = malloc,
    .deallocate_memory = free,

    .get_os_pid =
        [](amd_dbgapi_client_process_id_t client_process_id, pid_t *pid){
            *pid = getpid();
            return AMD_DBGAPI_STATUS_SUCCESS;
        },

    .insert_breakpoint =
        [](amd_dbgapi_client_process_id_t client_process_id,
           amd_dbgapi_global_address_t address,
           amd_dbgapi_breakpoint_id_t breakpoint_id)
    {
        return AMD_DBGAPI_STATUS_SUCCESS;
    },

    .remove_breakpoint =
        [](amd_dbgapi_client_process_id_t client_process_id,
           amd_dbgapi_breakpoint_id_t breakpoint_id)
    {
        return AMD_DBGAPI_STATUS_SUCCESS;
    },

    .log_message =
        [](amd_dbgapi_log_level_t level, const char *message) {}
};


void Sibir::hip_startup_callback(void* cb_data, sibir_api_phase_t phase, int api_id) {
//    static std::vector<hip___hipRegisterFatBinary_api_args_t*>
    static const void* lastSavedFatBinary{};
    if (phase == SIBIR_API_PHASE_EXIT) {
        if (api_id == HIP_PRIVATE_API_ID___hipRegisterFatBinary) {
            auto args = reinterpret_cast<hip___hipRegisterFatBinary_api_args_t*>(cb_data);
            lastSavedFatBinary = args->data;
        }
        else if (api_id == HIP_PRIVATE_API_ID___hipRegisterFunction) {
            auto args = reinterpret_cast<hip___hipRegisterFunction_api_args_t*>(cb_data);
            auto& coManager = SibirCodeObjectManager::Instance();

            // If the function doesn't have __sibir_wrap__ in its name then it belongs to the instrumented application
            // Give control to the user-defined callback
            if (std::string(args->deviceFunction).find("__sibir_wrap__") == std::string::npos)
                SibirHipInterceptor::Instance().SetCallback(Sibir::hip_api_callback);
            else {
                coManager.registerFatBinary(lastSavedFatBinary);
                coManager.registerFunction(lastSavedFatBinary, args->deviceFunction, args->hostFunction, args->deviceName);
            }
        }
        else if (api_id == HIP_PRIVATE_API_ID___hipRegisterManagedVar) {
        }
        else if (api_id == HIP_PRIVATE_API_ID___hipRegisterSurface) {

        }
        else if (api_id == HIP_PRIVATE_API_ID___hipRegisterTexture) {

        }
        else if (api_id == HIP_PRIVATE_API_ID___hipRegisterVar) {

        }
    }
}


void __attribute__((constructor)) Sibir::init() {
    std::cout << "Initializing Sibir...." << std::endl << std::flush;
    assert(amd_dbgapi_initialize(&amd_dbgapi_callbacks) == AMD_DBGAPI_STATUS_SUCCESS);
    assert(SibirHipInterceptor::Instance().IsEnabled());
    sibir_at_init();
    SibirHipInterceptor::Instance().SetCallback(Sibir::hip_startup_callback);
    SibirHsaInterceptor::Instance().SetCallback(Sibir::hsa_api_callback);
}

__attribute__((destructor)) void Sibir::finalize() {
    assert(amd_dbgapi_finalize() == AMD_DBGAPI_STATUS_SUCCESS);
    sibir_at_term();
    std::cout << "Sibir Terminated." << std::endl << std::flush;
}

const HsaApiTable* sibir_get_hsa_table() {
    return &SibirHsaInterceptor::Instance().getSavedHsaTables().root;
}

const hsa_ven_amd_loader_1_03_pfn_s* sibir_get_hsa_ven_amd_loader() {
    return &SibirHsaInterceptor::Instance().getHsaVenAmdLoaderTable();
}

void Sibir::hip_api_callback(void* cb_data, sibir_api_phase_t phase, int api_id) {
    sibir_at_hip_event(cb_data, phase, api_id);
}

void Sibir::hsa_api_callback(hsa_api_args_t* cb_data, sibir_api_phase_t phase, hsa_api_id_t api_id) {
    sibir_at_hsa_event(cb_data, phase, api_id);
}
std::vector<std::pair<std::string, std::vector<std::byte>>> sibir_disassemble_kernel_object(uint64_t kernel_object) {
    const kernel_descriptor_t *kernelDescriptor = nullptr;
    auto amdTable = SibirHsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    SIBIR_HSA_CHECK(amdTable.hsa_ven_amd_loader_query_host_address(reinterpret_cast<const void *>(kernel_object),
                                                                   reinterpret_cast<const void **>(&kernelDescriptor)));
    auto kernelEntryPoint =
        reinterpret_cast<amd_dbgapi_global_address_t>(kernel_object) + kernelDescriptor->kernel_code_entry_byte_offset;

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
    auto curr_address = kernelEntryPoint;
    amd_dbgapi_size_t instrSize;

    std::vector<std::pair<std::string, std::vector<std::byte>>> instList;
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

        free(instChar);
        instList.emplace_back(instStr, instBytes);

        curr_address += instrSize;
        is_end = instStr.find("s_endpgm") != std::string::npos;
    }
    return instList;
}

//void print_instructions(const std::vector<Inst>& isa) {
//
//        std::cout << "Decoded by rocdbg-api: " << instruction << " Instruction Size: " << instrSize << " Address: " << std::hex <<
//            curr_address << " Bytes: ";
//        for (std::byte &el: instBytes) {
//            std::cout << std::hex << std::setfill('0') << std::setw(1) << uint16_t(el) << " ";
//        }
//        std::cout << std::dec << std::endl;
////    InstPrinter printer{};
////    for (const auto& inst: isa)
////        std::cout << printer.print(inst) << std::endl;
//}

void* sibir_get_hip_function(const char* funcName) {
    return SibirHipInterceptor::Instance().GetHipFunction(funcName);
}


void sibir_insert_call(const Instr *instr, const char *dev_func_name, sibir_ipoint_t point) {


}

extern "C" {

ROCTRACER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 49;

ROCTRACER_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                             uint64_t failed_tool_count, const char *const *failed_tool_names) {
    [](auto &&...) {}(runtime_version, failed_tool_count, failed_tool_names);
    return SibirHsaInterceptor::Instance().captureHsaApiTable(table);
}

ROCTRACER_EXPORT void OnUnload() {}
}