#include "sibir_impl.hpp"
#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "context_manager.hpp"
#include "disassembler.hpp"
#include "hsa_intercept.hpp"
#include <iomanip>
#include <roctracer/roctracer.h>
#include <sibir.h>

void sibir::impl::hipStartupCallback(void *cb_data, sibir_api_phase_t phase, int api_id) {
    //    static std::vector<hip___hipRegisterFatBinary_api_args_t*>
    static const void *lastSavedFatBinary{};
    if (phase == SIBIR_API_PHASE_EXIT) {
        if (api_id == HIP_PRIVATE_API_ID___hipRegisterFatBinary) {
            auto args = reinterpret_cast<hip___hipRegisterFatBinary_api_args_t *>(cb_data);
            lastSavedFatBinary = args->data;
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterFunction) {
            auto args = reinterpret_cast<hip___hipRegisterFunction_api_args_t *>(cb_data);
            auto &coManager = CodeObjectManager::Instance();

            // If the function doesn't have __sibir_wrap__ in its name then it belongs to the instrumented application
            // Give control to the user-defined callback
            if (std::string(args->deviceFunction).find("__sibir_wrap__") == std::string::npos)
                HipInterceptor::Instance().SetCallback(sibir::impl::hipApiCallback);
            else {
                coManager.registerFatBinary(lastSavedFatBinary);
                coManager.registerFunction(lastSavedFatBinary, args->deviceFunction, args->hostFunction, args->deviceName);
            }
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterManagedVar) {
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterSurface) {

        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterTexture) {

        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterVar) {
        }
    }
}

__attribute__((constructor)) void sibir::impl::init() {
    std::cout << "Initializing Sibir...." << std::endl
              << std::flush;
    assert(HipInterceptor::Instance().IsEnabled());
    sibir_at_init();
    HipInterceptor::Instance().SetCallback(sibir::impl::hipStartupCallback);
    HsaInterceptor::Instance().SetCallback(sibir::impl::hsaApiCallback);
}

__attribute__((destructor)) void sibir::impl::finalize() {
    sibir_at_term();
    std::cout << "Sibir Terminated." << std::endl
              << std::flush;
}

const HsaApiTable *sibir_get_hsa_table() {
    return &sibir::HsaInterceptor::Instance().getSavedHsaTables().root;
}

const hsa_ven_amd_loader_1_03_pfn_s *sibir_get_hsa_ven_amd_loader() {
    return &sibir::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
}

void sibir::impl::hipApiCallback(void *cb_data, sibir_api_phase_t phase, int api_id) {
    sibir_at_hip_event(cb_data, phase, api_id);
}

void sibir::impl::hsaApiCallback(hsa_api_args_t *cb_data, sibir_api_phase_t phase, hsa_api_id_t api_id) {
    sibir_at_hsa_event(cb_data, phase, api_id);
}

std::vector<sibir::Instr> sibir_disassemble_kernel_object(uint64_t kernel_object) {
    return sibir::Disassembler::Instance().disassemble(kernel_object);
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

void *sibir_get_hip_function(const char *funcName) {
    return sibir::HipInterceptor::Instance().GetHipFunction(funcName);
}

void sibir_insert_call(sibir::Instr *instr, const char *dev_func_name, sibir_ipoint_t point) {

    auto agent = instr->getAgent();


    const char* codeObjectPtr;
    size_t codeObjectSize;

    std::string instrumentationFunc = sibir::CodeObjectManager::Instance().getCodeObjectOfInstrumentationFunction(dev_func_name, agent);

    sibir::CodeGenerator::instrument(*instr, instrumentationFunc, point);


//    // COMGR symbol iteration things
//    amd_comgr_data_t coData;
//    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &coData));
//    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(coData, codeObjectSize,
//                                             codeObjectPtr));
//
//    auto symbolIterator = [](amd_comgr_symbol_t s, void *data) {
//        uint64_t nameLength;
//        auto status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH, &nameLength));
//        if (status != AMD_COMGR_STATUS_SUCCESS)
//            return status;
//        std::string name;
//        name.resize(nameLength);
//        status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_NAME, name.data()));
//        if (status != AMD_COMGR_STATUS_SUCCESS)
//            return status;
//
//        std::cout << "AMD COMGR symbol name: " << name << std::endl;
//
//        amd_comgr_symbol_type_t type;
//        status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_TYPE, &type));
//        if (status != AMD_COMGR_STATUS_SUCCESS)
//            return status;
//
//        std::cout << "AMD COMGR symbol type: " << type << std::endl;
//
//        uint64_t size;
//        status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_SIZE, &size));
//        if (status != AMD_COMGR_STATUS_SUCCESS)
//            return status;
//
//        std::cout << "AMD COMGR symbol size: " << size << std::endl;
//
//        bool isDefined;
//        status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED, &isDefined));
//        if (status != AMD_COMGR_STATUS_SUCCESS)
//            return status;
//
//        std::cout << "AMD COMGR symbol is defined: " << size << std::endl;
//
//        uint64_t value;
//        status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_VALUE, &value));
//        if (status != AMD_COMGR_STATUS_SUCCESS)
//            return status;
//
//        std::cout << "AMD COMGR symbol value: " << reinterpret_cast<void*>(value) << std::endl;
//
//        return AMD_COMGR_STATUS_SUCCESS;
//    };
//
//    amd_comgr_iterate_symbols(coData, symbolIterator, nullptr);


//    auto coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
//    hsa_code_object_reader_t coReader;
//    hsa_executable_t executable;
//    SIBIR_HSA_CHECK(coreApi.hsa_code_object_reader_create_from_memory_fn(codeObjectPtr,
//                                                                         codeObjectSize, &coReader));
//
//    SIBIR_HSA_CHECK(coreApi.hsa_executable_create_alt_fn(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &executable));
//
//    SIBIR_HSA_CHECK(coreApi.hsa_executable_load_agent_code_object_fn(executable, agent, coReader, nullptr, nullptr));
//
//    SIBIR_HSA_CHECK(coreApi.hsa_executable_freeze_fn(executable, nullptr));
//    return executable;
}

void sibir_enable_instrumented(hsa_kernel_dispatch_packet_t* dispatch_packet, const sibir_address_t func, bool flag) {
    if (flag) {
        auto instrumentedKd = sibir::CodeObjectManager::Instance().getInstrumentedFunctionOfKD(func);
        dispatch_packet->kernel_object = reinterpret_cast<uint64_t>(instrumentedKd);
    }
    else
        dispatch_packet->kernel_object = reinterpret_cast<uint64_t>(func);
}

extern "C" {

ROCTRACER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 49;

ROCTRACER_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                             uint64_t failed_tool_count, const char *const *failed_tool_names) {
    [](auto &&...) {}(runtime_version, failed_tool_count, failed_tool_names);
    return sibir::HsaInterceptor::Instance().captureHsaApiTable(table);
}

ROCTRACER_EXPORT void OnUnload() {}
}