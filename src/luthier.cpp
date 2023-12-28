#include "luthier.h"

#include <optional>

#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "error.h"
#include "hip_intercept.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_instr.hpp"
#include "hsa_intercept.hpp"
#include "log.hpp"

namespace luthier::impl {

void hipApiInternalCallback(void *cb_data, luthier_api_evt_phase_t phase, int api_id, bool *skip_func,
                            std::optional<std::any> *out) {
    LUTHIER_LOG_FUNCTION_CALL_START
    // Logic for intercepting the __hipRegister* functions
    static bool isHipRegistrationOver{
        false};//> indicates if the registration step of the HIP runtime is over. It is set when the
               // current API ID is not a registration function anymore
    static std::vector<std::tuple<const void *, const char *>>
        coManagerArgs;//> List of device functions to be managed by CodeObjectManager

    if (!isHipRegistrationOver && phase == LUTHIER_API_EVT_PHASE_ENTER) {
        if (api_id == HIP_PRIVATE_API_ID___hipRegisterFunction) {
            auto lastRFuncArgs = reinterpret_cast<hip___hipRegisterFunction_api_args_t *>(cb_data);
            // If the function doesn't have __luthier_wrap__ in its name then it belongs to the instrumented application or
            // HIP can manage on its own since no device function is present to strip from it
            if (std::string(lastRFuncArgs->deviceFunction).find(LUTHIER_DEVICE_FUNCTION_WRAP) != std::string::npos) {
                coManagerArgs.emplace_back(lastRFuncArgs->hostFunction, lastRFuncArgs->deviceFunction);
            }
        } else if (api_id != HIP_PRIVATE_API_ID___hipRegisterFatBinary
                   && api_id != HIP_PRIVATE_API_ID___hipRegisterManagedVar
                   && api_id != HIP_PRIVATE_API_ID___hipRegisterVar && api_id != HIP_PRIVATE_API_ID___hipRegisterSurface
                   && api_id != HIP_PRIVATE_API_ID___hipRegisterTexture) {
            auto &coManager = CodeObjectManager::instance();
            if (!coManagerArgs.empty()) {
                coManager.registerHipWrapperKernelsOfInstrumentationFunctions(coManagerArgs);
            }
            isHipRegistrationOver = true;
        }
    }
    LUTHIER_LOG_FUNCTION_CALL_END
}

void hipApiUserCallback(void *cb_data, luthier_api_evt_phase_t phase, int api_id) {
    ::luthier_at_hip_event(cb_data, phase, api_id);
}

void hsaApiUserCallback(hsa_api_evt_args_t *cb_data, luthier_api_evt_phase_t phase, hsa_api_evt_id_t api_id) {
    ::luthier_at_hsa_event(cb_data, phase, api_id);
}

void queueSubmitWriteInterceptor(const void *packets, uint64_t pktCount, uint64_t userPktIndex, void *data,
                                 hsa_amd_queue_intercept_packet_writer writer) {
    auto &hsaInterceptor = luthier::HsaInterceptor::instance();
    auto &hsaUserCallback = hsaInterceptor.getUserCallback();
    auto &hsaInternalCallback = hsaInterceptor.getInternalCallback();
    auto apiId = HSA_EVT_ID_hsa_queue_packet_submit;
    hsa_api_evt_args_t args;

    // Copy the packets to a non-const buffer
    std::vector<luthier_hsa_aql_packet_t> modifiedPackets(
        reinterpret_cast<const luthier_hsa_aql_packet_t *>(packets),
        reinterpret_cast<const luthier_hsa_aql_packet_t *>(packets) + pktCount);

    args.evt_args.hsa_queue_packet_submit.packets = modifiedPackets.data();
    args.evt_args.hsa_queue_packet_submit.pkt_count = pktCount;
    args.evt_args.hsa_queue_packet_submit.user_pkt_index = userPktIndex;

    hsaUserCallback(&args, LUTHIER_API_EVT_PHASE_ENTER, apiId);
    hsaInternalCallback(&args, LUTHIER_API_EVT_PHASE_ENTER, apiId, nullptr);

    // Write the packets to hardware queue
    // Even if the packets are not modified, this call has to be made to ensure the packets are copied to the hardware queue
    writer(modifiedPackets.data(), pktCount);
}

void hsaApiInternalCallback(hsa_api_evt_args_t *cb_data, luthier_api_evt_phase_t phase, hsa_api_evt_id_t api_id,
                            bool *skipFunction) {
    if (phase == LUTHIER_API_EVT_PHASE_ENTER) {
        if (api_id == HSA_API_ID_hsa_queue_create) {
            auto args = cb_data->api_args.hsa_queue_create;
            LUTHIER_HSA_CHECK(luthier_get_hsa_table()->amd_ext_->hsa_amd_queue_intercept_create_fn(
                args.agent, args.size, args.type, args.callback, args.data, args.private_segment_size,
                args.group_segment_size, args.queue));
            LUTHIER_HSA_CHECK(
                luthier_get_hsa_table()->amd_ext_->hsa_amd_profiling_set_profiler_enabled_fn(*args.queue, true));
            LUTHIER_HSA_CHECK(luthier_get_hsa_table()->amd_ext_->hsa_amd_queue_intercept_register_fn(
                *args.queue, queueSubmitWriteInterceptor, *args.queue));
            *skipFunction = true;
        }
    }
}

__attribute__((constructor)) void init() {
    LUTHIER_LOG_FUNCTION_CALL_START
    assert(HipInterceptor::Instance().IsEnabled());
    luthier_at_init();
    HipInterceptor::Instance().SetInternalCallback(luthier::impl::hipApiInternalCallback);
    HipInterceptor::Instance().SetUserCallback(luthier::impl::hipApiUserCallback);
    HsaInterceptor::instance().setInternalCallback(luthier::impl::hsaApiInternalCallback);
    HsaInterceptor::instance().setUserCallback(luthier::impl::hsaApiUserCallback);
    LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((destructor)) void finalize() {
    LUTHIER_LOG_FUNCTION_CALL_START
    luthier_at_term();
    LUTHIER_LOG_FUNCTION_CALL_END
}

}// namespace luthier::impl

const HsaApiTable *luthier_get_hsa_table() { return &luthier::HsaInterceptor::instance().getSavedHsaTables().root; }

const hsa_ven_amd_loader_1_03_pfn_s *luthier_get_hsa_ven_amd_loader() {
    return &luthier::HsaInterceptor::instance().getHsaVenAmdLoaderTable();
}

luthier_instruction_t *luthier_disassemble_kernel_object(uint64_t kernel_object, void *(*alloc_callback)(size_t size),
                                                         size_t *size) {
    auto kdSymbol = luthier::hsa::ExecutableSymbol::fromKernelDescriptor(
        reinterpret_cast<const luthier::hsa::KernelDescriptor *>(kernel_object));
    auto instructions = luthier::Disassembler::instance().disassemble(kdSymbol);
    *size = instructions.size();
    auto *out =
        reinterpret_cast<luthier_instruction_t *>(alloc_callback(instructions.size() * sizeof(luthier_instruction_t)));
    assert(out);
    for (int i = 0; i < *size; i++) {
        out[i] = luthier::hsa::Instr::toHandle(instructions[i]);
    }
    return out;
}

void luthier_instructions_handles_destroy(luthier_instruction_t* instrs, size_t size) {
    for (int i = 0; i < size; i++) {
        auto inst = luthier::hsa::Instr::fromHandle(instrs[i]);
        luthier::Disassembler::instance().destroyInstr(inst);
    }
}

void *luthier_get_hip_function(const char *funcName) {
    return luthier::HipInterceptor::Instance().GetHipFunction(funcName);
}

void luthier_insert_call(luthier_instruction_t instr, const void *dev_func, luthier_ipoint_t point) {
    luthier::CodeGenerator::instance().instrument(*luthier::hsa::Instr::fromHandle(instr), dev_func, point);
}

void luthier_override_with_instrumented(hsa_kernel_dispatch_packet_t *dispatch_packet) {
    const auto instrumentedKernel = luthier::CodeObjectManager::instance().getInstrumentedKernel(
        luthier::hsa::ExecutableSymbol::fromKernelDescriptor(
            reinterpret_cast<const luthier::hsa::KernelDescriptor *>(dispatch_packet->kernel_object)));
    dispatch_packet->kernel_object = reinterpret_cast<uint64_t>(instrumentedKernel.getKernelDescriptor());
    fmt::println("Kernel Object address: {:#x}", dispatch_packet->kernel_object);
}

extern "C" {

__attribute__((visibility("default"))) extern const uint32_t HSA_AMD_TOOL_PRIORITY = 49;

__attribute__((visibility("default"))) bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                                   uint64_t failed_tool_count, const char *const *failed_tool_names) {
    [](auto &&...) {}(runtime_version, failed_tool_count, failed_tool_names);
    return luthier::HsaInterceptor::instance().captureHsaApiTable(table);
}

__attribute__((visibility("default"))) void OnUnload() {}
}