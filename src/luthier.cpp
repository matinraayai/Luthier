#include "luthier.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FormatVariadic.h>

#include <optional>

#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "disassembler.hpp"
#include "error.hpp"
#include "hip_intercept.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_instr.hpp"
#include "hsa_intercept.hpp"
#include "log.hpp"

namespace luthier::impl {

void hipApiInternalCallback(void *cbData, luthier_api_evt_phase_t phase, int apiId, bool *skipFunc,
                            std::optional<std::any> *out) {
    LUTHIER_LOG_FUNCTION_CALL_START
    if (phase == LUTHIER_API_EVT_PHASE_ENTER) {
        if (apiId == HIP_PRIVATE_API_ID___hipRegisterFunction) {
            auto &coManager = CodeObjectManager::instance();
            auto lastRFuncArgs = reinterpret_cast<hip___hipRegisterFunction_api_args_t *>(cbData);
            // If the function doesn't have __luthier_wrap__ in its name then it belongs to the instrumented application
            // or HIP can manage it on its own since no device function is present to strip from it
            if (llvm::StringRef(lastRFuncArgs->deviceFunction).find(LUTHIER_DEVICE_FUNCTION_WRAP)
                != llvm::StringRef::npos) {
                coManager.registerInstrumentationFunctionWrapper(lastRFuncArgs->hostFunction,
                                                                 lastRFuncArgs->deviceFunction);
            }
        }
    }
    LUTHIER_LOG_FUNCTION_CALL_END
}

void hipApiUserCallback(void *cbData, luthier_api_evt_phase_t phase, int apiId) {
    ::luthier_at_hip_event(cbData, phase, apiId);
}

void hsaApiUserCallback(hsa_api_evt_args_t *cbData, luthier_api_evt_phase_t phase, hsa_api_evt_id_t apiId) {
    ::luthier_at_hsa_event(cbData, phase, apiId);
}

void queueSubmitWriteInterceptor(const void *packets, uint64_t pktCount, uint64_t userPktIndex, void *data,
                                 hsa_amd_queue_intercept_packet_writer writer) {
    auto &hsaInterceptor = luthier::HsaInterceptor::instance();
    auto &hsaUserCallback = hsaInterceptor.getUserCallback();
    auto &hsaInternalCallback = hsaInterceptor.getInternalCallback();
    auto apiId = HSA_EVT_ID_hsa_queue_packet_submit;
    hsa_api_evt_args_t args;
    bool isUserCallbackEnabled = hsaInterceptor.isUserCallbackEnabled(apiId);
    bool isInternalCallbackEnabled = hsaInterceptor.isInternalCallbackEnabled(apiId);
    if (isUserCallbackEnabled || isInternalCallbackEnabled) {
        // Copy the packets to a non-const buffer
        std::vector<luthier_hsa_aql_packet_t> modifiedPackets(
            reinterpret_cast<const luthier_hsa_aql_packet_t *>(packets),
            reinterpret_cast<const luthier_hsa_aql_packet_t *>(packets) + pktCount);

        args.evt_args.hsa_queue_packet_submit.packets = modifiedPackets.data();
        args.evt_args.hsa_queue_packet_submit.pkt_count = pktCount;
        args.evt_args.hsa_queue_packet_submit.user_pkt_index = userPktIndex;
        if (isUserCallbackEnabled) hsaUserCallback(&args, LUTHIER_API_EVT_PHASE_ENTER, apiId);
        if (isInternalCallbackEnabled) hsaInternalCallback(&args, LUTHIER_API_EVT_PHASE_ENTER, apiId, nullptr);

        // Write the packets to hardware queue
        // Even if the packets are not modified, this call has to be made to ensure the packets are copied to
        // the hardware queue
        writer(modifiedPackets.data(), pktCount);
    } else {
        writer(packets, pktCount);
    }
}

void hsaApiInternalCallback(hsa_api_evt_args_t *cbData, luthier_api_evt_phase_t phase, hsa_api_evt_id_t apiId,
                            bool *skipFunction) {
    LUTHIER_LOG_FUNCTION_CALL_START
    if (phase == LUTHIER_API_EVT_PHASE_ENTER) {
        if (apiId == HSA_API_ID_hsa_queue_create) {
            auto args = cbData->api_args.hsa_queue_create;
            LUTHIER_HSA_CHECK(luthier_get_hsa_table()->amd_ext_->hsa_amd_queue_intercept_create_fn(
                args.agent, args.size, args.type, args.callback, args.data, args.private_segment_size,
                args.group_segment_size, args.queue));
            LUTHIER_HSA_CHECK(luthier_get_hsa_table()->amd_ext_->hsa_amd_queue_intercept_register_fn(
                *args.queue, queueSubmitWriteInterceptor, *args.queue));
            *skipFunction = true;
        }
    }
    LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((constructor)) void init() {
    LUTHIER_LOG_FUNCTION_CALL_START
    auto &hipInterceptor = HipInterceptor::instance();
    LUTHIER_CHECK_WITH_MSG(hipInterceptor.isEnabled(), "HIP Interceptor failed to initialize");
    hipInterceptor.setInternalCallback(luthier::impl::hipApiInternalCallback);
    hipInterceptor.setUserCallback(luthier::impl::hipApiUserCallback);
    hipInterceptor.enableInternalCallback(HIP_PRIVATE_API_ID___hipRegisterFunction);
    LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((destructor)) void finalize() {
    LUTHIER_LOG_FUNCTION_CALL_START
    luthier_at_term();
    LUTHIER_LOG_FUNCTION_CALL_END
}

}// namespace luthier::impl

extern "C" {
const HsaApiTable *luthier_get_hsa_table() { return &luthier::HsaInterceptor::instance().getSavedHsaTables().root; }

const hsa_ven_amd_loader_1_03_pfn_s *luthier_get_hsa_ven_amd_loader() {
    return &luthier::HsaInterceptor::instance().getHsaVenAmdLoaderTable();
}

void luthier_disassemble_kernel_object(uint64_t kernel_object, size_t *size, luthier_instruction_t *instructions) {
    auto kdSymbol = luthier::hsa::ExecutableSymbol::fromKernelDescriptor(
        reinterpret_cast<const luthier::hsa::KernelDescriptor *>(kernel_object));
    auto hsaInstructions = luthier::Disassembler::instance().disassemble(kdSymbol);
    if (instructions == nullptr) {
        *size = hsaInstructions->size();
    } else {
        for (unsigned int i = 0; i < *size; i++) {
            instructions[i] = luthier::hsa::Instr::toHandle(&(*hsaInstructions)[i]);
        }
    }
}

void *luthier_get_hip_function(const char *funcName) {
    return luthier::HipInterceptor::instance().getHipFunction(funcName);
}

void luthier_insert_call(luthier_instruction_t instr, const void *dev_func, luthier_ipoint_t point) {
    luthier::CodeGenerator::instance().instrument(*luthier::hsa::Instr::fromHandle(instr), dev_func, point);
}

hsa_packet_type_t luthier_get_packet_type(luthier_hsa_aql_packet_t *aql_packet) {
    return static_cast<hsa_packet_type_t>((aql_packet->packet.header >> HSA_PACKET_HEADER_TYPE)
                                          & ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1));
}

void luthier_enable_hsa_op_callback(hsa_api_evt_id_t op) { luthier::HsaInterceptor::instance().enableUserCallback(op); }

void luthier_disable_hsa_op_callback(hsa_api_evt_id_t op) {
    luthier::HsaInterceptor::instance().disableUserCallback(op);
}

void luthier_enable_all_hsa_callbacks() { luthier::HsaInterceptor::instance().enableAllUserCallbacks(); }

void luthier_disable_all_hsa_callbacks() { luthier::HsaInterceptor::instance().disableAllUserCallbacks(); }

void luthier_enable_hip_op_callback(uint32_t op) { luthier::HipInterceptor::instance().enableUserCallback(op); }

void luthier_disable_hip_op_callback(uint32_t op) { luthier::HipInterceptor::instance().disableUserCallback(op); }

void luthier_enable_all_hip_callbacks() { luthier::HipInterceptor::instance().enableAllUserCallbacks(); }

void luthier_diable_all_hip_callbacks() { luthier::HipInterceptor::instance().disableAllUserCallbacks(); }

void luthier_override_with_instrumented(hsa_kernel_dispatch_packet_t *dispatch_packet) {
    const auto instrumentedKernel = luthier::CodeObjectManager::instance().getInstrumentedKernel(
        luthier::hsa::ExecutableSymbol::fromKernelDescriptor(
            reinterpret_cast<const luthier::hsa::KernelDescriptor *>(dispatch_packet->kernel_object)));
    dispatch_packet->kernel_object = reinterpret_cast<uint64_t>(instrumentedKernel.getKernelDescriptor());
    llvm::outs() << llvm::formatv("Kernel Object Address: {0:x}\n", dispatch_packet->kernel_object);
}

// NOLINTBEGIN

__attribute__((visibility("default"))) extern const uint32_t HSA_AMD_TOOL_PRIORITY = 49;

__attribute__((visibility("default"))) bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                                                   uint64_t failed_tool_count, const char *const *failed_tool_names) {
    LUTHIER_LOG_FUNCTION_CALL_START
    [](auto &&...) {}(runtime_version, failed_tool_count, failed_tool_names);
    bool res = luthier::HsaInterceptor::instance().captureHsaApiTable(table);
    luthier_at_init();
    auto &hsaInterceptor = luthier::HsaInterceptor::instance();
    hsaInterceptor.setInternalCallback(luthier::impl::hsaApiInternalCallback);
    hsaInterceptor.setUserCallback(luthier::impl::hsaApiUserCallback);
    hsaInterceptor.enableInternalCallback(HSA_API_ID_hsa_queue_create);
    return res;
    LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((visibility("default"))) void OnUnload() {
    LUTHIER_LOG_FUNCTION_CALL_START
    luthier::HsaInterceptor::instance().uninstallApiTables();
    LUTHIER_LOG_FUNCTION_CALL_END
}
}
// NOLINTEND