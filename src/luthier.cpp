#include "luthier.h"
#include "code_generator.hpp"
#include "code_object_manager.hpp"
#include "code_object_manipulation.hpp"
#include "disassembler.hpp"
#include "error.h"
#include "hip_intercept.hpp"
#include "hsa_intercept.hpp"
#include "log.hpp"
#include "luthier_types.hpp"
#include <optional>
#include <roctracer/roctracer.h>

namespace luthier::impl {

void resetHipRegistrationArgs(std::optional<hip___hipRegisterFatBinary_api_args_t> &rFatBinArgs,
                              std::vector<hip___hipRegisterFunction_api_args_t> &rFuncArgs,
                              std::vector<hip___hipRegisterManagedVar_api_args_t> &rManagedVarArgs,
                              std::vector<hip___hipRegisterSurface_api_args_t> &rSurfaceArgs,
                              std::vector<hip___hipRegisterTexture_api_args_t> &rTextureArgs,
                              std::vector<hip___hipRegisterVar_api_args_t> &rVarArgs) {
    rFatBinArgs = std::nullopt;
    rFuncArgs.clear();
    rManagedVarArgs.clear();
    rSurfaceArgs.clear();
    rTextureArgs.clear();
    rVarArgs.clear();
}

void hijackedHipRegister(std::optional<hip___hipRegisterFatBinary_api_args_t> &rFatBinArgs,
                         std::vector<hip___hipRegisterFunction_api_args_t> &rFuncArgs,
                         std::vector<hip___hipRegisterManagedVar_api_args_t> &rManagedVarArgs,
                         std::vector<hip___hipRegisterSurface_api_args_t> &rSurfaceArgs,
                         std::vector<hip___hipRegisterTexture_api_args_t> &rTextureArgs,
                         std::vector<hip___hipRegisterVar_api_args_t> &rVarArgs) {
    const auto &hipInterceptor = HipInterceptor::Instance();
    auto rFatBinFunc = hipInterceptor.GetHipFunction<hip::FatBinaryInfo **(*) (const void *)>("__hipRegisterFatBinary");
    if (rFatBinArgs.has_value()) {
        auto fatBinaryInfo = rFatBinFunc(rFatBinArgs->data);
        if (!rManagedVarArgs.empty()) {
            auto rManagedVarFunc = hipInterceptor.GetHipFunction<void (*)(void *, void **, void *,
                                                                          const char *, size_t, unsigned)>("__hipRegisterManagedVar");
            for (const auto &arg: rManagedVarArgs) {
                rManagedVarFunc(reinterpret_cast<void *>(fatBinaryInfo), arg.pointer, arg.init_value,
                                arg.name, arg.size, arg.align);
            }
        }
        if (!rSurfaceArgs.empty()) {
            auto rSurfaceFunc = hipInterceptor.GetHipFunction<void (*)(hip::FatBinaryInfo **, void *, char *,
                                                                       char *, int, int)>("__hipRegisterSurface");
            for (const auto &arg: rSurfaceArgs) {
                rSurfaceFunc(fatBinaryInfo, arg.var, arg.hostVar, arg.deviceVar,
                             arg.type, arg.ext);
            }
        }
        if (!rVarArgs.empty()) {
            auto rVarFunc = hipInterceptor.GetHipFunction<void (*)(hip::FatBinaryInfo **modules,
                                                                   void *var,
                                                                   char *hostVar,
                                                                   char *deviceVar,
                                                                   int ext,
                                                                   size_t size,
                                                                   int constant,
                                                                   int global)>("__hipRegisterVar");
            for (const auto &arg: rVarArgs) {
                rVarFunc(fatBinaryInfo, arg.var, arg.hostVar, arg.deviceVar, arg.ext, arg.size,
                         arg.constant, arg.global);
            }
        }
    }
    if (!rFuncArgs.empty()) {
        // Force the HIP runtime to load the static HIP Modules containing the Luthier instrumentation code objects into HSA executables
        // and freeze them by launching a nullptr hipFunction_t
        // This has to be done once per HIP device
        // For HIP to ignore lazy module loading, the tool's HIP FAT binary should have at least one static managed variable
        // This will not trigger the launch of any kernels and should return quickly after checking that the hipFunction_t is nullptr
        auto hipLaunchKernelFunc = hipInterceptor.GetHipFunction<hipError_t (*)(hipFunction_t f,
                                                                                uint32_t gridDimX, uint32_t gridDimY,
                                                                                uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
                                                                                uint32_t blockDimZ, uint32_t sharedMemBytes, hipStream_t hStream,
                                                                                void **kernelParams, void **extra)>("hipModuleLaunchKernel");
        auto hipGetDeviceCountFunc = hipInterceptor.GetHipFunction<hipError_t (*)(int *count)>("hipGetDeviceCount");
        auto hipSetDeviceFunc = hipInterceptor.GetHipFunction<hipError_t (*)(int deviceId)>("hipSetDevice");
        int deviceCount;
        LUTHIER_HIP_CHECK(hipGetDeviceCountFunc(&deviceCount));
        for (int i = 0; i < deviceCount; i++) {
            LUTHIER_HIP_CHECK(hipSetDeviceFunc(i));
            auto status = hipLaunchKernelFunc(nullptr, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr, nullptr);
            // Check that the launch returned ONLY because the hipFunction_t was a nullptr, nothing else
            assert(status == hipErrorInvalidResourceHandle || status == hipErrorInvalidImage);
        }

        auto &coManager = CodeObjectManager::instance();
        std::vector<std::tuple<const void *, const char *>> coManagerArgs;
        coManagerArgs.reserve(rFuncArgs.size());
        for (const auto &arg: rFuncArgs) {
            coManagerArgs.emplace_back(arg.hostFunction, arg.deviceFunction);
        }
        coManager.registerHipWrapperKernelsOfInstrumentationFunctions(coManagerArgs);
    }

    resetHipRegistrationArgs(rFatBinArgs,
                             rFuncArgs,
                             rManagedVarArgs,
                             rSurfaceArgs,
                             rTextureArgs,
                             rVarArgs);
}

void normalHipRegister(std::optional<hip___hipRegisterFatBinary_api_args_t> &rFatBinArgs,
                       std::vector<hip___hipRegisterFunction_api_args_t> &rFuncArgs,
                       std::vector<hip___hipRegisterManagedVar_api_args_t> &rManagedVarArgs,
                       std::vector<hip___hipRegisterSurface_api_args_t> &rSurfaceArgs,
                       std::vector<hip___hipRegisterTexture_api_args_t> &rTextureArgs,
                       std::vector<hip___hipRegisterVar_api_args_t> &rVarArgs) {
    const auto &hipInterceptor = HipInterceptor::Instance();
    auto rFatBinFunc = hipInterceptor.GetHipFunction<hip::FatBinaryInfo **(*) (const void *)>("__hipRegisterFatBinary");
    if (rFatBinArgs.has_value()) {
        auto fatBinaryInfo = rFatBinFunc(rFatBinArgs->data);
        if (!rFuncArgs.empty()) {
            auto rMethodFunc = hipInterceptor.GetHipFunction<void (*)(hip::FatBinaryInfo **,
                                                                      const void *,
                                                                      char *,
                                                                      const char *,
                                                                      unsigned int,
                                                                      uint3 *, uint3 *, dim3 *,
                                                                      dim3 *, int *)>("__hipRegisterFunction");
            for (const auto &args: rFuncArgs) {
                rMethodFunc(fatBinaryInfo, args.hostFunction, args.deviceFunction, args.deviceName,
                            args.threadLimit, args.tid, args.bid, args.blockDim,
                            args.gridDim, args.wSize);
            }
        }
        if (!rManagedVarArgs.empty()) {
            auto rManagedVarFunc = hipInterceptor.GetHipFunction<void (*)(void *, void **, void *,
                                                                          const char *, size_t, unsigned)>("__hipRegisterManagedVar");
            for (const auto &arg: rManagedVarArgs) {
                rManagedVarFunc(reinterpret_cast<void *>(fatBinaryInfo), arg.pointer, arg.init_value,
                                arg.name, arg.size, arg.align);
            }
        }
        if (!rSurfaceArgs.empty()) {
            auto rSurfaceFunc = hipInterceptor.GetHipFunction<void (*)(hip::FatBinaryInfo **, void *, char *,
                                                                       char *, int, int)>("__hipRegisterSurface");
            for (const auto &arg: rSurfaceArgs) {
                rSurfaceFunc(fatBinaryInfo, arg.var, arg.hostVar, arg.deviceVar,
                             arg.type, arg.ext);
            }
        }
        if (!rVarArgs.empty()) {
            auto rVarFunc = hipInterceptor.GetHipFunction<void (*)(hip::FatBinaryInfo **modules,
                                                                   void *var,
                                                                   char *hostVar,
                                                                   char *deviceVar,
                                                                   int ext,
                                                                   size_t size,
                                                                   int constant,
                                                                   int global)>("__hipRegisterVar");
            for (const auto &arg: rVarArgs) {
                rVarFunc(fatBinaryInfo, arg.var, arg.hostVar, arg.deviceVar, arg.ext, arg.size,
                         arg.constant, arg.global);
            }
        }
    }
    resetHipRegistrationArgs(rFatBinArgs,
                             rFuncArgs,
                             rManagedVarArgs,
                             rSurfaceArgs,
                             rTextureArgs,
                             rVarArgs);
}

void hipApiInternalCallback(void *cb_data, luthier_api_phase_t phase, int api_id, bool *skip_func) {
    LUTHIER_LOG_FUNCTION_CALL_START
    // Logic for intercepting the __hipRegister* functions
    static bool isHipRegistrationOver{false};           //> indicates if the registration step of the HIP runtime is over. It is set when the
                                                        // current API ID is not a registration function anymore
    static bool hijackRegistrationIterForLuthier{false};//> indicates if the past/current arguments to the registration functions
                                                        // contain luthier device code (functions wrapped in __luthier_wrap__).
                                                        // Such scenarios require removal of the device code from the Fat binary
                                                        // and loading the modified binary to the HIP runtime as a dynamic module.
                                                        // This ensures that instrumentation functions can have a single copy of them
                                                        // and have them as a call argument
    static std::optional<hip___hipRegisterFatBinary_api_args_t>
        lastRFatBinArgs{std::nullopt};//> if __hipRegisterFatBinary was called, holds the arguments
                                      // of the call; Otherwise holds std::nullopt
    static std::vector<hip___hipRegisterFunction_api_args_t>
        lastRFuncArgs{};//> list of functions to register for the last saved Fat Binary
    static std::vector<hip___hipRegisterManagedVar_api_args_t>
        lastRManagedVarArgs{};//> list of managed variables to register for the last saved Fat Binary
    static std::vector<hip___hipRegisterSurface_api_args_t>
        lastRSurfaceArgs{};//> list of surface variables to register for the last saved Fat Binary
    static std::vector<hip___hipRegisterTexture_api_args_t>
        lastRTextureArgs{};//> list of texture variables to register for the last saved Fat Binary
    static std::vector<hip___hipRegisterVar_api_args_t>
        lastRVarArgs{};//> list of global variables to register for the last Fat saved Binary

    if (!isHipRegistrationOver && phase == LUTHIER_API_PHASE_ENTER) {
        if (api_id == HIP_PRIVATE_API_ID___hipRegisterFatBinary) {
            if (hijackRegistrationIterForLuthier) {
                hijackedHipRegister(lastRFatBinArgs, lastRFuncArgs, lastRManagedVarArgs,
                                    lastRSurfaceArgs, lastRTextureArgs, lastRVarArgs);
                hijackRegistrationIterForLuthier = false;
            } else {
                normalHipRegister(lastRFatBinArgs, lastRFuncArgs, lastRManagedVarArgs,
                                  lastRSurfaceArgs, lastRTextureArgs, lastRVarArgs);
            }
            lastRFatBinArgs = *reinterpret_cast<hip___hipRegisterFatBinary_api_args_t *>(cb_data);
            *skip_func = true;
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterFunction) {
            lastRFuncArgs.push_back(*reinterpret_cast<hip___hipRegisterFunction_api_args_t *>(cb_data));
            *skip_func = true;
            // If the function doesn't have __luthier_wrap__ in its name then it belongs to the instrumented application or
            // HIP can manage on its own since no device function is present to strip from it
            if (std::string(lastRFuncArgs.front().deviceFunction).find("__luthier_wrap__") != std::string::npos)
                hijackRegistrationIterForLuthier = true;
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterManagedVar) {
            lastRManagedVarArgs.push_back(*reinterpret_cast<hip___hipRegisterManagedVar_api_args_t *>(cb_data));
            *skip_func = true;
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterSurface) {
            lastRSurfaceArgs.push_back(*reinterpret_cast<hip___hipRegisterSurface_api_args_t *>(cb_data));
            *skip_func = true;
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterTexture) {
            lastRTextureArgs.push_back(*reinterpret_cast<hip___hipRegisterTexture_api_args_t *>(cb_data));
            *skip_func = true;
        } else if (api_id == HIP_PRIVATE_API_ID___hipRegisterVar) {
            lastRVarArgs.push_back(*reinterpret_cast<hip___hipRegisterVar_api_args_t *>(cb_data));
            *skip_func = true;
        } else {
            if (hijackRegistrationIterForLuthier) {
                hijackedHipRegister(lastRFatBinArgs, lastRFuncArgs, lastRManagedVarArgs,
                                    lastRSurfaceArgs, lastRTextureArgs, lastRVarArgs);
                hijackRegistrationIterForLuthier = false;
            } else {
                normalHipRegister(lastRFatBinArgs, lastRFuncArgs, lastRManagedVarArgs,
                                  lastRSurfaceArgs, lastRTextureArgs, lastRVarArgs);
            }
            isHipRegistrationOver = true;
        }
    }
    LUTHIER_LOG_FUNCTION_CALL_END
}

void hipApiUserCallback(void *cb_data, luthier_api_phase_t phase, int api_id) {
    ::luthier_at_hip_event(cb_data, phase, api_id);
}

void hsaApiCallback(hsa_api_args_t *cb_data, luthier_api_phase_t phase, hsa_api_id_t api_id) {
    ::luthier_at_hsa_event(cb_data, phase, api_id);
}

__attribute__((constructor)) void init() {
    LUTHIER_LOG_FUNCTION_CALL_START
    assert(HipInterceptor::Instance().IsEnabled());
    luthier_at_init();
    HipInterceptor::Instance().SetInternalCallback(luthier::impl::hipApiInternalCallback);
    HipInterceptor::Instance().SetUserCallback(luthier::impl::hipApiUserCallback);
    HsaInterceptor::Instance().SetCallback(luthier::impl::hsaApiCallback);
    LUTHIER_LOG_FUNCTION_CALL_END
}

__attribute__((destructor)) void finalize() {
    LUTHIER_LOG_FUNCTION_CALL_START
    luthier_at_term();
    LUTHIER_LOG_FUNCTION_CALL_END
}

}// namespace luthier::impl

const HsaApiTable *luthier_get_hsa_table() {
    return &luthier::HsaInterceptor::Instance().getSavedHsaTables().root;
}

const hsa_ven_amd_loader_1_03_pfn_s *luthier_get_hsa_ven_amd_loader() {
    return &luthier::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
}

std::vector<luthier::Instr> luthier_disassemble_kernel_object(uint64_t kernel_object) {
    return luthier::Disassembler::instance().disassemble(kernel_object);
}

void *luthier_get_hip_function(const char *funcName) {
    return luthier::HipInterceptor::Instance().GetHipFunction(funcName);
}

void luthier_insert_call(luthier::Instr *instr, const void *dev_func, luthier_ipoint_t point) {
    luthier::CodeGenerator::instance().instrument(*instr, dev_func, point);
}

void luthier_insert_call(luthier::Instr *instr, void *my_addr) {
    luthier::CodeGenerator::instance().modify(*instr, my_addr);
}

void luthier_override_with_instrumented(hsa_kernel_dispatch_packet_t *dispatch_packet) {
    const auto instrumentedKd = luthier::CodeObjectManager::instance().getInstrumentedKernelKD(
        reinterpret_cast<const kernel_descriptor_t *>(dispatch_packet->kernel_object));
    dispatch_packet->kernel_object = reinterpret_cast<uint64_t>(instrumentedKd);
    fmt::println("Kernel Object address: {:#x}", dispatch_packet->kernel_object);
}

extern "C" {

ROCTRACER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 49;

ROCTRACER_EXPORT bool OnLoad(HsaApiTable *table, uint64_t runtime_version,
                             uint64_t failed_tool_count, const char *const *failed_tool_names) {
    [](auto &&...) {}(runtime_version, failed_tool_count, failed_tool_names);
    return luthier::HsaInterceptor::Instance().captureHsaApiTable(table);
}

ROCTRACER_EXPORT void OnUnload() {}
}