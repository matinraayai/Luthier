#include "instr.hpp"
#include "error.h"
#include "hsa_intercept.hpp"

luthier_address_t luthier::Instr::getHostAddress() {
//    if (kd_ != nullptr && hostAddress_ == luthier_address_t{}) {
//        LUTHIER_HSA_CHECK(
//            LuthierHsaInterceptor::Instance().getHsaVenAmdLoaderTable().
//            hsa_ven_amd_loader_query_host_address(
//                reinterpret_cast<const void*>(deviceAddress_),
//                reinterpret_cast<const void**>(&hostAddress_)
//            )
//        );
//    }
    return hostAddress_;
}
hsa_executable_t luthier::Instr::getExecutable() {
    return executable_;
}
const kernel_descriptor_t *luthier::Instr::getKernelDescriptor() {
    const kernel_descriptor_t *kernelDescriptor{nullptr};

    auto coreApi = HsaInterceptor::Instance().getSavedHsaTables().core;
    LUTHIER_HSA_CHECK(coreApi.hsa_executable_symbol_get_info_fn(executableSymbol_,
                                                              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                                              reinterpret_cast<luthier_address_t*>(&kernelDescriptor)));
    return kernelDescriptor;
}

luthier::Instr::Instr(std::string instStr, hsa_agent_t agent,
                    hsa_executable_t executable,
                    hsa_executable_symbol_t symbol,
                    luthier_address_t DeviceAccessibleInstrAddress,
                    size_t instrSize): executable_(executable), deviceAddress_(DeviceAccessibleInstrAddress),
                                       instStr_(std::move(instStr)), size_(instrSize),
                                       agent_(agent),
                                       executableSymbol_(symbol) {
    HsaInterceptor::Instance().getHsaVenAmdLoaderTable().hsa_ven_amd_loader_query_host_address(
                                                reinterpret_cast<const void*>(DeviceAccessibleInstrAddress),
                                                reinterpret_cast<const void**>(&hostAddress_)
                                                );
};
