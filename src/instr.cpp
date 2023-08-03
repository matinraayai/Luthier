#include "instr.hpp"
#include "error.h"
#include "hsa_intercept.hpp"

sibir_address_t sibir::Instr::getHostAddress() {
//    if (kd_ != nullptr && hostAddress_ == sibir_address_t{}) {
//        SIBIR_HSA_CHECK(
//            SibirHsaInterceptor::Instance().getHsaVenAmdLoaderTable().
//            hsa_ven_amd_loader_query_host_address(
//                reinterpret_cast<const void*>(deviceAddress_),
//                reinterpret_cast<const void**>(&hostAddress_)
//            )
//        );
//    }
    return hostAddress_;
}
hsa_executable_t sibir::Instr::getExecutable() {
    return executable_;
}
const kernel_descriptor_t *sibir::Instr::getKernelDescriptor() {
    const kernel_descriptor_t *kernelDescriptor{nullptr};

    auto coreApi = HsaInterceptor::Instance().getSavedHsaTables().core;
    SIBIR_HSA_CHECK(coreApi.hsa_executable_symbol_get_info_fn(executableSymbol_,
                                                              HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                                              reinterpret_cast<sibir_address_t*>(&kernelDescriptor)));
    return kernelDescriptor;
}

sibir::Instr::Instr(std::string instStr, hsa_agent_t  agent,
                    hsa_executable_t executable,
                    hsa_executable_symbol_t symbol,
                    sibir_address_t DeviceAccessibleInstrAddress,
                    size_t instrSize): executable_(executable), deviceAddress_(DeviceAccessibleInstrAddress),
                                       instStr_(std::move(instStr)), size_(instrSize),
                                       agent_(agent),
                                       executableSymbol_(symbol) {
    HsaInterceptor::Instance().getHsaVenAmdLoaderTable().hsa_ven_amd_loader_query_host_address(
                                                reinterpret_cast<const void*>(DeviceAccessibleInstrAddress),
                                                reinterpret_cast<const void**>(&hostAddress_)
                                                );
};
