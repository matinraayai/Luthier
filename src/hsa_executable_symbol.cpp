#include "hsa_executable_symbol.hpp"
#include "hsa_executable.hpp"
#include "context_manager.hpp"

hsa_symbol_kind_t luthier::hsa::ExecutableSymbol::getType() const {
    hsa_symbol_kind_t out;
    LUTHIER_HSA_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &out));
    return out;
}
std::string luthier::hsa::ExecutableSymbol::getName() const {
    uint32_t nameLength;
    LUTHIER_HSA_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameLength));
    std::string out;
    out.resize(nameLength);
    LUTHIER_HSA_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME, out.data()));
    return out;
}

hsa_symbol_linkage_t luthier::hsa::ExecutableSymbol::getLinkage() const {
    return HSA_SYMBOL_LINKAGE_MODULE;
}

luthier_address_t luthier::hsa::ExecutableSymbol::getVariableAddress() const {
    luthier_address_t out;
    LUTHIER_HSA_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &out));
    return out;
}
const kernel_descriptor_t *luthier::hsa::ExecutableSymbol::getKernelDescriptor() const {
    luthier_address_t kernelObject;
    LUTHIER_HSA_CHECK(
        getApiTable().core.hsa_executable_symbol_get_info_fn(this->asHsaType(),
                                                             HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                                             &kernelObject));
    return reinterpret_cast<const kernel_descriptor_t *>(kernelObject);
}
luthier::hsa::ExecutableSymbol luthier::hsa::ExecutableSymbol::fromKernelDescriptor(const kernel_descriptor_t *kd) {
    hsa_executable_t executable;
    const auto &loaderTable = HsaInterceptor::instance().getHsaVenAmdLoaderTable();

    // Check which executable this kernel object (address) belongs to
    LUTHIER_HSA_CHECK(loaderTable.hsa_ven_amd_loader_query_executable(reinterpret_cast<const void *>(kd),
                                                                      &executable));
    for (const auto& a: ContextManager::Instance().getHsaAgents()) {
        for (const auto& s: hsa::Executable(executable).getSymbols(a)) {
            if (s.getKernelDescriptor() == kd)
                return s;
        }
    }
}
luthier::hsa::GpuAgent luthier::hsa::ExecutableSymbol::getAgent() const {
    return luthier::hsa::GpuAgent(agent_);
}
luthier::hsa::Executable luthier::hsa::ExecutableSymbol::getExecutable() const {
    return luthier::hsa::Executable(executable_);
}
