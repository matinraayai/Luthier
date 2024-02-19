#include "hsa_executable_symbol.hpp"

#include <llvm/Object/ELFObjectFile.h>
#include <llvm/Support/ErrorHandling.h>

#include "hsa.hpp"
#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_loaded_code_object.hpp"
#include "object_utils.hpp"

hsa_symbol_kind_t luthier::hsa::ExecutableSymbol::getType() const {
    hsa_symbol_kind_t out;
    if (indirectFunctionName_.has_value() && indirectFunctionCode_.has_value()) out = HSA_SYMBOL_KIND_INDIRECT_FUNCTION;
    else {
        LUTHIER_HSA_CHECK(
            getApiTable().core.hsa_executable_symbol_get_info_fn(asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &out));
    }
    return out;
}
std::string luthier::hsa::ExecutableSymbol::getName() const {
    if (indirectFunctionName_.has_value()) return *indirectFunctionName_;
    else {
        uint32_t nameLength;
        LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_symbol_get_info_fn(
            asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameLength));
        std::string out(nameLength, '\0');
        LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_symbol_get_info_fn(
            asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_NAME, &out.front()));
        return out;
    }
}

hsa_symbol_linkage_t luthier::hsa::ExecutableSymbol::getLinkage() const { return HSA_SYMBOL_LINKAGE_MODULE; }

luthier_address_t luthier::hsa::ExecutableSymbol::getVariableAddress() const {
    luthier_address_t out;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_symbol_get_info_fn(
        asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &out));
    return out;
}
const luthier::hsa::KernelDescriptor *luthier::hsa::ExecutableSymbol::getKernelDescriptor() const {
    luthier_address_t kernelObject;
    LUTHIER_HSA_CHECK(getApiTable().core.hsa_executable_symbol_get_info_fn(
        this->asHsaType(), HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernelObject));
    return reinterpret_cast<const luthier::hsa::KernelDescriptor *>(kernelObject);
}
luthier::hsa::ExecutableSymbol luthier::hsa::ExecutableSymbol::fromKernelDescriptor(const hsa::KernelDescriptor *kd) {
    hsa_executable_t executable;
    const auto &loaderTable = HsaInterceptor::instance().getHsaVenAmdLoaderTable();

    // Check which executable this kernel object (address) belongs to
    LUTHIER_HSA_CHECK(loaderTable.hsa_ven_amd_loader_query_executable(reinterpret_cast<const void *>(kd), &executable));
    llvm::SmallVector<GpuAgent> agents;
    hsa::getGpuAgents(agents);

    for (const auto &a: agents) {
        for (const auto &s: hsa::Executable(executable).getSymbols(a)) {
            if (s.getKernelDescriptor() == kd) return s;
        }
    }
    llvm::report_fatal_error(llvm::formatv("Kernel descriptor {0:x} does not have a symbol associated with it.",
                                           reinterpret_cast<const void *>(kd)));
}
luthier::hsa::GpuAgent luthier::hsa::ExecutableSymbol::getAgent() const { return luthier::hsa::GpuAgent(agent_); }
luthier::hsa::Executable luthier::hsa::ExecutableSymbol::getExecutable() const {
    return luthier::hsa::Executable(executable_);
}

llvm::ArrayRef<uint8_t> luthier::hsa::ExecutableSymbol::getIndirectFunctionCode() const {
    return *indirectFunctionCode_;
}

llvm::ArrayRef<uint8_t> luthier::hsa::ExecutableSymbol::getKernelCode() const {
    auto loadedCodeObjects = Executable(executable_).getLoadedCodeObjects();
    auto kdSymbolName = getName();
    auto kernelSymbolName = kdSymbolName.substr(0, kdSymbolName.find(".kd"));
    for (const auto &lco: loadedCodeObjects) {
        auto hostElfOrError = getELFObjectFileBase(lco.getStorageMemory());
        LUTHIER_CHECK_WITH_MSG(hostElfOrError == true, "Failed to create an ELF");

        auto hostElf = hostElfOrError->get();

        //TODO: Replace this with a hash lookup
        auto Syms = hostElf->symbols();
        for (llvm::object::ELFSymbolRef elfSymbol: Syms) {
            auto nameOrError = elfSymbol.getName();
            LUTHIER_CHECK_WITH_MSG(nameOrError == true, "Failed to get the name of the symbol");
            if (nameOrError.get() == kernelSymbolName) {
                auto addressOrError = elfSymbol.getAddress();
                LUTHIER_CHECK_WITH_MSG(addressOrError == true, "Failed to get the address of the symbol");
                return {reinterpret_cast<const uint8_t *>(addressOrError.get() + lco.getLoadedMemory().data()),
                        elfSymbol.getSize()};
            }
        }
    }
    return {};
}
