#include "code_object_manager.hpp"
#include "amdgpu_elf.hpp"
#include "context_manager.hpp"
#include "disassembler.hpp"
#include "hsa_intercept.hpp"
#include <assert.h>
#include <elfio/elfio.hpp>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>

namespace {
struct __CudaFatBinaryWrapper {
    unsigned int magic;
    unsigned int version;
    void *binary;
    void *dummy1;
};

constexpr unsigned __hipFatMAGIC2 = 0x48495046;// "HIPF"


std::string getDemangledName(const char *mangledName) {
    amd_comgr_data_t mangledNameData;
    amd_comgr_data_t demangledNameData;
    std::string out;

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_BYTES, &mangledNameData));

    size_t size = strlen(mangledName);
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(mangledNameData, size, mangledName));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_demangle_symbol_name(mangledNameData, &demangledNameData));

    size_t demangledNameSize = 0;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data(demangledNameData, &demangledNameSize, nullptr));

    out.resize(demangledNameSize);

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data(demangledNameData, &demangledNameSize, out.data()));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_release_data(mangledNameData));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_release_data(demangledNameData));

    return out;
}


std::string stripOffKernelLaunch(const std::string& elf, const std::string& demangledName) {

    std::istringstream ss{elf};
    ELFIO::elfio elfio;
    elfio.load(ss);
    auto numSymbols = luthier::elf::getSymbolNum(elfio);
    for (unsigned int i = 0; i < numSymbols; i++) {
        luthier::elf::SymbolInfo info;
        luthier::elf::getSymbolInfo(elfio, i, info);
        std::string demangledSymName = getDemangledName(info.sym_name.c_str());
        std::cout << "Symbol name: " << info.sym_name.c_str() << std::endl;
        std::cout << "Symbol size: " << info.size << std::endl;
        std::cout << "Symbol Addr: " << reinterpret_cast<const void*>(info.address) << std::endl;
    }
    for (unsigned int i = 0; i < numSymbols; i++) {
        luthier::elf::SymbolInfo info;
        luthier::elf::getSymbolInfo(elfio, i, info);
        std::string demangledSymName = getDemangledName(info.sym_name.c_str());
        std::cout << "Symbol name: " << info.sym_name.c_str() << std::endl;
        std::cout << "Symbol size: " << info.size << std::endl;
        std::cout << "Symbol Addr: " << reinterpret_cast<const void*>(info.address) << std::endl;
        if (demangledSymName.find("__luthier_wrap__") == std::string::npos and demangledSymName.find(demangledName) != std::string::npos) {
            std::cout << "Symbol name: " << getDemangledName(info.sym_name.c_str()) << std::endl;
            std::cout << "Symbol size: " << info.size << std::endl;
            std::cout << "Symbol Addr: " << reinterpret_cast<const void*>(info.address) << std::endl;
            return {info.address, info.size};
        }

    }

    return {};
}
}// namespace

void luthier::CodeObjectManager::registerFatBinary(const void *data) {
    assert(data != nullptr);
    auto fbWrapper = reinterpret_cast<const __CudaFatBinaryWrapper *>(data);
    assert(fbWrapper->magic == __hipFatMAGIC2 && fbWrapper->version == 1);
    auto fatBinary = fbWrapper->binary;
    if (!fatBinaries_.contains(fatBinary)) {
        amd_comgr_data_t fbData;
        LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &fbData));
        LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(fbData, 4096, reinterpret_cast<const char *>(fatBinary)));
        fatBinaries_.insert({fatBinary, fbData});
    }
}

void luthier::CodeObjectManager::registerFunction(const void *fbWrapper,
                                                  const char *funcName, const void *hostFunction, const char *deviceName) {
    const auto& loaderApi = HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    typedef std::unordered_map<decltype(hsa_agent_t::handle), hsa_executable_t> loader_cb_t;
    loader_cb_t luthierExecutables;
    auto findLuthierExecutable = [](hsa_executable_t exec, void* data){
        const auto& coreApi = HsaInterceptor::Instance().getSavedHsaTables().core;
        std::vector<hsa_agent_t> agents = ContextManager::Instance().getHsaAgents();
        auto symbolIterator = [](hsa_executable_t exec, hsa_agent_t agent, hsa_executable_symbol_t symbol,void * data) {
            const auto& coreTable = HsaInterceptor::Instance().getSavedHsaTables().core;
            hsa_symbol_kind_t symbolKind;
            LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbolKind));

            std::cout << "Symbol kind: " << symbolKind << std::endl;

            uint32_t nameSize;
            LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &nameSize));
            std::cout << "Symbol name size: " << nameSize << std::endl;
            std::string name;
            name.resize(nameSize);
            LUTHIER_HSA_CHECK(coreTable.hsa_executable_symbol_get_info_fn(symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()));
            std::cout << "Symbol Name: " << name << std::endl;
            if (name == "__luthier_reserved") {
                auto out = reinterpret_cast<loader_cb_t*>(data);
                out->insert({agent.handle, exec});
            }
            return HSA_STATUS_SUCCESS;
        };
        for (const auto& a: agents) {
            coreApi.hsa_executable_iterate_agent_symbols_fn(exec, a, symbolIterator, data);
        }
        return HSA_STATUS_SUCCESS;
    };
    loaderApi.hsa_ven_amd_loader_iterate_executables(findLuthierExecutable, &luthierExecutables);
    fmt::println("Number of executables captured: {}", luthierExecutables.size());
    assert(fbWrapper != nullptr);
    auto fbWrapperData = reinterpret_cast<const __CudaFatBinaryWrapper *>(fbWrapper);
    assert(fbWrapperData->magic == __hipFatMAGIC2 && fbWrapperData->version == 1);
    auto fatBinary = fbWrapperData->binary;
    if (!fatBinaries_.contains(fatBinary))
        registerFatBinary(fatBinary);

    std::string demangledName = getDemangledName(funcName);
    demangledName = demangledName.substr(0, demangledName.find('('));
    if (!functions_.contains(hostFunction))
        functions_.insert({hostFunction, {luthierExecutables, std::string(funcName), std::string(deviceName), fatBinary}});
}

std::string luthier::CodeObjectManager::getCodeObjectOfInstrumentationFunction(const void *function, hsa_agent_t agent) {
//    std::string funcNameKey = "__luthier_wrap__" + std::string(function);

    auto fb = functions_[function].parentFatBinary;
    auto fbData = fatBinaries_[fb];

    auto agentIsa = luthier::ContextManager::Instance().getHsaAgentInfo(agent)->getIsaName();

    std::vector<amd_comgr_code_object_info_t> isaInfo{{agentIsa.c_str(), 0, 0}};

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_lookup_code_object(fbData, isaInfo.data(), isaInfo.size()));

    // Strip off the kernel launch portion of the code object
//    return {reinterpret_cast<const char*>(fb) + isaInfo[0].offset, isaInfo[0].size};
//    stripOffKernelLaunch({reinterpret_cast<const char*>(fb) + isaInfo[0].offset, isaInfo[0].size}, function);
    return {reinterpret_cast<const char*>(fb) + isaInfo[0].offset, isaInfo[0].size};
}
void luthier::CodeObjectManager::registerKD(luthier_address_t originalCode, luthier_address_t instrumentedCode) {
    if (!instrumentedKernels_.contains(originalCode))
        instrumentedKernels_.insert({originalCode, instrumentedCode});
}
