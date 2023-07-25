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
    amd_comgr_status_t status;

    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_BYTES, &mangledNameData));

    size_t size = strlen(mangledName);
    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(mangledNameData, size, mangledName));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_demangle_symbol_name(mangledNameData, &demangledNameData));

    size_t demangledNameSize = 0;
    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_data(demangledNameData, &demangledNameSize, nullptr));

    out.resize(demangledNameSize);

    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_data(demangledNameData, &demangledNameSize, out.data()));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_release_data(mangledNameData));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_release_data(demangledNameData));

    return out;
}


std::string stripOffKernelLaunch(const std::string& elf, const std::string& demangledName) {

    std::istringstream ss{elf};
    ELFIO::elfio elfio;
    elfio.load(ss);
    auto numSymbols = sibir::elf::getSymbolNum(elfio);
    for (int i = 0; i < numSymbols; i++) {
        sibir::elf::SymbolInfo info;
        sibir::elf::getSymbolInfo(elfio, i, info);
        std::string demangledSymName = getDemangledName(info.sym_name.c_str());
        if (demangledSymName.find("__sibir_wrap__") == std::string::npos and demangledSymName.find(demangledName) != std::string::npos) {
            std::cout << "Symbol name: " << getDemangledName(info.sym_name.c_str()) << std::endl;
            std::cout << "Symbol size: " << info.size << std::endl;
            std::cout << "Symbol Addr: " << reinterpret_cast<const void*>(info.address) << std::endl;
            return {info.address, info.size};
        }

    }
    return {};
}
}// namespace

void sibir::CodeObjectManager::registerFatBinary(const void *data) {
    assert(data != nullptr);
    auto fbWrapper = reinterpret_cast<const __CudaFatBinaryWrapper *>(data);
    assert(fbWrapper->magic == __hipFatMAGIC2 && fbWrapper->version == 1);
    auto fatBinary = fbWrapper->binary;
    if (!fatBinaries_.contains(fatBinary)) {
        amd_comgr_data_t fbData;
        SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_FATBIN, &fbData));
        SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(fbData, 4096, reinterpret_cast<const char *>(fatBinary)));
        fatBinaries_.insert({fatBinary, fbData});
    }
}

void sibir::CodeObjectManager::registerFunction(const void *fbWrapper,
                                              const char *funcName, const void *hostFunction, const char *deviceName) {
    auto fatBinary = reinterpret_cast<const __CudaFatBinaryWrapper *>(fbWrapper)->binary;
    if (!fatBinaries_.contains(fatBinary))
        registerFatBinary(fatBinary);
    std::string demangledName = getDemangledName(funcName);
    demangledName = demangledName.substr(0, demangledName.find('('));
    if (!functions_.contains(demangledName))
        functions_.insert({demangledName, {std::string(funcName), hostFunction, std::string(deviceName), fatBinary}});
}

std::string sibir::CodeObjectManager::getCodeObjectOfInstrumentationFunction(const char *functionName, hsa_agent_t agent) {
    std::string funcNameKey = "__sibir_wrap__" + std::string(functionName);

    auto fb = functions_[funcNameKey].parentFatBinary;
    auto fbData = fatBinaries_[fb];

    auto agentIsa = sibir::ContextManager::Instance().getHsaAgentInfo(agent)->getIsaName();

    std::vector<amd_comgr_code_object_info_t> isaInfo{{agentIsa.c_str(), 0, 0}};

    SIBIR_AMD_COMGR_CHECK(amd_comgr_lookup_code_object(fbData, isaInfo.data(), isaInfo.size()));

    // Strip off the kernel launch portion of the code object
    return stripOffKernelLaunch({reinterpret_cast<const char*>(fb) + isaInfo[0].offset, isaInfo[0].size}, functionName);
//    return {reinterpret_cast<const char*>(fb) + isaInfo[0].offset, isaInfo[0].size};
}
void sibir::CodeObjectManager::registerKD(sibir_address_t originalCode, sibir_address_t instrumentedCode) {
    if (!instrumentedKernels_.contains(originalCode))
        instrumentedKernels_.insert({originalCode, instrumentedCode});
}
