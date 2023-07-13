#include "code_object_manager.hpp"
#include "hsa_intercept.h"
#include <assert.h>
#include <elfio/elfio.hpp>
#include "amdgpu_elf.hpp"
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>
#include "disassembler.hpp"

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
    if (!functions_.contains(demangledName))
        functions_.insert({demangledName, {std::string(funcName), hostFunction, std::string(deviceName), fatBinary}});
}

hsa_executable_t sibir::CodeObjectManager::getInstrumentationFunction(const char *functionName, hsa_agent_t agent) {
    std::string funcNameKey = "__sibir_wrap__" + std::string(functionName);

    auto fb = functions_[funcNameKey].parentFatBinary;
    auto fbData = fatBinaries_[fb];

    //TODO: Make this work with hsa_agent's ISA
    //TODO: Put the hsa_agent map somewhere accessible (rethink abstraction)
    auto coreHsaApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;

    std::vector<amd_comgr_code_object_info_t> isaInfo{{"amdgcn-amd-amdhsa--gfx908", 0, 0}};

    SIBIR_AMD_COMGR_CHECK(amd_comgr_lookup_code_object(fbData, isaInfo.data(), isaInfo.size()));

    auto coBinary = reinterpret_cast<const char *>(fb) + isaInfo[0].offset;
    auto coSize = isaInfo[0].size;

    ELFIO::elfio coElfIO;


    std::istringstream coStrStream{std::string(coBinary, coSize)};

    coElfIO.load(coStrStream);

    std::cout << "Using ELFIO to iterate over the sections of the ELF" << std::endl;

    for (auto it = coElfIO.sections.begin(); it != coElfIO.sections.end(); it++) {
        auto sec = it->get();
        std::cout << "Name: " << sec->get_name() << std::endl;
        std::cout << "Address: " << sec->get_address() << std::endl;
    }

    std::cout << "Read the notes section: " << std::endl;

    ELFIO::section* noteSec = coElfIO.sections[".note"];

    ELFIO::note_section_accessor note_reader(coElfIO, noteSec);

    auto num = note_reader.get_notes_num();
    ELFIO::Elf_Word type = 0;
    char* desc = nullptr;
    ELFIO::Elf_Word descSize = 0;

    for (unsigned int i = 0; i < num; i++) {
        std::string name;
        if(note_reader.get_note(i, type, name, desc, descSize)) {
            std::cout << "Note name: " << name << std::endl;
//            auto f = std::fstream("./note_content", std::ios::out);
            std::string content(desc, descSize);
            std::cout << "Note content" << content << std::endl;
//            f << content;
//            f.close();
        }
    }

    for (unsigned int i = 0; i < sibir::elf::getSymbolNum(coElfIO); i++) {
        sibir::elf::SymbolInfo info;
        sibir::elf::getSymbolInfo(coElfIO, i, info);
        std::cout << "Symbol Name: " << info.sym_name << std::endl;
        std::cout << "Section Name: " << info.sec_name << std::endl;
        std::cout << "Address: " << reinterpret_cast<const void*>(info.address) << std::endl;
        std::cout << "Sec address: " << reinterpret_cast<const void*>(info.sec_addr) << std::endl;
        std::cout << "Size: " << info.size << std::endl;
        std::cout << "Sec Size: " << info.sec_size << std::endl;
        if (info.sym_name.find("instrumentation_kernel") != std::string::npos and info.sym_name.find("kd") == std::string::npos) {
            auto insts = sibir::Disassembler::Instance().disassemble(agent, reinterpret_cast<sibir_address_t>(info.address), info.size);
            for (auto i : insts)
                std::cout << std::hex << i.addr << std::dec << ": " << i.instr << std::endl;
        }
        else if (info.sym_name.find("kd") != std::string::npos) {

            const auto kd = reinterpret_cast<const kernel_descriptor_t*>(info.address);
            std::cout << "KD: " << reinterpret_cast<const void *>(kd) << std::endl;
            std::cout << "Offset in KD: " << kd->kernel_code_entry_byte_offset << std::endl;
        }
    }


    // COMGR symbol iteration things
    amd_comgr_data_t coData;
    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &coData));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(coData, isaInfo[0].size,
                                             reinterpret_cast<const char *>(fb) + isaInfo[0].offset));

    auto symbolIterator = [](amd_comgr_symbol_t s, void *data) {
        uint64_t nameLength;
        auto status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH, &nameLength));
        if (status != AMD_COMGR_STATUS_SUCCESS)
            return status;
        std::string name;
        name.resize(nameLength);
        status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_NAME, name.data()));
        if (status != AMD_COMGR_STATUS_SUCCESS)
            return status;

        std::cout << "AMD COMGR symbol name: " << name << std::endl;

        amd_comgr_symbol_type_t type;
        status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_TYPE, &type));
        if (status != AMD_COMGR_STATUS_SUCCESS)
            return status;

        std::cout << "AMD COMGR symbol type: " << type << std::endl;

        uint64_t size;
        status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_SIZE, &size));
        if (status != AMD_COMGR_STATUS_SUCCESS)
            return status;

        std::cout << "AMD COMGR symbol size: " << size << std::endl;

        bool isDefined;
        status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED, &isDefined));
        if (status != AMD_COMGR_STATUS_SUCCESS)
            return status;

        std::cout << "AMD COMGR symbol is defined: " << size << std::endl;

        uint64_t value;
        status = SIBIR_AMD_COMGR_CHECK(amd_comgr_symbol_get_info(s, AMD_COMGR_SYMBOL_INFO_VALUE, &value));
        if (status != AMD_COMGR_STATUS_SUCCESS)
            return status;

        std::cout << "AMD COMGR symbol value: " << reinterpret_cast<void*>(value) << std::endl;

        return AMD_COMGR_STATUS_SUCCESS;
    };

    amd_comgr_iterate_symbols(coData, symbolIterator, nullptr);

    auto coreApi = SibirHsaInterceptor::Instance().getSavedHsaTables().core;
    hsa_code_object_reader_t coReader;
    hsa_executable_t executable;
    SIBIR_HSA_CHECK(coreApi.hsa_code_object_reader_create_from_memory_fn(reinterpret_cast<const unsigned char *>(fb) + isaInfo[0].offset,
                                                                         isaInfo[0].size, &coReader));

    SIBIR_HSA_CHECK(coreApi.hsa_executable_create_alt_fn(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &executable));

    SIBIR_HSA_CHECK(coreApi.hsa_executable_load_agent_code_object_fn(executable, agent, coReader, nullptr, nullptr));

    SIBIR_HSA_CHECK(coreApi.hsa_executable_freeze_fn(executable, nullptr));

    return executable;
}
