/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "code_object_manipulation.hpp"
#include "error.h"
#include "hsa_intercept.hpp"
#include "log.hpp"
#include <elfio/elfio_dump.hpp>


#include <llvm/Support/BinaryStreamReader.h>

#include <string>

#include <thread>

namespace luthier::co_manip {
using namespace ELFIO;

// Taken from the hipamd project
struct CudaFatBinaryWrapper {
    unsigned int magic;
    unsigned int version;
    void *binary;
    void *dummy1;
};

// Taken from the hipamd project
constexpr unsigned hipFatMAGIC2 = 0x48495046;



static size_t constexpr strLiteralLength(char const *str) {
    size_t I = 0;
    while (str[I]) {
        ++I;
    }
    return I;
}

static constexpr const char *OFFLOAD_KIND_HIP = "hip";
static constexpr const char *OFFLOAD_KIND_HIPV4 = "hipv4";
static constexpr const char *OFFLOAD_KIND_HCC = "hcc";
static constexpr const char *CLANG_OFFLOAD_BUNDLER_MAGIC =
    "__CLANG_OFFLOAD_BUNDLE__";
static constexpr size_t OffloadBundleMagicLen =
    strLiteralLength(CLANG_OFFLOAD_BUNDLER_MAGIC);



#if !defined(ELFMAG)
#define ELFMAG "\177ELF"
#define SELFMAG 4
#endif

typedef enum {
    LLVMIR = 0,
    SOURCE,
    ILTEXT,
    ASTEXT,
    CAL,
    DLL,
    STRTAB,
    SYMTAB,
    RODATA,
    SHSTRTAB,
    NOTES,
    COMMENT,
    ILDEBUG,
    DEBUG_INFO,
    DEBUG_ABBREV,
    DEBUG_LINE,
    DEBUG_PUBNAMES,
    DEBUG_PUBTYPES,
    DEBUG_LOC,
    DEBUG_ARANGES,
    DEBUG_RANGES,
    DEBUG_MACINFO,
    DEBUG_STR,
    DEBUG_FRAME,
    JITBINARY,
    CODEGEN,
    TEXT,
    INTERNAL,
    SPIR,
    SPIRV,
    RUNTIME_METADATA,
    ELF_SECTIONS_LAST
} ElfSections;

typedef struct {
    ElfSections id;
    const char *name;
    uint64_t d_align;   // section alignment in bytes
    Elf32_Word sh_type; // section type
    Elf32_Word sh_flags;// section flags
    const char *desc;
} ElfSectionsDesc;



namespace {
// Objects that are visible only within this module
constexpr ElfSectionsDesc ElfSecDesc[] =
    {
        {LLVMIR, ".llvmir", 1, SHT_PROGBITS, 0, "ASIC-independent LLVM IR"},
        {SOURCE, ".source", 1, SHT_PROGBITS, 0, "OpenCL source"},
        {ILTEXT, ".amdil", 1, SHT_PROGBITS, 0, "AMD IL text"},
        {ASTEXT, ".astext", 1, SHT_PROGBITS, 0, "X86 assembly text"},
        {CAL, ".text", 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, "AMD CalImage"},
        {DLL, ".text", 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, "x86 dll"},
        {STRTAB, ".strtab", 1, SHT_STRTAB, SHF_STRINGS, "String table"},
        {SYMTAB, ".symtab", sizeof(Elf_Xword), SHT_SYMTAB, 0, "Symbol table"},
        {RODATA, ".rodata", 1, SHT_PROGBITS, SHF_ALLOC, "Read-only data"},
        {SHSTRTAB, ".shstrtab", 1, SHT_STRTAB, SHF_STRINGS, "Section names"},
        {NOTES, ".note", 1, SHT_NOTE, 0, "used by loader for notes"},
        {COMMENT, ".comment", 1, SHT_PROGBITS, 0, "Version string"},
        {ILDEBUG, ".debugil", 1, SHT_PROGBITS, 0, "AMD Debug IL"},
        {DEBUG_INFO, ".debug_info", 1, SHT_PROGBITS, 0, "Dwarf debug info"},
        {DEBUG_ABBREV, ".debug_abbrev", 1, SHT_PROGBITS, 0, "Dwarf debug abbrev"},
        {DEBUG_LINE, ".debug_line", 1, SHT_PROGBITS, 0, "Dwarf debug line"},
        {DEBUG_PUBNAMES, ".debug_pubnames", 1, SHT_PROGBITS, 0, "Dwarf debug pubnames"},
        {DEBUG_PUBTYPES, ".debug_pubtypes", 1, SHT_PROGBITS, 0, "Dwarf debug pubtypes"},
        {DEBUG_LOC, ".debug_loc", 1, SHT_PROGBITS, 0, "Dwarf debug loc"},
        {DEBUG_ARANGES, ".debug_aranges", 1, SHT_PROGBITS, 0, "Dwarf debug aranges"},
        {DEBUG_RANGES, ".debug_ranges", 1, SHT_PROGBITS, 0, "Dwarf debug ranges"},
        {DEBUG_MACINFO, ".debug_macinfo", 1, SHT_PROGBITS, 0, "Dwarf debug macinfo"},
        {DEBUG_STR, ".debug_str", 1, SHT_PROGBITS, 0, "Dwarf debug str"},
        {DEBUG_FRAME, ".debug_frame", 1, SHT_PROGBITS, 0, "Dwarf debug frame"},
        {JITBINARY, ".text", 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, "x86 JIT Binary"},
        {CODEGEN, ".cg", 1, SHT_PROGBITS, 0, "Target dependent IL"},
        {TEXT, ".text", 1, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, "Device specific ISA"},
        {INTERNAL, ".internal", 1, SHT_PROGBITS, 0, "Internal usage"},
        {SPIR, ".spir", 1, SHT_PROGBITS, 0, "Vendor/Device-independent LLVM IR"},
        {SPIRV, ".spirv", 1, SHT_PROGBITS, 0, "SPIR-V Binary"},
        {RUNTIME_METADATA, ".AMDGPU.runtime_metadata", 1, SHT_PROGBITS, 0, "AMDGPU runtime metadata"},
};
}// namespace


unsigned int getSymbolNum(const std::shared_ptr<co_manip::ElfView>& io) {
    auto& elfIo = io->getElfIo();
    symbol_section_accessor symbol_reader(elfIo, elfIo.sections[ElfSecDesc[SYMTAB].name]);
    return symbol_reader.get_symbols_num() - 1;// Exclude the first dummy symbol
}

SymbolView::SymbolView(const std::shared_ptr<ElfView>& elf, unsigned int index) : elf_(elf) {
    auto& elfIo = elf_->getElfIo();
    symbol_section_accessor symbol_reader(elfIo, elfIo.sections[ElfSecDesc[SYMTAB].name]);

    unsigned int num = getSymbolNum(elf);

    if (index >= num) {
        throw std::runtime_error(fmt::format("Failed to get symbol info: Index {} >= total number of symbols {}", index, num));
    }

    unsigned char bind = 0;
    Elf_Half sec_index = 0;
    unsigned char other = 0;
    size_t size = 0;

    // index++ for real index on top of the first dummy symbol
    bool ret = symbol_reader.get_symbol(++index, name_, value_, size, bind, type_,
                                        sec_index, other);
    if (!ret) {
        throw std::runtime_error(fmt::format("Failed to get symbol info for index {}.", index));
    }
    section_ = elfIo.sections[sec_index];
    if (section_ == nullptr) {
        throw std::runtime_error(fmt::format("Section for symbol index {} was "
                                             "reported as nullptr by the ELFIO library.", index));
    }

    data_ = code_view_t{const_cast<std::byte*>(elf->GetView().data() + section_->get_address() + (size_t) value_ - (size_t) section_->get_offset()),
                        size};
}

//SymbolInfo getSymbolInfo(const elfio &io, unsigned int index) {
//    symbol_section_accessor symbol_reader(io, io.sections[ElfSecDesc[SYMTAB].name]);
//
//    unsigned int num = getSymbolNum(io);
//
//    if (index >= num) {
//        throw std::runtime_error(fmt::format("Failed to get symbol info: Index {} >= total number of symbols {}", index, num));
//    }
//
//    std::string name;
//    Elf64_Addr value = 0;
//    Elf_Xword size = 0;
//    unsigned char bind = 0;
//    unsigned char type = 0;
//    Elf_Half sec_index = 0;
//    unsigned char other = 0;
//
//    // index++ for real index on top of the first dummy symbol
//    bool ret = symbol_reader.get_symbol(++index, name, value, size, bind, type,
//                                        sec_index, other);
//    if (!ret) {
//        throw std::runtime_error(fmt::format("Failed to get symbol info for index {}.", index));
//    }
//    section *sec = io.sections[sec_index];
//    if (sec == nullptr) {
//        throw std::runtime_error(fmt::format("Section for symbol index {} was "
//                                             "reported as nullptr by the ELFIO library.", index));
//    }
//
//    luthier_address_t address = sec->get_address() + (size_t) value - (size_t) sec->get_offset();
//
//    return {sec, name, address, size, value, type};
//}


amd_comgr_status_t printEntryCo(amd_comgr_metadata_node_t key,
                                amd_comgr_metadata_node_t value, void *data) {
    amd_comgr_metadata_kind_t kind;
    amd_comgr_metadata_node_t son;
    size_t size;
    char *keybuf;
    char *valbuf;
    int *indent = (int *)data;

    // assume key to be string in this test function
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_kind(key, &kind));
    if (kind != AMD_COMGR_METADATA_KIND_STRING)
        return AMD_COMGR_STATUS_ERROR;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(key, &size, NULL));
    keybuf = (char *)calloc(size, sizeof(char));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(key, &size, keybuf));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_kind(value, &kind));
    for (int i = 0; i < *indent; i++)
        printf("  ");

    switch (kind) {
        case AMD_COMGR_METADATA_KIND_STRING: {
            printf("%s  :  ", size ? keybuf : "");
            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(value, &size, NULL));
            valbuf = (char *)calloc(size, sizeof(char));
            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(value, &size, valbuf));
            printf(" %s\n", valbuf);
            free(valbuf);
            break;
        }
        case AMD_COMGR_METADATA_KIND_LIST: {
            *indent += 1;
            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_list_size(value, &size));
            printf("LIST %s %zd entries = \n", keybuf, size);
            for (size_t i = 0; i < size; i++) {
                LUTHIER_AMD_COMGR_CHECK(amd_comgr_index_list_metadata(value, i, &son));
                LUTHIER_AMD_COMGR_CHECK(printEntryCo(key, son, data));
                LUTHIER_AMD_COMGR_CHECK(amd_comgr_destroy_metadata(son));
            }
            *indent = *indent > 0 ? *indent - 1 : 0;
            break;
        }
        case AMD_COMGR_METADATA_KIND_MAP: {
            *indent += 1;
            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_map_size(value, &size));
            printf("MAP %zd entries = \n", size);
            LUTHIER_AMD_COMGR_CHECK(amd_comgr_iterate_map_metadata(value, printEntryCo, data));
            *indent = *indent > 0 ? *indent - 1 : 0;
            break;
        }
        default:
            free(keybuf);
            return AMD_COMGR_STATUS_ERROR;
    } // switch

    free(keybuf);
    return AMD_COMGR_STATUS_SUCCESS;
};


void iterateCodeObjectMetaData(luthier_address_t codeObjectData, size_t codeObjectSize) {
    // COMGR symbol iteration things
    amd_comgr_data_t coData;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &coData));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(coData, codeObjectSize, reinterpret_cast<const char*>(codeObjectData)));
    amd_comgr_metadata_node_t meta;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data_metadata(coData, &meta));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data_name(coData, "my-data.s"));
    int Indent = 1;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_iterate_map_metadata(meta, printEntryCo, (void *)&Indent));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_destroy_metadata(meta));
}

amd_comgr_status_t getCodeObjectElfsFromFatBinary(const void *data, std::vector<elfio>& fatBinaryElfs) {
    auto fbWrapper = reinterpret_cast<const CudaFatBinaryWrapper *>(data);
    assert(fbWrapper->magic == hipFatMAGIC2 && fbWrapper->version == 1);
    auto fatBinary = fbWrapper->binary;

    llvm::BinaryStreamReader Reader(llvm::StringRef(reinterpret_cast<const char*>(fatBinary), 4096),
                                    llvm::support::little);
    llvm::StringRef Magic;
    auto EC = Reader.readFixedString(Magic, OffloadBundleMagicLen);
    if (EC) {
        return AMD_COMGR_STATUS_ERROR;
    }
    if (Magic != CLANG_OFFLOAD_BUNDLER_MAGIC) {
        return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
    uint64_t NumOfCodeObjects;
    EC = Reader.readInteger(NumOfCodeObjects);
    if (EC) {
        return AMD_COMGR_STATUS_ERROR;
    }
    // For each code object, extract BundleEntryID information, and check that
    // against each ISA in the QueryList
    fatBinaryElfs.resize(NumOfCodeObjects);
    for (uint64_t I = 0; I < NumOfCodeObjects; I++) {
        uint64_t BundleEntryCodeObjectSize;
        uint64_t BundleEntryCodeObjectOffset;
        uint64_t BundleEntryIDSize;
        llvm::StringRef BundleEntryID;

        if (auto EC = Reader.readInteger(BundleEntryCodeObjectOffset)) {
            return AMD_COMGR_STATUS_ERROR;
        }

        if (auto Status = Reader.readInteger(BundleEntryCodeObjectSize)) {
            return AMD_COMGR_STATUS_ERROR;
        }

        if (auto Status = Reader.readInteger(BundleEntryIDSize)) {
            return AMD_COMGR_STATUS_ERROR;
        }

        if (Reader.readFixedString(BundleEntryID, BundleEntryIDSize)) {
            return AMD_COMGR_STATUS_ERROR;
        }

        const auto OffloadAndTargetId = BundleEntryID.split('-');
        fmt::println("Target: {}", OffloadAndTargetId.second.str());
        if (OffloadAndTargetId.first != OFFLOAD_KIND_HIP &&
            OffloadAndTargetId.first != OFFLOAD_KIND_HIPV4 &&
            OffloadAndTargetId.first != OFFLOAD_KIND_HCC) {
            continue;
        }
        std::stringstream ss{std::string(reinterpret_cast<const char*>(fatBinary) + BundleEntryCodeObjectOffset,
                                         BundleEntryCodeObjectSize)};
        if (!fatBinaryElfs.at(I).load(ss, false)) {
            fmt::println("Size of the code object: {}", BundleEntryCodeObjectSize);
            fmt::println("Failed to parse the ELF.");
            return AMD_COMGR_STATUS_ERROR;
        }

    }

    return AMD_COMGR_STATUS_SUCCESS;
}

code_view_t getFunctionFromSymbol(const std::shared_ptr<ElfView>& elfView, const std::string &functionName) {
    auto& elfIo = elfView->getElfIo();
    symbol_section_accessor symbol_reader(elfIo, elfIo.sections[ElfSecDesc[SYMTAB].name]);

    auto num = getSymbolNum(elfView);

//    symInfo.address = symInfo.sec_addr + (size_t) value - (size_t) sec->get_offset();
    std::string sym_name;
    Elf64_Addr value = 0;
    Elf_Xword size = 0;
    unsigned char bind = 0;
    unsigned char type = 0;
    Elf_Half sec_index = 0;
    unsigned char other = 0;

    fmt::println("Number of sections: {}", elfIo.sections.size());
    for (const auto& s: elfIo.sections) {
        fmt::println("Name of the section: {}", s->get_name());
        fmt::println("Address: {:#x}", s->get_address());
        fmt::println("Section size: {}", s->get_size());
    }
    fmt::println("Number of sections: {}", elfIo.segments.size());
    for (const auto& s: elfIo.segments) {
        fmt::println("Address: {:#x}", s->get_virtual_address());
    }

    for (unsigned int i = 1; i < num; i++) {
        bool ret = symbol_reader.get_symbol(i, sym_name, value, size, bind, type,
                                            sec_index, other);
        section *sec = elfIo.sections[sec_index];
        fmt::println("Section address: {:#x}", sec->get_address());
        fmt::println("Symbol name: {}", sym_name);
        fmt::println("Section name: {}", sec->get_name());
        fmt::println("Section size: {}", sec->get_size());
    }
    // index++ for real index on top of the first dummy symbol

    //    for (unsigned int i = 0; i < secNumbers; i++) {
    //        //                    luthier::elf::SymbolInfo info;
    //        //                    luthier::elf::getSymbolInfo(elfs[1], i, info);
    //        //                    fmt::println("Symbol's name and value: {}, {}", info.sym_name, info.value);
    //        //                    fmt::println("Symbol's address: {:#x}", reinterpret_cast<luthier_address_t>(info.address));
    //        //                    fmt::println("Symbol's content: {}", *reinterpret_cast<const int*>(info.address));
    //    }
}


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

std::vector<luthier::co_manip::code_view_t> getDeviceLoadedCodeObjectOfExecutable(hsa_executable_t executable,
                                                                                           hsa_agent_t agent) {
    auto amdTable = luthier::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void* data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t>*>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, & loadedCodeObjects);

    std::vector<luthier::co_manip::code_view_t> out;
    for (const auto &lco: loadedCodeObjects) {
        hsa_ven_amd_loader_loaded_code_object_kind_t coKind;
        LUTHIER_HSA_CHECK(amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(lco,
                                                                                  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND,
                                                                                  &coKind));
        assert(coKind == HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_AGENT);
        hsa_agent_t coAgent;
        LUTHIER_HSA_CHECK(amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(lco,
                                                                                  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
                                                                                  &coAgent));
        if (coAgent.handle == agent.handle) {
            uint64_t lcoBaseAddrDevice;
            amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                                                                    &lcoBaseAddrDevice);
            // Query the size of the loaded code object
            uint64_t lcoSizeDevice;
            amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                                                                    &lcoSizeDevice);
            out.emplace_back(reinterpret_cast<std::byte*>(lcoBaseAddrDevice), lcoSizeDevice);
        }
    }
    return out;
}

std::vector<co_manip::code_view_t> getHostLoadedCodeObjectOfExecutable(hsa_executable_t executable, hsa_agent_t agent) {
    auto amdTable = luthier::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void* data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t>*>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    LUTHIER_HSA_CHECK(amdTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, &loadedCodeObjects));

    std::vector<luthier::co_manip::code_view_t> out;
    for (const auto &lco: loadedCodeObjects) {
        hsa_ven_amd_loader_loaded_code_object_kind_t coKind;
        LUTHIER_HSA_CHECK(amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(lco,
                                                                                  HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_KIND,
                                                                                  &coKind));
        assert(coKind == HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_KIND_AGENT);
        hsa_agent_t coAgent;
        LUTHIER_HSA_CHECK(amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(lco,
                                                                HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
                                                                &coAgent));
        if (coAgent.handle == agent.handle) {
            uint64_t lcoBaseAddr;
            amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
                                                                    &lcoBaseAddr);
            // Query the size of the loaded code object
            uint64_t lcoSize;
            amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
                                                                    &lcoSize);
            out.emplace_back(reinterpret_cast<std::byte*>(lcoBaseAddr), lcoSize);
//            out.push_back({reinterpret_cast<luthier_address_t>(lcoBaseAddr), static_cast<size_t>(lcoSize)});
        }
    }
    return out;
}

void printRSR1(const kernel_descriptor_t *kd) {
    auto rsrc1 = kd->compute_pgm_rsrc1;
#define PRINT_RSRC1(rsrc1, prop) fmt::println(#prop": {}", AMD_HSA_BITS_GET(rsrc1, prop));
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_PRIV)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_BULKY)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER)
    PRINT_RSRC1(rsrc1, AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1)
#undef PRINT_RSRC1
}

void printRSR2(const kernel_descriptor_t *kd) {
    auto rsrc2 = kd->compute_pgm_rsrc2;
#define PRINT_RSRC2(rsrc2, prop) fmt::println(#prop": {}", AMD_HSA_BITS_GET(rsrc2, prop));
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO)
        PRINT_RSRC2(rsrc2, AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1)
#undef PRINT_RSRC2
}

void printCodeProperties(const kernel_descriptor_t *kd) {
        auto codeProperties = kd->kernel_code_properties;
#define PRINT_CODE_PROPERTIES(codeProperties, prop) fmt::println(#prop": {}", AMD_HSA_BITS_GET(codeProperties, prop));
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_RESERVED1)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_IS_PTR64)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED)
        PRINT_CODE_PROPERTIES(codeProperties, AMD_KERNEL_CODE_PROPERTIES_RESERVED2)
#undef PRINT_CODE_PROPERTIES
}

ELFIO::elfio createAuxilaryInstrumentationElf() {
    elfio out;
    out.create(ELFCLASS64, ELFDATA2LSB);
    out.set_os_abi(ELFOSABI_AMDGPU_HSA);
    out.set_flags(ET_DYN);
    out.set_abi_version(ELFABIVERSION_AMDGPU_HSA_V4);
    out.set_entry(0);
    return out;
}

//std::shared_ptr<ELFIO::elfio> elfioFromMemory(const co_manip::code_object_region_t& elf, bool lazy) {
//    std::string_view my_string{reinterpret_cast<const char*>(elf.data), elf.size};
//    boost::iostreams::stream<boost::iostreams::basic_array_source<char>> ss(my_string.begin(), my_string.end());
////    std::string a;
////    std::basic_string<char> ss;
//
////    std::istringstream my_ss(std::string_view{my_string});
////    fmt::println("String stream content before erase: {}", my_ss.str());
////    my_string[1] = ' ';
////    my_string[5] = ' ';
////    fmt::println("String stream content after erase: {}", my_ss.str());
//    auto out = std::make_shared<ELFIO::elfio>();
////    auto ss = std::make_shared<std::istringstream>(std::string(reinterpret_cast<const char*>(elf.data), elf.size));
//    out->load(ss, false);
//    return out;
//}


}// namespace luthier::elf
