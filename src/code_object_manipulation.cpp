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


unsigned int getSymbolNum(const elfio &io) {
    symbol_section_accessor symbol_reader(io, io.sections[ElfSecDesc[SYMTAB].name]);
    auto num = symbol_reader.get_symbols_num() - 1;// Exclude the first dummy symbol
    return num;
}

bool getSymbolInfo(const elfio &io, unsigned int index, SymbolInfo &symInfo) {

    symbol_section_accessor symbol_reader(io, io.sections[ElfSecDesc[SYMTAB].name]);

    auto num = getSymbolNum(io);

    if (index >= num) {
        LuthierErrorFmt(" failed: wrong index {} >= symbols num {}", index, num);
        return false;
    }

    std::string sym_name;
    Elf64_Addr value = 0;
    Elf_Xword size = 0;
    unsigned char bind = 0;
    unsigned char type = 0;
    Elf_Half sec_index = 0;
    unsigned char other = 0;

    // index++ for real index on top of the first dummy symbol
    bool ret = symbol_reader.get_symbol(++index, sym_name, value, size, bind, type,
                                        sec_index, other);
    if (!ret) {
        LuthierErrorFmt("failed to get_symbol({})", index);
        return false;
    }
    section *sec = io.sections[sec_index];
    if (sec == nullptr) {
        LuthierErrorFmt("failed: null section at {}", sec_index);
        return false;
    }

    symInfo.sec_addr = reinterpret_cast<const char *>(sec->get_address());
    symInfo.sec_size = sec->get_size();
    //  std::cout << "Offset: " << sec->get_offset() << std::endl;
    //  std::cout << "get Address: " << sec->get_address() << std::endl;
    //  std::cout << "align address: " << sec->get_addr_align() << std::endl;
    //  std::cout << "entry size: " << sec->get_entry_size() << std::endl;
    symInfo.address = symInfo.sec_addr + (size_t) value - (size_t) sec->get_offset();
    symInfo.size = (uint64_t) size;
    symInfo.value = (size_t) value;

    symInfo.sec_name = sec->get_name();
    symInfo.sym_name = sym_name;

    return true;
}


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

code_object_region_t getFunctionFromSymbol(ELFIO::elfio &elfio, const std::string &functionName) {
    symbol_section_accessor symbol_reader(elfio, elfio.sections[ElfSecDesc[SYMTAB].name]);

    auto num = getSymbolNum(elfio);

//    symInfo.address = symInfo.sec_addr + (size_t) value - (size_t) sec->get_offset();
    std::string sym_name;
    Elf64_Addr value = 0;
    Elf_Xword size = 0;
    unsigned char bind = 0;
    unsigned char type = 0;
    Elf_Half sec_index = 0;
    unsigned char other = 0;

    fmt::println("Number of sections: {}", elfio.sections.size());
    for (const auto& s: elfio.sections) {
        fmt::println("Name of the section: {}", s->get_name());
        fmt::println("Address: {:#x}", s->get_address());
        fmt::println("Section size: {}", s->get_size());
    }
    fmt::println("Number of sections: {}", elfio.segments.size());
    for (const auto& s: elfio.segments) {
        fmt::println("Address: {:#x}", s->get_virtual_address());
    }

    for (unsigned int i = 1; i < num; i++) {
        bool ret = symbol_reader.get_symbol(i, sym_name, value, size, bind, type,
                                            sec_index, other);
        section *sec = elfio.sections[sec_index];
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

luthier::co_manip::code_object_region_t getDeviceLoadedCodeObjectOfExecutable(hsa_executable_t executable) {
    auto amdTable = luthier::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void* data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t>*>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, & loadedCodeObjects);

    // Can be removed
    assert(loadedCodeObjects.size() == 1);


    uint64_t lcoBaseAddrDevice;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                                                            &lcoBaseAddrDevice);
    // Query the size of the loaded code object
    uint64_t lcoSizeDevice;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
                                                            &lcoSizeDevice);
    return {reinterpret_cast<luthier_address_t>(lcoBaseAddrDevice), static_cast<size_t>(lcoSizeDevice)};
}

luthier::co_manip::code_object_region_t getHostLoadedCodeObjectOfExecutable(hsa_executable_t executable) {
    auto amdTable = luthier::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void* data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t>*>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, & loadedCodeObjects);

    // Can be removed
    assert(loadedCodeObjects.size() == 1);


    uint64_t lcoBaseAddr;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
                                                            &lcoBaseAddr);
    // Query the size of the loaded code object
    uint64_t lcoSize;
    amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
                                                            &lcoSize);
    return {reinterpret_cast<luthier_address_t>(lcoBaseAddr), static_cast<size_t>(lcoSize)};
}

}// namespace luthier::elf
