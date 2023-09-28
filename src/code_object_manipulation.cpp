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
#include "context_manager.hpp"
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

unsigned int getSymbolNum(const ElfView &elfView) {
    auto &elfIo = elfView->getElfIo();
    symbol_section_accessor symbol_reader(elfIo, elfIo.sections[ElfSecDesc[SYMTAB].name]);
    return symbol_reader.get_symbols_num() - 1;// Exclude the first dummy symbol
}

SymbolView::SymbolView(const ElfView &elf, unsigned int index) : elf_(elf) {
    auto &elfIo = elf_->getElfIo();
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
                                             "reported as nullptr by the ELFIO library.",
                                             index));
    }

    data_ = code_view_t{const_cast<std::byte *>(elf->getView().data() + section_->get_address() + (size_t) value_ - (size_t) section_->get_offset()),
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



//void iterateCodeObjectMetaData(luthier_address_t codeObjectData, size_t codeObjectSize) {
//    // COMGR symbol iteration things
//    amd_comgr_data_t coData;
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &coData));
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(coData, codeObjectSize, reinterpret_cast<const char *>(codeObjectData)));
//    amd_comgr_metadata_node_t meta;
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data_metadata(coData, &meta));
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data_name(coData, "my-data.s"));
//    int Indent = 1;
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_iterate_map_metadata(meta, extractCodeObjectMetaDataMap, (void *) &Indent));
//    LUTHIER_AMD_COMGR_CHECK(amd_comgr_destroy_metadata(meta));
//}

amd_comgr_status_t getCodeObjectElfsFromFatBinary(const void *data, std::vector<elfio> &fatBinaryElfs) {
    auto fbWrapper = reinterpret_cast<const CudaFatBinaryWrapper *>(data);
    assert(fbWrapper->magic == hipFatMAGIC2 && fbWrapper->version == 1);
    auto fatBinary = fbWrapper->binary;

    llvm::BinaryStreamReader Reader(llvm::StringRef(reinterpret_cast<const char *>(fatBinary), 4096),
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
        if (OffloadAndTargetId.first != OFFLOAD_KIND_HIP && OffloadAndTargetId.first != OFFLOAD_KIND_HIPV4 && OffloadAndTargetId.first != OFFLOAD_KIND_HCC) {
            continue;
        }
        std::stringstream ss{std::string(reinterpret_cast<const char *>(fatBinary) + BundleEntryCodeObjectOffset,
                                         BundleEntryCodeObjectSize)};
        if (!fatBinaryElfs.at(I).load(ss, false)) {
            fmt::println("Size of the code object: {}", BundleEntryCodeObjectSize);
            fmt::println("Failed to parse the ELF.");
            return AMD_COMGR_STATUS_ERROR;
        }
    }

    return AMD_COMGR_STATUS_SUCCESS;
}

code_view_t getFunctionFromSymbol(const ElfView &elfView, const std::string &functionName) {
    auto &elfIo = elfView->getElfIo();
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
    for (const auto &s: elfIo.sections) {
        fmt::println("Name of the section: {}", s->get_name());
        fmt::println("Address: {:#x}", s->get_address());
        fmt::println("Section size: {}", s->get_size());
    }
    fmt::println("Number of sections: {}", elfIo.segments.size());
    for (const auto &s: elfIo.segments) {
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

std::string getDemangledName(const std::string &mangledName) {
    amd_comgr_data_t mangledNameData;
    amd_comgr_data_t demangledNameData;
    std::string out;

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_BYTES, &mangledNameData));

    size_t size = mangledName.size();
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(mangledNameData, size, mangledName.data()));

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
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void *data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t> *>(data);
        out->push_back(lco);
        return HSA_STATUS_SUCCESS;
    };
    amdTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, iterator, &loadedCodeObjects);

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
            out.emplace_back(reinterpret_cast<std::byte *>(lcoBaseAddrDevice), lcoSizeDevice);
        }
    }
    return out;
}

std::vector<co_manip::code_view_t> getHostLoadedCodeObjectOfExecutable(hsa_executable_t executable, hsa_agent_t agent) {
    auto amdTable = luthier::HsaInterceptor::Instance().getHsaVenAmdLoaderTable();
    // Get a list of loaded code objects inside the executable
    std::vector<hsa_loaded_code_object_t> loadedCodeObjects;
    auto iterator = [](hsa_executable_t e, hsa_loaded_code_object_t lco, void *data) -> hsa_status_t {
        auto out = reinterpret_cast<std::vector<hsa_loaded_code_object_t> *>(data);
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
            out.emplace_back(reinterpret_cast<std::byte *>(lcoBaseAddr), lcoSize);
            //            out.push_back({reinterpret_cast<luthier_address_t>(lcoBaseAddr), static_cast<size_t>(lcoSize)});
        }
    }
    return out;
}

void printRSR1(const kernel_descriptor_t *kd) {
    auto rsrc1 = kd->compute_pgm_rsrc1;
#define PRINT_RSRC1(rsrc1, prop) fmt::println(#prop ": {}", AMD_HSA_BITS_GET(rsrc1, prop));
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
#define PRINT_RSRC2(rsrc2, prop) fmt::println(#prop ": {}", AMD_HSA_BITS_GET(rsrc2, prop));
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
#define PRINT_CODE_PROPERTIES(codeProperties, prop) fmt::println(#prop ": {}", AMD_HSA_BITS_GET(codeProperties, prop));
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

void setupSHdr(
    ELFIO::elfio &elfIo,
    ElfSections id,
    section *section,
    Elf64_Word shlink = 0) {
    section->set_addr_align(ElfSecDesc[id].d_align);
    section->set_type(ElfSecDesc[id].sh_type);
    section->set_flags(ElfSecDesc[id].sh_flags);
    section->set_link(shlink);

    auto class_num = elfIo.get_class();
    size_t entry_size = 0;
    switch (id) {
        case SYMTAB:
            if (class_num == ELFCLASS32) {
                entry_size = sizeof(Elf32_Sym);
            } else {
                entry_size = sizeof(Elf64_Sym);
            }
            break;
        default:
            // .dynsym and .relaNAME also have table entries
            break;
    }
    if (entry_size > 0) {
        section->set_entry_size(entry_size);
    }
}

section *newSection(
    ELFIO::elfio &elfIo,
    ElfSections id,
    co_manip::code_view_t data) {
    assert(ElfSecDesc[id].id == id && "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");

    section *sec = elfIo.sections[ElfSecDesc[id].name];
    if (sec == nullptr) {
        sec = elfIo.sections.add(ElfSecDesc[id].name);
    }
    if (sec == nullptr) {
        LuthierErrorFmt("failed: sections.add({:s}) = nullptr", ElfSecDesc[id].name);
        return sec;
    }

    sec->set_data(reinterpret_cast<const char*>(data.data()), data.size());

    auto shlink = (id == SYMTAB) ? elfIo.sections[ElfSecDesc[SYMTAB].name]->get_index() : 0;

    setupSHdr(elfIo, id, sec, shlink);

    return sec;
}

bool addSectionData(
    ELFIO::elfio& elfIo,
    Elf_Xword &outOffset,
    ElfSections id,
    co_manip::code_view_t data) {
    assert(ElfSecDesc[id].id == id && "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");

    outOffset = 0;
    section *sec = elfIo.sections[ElfSecDesc[id].name];
    assert(sec != nullptr);

    outOffset = sec->get_size();

    sec->append_data(reinterpret_cast<const char *>(data.data()), data.size());

    return true;
}

bool addSection(
    ELFIO::elfio &elfIo,
    ElfSections id,
    co_manip::code_view_t data) {
    assert(ElfSecDesc[id].id == id && "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");

    section *sec = elfIo.sections[ElfSecDesc[id].name];

    if (sec != nullptr) {
        Elf_Xword sec_offset = 0;
        addSectionData(elfIo, sec_offset, id, data);
    } else {
        sec = newSection(elfIo, id, data);
        if (sec == nullptr) {
            return false;
        }
    }

    return true;
}



bool addSymbol(
    ELFIO::elfio& elfIo,
    ElfSections id,
    const char *symbolName,
    code_view_t data) {
    assert(ElfSecDesc[id].id == id && "The order of ElfSecDesc[] and Elf::ElfSections mismatches.");

    section* symTabSection = elfIo.sections[ElfSecDesc[SYMTAB].name];
    assert(symTabSection != nullptr);

    const char *sectionName = ElfSecDesc[id].name;

    bool isFunction = ((id == CAL) || (id == DLL) || (id == JITBINARY) || (id == TEXT));

    // Get section index
    section *sec = elfIo.sections[sectionName];
    if (sec == nullptr) {
        // Create a new section.
        if ((sec = newSection(elfIo, id, {})) == nullptr) {
            LuthierErrorFmt("Failed in newSection(name={:s})", sectionName);
            return false;
        }
    }
    size_t sec_ndx = sec->get_index();
    if (sec_ndx == SHN_UNDEF) {
        LuthierErrorMsg("failed: sec->get_index() = SHN_UNDEF");
        return false;
    }

    // Put symbolName into .strtab section
    Elf_Xword strtab_offset = 0;
    addSectionData(elfIo, strtab_offset, STRTAB, {reinterpret_cast<const std::byte*>(symbolName),
                                                       strlen(symbolName) + 1});

    // Put buffer into section
    Elf_Xword sec_offset = 0;
    if (not data.empty()) {
        if (!addSectionData(elfIo, sec_offset, id, data)) {
//            LogElfError("failed in addSectionData(name=%s, buffer=%p, size=%zu)",
//                        sectionName, buffer, size);
            return false;
        }
    }

    symbol_section_accessor symbolWriter(elfIo, symTabSection);

    auto ret = symbolWriter.add_symbol(strtab_offset, sec_offset, data.size(), 0,
                                         (isFunction) ? STT_FUNC : STT_OBJECT, 0, sec_ndx);

//    LuthierLogDebug("{:s}: sectionName={:s} symbolName={:s} strtab_offset={:lu}, sec_offset={:lu}, "
//                "size={:zu}, sec_ndx={:zu}, ret={:d}",
//                ret >= 1 ? "succeeded" : "failed",
//                sectionName, symbolName, strtab_offset, sec_offset, data.size(), sec_ndx, ret);
    return ret >= 1;
}

ELFIO::elfio createAMDGPUElf(const ELFIO::elfio &elfIoIn, hsa_agent_t agent) {
    ELFIO::elfio elfIo;
    elfIo.create(elfIoIn.get_class(), elfIoIn.get_encoding());
    elfIo.set_os_abi(elfIoIn.get_os_abi());
    elfIo.set_flags(elfIoIn.get_flags());
    elfIo.set_abi_version(elfIoIn.get_abi_version());
    elfIo.set_entry(elfIoIn.get_entry());
    elfIo.set_machine(elfIoIn.get_machine());

    auto shStrTabSection = elfIo.sections[ElfSecDesc[SHSTRTAB].name];
    assert(shStrTabSection != nullptr);

    setupSHdr(elfIo, SHSTRTAB, shStrTabSection);

    //
    // 3. Create .strtab section
    //
    auto *strTabSec = elfIo.sections.add(ElfSecDesc[STRTAB].name);
    assert(strTabSec != nullptr);

    // adding null string data associated with section
    // index 0 is reserved and must be there (NULL name)
    constexpr char strtab[] = {
        /* index 0 */ '\0'};
    strTabSec->set_data(const_cast<char *>(strtab), sizeof(strtab));

    setupSHdr(elfIo, STRTAB, strTabSec);

    // 4. Create the symbol table

    // Create the first reserved dummy symbol (undefined symbol)
    size_t sym_sz = (elfIo.get_class() == ELFCLASS32) ? sizeof(Elf32_Sym) : sizeof(Elf64_Sym);
    auto sym = static_cast<std::byte*>(std::calloc(1, sym_sz));
    assert(sym != nullptr);

    auto *symTabSec = newSection(elfIo, SYMTAB, {sym, sym_sz});
    std::free(sym);
    assert(symTabSec != nullptr);

    return elfIo;
}

amd_comgr_status_t extractCodeObjectMetaDataMap(amd_comgr_metadata_node_t key,
                                                amd_comgr_metadata_node_t value, void *data) {
    amd_comgr_metadata_kind_t kind;
    amd_comgr_metadata_node_t node;
    size_t size;
    std::string keyStr;
    std::string valueStr;
    std::any valueAny;
    auto out = reinterpret_cast<std::any*>(data);

    // TODO: Keys should only be strings??
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_kind(key, &kind));
    if (kind != AMD_COMGR_METADATA_KIND_STRING)
        return AMD_COMGR_STATUS_ERROR;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(key, &size, nullptr));

    keyStr.resize(size);
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(key, &size, keyStr.data()));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_kind(value, &kind));

    fmt::println("Key: {}", keyStr);
    switch (kind) {
        case AMD_COMGR_METADATA_KIND_STRING: {
            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(value, &size, nullptr));
            valueStr.resize(size);
            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(value, &size, valueStr.data()));
            valueAny = valueStr;
            fmt::println("STRING: {}", valueStr);
            break;
        }
        case AMD_COMGR_METADATA_KIND_LIST: {
            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_list_size(value, &size));
            valueAny = std::vector<std::any>();
            fmt::println("Vector");
            for (size_t i = 0; i < size; i++) {
                LUTHIER_AMD_COMGR_CHECK(amd_comgr_index_list_metadata(value, i, &node));
                LUTHIER_AMD_COMGR_CHECK(extractCodeObjectMetaDataMap(key, node, &valueAny));
                LUTHIER_AMD_COMGR_CHECK(amd_comgr_destroy_metadata(node));
            }

            break;
        }
        case AMD_COMGR_METADATA_KIND_MAP: {
            fmt::println("Map");
            valueAny = std::unordered_map<std::string, std::any>();
            LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_map_size(value, &size));
            LUTHIER_AMD_COMGR_CHECK(amd_comgr_iterate_map_metadata(value, extractCodeObjectMetaDataMap, &valueAny));
            break;
        }
        default:
            return AMD_COMGR_STATUS_ERROR;
    }// switch

    if (out->type() == typeid(std::unordered_map<std::string, std::any>)) {
        std::any_cast<std::unordered_map<std::string, std::any>>(out)->insert({keyStr, valueAny});
    }
    else if (out->type() == typeid(std::vector<std::any>)) {
        std::any_cast<std::vector<std::any>>(out)->emplace_back(valueAny);
    }

    return AMD_COMGR_STATUS_SUCCESS;
};


std::unordered_map<std::string, std::any> parseElfNoteSection(const co_manip::ElfView& elfView) {
    std::any out{std::unordered_map<std::string, std::any>()};
    amd_comgr_data_t coData;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &coData));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(coData, elfView->getView().size(),
                                               reinterpret_cast<const char *>(elfView->getView().data())));
    amd_comgr_metadata_node_t meta;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data_metadata(coData, &meta));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_iterate_map_metadata(meta, extractCodeObjectMetaDataMap, &out));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_destroy_metadata(meta));
    return std::any_cast<std::unordered_map<std::string, std::any>>(out);

}

//    fmt::println("Number of sections: {}", elfIoIn.sections.size());
//    for (const auto& sec: elfIoIn.sections) {
//        fmt::println("Stream size: {:#x}", sec->get_stream_size());
//        fmt::println("Address: {:#x}", sec->get_address());
//        fmt::println("Name: {}", sec->get_name());
//        fmt::println("Size: {}", sec->get_size());
//        fmt::println("Name: {}", sec->get_name());
//        fmt::println("Index: {}", sec->get_index());
//        fmt::println("Offset: {:#x}", sec->get_offset());
//        fmt::println("Flags: {}", sec->get_flags());
//        fmt::println("Addr Align: {:#x}", sec->get_addr_align());
//        fmt::println("Info: {}", sec->get_info());
//        fmt::println("Link: {}", sec->get_link());
//        fmt::println("Name string offset: {:#x}", sec->get_name_string_offset());
//        fmt::println("Type: {}", sec->get_type());
//        fmt::println("------------");
//    }
//    fmt::println("Number of segments: {}", elfIoIn.segments.size());
//    for (const auto& seg: elfIoIn.segments) {
//        fmt::println("Section Num: {}", seg->get_sections_num());
//        for (unsigned int i = 0; i < seg->get_sections_num(); i++) {
//            fmt::println("Section name: {}", elfIoIn.sections[seg->get_section_index_at(i)]->get_name());
//        }
//        fmt::println("Flags: {:#x}", seg->get_flags());
//        fmt::println("Offset: {:#x}", seg->get_offset());
//        fmt::println("Align: {:#x}", seg->get_align());
//        fmt::println("File size: {}", seg->get_file_size());
//        fmt::println("Index: {}", seg->get_index());
//        fmt::println("Memory size: {}", seg->get_memory_size());
//        fmt::println("Physical Address: {:#x}", seg->get_physical_address());
//        fmt::println("Type: {:#x}", seg->get_type());
//        fmt::println("Virtual Address: {:#x}", seg->get_virtual_address());
//        fmt::println("------");
//    }


}// namespace luthier::co_manip
