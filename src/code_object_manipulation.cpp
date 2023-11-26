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

#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/BinaryStreamReader.h>
#include <llvm/Support/Error.h>

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
    auto amdTable = luthier::HsaInterceptor::instance().getHsaVenAmdLoaderTable();
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
    auto amdTable = luthier::HsaInterceptor::instance().getHsaVenAmdLoaderTable();
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
            amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(lco,
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
                                                                    &lcoBaseAddr);
            // Query the size of the loaded code object
            uint64_t lcoSize;
            amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(lco,
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
                                                                    &lcoSize);
            int64_t lcoDelta;
            amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(lco,
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
                                                                    &lcoDelta);

            uint64_t lcoBaseAddrDevice;
            amdTable.hsa_ven_amd_loader_loaded_code_object_get_info(loadedCodeObjects[0],
                                                                    HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
                                                                    &lcoBaseAddrDevice);

            fmt::println("Code object Delta: {:#x}", lcoDelta);
            fmt::println("Base storage: {:#x}", lcoBaseAddr);
            fmt::println("Base Device: {:#x}", lcoBaseAddrDevice);

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

    sec->set_data(reinterpret_cast<const char *>(data.data()), data.size());

    auto shlink = (id == SYMTAB) ? elfIo.sections[ElfSecDesc[SYMTAB].name]->get_index() : 0;

    setupSHdr(elfIo, id, sec, shlink);

    return sec;
}

bool addSectionData(
    ELFIO::elfio &elfIo,
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
    ELFIO::elfio &elfIo,
    ElfSections id,
    const char *symbolName,
    code_view_t data) {
    assert(ElfSecDesc[id].id == id && "The order of ElfSecDesc[] and Elf::ElfSections mismatches.");

    section *symTabSection = elfIo.sections[ElfSecDesc[SYMTAB].name];
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
    addSectionData(elfIo, strtab_offset, STRTAB, {reinterpret_cast<const std::byte *>(symbolName), strlen(symbolName) + 1});

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
    // Create relocatable stored in a code_t
    const std::string code = "s_nop 0";
    std::string isaName = luthier::ContextManager::Instance().getHsaAgentInfo(agent)->getIsaName();

    amd_comgr_data_t relocIn, relocOut;

    std::string relocName;
    size_t relocNameSize;

    amd_comgr_data_set_t relocDataSetIn, relocDataSetOut;
    amd_comgr_action_info_t relocDataAction;

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&relocDataSetIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &relocIn));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(relocIn, code.size(), code.data()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data_name(relocIn, "test_relocatable.s"));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_data_set_add(relocDataSetIn, relocIn));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data_set(&relocDataSetOut));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_action_info(&relocDataAction));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_isa_name(relocDataAction, isaName.c_str()));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_action_info_set_option_list(relocDataAction, nullptr, 0));
    LUTHIER_AMD_COMGR_CHECK(
        amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                            relocDataAction, relocDataSetIn, relocDataSetOut));
    amd_comgr_action_data_get_data(relocDataSetOut, AMD_COMGR_DATA_KIND_RELOCATABLE, 0, &relocOut);

    amd_comgr_get_data_name(relocOut, &relocNameSize, nullptr);
    relocName.resize(relocNameSize);
    amd_comgr_get_data_name(relocOut, &relocNameSize, relocName.data());

    luthier::co_manip::code_t relocElf;
    size_t relocOutSize;

    amd_comgr_get_data(relocOut, &relocOutSize, nullptr);
    relocElf.resize(relocOutSize);
    amd_comgr_get_data(relocOut, &relocOutSize, reinterpret_cast<char *>(relocElf.data()));

    // Load elfio from code_t
    ELFIO::elfio elfIo;

    luthier::co_manip::code_char_stream_t relocElfStream = makeCodeCharStream(relocElf);

    elfIo.load(relocElfStream, false);
    elfIo.set_os_abi(elfIoIn.get_os_abi());
    elfIo.set_flags(elfIoIn.get_flags());
    elfIo.set_abi_version(elfIoIn.get_abi_version());
    elfIo.set_entry(elfIoIn.get_entry());
    elfIo.set_machine(elfIoIn.get_machine());
    elfIo.set_type(ELFIO::ET_REL);

    // Create empty ELF sections in elfIo
    std::vector<ELFIO::section*> relocElfSections;
    // fmt::println("Initial ELF has {} Sections", elfIo.sections.size());
    for (int i = 0; i < elfIo.sections.size(); i++) {
        const ELFIO::section* psec = elfIo.sections[i];
        // fmt::println(" [{}] {}\t{}", i, psec->get_name(), psec->get_size());
        relocElfSections.emplace_back(elfIo.sections[i]);
    }
    // fmt::println("\nLine {}: Create sections for relocatable", __LINE__);
    // fmt::println("\nOriginal ELF has {} sections", elfIoIn.sections.size());
    for (int i = 0; i < elfIoIn.sections.size(); i++) {
        const ELFIO::section* psec = elfIoIn.sections[i];

        bool section_in_reloc = false;
        // fmt::println(" [{}] {}\t{}", i, psec->get_name(), psec->get_size());
        for (auto reloc_sec : relocElfSections) {
            if (!psec->get_name().compare(reloc_sec->get_name())) {
                section_in_reloc = true;
                break;
            }
        }
        if (!section_in_reloc)
            elfIo.sections.add(psec->get_name());
    }   //fmt::print("\n");
    // fmt::println("Final ELF has {} Sections", elfIo.sections.size());
    // for (int i = 0; i < elfIo.sections.size(); i++) {
    //     const ELFIO::section* psec = elfIo.sections[i];
    //     fmt::println(" [{}] {}\t{}", i, psec->get_name(), psec->get_size());
    // }
    return elfIo;
}

ElfViewImpl::ElfViewImpl(code_view_t elf) : data_(elf),
                                            // Convert the code_view_t to a string_view first, and then take its iterators to construct the dataStringStream_
                                            dataStringStream_(std::make_unique<boost_ios::stream<boost_ios::basic_array_source<char>>>(
                                                std::string_view(reinterpret_cast<const char *>(data_.data()), data_.size()).begin(),
                                                std::string_view(reinterpret_cast<const char *>(data_.data()), data_.size()).end())) {
}

amd_comgr_status_t extractCodeObjectMetaDataMap(amd_comgr_metadata_node_t key,
                                                amd_comgr_metadata_node_t value, void *data) {
    amd_comgr_metadata_kind_t kind;
    amd_comgr_metadata_node_t node;
    size_t size;
    std::string keyStr;
    std::string valueStr;
    std::any valueAny;
    auto out = reinterpret_cast<std::any *>(data);

    // TODO: Keys should only be strings??
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_kind(key, &kind));
    if (kind != AMD_COMGR_METADATA_KIND_STRING)
        return AMD_COMGR_STATUS_ERROR;
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(key, &size, nullptr));

    keyStr.resize(size);
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(key, &size, keyStr.data()));

    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_metadata_kind(value, &kind));

    fmt::print("Key: {} -> ", keyStr);
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
        fmt::println("Added to map: {}", keyStr);
        std::any_cast<std::unordered_map<std::string, std::any>>(out)->insert({keyStr, valueAny});
    } else if (out->type() == typeid(std::vector<std::any>)) {
        fmt::println("added entry to vector");
        std::any_cast<std::vector<std::any>>(out)->emplace_back(valueAny);
    }

    return AMD_COMGR_STATUS_SUCCESS;
};

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




//  for Code Object V3
enum class ArgField : uint8_t {
    Name = 0,
    TypeName = 1,
    Size = 2,
    Align = 3,
    ValueKind = 4,
    PointeeAlign = 5,
    AddrSpaceQual = 6,
    AccQual = 7,
    ActualAccQual = 8,
    IsConst = 9,
    IsRestrict = 10,
    IsVolatile = 11,
    IsPipe = 12,
    Offset = 13
};

enum class AttrField : uint8_t {
    ReqdWorkGroupSize = 0,
    WorkGroupSizeHint = 1,
    VecTypeHint = 2,
    RuntimeHandle = 3
};

enum class CodePropField : uint8_t {
    KernargSegmentSize = 0,
    GroupSegmentFixedSize = 1,
    PrivateSegmentFixedSize = 2,
    KernargSegmentAlign = 3,
    WavefrontSize = 4,
    NumSGPRs = 5,
    NumVGPRs = 6,
    MaxFlatWorkGroupSize = 7,
    IsDynamicCallStack = 8,
    IsXNACKEnabled = 9,
    NumSpilledSGPRs = 10,
    NumSpilledVGPRs = 11
};

static const std::map<std::string, ArgField> ArgFieldMap =
    {
        {"Name", ArgField::Name},
        {"TypeName", ArgField::TypeName},
        {"Size", ArgField::Size},
        {"Align", ArgField::Align},
        {"ValueKind", ArgField::ValueKind},
        {"PointeeAlign", ArgField::PointeeAlign},
        {"AddrSpaceQual", ArgField::AddrSpaceQual},
        {"AccQual", ArgField::AccQual},
        {"ActualAccQual", ArgField::ActualAccQual},
        {"IsConst", ArgField::IsConst},
        {"IsRestrict", ArgField::IsRestrict},
        {"IsVolatile", ArgField::IsVolatile},
        {"IsPipe", ArgField::IsPipe}};

static const std::map<std::string, uint32_t> ArgValueKind = {
    {"ByValue", KernelParameterDescriptor::ValueObject},
    {"GlobalBuffer", KernelParameterDescriptor::MemoryObject},
    {"DynamicSharedPointer", KernelParameterDescriptor::MemoryObject},
    {"Sampler", KernelParameterDescriptor::SamplerObject},
    {"Image", KernelParameterDescriptor::ImageObject},
    {"Pipe", KernelParameterDescriptor::MemoryObject},
    {"Queue", KernelParameterDescriptor::QueueObject},
    {"HiddenGlobalOffsetX", KernelParameterDescriptor::HiddenGlobalOffsetX},
    {"HiddenGlobalOffsetY", KernelParameterDescriptor::HiddenGlobalOffsetY},
    {"HiddenGlobalOffsetZ", KernelParameterDescriptor::HiddenGlobalOffsetZ},
    {"HiddenNone", KernelParameterDescriptor::HiddenNone},
    {"HiddenPrintfBuffer", KernelParameterDescriptor::HiddenPrintfBuffer},
    {"HiddenDefaultQueue", KernelParameterDescriptor::HiddenDefaultQueue},
    {"HiddenCompletionAction", KernelParameterDescriptor::HiddenCompletionAction},
    {"HiddenMultigridSyncArg", KernelParameterDescriptor::HiddenMultiGridSync},
    {"HiddenHostcallBuffer", KernelParameterDescriptor::HiddenHostcallBuffer}};

static const std::map<std::string, cl_kernel_arg_access_qualifier> ArgAccQual = {
    {"Default", CL_KERNEL_ARG_ACCESS_NONE},
    {"ReadOnly", CL_KERNEL_ARG_ACCESS_READ_ONLY},
    {"WriteOnly", CL_KERNEL_ARG_ACCESS_WRITE_ONLY},
    {"ReadWrite", CL_KERNEL_ARG_ACCESS_READ_WRITE}};

static const std::map<std::string, cl_kernel_arg_address_qualifier> ArgAddrSpaceQual = {
    {"Private", CL_KERNEL_ARG_ADDRESS_PRIVATE},
    {"Global", CL_KERNEL_ARG_ADDRESS_GLOBAL},
    {"Constant", CL_KERNEL_ARG_ADDRESS_CONSTANT},
    {"Local", CL_KERNEL_ARG_ADDRESS_LOCAL},
    {"Generic", CL_KERNEL_ARG_ADDRESS_GLOBAL},
    {"Region", CL_KERNEL_ARG_ADDRESS_PRIVATE}};

static const std::map<std::string, AttrField> AttrFieldMap =
    {
        {"ReqdWorkGroupSize", AttrField::ReqdWorkGroupSize},
        {"WorkGroupSizeHint", AttrField::WorkGroupSizeHint},
        {"VecTypeHint", AttrField::VecTypeHint},
        {"RuntimeHandle", AttrField::RuntimeHandle}};

static const std::map<std::string, CodePropField> CodePropFieldMap =
    {
        {"KernargSegmentSize", CodePropField::KernargSegmentSize},
        {"GroupSegmentFixedSize", CodePropField::GroupSegmentFixedSize},
        {"PrivateSegmentFixedSize", CodePropField::PrivateSegmentFixedSize},
        {"KernargSegmentAlign", CodePropField::KernargSegmentAlign},
        {"WavefrontSize", CodePropField::WavefrontSize},
        {"NumSGPRs", CodePropField::NumSGPRs},
        {"NumVGPRs", CodePropField::NumVGPRs},
        {"MaxFlatWorkGroupSize", CodePropField::MaxFlatWorkGroupSize},
        {"IsDynamicCallStack", CodePropField::IsDynamicCallStack},
        {"IsXNACKEnabled", CodePropField::IsXNACKEnabled},
        {"NumSpilledSGPRs", CodePropField::NumSpilledSGPRs},
        {"NumSpilledVGPRs", CodePropField::NumSpilledVGPRs}};

//  for Code Object V3
enum class KernelField : uint8_t {
    SymbolName = 0,
    ReqdWorkGroupSize = 1,
    WorkGroupSizeHint = 2,
    VecTypeHint = 3,
    DeviceEnqueueSymbol = 4,
    KernargSegmentSize = 5,
    GroupSegmentFixedSize = 6,
    PrivateSegmentFixedSize = 7,
    KernargSegmentAlign = 8,
    WavefrontSize = 9,
    NumSGPRs = 10,
    NumVGPRs = 11,
    MaxFlatWorkGroupSize = 12,
    NumSpilledSGPRs = 13,
    NumSpilledVGPRs = 14,
    Kind = 15,
    WgpMode = 16
};

static const std::map<std::string, ArgField> ArgFieldMapV3 =
    {
        {".name", ArgField::Name},
        {".type_name", ArgField::TypeName},
        {".size", ArgField::Size},
        {".offset", ArgField::Offset},
        {".value_kind", ArgField::ValueKind},
        {".pointee_align", ArgField::PointeeAlign},
        {".address_space", ArgField::AddrSpaceQual},
        {".access", ArgField::AccQual},
        {".actual_access", ArgField::ActualAccQual},
        {".is_const", ArgField::IsConst},
        {".is_restrict", ArgField::IsRestrict},
        {".is_volatile", ArgField::IsVolatile},
        {".is_pipe", ArgField::IsPipe}};

static const std::map<std::string, uint32_t> ArgValueKindV3 = {
    {"by_value", KernelParameterDescriptor::ValueObject},
    {"global_buffer", KernelParameterDescriptor::MemoryObject},
    {"dynamic_shared_pointer", KernelParameterDescriptor::MemoryObject},
    {"sampler", KernelParameterDescriptor::SamplerObject},
    {"image", KernelParameterDescriptor::ImageObject},
    {"pipe", KernelParameterDescriptor::MemoryObject},
    {"queue", KernelParameterDescriptor::QueueObject},
    {"hidden_global_offset_x", KernelParameterDescriptor::HiddenGlobalOffsetX},
    {"hidden_global_offset_y", KernelParameterDescriptor::HiddenGlobalOffsetY},
    {"hidden_global_offset_z", KernelParameterDescriptor::HiddenGlobalOffsetZ},
    {"hidden_none", KernelParameterDescriptor::HiddenNone},
    {"hidden_printf_buffer", KernelParameterDescriptor::HiddenPrintfBuffer},
    {"hidden_default_queue", KernelParameterDescriptor::HiddenDefaultQueue},
    {"hidden_completion_action", KernelParameterDescriptor::HiddenCompletionAction},
    {"hidden_multigrid_sync_arg", KernelParameterDescriptor::HiddenMultiGridSync},
    {"hidden_heap_v1", KernelParameterDescriptor::HiddenHeap},
    {"hidden_hostcall_buffer", KernelParameterDescriptor::HiddenHostcallBuffer},
    {"hidden_block_count_x", KernelParameterDescriptor::HiddenBlockCountX},
    {"hidden_block_count_y", KernelParameterDescriptor::HiddenBlockCountY},
    {"hidden_block_count_z", KernelParameterDescriptor::HiddenBlockCountZ},
    {"hidden_group_size_x", KernelParameterDescriptor::HiddenGroupSizeX},
    {"hidden_group_size_y", KernelParameterDescriptor::HiddenGroupSizeY},
    {"hidden_group_size_z", KernelParameterDescriptor::HiddenGroupSizeZ},
    {"hidden_remainder_x", KernelParameterDescriptor::HiddenRemainderX},
    {"hidden_remainder_y", KernelParameterDescriptor::HiddenRemainderY},
    {"hidden_remainder_z", KernelParameterDescriptor::HiddenRemainderZ},
    {"hidden_grid_dims", KernelParameterDescriptor::HiddenGridDims},
    {"hidden_private_base", KernelParameterDescriptor::HiddenPrivateBase},
    {"hidden_shared_base", KernelParameterDescriptor::HiddenSharedBase},
    {"hidden_queue_ptr", KernelParameterDescriptor::HiddenQueuePtr}};

static const std::map<std::string, cl_kernel_arg_access_qualifier> ArgAccQualV3 = {
    {"default", CL_KERNEL_ARG_ACCESS_NONE},
    {"read_only", CL_KERNEL_ARG_ACCESS_READ_ONLY},
    {"write_only", CL_KERNEL_ARG_ACCESS_WRITE_ONLY},
    {"read_write", CL_KERNEL_ARG_ACCESS_READ_WRITE}};

static const std::map<std::string, cl_kernel_arg_address_qualifier> ArgAddrSpaceQualV3 = {
    {"private", CL_KERNEL_ARG_ADDRESS_PRIVATE},
    {"global", CL_KERNEL_ARG_ADDRESS_GLOBAL},
    {"constant", CL_KERNEL_ARG_ADDRESS_CONSTANT},
    {"local", CL_KERNEL_ARG_ADDRESS_LOCAL},
    {"generic", CL_KERNEL_ARG_ADDRESS_GLOBAL},
    {"region", CL_KERNEL_ARG_ADDRESS_PRIVATE}};

static const std::map<std::string, KernelField> KernelFieldMapV3 =
    {
        {".symbol", KernelField::SymbolName},
        {".reqd_workgroup_size", KernelField::ReqdWorkGroupSize},
        {".workgroup_size_hint", KernelField::WorkGroupSizeHint},
        {".vec_type_hint", KernelField::VecTypeHint},
        {".device_enqueue_symbol", KernelField::DeviceEnqueueSymbol},
        {".kernarg_segment_size", KernelField::KernargSegmentSize},
        {".group_segment_fixed_size", KernelField::GroupSegmentFixedSize},
        {".private_segment_fixed_size", KernelField::PrivateSegmentFixedSize},
        {".kernarg_segment_align", KernelField::KernargSegmentAlign},
        {".wavefront_size", KernelField::WavefrontSize},
        {".sgpr_count", KernelField::NumSGPRs},
        {".vgpr_count", KernelField::NumVGPRs},
        {".max_flat_workgroup_size", KernelField::MaxFlatWorkGroupSize},
        {".sgpr_spill_count", KernelField::NumSpilledSGPRs},
        {".vgpr_spill_count", KernelField::NumSpilledVGPRs},
        {".kind", KernelField::Kind},
        {".workgroup_processor_mode", KernelField::WgpMode}};

amd_comgr_status_t getMetaBuf(const amd_comgr_metadata_node_t meta,
                              std::string &str) {
    size_t size = 0;
    amd_comgr_status_t status = amd_comgr_get_metadata_string(meta, &size, nullptr);

    if (status == AMD_COMGR_STATUS_SUCCESS) {
        str.resize(size - 1);// minus one to discount the null character
        status = amd_comgr_get_metadata_string(meta, &size, str.data());
    }
    return status;
}

static amd_comgr_status_t populateArgs(const amd_comgr_metadata_node_t key,
                                       const amd_comgr_metadata_node_t value,
                                       void *data) {
    amd_comgr_status_t status;
    amd_comgr_metadata_kind_t kind;
    std::string buf;

    // get the key of the argument field
    size_t size = 0;
    status = amd_comgr_get_metadata_kind(key, &kind);
    if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
        status = getMetaBuf(key, buf);
    }

    if (status != AMD_COMGR_STATUS_SUCCESS) {
        return AMD_COMGR_STATUS_ERROR;
    }

    auto itArgField = ArgFieldMap.find(buf);
    if (itArgField == ArgFieldMap.end()) {
        return AMD_COMGR_STATUS_ERROR;
    }

    // get the value of the argument field
    status = getMetaBuf(value, buf);

    auto *lcArg = static_cast<KernelParameterDescriptor *>(data);

    switch (itArgField->second) {
        case ArgField::Name:
            lcArg->name_ = buf;
            break;
        case ArgField::TypeName:
            lcArg->typeName_ = buf;
            break;
        case ArgField::Size:
            lcArg->size_ = atoi(buf.c_str());
            break;
        case ArgField::Align:
            lcArg->alignment_ = atoi(buf.c_str());
            break;
        case ArgField::ValueKind: {
            auto itValueKind = ArgValueKind.find(buf);
            if (itValueKind == ArgValueKind.end()) {
                lcArg->info_.hidden_ = true;
                return AMD_COMGR_STATUS_ERROR;
            }
            lcArg->info_.oclObject_ = itValueKind->second;
            switch (lcArg->info_.oclObject_) {
                case KernelParameterDescriptor::MemoryObject:
                    if (itValueKind->first.compare("DynamicSharedPointer") == 0) {
                        lcArg->info_.shared_ = true;
                    }
                    break;
                case KernelParameterDescriptor::HiddenGlobalOffsetX:
                case KernelParameterDescriptor::HiddenGlobalOffsetY:
                case KernelParameterDescriptor::HiddenGlobalOffsetZ:
                case KernelParameterDescriptor::HiddenPrintfBuffer:
                case KernelParameterDescriptor::HiddenHostcallBuffer:
                case KernelParameterDescriptor::HiddenDefaultQueue:
                case KernelParameterDescriptor::HiddenCompletionAction:
                case KernelParameterDescriptor::HiddenMultiGridSync:
                case KernelParameterDescriptor::HiddenNone:
                    lcArg->info_.hidden_ = true;
                    break;
            }
        } break;
        case ArgField::PointeeAlign:
            lcArg->info_.arrayIndex_ = atoi(buf.c_str());
            break;
        case ArgField::AddrSpaceQual: {
            auto itAddrSpaceQual = ArgAddrSpaceQual.find(buf);
            if (itAddrSpaceQual == ArgAddrSpaceQual.end()) {
                return AMD_COMGR_STATUS_ERROR;
            }
            lcArg->addressQualifier_ = itAddrSpaceQual->second;
        } break;
        case ArgField::AccQual: {
            auto itAccQual = ArgAccQual.find(buf);
            if (itAccQual == ArgAccQual.end()) {
                return AMD_COMGR_STATUS_ERROR;
            }
            lcArg->accessQualifier_ = itAccQual->second;
            lcArg->info_.readOnly_ =
                (lcArg->accessQualifier_ == CL_KERNEL_ARG_ACCESS_READ_ONLY) ? true : false;
        } break;
        case ArgField::ActualAccQual: {
            auto itAccQual = ArgAccQual.find(buf);
            if (itAccQual == ArgAccQual.end()) {
                return AMD_COMGR_STATUS_ERROR;
            }
            // lcArg->mActualAccQual = itAccQual->second;
        } break;
        case ArgField::IsConst:
            lcArg->typeQualifier_ |= (buf.compare("true") == 0) ? CL_KERNEL_ARG_TYPE_CONST : 0;
            break;
        case ArgField::IsRestrict:
            lcArg->typeQualifier_ |= (buf.compare("true") == 0) ? CL_KERNEL_ARG_TYPE_RESTRICT : 0;
            break;
        case ArgField::IsVolatile:
            lcArg->typeQualifier_ |= (buf.compare("true") == 0) ? CL_KERNEL_ARG_TYPE_VOLATILE : 0;
            break;
        case ArgField::IsPipe:
            lcArg->typeQualifier_ |= (buf.compare("true") == 0) ? CL_KERNEL_ARG_TYPE_PIPE : 0;
            break;
        default:
            return AMD_COMGR_STATUS_ERROR;
    }
    return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t populateAttrs(const amd_comgr_metadata_node_t key,
                                        const amd_comgr_metadata_node_t value,
                                        void *data) {
    amd_comgr_status_t status;
    amd_comgr_metadata_kind_t kind;
    size_t size = 0;
    std::string buf;

    // get the key of the argument field
    status = amd_comgr_get_metadata_kind(key, &kind);
    if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
        status = getMetaBuf(key, buf);
    }

    if (status != AMD_COMGR_STATUS_SUCCESS) {
        return AMD_COMGR_STATUS_ERROR;
    }

    auto itAttrField = AttrFieldMap.find(buf);
    if (itAttrField == AttrFieldMap.end()) {
        return AMD_COMGR_STATUS_ERROR;
    }

    auto kernelMetaData = static_cast<WorkGroupInfo*>(data);
    switch (itAttrField->second) {
        case AttrField::ReqdWorkGroupSize: {
            status = amd_comgr_get_metadata_list_size(value, &size);
            if (size == 3 && status == AMD_COMGR_STATUS_SUCCESS) {
                std::vector<size_t> wrkSize;
                for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
                    amd_comgr_metadata_node_t workgroupSize;
                    status = amd_comgr_index_list_metadata(value, i, &workgroupSize);

                    if (status == AMD_COMGR_STATUS_SUCCESS && getMetaBuf(workgroupSize, buf) == AMD_COMGR_STATUS_SUCCESS) {
                        wrkSize.push_back(atoi(buf.c_str()));
                    }
                    amd_comgr_destroy_metadata(workgroupSize);
                }
                if (!wrkSize.empty()) {
                    kernelMetaData->compileSize_[0] = wrkSize[0];
                    kernelMetaData->compileSize_[1] = wrkSize[1];
                    kernelMetaData->compileSize_[2] = wrkSize[2];
                }
            }
        } break;
        case AttrField::WorkGroupSizeHint: {
            status = amd_comgr_get_metadata_list_size(value, &size);
            if (status == AMD_COMGR_STATUS_SUCCESS && size == 3) {
                std::vector<size_t> hintSize;
                for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
                    amd_comgr_metadata_node_t workgroupSizeHint;
                    status = amd_comgr_index_list_metadata(value, i, &workgroupSizeHint);

                    if (status == AMD_COMGR_STATUS_SUCCESS && getMetaBuf(workgroupSizeHint, buf) == AMD_COMGR_STATUS_SUCCESS) {
                        hintSize.push_back(atoi(buf.c_str()));
                    }
                    amd_comgr_destroy_metadata(workgroupSizeHint);
                }
                if (!hintSize.empty()) {
                    kernelMetaData->compileSizeHint_[0] = hintSize[0];
                    kernelMetaData->compileSizeHint_[1] = hintSize[1];
                    kernelMetaData->compileSizeHint_[2] = hintSize[2];
                }
            }
        } break;
        case AttrField::VecTypeHint:
            if (getMetaBuf(value, buf) == AMD_COMGR_STATUS_SUCCESS) {
                kernelMetaData->compileVecTypeHint_ = buf;
            }
            break;
        case AttrField::RuntimeHandle:
            if (getMetaBuf(value, buf) == AMD_COMGR_STATUS_SUCCESS) {
                kernelMetaData->runtimeHandle_ = buf;
            }
            break;
        default:
            return AMD_COMGR_STATUS_ERROR;
    }

    return status;
}

static amd_comgr_status_t populateCodeProps(const amd_comgr_metadata_node_t key,
                                            const amd_comgr_metadata_node_t value,
                                            void *data) {
    amd_comgr_status_t status;
    amd_comgr_metadata_kind_t kind;
    std::string buf;

    // get the key of the argument field
    status = amd_comgr_get_metadata_kind(key, &kind);
    if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
        status = getMetaBuf(key, buf);
    }

    if (status != AMD_COMGR_STATUS_SUCCESS) {
        return AMD_COMGR_STATUS_ERROR;
    }

    auto itCodePropField = CodePropFieldMap.find(buf);
    if (itCodePropField == CodePropFieldMap.end()) {
        return AMD_COMGR_STATUS_ERROR;
    }

    // get the value of the argument field
    if (status == AMD_COMGR_STATUS_SUCCESS) {
        status = getMetaBuf(value, buf);
    }

    auto kernelMetaData = static_cast<WorkGroupInfo *>(data);
    switch (itCodePropField->second) {
        case CodePropField::KernargSegmentSize:
            kernelMetaData->kernargSegmentByteSize_ = atoi(buf.c_str());
            break;
        case CodePropField::GroupSegmentFixedSize:
            kernelMetaData->workgroupGroupSegmentByteSize_ = atoi(buf.c_str());
            break;
        case CodePropField::PrivateSegmentFixedSize:
            kernelMetaData->workitemPrivateSegmentByteSize_ = atoi(buf.c_str());
            break;
        case CodePropField::KernargSegmentAlign:
            kernelMetaData->kernargSegmentAlignment_ = atoi(buf.c_str());
            break;
        case CodePropField::WavefrontSize:
            kernelMetaData->wavefrontSize_ = atoi(buf.c_str());
            break;
        case CodePropField::NumSGPRs:
            kernelMetaData->usedSGPRs_ = atoi(buf.c_str());
            break;
        case CodePropField::NumVGPRs:
            kernelMetaData->usedVGPRs_ = atoi(buf.c_str());
            break;
        case CodePropField::MaxFlatWorkGroupSize:
            kernelMetaData->size_ = atoi(buf.c_str());
            break;
        case CodePropField::IsDynamicCallStack: {
            kernelMetaData->isDynamicCallStack_ = buf == "true";
        } break;
        case CodePropField::IsXNACKEnabled: {
            kernelMetaData->isXNACKEnabled_ = buf == "true";
        } break;
        case CodePropField::NumSpilledSGPRs: {
            kernelMetaData->numSpilledSGPRs_ = atoi(buf.c_str());
        } break;
        case CodePropField::NumSpilledVGPRs: {
            kernelMetaData->numSpilledVGPRs_ = atoi(buf.c_str());
        } break;
        default:
            return AMD_COMGR_STATUS_ERROR;
    }
    return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t populateArgsV3(const amd_comgr_metadata_node_t key,
                                         const amd_comgr_metadata_node_t value,
                                         void *data) {
    amd_comgr_status_t status;
    amd_comgr_metadata_kind_t kind;
    std::string buf;

    // get the key of the argument field
    size_t size = 0;
    status = amd_comgr_get_metadata_kind(key, &kind);
    if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
        status = getMetaBuf(key, buf);
        fmt::println("KEY FOUND: {}", buf);
    }

    if (status != AMD_COMGR_STATUS_SUCCESS) {
        return AMD_COMGR_STATUS_ERROR;
    }

    auto itArgField = ArgFieldMapV3.find(buf);
    if (itArgField == ArgFieldMapV3.end()) {
        return AMD_COMGR_STATUS_ERROR;
    }

    // get the value of the argument field
    status = getMetaBuf(value, buf);

    auto lcArg = static_cast<KernelParameterDescriptor *>(data);

    switch (itArgField->second) {
        case ArgField::Name:
            lcArg->name_ = buf;
            break;
        case ArgField::TypeName:
            lcArg->typeName_ = buf;
            break;
        case ArgField::Size:
            lcArg->size_ = atoi(buf.c_str());
            break;
        case ArgField::Offset:
            lcArg->offset_ = atoi(buf.c_str());
            break;
        case ArgField::ValueKind: {
            auto itValueKind = ArgValueKindV3.find(buf);
            if (itValueKind == ArgValueKindV3.end()) {
                return AMD_COMGR_STATUS_ERROR;
            }
            lcArg->info_.oclObject_ = itValueKind->second;
            if (lcArg->info_.oclObject_ == KernelParameterDescriptor::MemoryObject) {
                if (itValueKind->first.compare("dynamic_shared_pointer") == 0) {
                    lcArg->info_.shared_ = true;
                }
            } else if ((lcArg->info_.oclObject_ >= KernelParameterDescriptor::HiddenNone) && (lcArg->info_.oclObject_ < KernelParameterDescriptor::HiddenLast)) {
                lcArg->info_.hidden_ = true;
            }
        } break;
        case ArgField::PointeeAlign:
            lcArg->info_.arrayIndex_ = atoi(buf.c_str());
            break;
        case ArgField::AddrSpaceQual: {
            auto itAddrSpaceQual = ArgAddrSpaceQualV3.find(buf);
            if (itAddrSpaceQual == ArgAddrSpaceQualV3.end()) {
                return AMD_COMGR_STATUS_ERROR;
            }
            lcArg->addressQualifier_ = itAddrSpaceQual->second;
        } break;
        case ArgField::AccQual: {
            auto itAccQual = ArgAccQualV3.find(buf);
            if (itAccQual == ArgAccQualV3.end()) {
                return AMD_COMGR_STATUS_ERROR;
            }
            lcArg->accessQualifier_ = itAccQual->second;
            lcArg->info_.readOnly_ =
                (lcArg->accessQualifier_ == CL_KERNEL_ARG_ACCESS_READ_ONLY) ? true : false;
        } break;
        case ArgField::ActualAccQual: {
            auto itAccQual = ArgAccQualV3.find(buf);
            if (itAccQual == ArgAccQualV3.end()) {
                return AMD_COMGR_STATUS_ERROR;
            }
            //lcArg->mActualAccQual = itAccQual->second;
        } break;
        case ArgField::IsConst:
            lcArg->typeQualifier_ |= (buf.compare("1") == 0) ? CL_KERNEL_ARG_TYPE_CONST : 0;
            break;
        case ArgField::IsRestrict:
            lcArg->typeQualifier_ |= (buf.compare("1") == 0) ? CL_KERNEL_ARG_TYPE_RESTRICT : 0;
            break;
        case ArgField::IsVolatile:
            lcArg->typeQualifier_ |= (buf.compare("1") == 0) ? CL_KERNEL_ARG_TYPE_VOLATILE : 0;
            break;
        case ArgField::IsPipe:
            lcArg->typeQualifier_ |= (buf.compare("1") == 0) ? CL_KERNEL_ARG_TYPE_PIPE : 0;
            break;
        default:
            return AMD_COMGR_STATUS_ERROR;
    }
    return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t populateKernelMetaV3(const amd_comgr_metadata_node_t key,
                                               const amd_comgr_metadata_node_t value,
                                               void *data) {
    amd_comgr_status_t status;
    amd_comgr_metadata_kind_t kind;
    size_t size = 0;
    std::string buf;
    // get the key of the argument field
    status = amd_comgr_get_metadata_kind(key, &kind);
    if (kind == AMD_COMGR_METADATA_KIND_STRING && status == AMD_COMGR_STATUS_SUCCESS) {
        status = getMetaBuf(key, buf);
        fmt::println("KEY FOUND: {}", buf);
    }

    if (status != AMD_COMGR_STATUS_SUCCESS) {
        return AMD_COMGR_STATUS_ERROR;
    }

    auto itKernelField = KernelFieldMapV3.find(buf);
    if (itKernelField == KernelFieldMapV3.end()) {
        return AMD_COMGR_STATUS_ERROR;
    }

    if (itKernelField->second != KernelField::ReqdWorkGroupSize && itKernelField->second != KernelField::WorkGroupSizeHint) {
        status = getMetaBuf(value, buf);
    }
    if (status != AMD_COMGR_STATUS_SUCCESS) {
        return AMD_COMGR_STATUS_ERROR;
    }

    auto kernelMetaData = static_cast<WorkGroupInfo *>(data);
    switch (itKernelField->second) {
        case KernelField::ReqdWorkGroupSize:
            status = amd_comgr_get_metadata_list_size(value, &size);
            if (size == 3 && status == AMD_COMGR_STATUS_SUCCESS) {
                std::vector<size_t> wrkSize;
                for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
                    amd_comgr_metadata_node_t workgroupSize;
                    status = amd_comgr_index_list_metadata(value, i, &workgroupSize);

                    if (status == AMD_COMGR_STATUS_SUCCESS && getMetaBuf(workgroupSize, buf) == AMD_COMGR_STATUS_SUCCESS) {
                        wrkSize.push_back(atoi(buf.c_str()));
                    }
                    amd_comgr_destroy_metadata(workgroupSize);
                }
                if (!wrkSize.empty()) {
                    kernelMetaData->compileSize_[0] = wrkSize[0];
                    kernelMetaData->compileSize_[1] = wrkSize[1];
                    kernelMetaData->compileSize_[2] = wrkSize[2];
                }
            }
            break;
        case KernelField::WorkGroupSizeHint:
            status = amd_comgr_get_metadata_list_size(value, &size);
            if (status == AMD_COMGR_STATUS_SUCCESS && size == 3) {
                std::vector<size_t> hintSize;
                for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
                    amd_comgr_metadata_node_t workgroupSizeHint;
                    status = amd_comgr_index_list_metadata(value, i, &workgroupSizeHint);

                    if (status == AMD_COMGR_STATUS_SUCCESS && getMetaBuf(workgroupSizeHint, buf) == AMD_COMGR_STATUS_SUCCESS) {
                        hintSize.push_back(atoi(buf.c_str()));
                    }
                    amd_comgr_destroy_metadata(workgroupSizeHint);
                }
                if (!hintSize.empty()) {
                    kernelMetaData->compileSizeHint_[0] = hintSize[0];
                    kernelMetaData->compileSizeHint_[1] = hintSize[1];
                    kernelMetaData->compileSizeHint_[2] = hintSize[2];
                }
            }
            break;
        case KernelField::VecTypeHint:
            kernelMetaData->compileVecTypeHint_ = buf;
            break;
        case KernelField::DeviceEnqueueSymbol:
            kernelMetaData->runtimeHandle_ = buf;
            break;
        case KernelField::KernargSegmentSize:
            kernelMetaData->kernargSegmentByteSize_ = atoi(buf.c_str());
            break;
        case KernelField::GroupSegmentFixedSize:
            kernelMetaData->workgroupGroupSegmentByteSize_ = atoi(buf.c_str());
            break;
        case KernelField::PrivateSegmentFixedSize:
            kernelMetaData->workitemPrivateSegmentByteSize_ = atoi(buf.c_str());
            break;
        case KernelField::KernargSegmentAlign:
            kernelMetaData->kernargSegmentAlignment_ = atoi(buf.c_str());
            break;
        case KernelField::WavefrontSize:
            kernelMetaData->wavefrontSize_ = atoi(buf.c_str());
            break;
        case KernelField::NumSGPRs:
            kernelMetaData->usedSGPRs_ = atoi(buf.c_str());
            break;
        case KernelField::NumVGPRs:
            kernelMetaData->usedVGPRs_ = atoi(buf.c_str());
            break;
        case KernelField::MaxFlatWorkGroupSize:
            kernelMetaData->size_ = atoi(buf.c_str());
            break;
        case KernelField::NumSpilledSGPRs: {
            size_t mNumSpilledSGPRs = atoi(buf.c_str());
        } break;
        case KernelField::NumSpilledVGPRs: {
            size_t mNumSpilledVGPRs = atoi(buf.c_str());
        } break;
        case KernelField::SymbolName:
            kernelMetaData->symbolName_ = buf;
            break;
        case KernelField::Kind:
            kernelMetaData->SetKernelKind(buf);
            break;
        case KernelField::WgpMode:
            kernelMetaData->isWGPMode_ = buf == "true";
            break;
        default:
            return AMD_COMGR_STATUS_ERROR;
    }

    return status;
}

template <typename T> inline T alignDown(T value, size_t alignment) {
    return (T)(value & ~(alignment - 1));
}

template <typename T> inline T* alignDown(T* value, size_t alignment) {
    return (T*)alignDown((intptr_t)value, alignment);
}


template <typename T> inline T alignUp(T value, size_t alignment) {
    return alignDown((T)(value + alignment - 1), alignment);
}

void InitParameters(const ElfView& elfView, const amd_comgr_metadata_node_t kernelMD, WorkGroupInfo& workGroupInfo) {
    // Iterate through the arguments and insert into parameterList
    size_t offset = 0;

    amd_comgr_metadata_node_t argsMeta;
    bool hsaArgsMeta = false;
    size_t argsSize = 0;

    amd_comgr_status_t status =  amd_comgr_metadata_lookup(
        kernelMD,
        (elfView->getCodeObjectVersion() == 2) ? "Args" : ".args",
        &argsMeta);
    // Assume no arguments if lookup fails.
    if (status == AMD_COMGR_STATUS_SUCCESS) {
        hsaArgsMeta = true;
        status = amd_comgr_get_metadata_list_size(argsMeta, &argsSize);
    }

    for (size_t i = 0; i < argsSize; ++i) {
        KernelParameterDescriptor desc = {};

        amd_comgr_metadata_node_t argsNode;
        amd_comgr_metadata_kind_t kind = AMD_COMGR_METADATA_KIND_NULL;
        bool hsaArgsNode = false;

        status = amd_comgr_index_list_metadata(argsMeta, i, &argsNode);

        if (status == AMD_COMGR_STATUS_SUCCESS) {
            hsaArgsNode = true;
            status = amd_comgr_get_metadata_kind(argsNode, &kind);
        }
        if (kind != AMD_COMGR_METADATA_KIND_MAP) {
            status = AMD_COMGR_STATUS_ERROR;
        }
        if (status == AMD_COMGR_STATUS_SUCCESS) {
            void *data = static_cast<void*>(&desc);
            if (elfView->getCodeObjectVersion() == 2) {
                status = amd_comgr_iterate_map_metadata(argsNode, populateArgs, data);
            }
            else if (elfView->getCodeObjectVersion() >= 3) {
                status = amd_comgr_iterate_map_metadata(argsNode, populateArgsV3, data);
            }
        }

        if (hsaArgsNode) {
            amd_comgr_destroy_metadata(argsNode);
        }

        if (status != AMD_COMGR_STATUS_SUCCESS) {
            if (hsaArgsMeta) {
                amd_comgr_destroy_metadata(argsMeta);
            }
            return;
        }

        // COMGR has unclear/undefined order of the fields filling.
        // Correct the types for the abstraciton layer after all fields are available
        if (desc.info_.oclObject_ != KernelParameterDescriptor::ValueObject) {
            switch (desc.info_.oclObject_) {
                case KernelParameterDescriptor::MemoryObject:
                case KernelParameterDescriptor::ImageObject:
                    desc.type_ = T_POINTER;
                    if (desc.info_.shared_) {
                        if (desc.info_.arrayIndex_ == 0) {
                            fmt::println("Missing DynamicSharedPointer alignment");
                            desc.info_.arrayIndex_ = 128; /* worst case alignment */
                        }
                    } else {
                        desc.info_.arrayIndex_ = 1;
                    }
                    break;
                case KernelParameterDescriptor::SamplerObject:
                    desc.type_ = T_SAMPLER;
                    desc.addressQualifier_ = CL_KERNEL_ARG_ADDRESS_PRIVATE;
                    break;
                case KernelParameterDescriptor::QueueObject:
                    desc.type_ = T_QUEUE;
                    break;
                default:
                    desc.type_ = T_VOID;
                    break;
            }
        }

        // LC doesn't report correct address qualifier for images and pipes,
        // hence overwrite it
        if ((desc.info_.oclObject_ == KernelParameterDescriptor::ImageObject) ||
            (desc.typeQualifier_  & CL_KERNEL_ARG_TYPE_PIPE)) {
            desc.addressQualifier_ = CL_KERNEL_ARG_ADDRESS_GLOBAL;

        }
        size_t size = desc.size_;

        // Allocate the hidden arguments, but abstraction layer will skip them
        if (desc.info_.hidden_) {
            if (desc.info_.oclObject_ == KernelParameterDescriptor::HiddenCompletionAction) {
                workGroupInfo.flags_.dynamicParallelism_ = true;
            }
            if (elfView->getCodeObjectVersion() == 2) {
                desc.offset_ = alignUp(offset, desc.alignment_);
                offset += size;
            }
            workGroupInfo.hiddenParameters_.push_back(desc);
            continue;
        }

        // These objects have forced data size to uint64_t
        if (elfView->getCodeObjectVersion() == 2) {
            if ((desc.info_.oclObject_ == KernelParameterDescriptor::ImageObject) ||
                (desc.info_.oclObject_ == KernelParameterDescriptor::SamplerObject) ||
                (desc.info_.oclObject_ == KernelParameterDescriptor::QueueObject)) {
                offset = alignUp(offset, sizeof(uint64_t));
                desc.offset_ = offset;
                offset += sizeof(uint64_t);
            }
            else {
                offset = alignUp(offset, desc.alignment_);
                desc.offset_ = offset;
                offset += size;
            }
        }

        workGroupInfo.parameters_.push_back(desc);

        if (desc.info_.oclObject_ == KernelParameterDescriptor::ImageObject) {
            workGroupInfo.flags_.imageEna_ = true;
            if (desc.accessQualifier_ != CL_KERNEL_ARG_ACCESS_READ_ONLY) {
                workGroupInfo.flags_.imageWriteEna_ = true;
            }
        }
    }

    if (hsaArgsMeta) {
        amd_comgr_destroy_metadata(argsMeta);
    }
}


WorkGroupInfo GetAttrCodePropMetadata(const ElfView& elfView, amd_comgr_metadata_node_t kernelMetaNode) {
    WorkGroupInfo workGroupInfo;
//     Set the workgroup information for the kernel
//    workGroupInfo.availableLDSSize_ = device().info().localMemSizePerCU_;
    workGroupInfo.availableSGPRs_ = 104;
    workGroupInfo.availableVGPRs_ = 256;

    // extract the attribute metadata if there is any
    amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;

    switch (elfView->getCodeObjectVersion()) {
        case 2: {
            amd_comgr_metadata_node_t symbolName;
            status = amd_comgr_metadata_lookup(kernelMetaNode, "SymbolName", &symbolName);
            if (status == AMD_COMGR_STATUS_SUCCESS) {
                std::string name;
                status = getMetaBuf(symbolName, name);
                amd_comgr_destroy_metadata(symbolName);
                workGroupInfo.symbolName_ = name;
            }

            amd_comgr_metadata_node_t attrMeta;
            if (status == AMD_COMGR_STATUS_SUCCESS) {
                if (amd_comgr_metadata_lookup(kernelMetaNode, "Attrs", &attrMeta) == AMD_COMGR_STATUS_SUCCESS) {
                    status = amd_comgr_iterate_map_metadata(attrMeta, populateAttrs,
                                                              static_cast<void *>(&workGroupInfo));
                    amd_comgr_destroy_metadata(attrMeta);
                }
            }

            // extract the code properties metadata
            amd_comgr_metadata_node_t codePropsMeta;
            if (status == AMD_COMGR_STATUS_SUCCESS) {
                status = amd_comgr_metadata_lookup(kernelMetaNode, "CodeProps", &codePropsMeta);
            }

            if (status == AMD_COMGR_STATUS_SUCCESS) {
                status = amd_comgr_iterate_map_metadata(codePropsMeta, populateCodeProps,
                                                          static_cast<void *>(&workGroupInfo));
                amd_comgr_destroy_metadata(codePropsMeta);
            }
        } break;
        default:
            status = amd_comgr_iterate_map_metadata(kernelMetaNode, populateKernelMetaV3,
                                                      static_cast<void *>(&workGroupInfo));
    }

    assert(status == AMD_COMGR_STATUS_SUCCESS);
    InitParameters(elfView, kernelMetaNode, workGroupInfo);

    return workGroupInfo;
}

amd_comgr_status_t ElfViewImpl::initializeComgrMetaData() const {
    getElfIo();
    auto comgrDataKind = io_->get_type() != ELFIO::ET_DYN ? AMD_COMGR_DATA_KIND_EXECUTABLE : AMD_COMGR_DATA_KIND_RELOCATABLE;
    comgrData_.emplace();
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_create_data(comgrDataKind, &*comgrData_));
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_set_data(*comgrData_, data_.size(),
                                               reinterpret_cast<const char *>(data_.data())));


    amd_comgr_status_t status;

    metadata_.emplace();
    LUTHIER_AMD_COMGR_CHECK(amd_comgr_get_data_metadata(*comgrData_, &*metadata_));

    bool hasKernelMD{false};
    size_t size = 0;

    kernelsMetadata_.emplace();
    status = amd_comgr_metadata_lookup(metadata_.value(), "Kernels", &*kernelsMetadata_);
    if (status == AMD_COMGR_STATUS_SUCCESS) {
        hasKernelMD = true;
        codeObjectVer_ = 2;
    } else {
        amd_comgr_metadata_node_t versionMD, versionNode;
        char major_version, minor_version;

        LUTHIER_AMD_COMGR_CHECK(amd_comgr_metadata_lookup(metadata_.value(), "amdhsa.version", &versionMD));

        status = amd_comgr_index_list_metadata(versionMD, 0, &versionNode);
        if (status != AMD_COMGR_STATUS_SUCCESS) {
            amd_comgr_destroy_metadata(versionMD);
            throw std::runtime_error("Cannot get code object metadata major version node.");
        }

        size = 1;
        status = amd_comgr_get_metadata_string(versionNode, &size, &major_version);
        if (status != AMD_COMGR_STATUS_SUCCESS) {
            amd_comgr_destroy_metadata(versionNode);
            amd_comgr_destroy_metadata(versionMD);
            throw std::runtime_error("Cannot get code object metadata major version.");
        }
        amd_comgr_destroy_metadata(versionNode);

        status = amd_comgr_index_list_metadata(versionMD, 1, &versionNode);
        if (status != AMD_COMGR_STATUS_SUCCESS) {
            amd_comgr_destroy_metadata(versionMD);
            throw std::runtime_error("Cannot get code object metadata minor version node.");
        }

        size = 1;
        status = amd_comgr_get_metadata_string(versionNode, &size, &minor_version);
        if (status != AMD_COMGR_STATUS_SUCCESS) {
            amd_comgr_destroy_metadata(versionNode);
            amd_comgr_destroy_metadata(versionMD);
            throw std::runtime_error("Cannot get code object metadata minor version.");
        }
        amd_comgr_destroy_metadata(versionNode);

        amd_comgr_destroy_metadata(versionMD);

        if (major_version == '1') {
            if (minor_version == '0') {
                codeObjectVer_ = 3;
            } else if (minor_version == '1') {
                codeObjectVer_ = 4;
            } else if (minor_version == '2') {
                codeObjectVer_ = 5;
            } else {
                codeObjectVer_ = 0;
            }
        } else {
            codeObjectVer_ = 0;
        }

        status = amd_comgr_metadata_lookup(metadata_.value(), "amdhsa.kernels", &kernelsMetadata_.value());

        if (status == AMD_COMGR_STATUS_SUCCESS) {
            hasKernelMD = true;
        }
    }

    if (status == AMD_COMGR_STATUS_SUCCESS) {
        status = amd_comgr_get_metadata_list_size(*kernelsMetadata_, &size);
    } else {
        // Assume an empty binary. HIP may have binaries with just global variables
        return AMD_COMGR_STATUS_SUCCESS;
    }

    kernelMetadataMap_ = std::unordered_map<std::string, amd_comgr_metadata_node_t>{};
    for (size_t i = 0; i < size && status == AMD_COMGR_STATUS_SUCCESS; i++) {
        amd_comgr_metadata_node_t nameMeta;
        bool hasNameMeta = false;
        bool hasKernelNode = false;

        amd_comgr_metadata_node_t kernelNode;

        std::string kernelName;
        status = amd_comgr_index_list_metadata(*kernelsMetadata_, i, &kernelNode);

        if (status == AMD_COMGR_STATUS_SUCCESS) {
            hasKernelNode = true;
            status = amd_comgr_metadata_lookup(kernelNode,
                                               (codeObjectVer_ == 2) ? "Name" : ".name",
                                               &nameMeta);
        }

        if (status == AMD_COMGR_STATUS_SUCCESS) {
            hasNameMeta = true;
            status = getMetaBuf(nameMeta, kernelName);
        }

        if (status == AMD_COMGR_STATUS_SUCCESS) {
            (*kernelMetadataMap_)[kernelName] = kernelNode;
        } else {
            if (hasKernelNode) {
                amd_comgr_destroy_metadata(kernelNode);
            }
            for (auto const &kernelMeta: *kernelMetadataMap_) {
                amd_comgr_destroy_metadata(kernelMeta.second);
            }
            kernelMetadataMap_->clear();
        }

        if (hasNameMeta) {
            amd_comgr_destroy_metadata(nameMeta);
        }
    }

    return AMD_COMGR_STATUS_SUCCESS;
}

}// namespace luthier::co_manip
