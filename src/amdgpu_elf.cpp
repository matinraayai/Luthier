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

#include "amdgpu_elf.hpp"
#include "error_and_log.hpp"
#include "log.hpp"

#include <string>

#include <thread>

namespace sibir::elf {
using namespace ELFIO;

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
        SibirErrorFmt(" failed: wrong index {} >= symbols num {}", index, num);
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
        SibirErrorFmt("failed to get_symbol({})", index);
        return false;
    }
    section *sec = io.sections[sec_index];
    if (sec == nullptr) {
        SibirErrorFmt("failed: null section at {}", sec_index);
        return false;
    }

    symInfo.sec_addr = sec->get_data();
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
    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_kind(key, &kind));
    if (kind != AMD_COMGR_METADATA_KIND_STRING)
        return AMD_COMGR_STATUS_ERROR;
    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(key, &size, NULL));
    keybuf = (char *)calloc(size, sizeof(char));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(key, &size, keybuf));

    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_kind(value, &kind));
    for (int i = 0; i < *indent; i++)
        printf("  ");

    switch (kind) {
        case AMD_COMGR_METADATA_KIND_STRING: {
            printf("%s  :  ", size ? keybuf : "");
            SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(value, &size, NULL));
            valbuf = (char *)calloc(size, sizeof(char));
            SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_string(value, &size, valbuf));
            printf(" %s\n", valbuf);
            free(valbuf);
            break;
        }
        case AMD_COMGR_METADATA_KIND_LIST: {
            *indent += 1;
            SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_list_size(value, &size));
            printf("LIST %s %zd entries = \n", keybuf, size);
            for (size_t i = 0; i < size; i++) {
                SIBIR_AMD_COMGR_CHECK(amd_comgr_index_list_metadata(value, i, &son));
                SIBIR_AMD_COMGR_CHECK(printEntryCo(key, son, data));
                SIBIR_AMD_COMGR_CHECK(amd_comgr_destroy_metadata(son));
            }
            *indent = *indent > 0 ? *indent - 1 : 0;
            break;
        }
        case AMD_COMGR_METADATA_KIND_MAP: {
            *indent += 1;
            SIBIR_AMD_COMGR_CHECK(amd_comgr_get_metadata_map_size(value, &size));
            printf("MAP %zd entries = \n", size);
            SIBIR_AMD_COMGR_CHECK(amd_comgr_iterate_map_metadata(value, printEntryCo, data));
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


void iterateCodeObjectMetaData(sibir_address_t codeObjectData, size_t codeObjectSize) {
    // COMGR symbol iteration things
    amd_comgr_data_t coData;
    SIBIR_AMD_COMGR_CHECK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &coData));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data(coData, codeObjectSize, reinterpret_cast<const char*>(codeObjectData)));
    amd_comgr_metadata_node_t meta;
    SIBIR_AMD_COMGR_CHECK(amd_comgr_get_data_metadata(coData, &meta));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_set_data_name(coData, "my-data.s"));
    int Indent = 1;
    SIBIR_AMD_COMGR_CHECK(amd_comgr_iterate_map_metadata(meta, printEntryCo, (void *)&Indent));
    SIBIR_AMD_COMGR_CHECK(amd_comgr_destroy_metadata(meta));
}

}// namespace sibir::elf
