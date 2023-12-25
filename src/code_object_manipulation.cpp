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
#include "log.hpp"
#include "hsa_isa.hpp"

#include <string>

namespace luthier::co_manip {
using namespace ELFIO;


#if !defined(ELFMAG)
#define ELFMAG "\177ELF"
#define SELFMAG 4
#endif

void setupSHdr(
    ELFIO::elfio &elfIo,
    luthier::code::ElfSections id,
    section *section,
    Elf64_Word shlink = 0) {
    section->set_addr_align(luthier::code::ElfSecDesc[id].d_align);
    section->set_type(luthier::code::ElfSecDesc[id].sh_type);
    section->set_flags(luthier::code::ElfSecDesc[id].sh_flags);
    section->set_link(shlink);

    auto class_num = elfIo.get_class();
    size_t entry_size = 0;
    switch (id) {
        case luthier::code::SYMTAB:
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
    luthier::code::ElfSections id,
    luthier::byte_string_view data) {
    assert(luthier::code::ElfSecDesc[id].id == id && "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");

    section *sec = elfIo.sections[luthier::code::ElfSecDesc[id].name];
    if (sec == nullptr) {
        sec = elfIo.sections.add(luthier::code::ElfSecDesc[id].name);
    }
    if (sec == nullptr) {
        LuthierErrorFmt("failed: sections.add({:s}) = nullptr", luthier::code::ElfSecDesc[id].name);
        return sec;
    }

    sec->set_data(reinterpret_cast<const char *>(data.data()), data.size());

    auto shlink = (id == luthier::code::SYMTAB) ? elfIo.sections[luthier::code::ElfSecDesc[luthier::code::ElfSections::SYMTAB].name]->get_index() : 0;

    setupSHdr(elfIo, id, sec, shlink);

    return sec;
}

bool addSectionData(
    ELFIO::elfio &elfIo,
    Elf_Xword &outOffset,
    luthier::code::ElfSections id,
    luthier::byte_string_view data) {
    assert(luthier::code::ElfSecDesc[id].id == id && "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");

    outOffset = 0;
    section *sec = elfIo.sections[luthier::code::ElfSecDesc[id].name];
    assert(sec != nullptr);

    outOffset = sec->get_size();

    sec->append_data(reinterpret_cast<const char *>(data.data()), data.size());

    return true;
}

bool addSection(
    ELFIO::elfio &elfIo,
    luthier::code::ElfSections id,
    luthier::byte_string_view data) {
    assert(luthier::code::ElfSecDesc[id].id == id && "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");

    section *sec = elfIo.sections[luthier::code::ElfSecDesc[id].name];

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
    luthier::code::ElfSections id,
    const char *symbolName,
    luthier::byte_string_view data) {
    assert(luthier::code::ElfSecDesc[id].id == id && "The order of ElfSecDesc[] and Elf::ElfSections mismatches.");

    section *symTabSection = elfIo.sections[luthier::code::ElfSecDesc[luthier::code::SYMTAB].name];
    assert(symTabSection != nullptr);

    const char *sectionName = luthier::code::ElfSecDesc[id].name;

    bool isFunction = ((id == luthier::code::CAL) || (id == luthier::code::DLL) || (id == luthier::code::JITBINARY) || (id == luthier::code::TEXT));

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
    addSectionData(elfIo, strtab_offset, luthier::code::STRTAB, {reinterpret_cast<const std::byte *>(symbolName), strlen(symbolName) + 1});

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

ELFIO::elfio createAMDGPUElf(const ELFIO::elfio &elfIoIn, const hsa::GpuAgent &agent) {
    // Create relocatable stored in a code_t
    const std::string code = "s_nop 0";
    std::string isaName = agent.getIsa()[0].getName();

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

    luthier::byte_string_t relocElf;
    size_t relocOutSize;

    amd_comgr_get_data(relocOut, &relocOutSize, nullptr);
    relocElf.resize(relocOutSize);
    amd_comgr_get_data(relocOut, &relocOutSize, reinterpret_cast<char *>(relocElf.data()));

    // Load elfio from code_t
    ELFIO::elfio elfIo;

    luthier::byte_char_stream_t relocElfStream = luthier::code::byteCharStream(relocElf);

    elfIo.load(relocElfStream, false);
    elfIo.set_os_abi(elfIoIn.get_os_abi());
    elfIo.set_flags(elfIoIn.get_flags());
    elfIo.set_abi_version(elfIoIn.get_abi_version());
    elfIo.set_entry(elfIoIn.get_entry());
    elfIo.set_machine(elfIoIn.get_machine());
    elfIo.set_type(ELFIO::ET_REL);

    // Create empty ELF sections in elfIo
    std::vector<ELFIO::section *> relocElfSections;
    // fmt::println("Initial ELF has {} Sections", elfIo.sections.size());
    for (int i = 0; i < elfIo.sections.size(); i++) {
        const ELFIO::section *psec = elfIo.sections[i];
        // fmt::println(" [{}] {}\t{}", i, psec->get_name(), psec->get_size());
        relocElfSections.emplace_back(elfIo.sections[i]);
    }
    // fmt::println("\nLine {}: Create sections for relocatable", __LINE__);
    // fmt::println("\nOriginal ELF has {} sections", elfIoIn.sections.size());
    for (int i = 0; i < elfIoIn.sections.size(); i++) {
        const ELFIO::section *psec = elfIoIn.sections[i];

        bool section_in_reloc = false;
        // fmt::println(" [{}] {}\t{}", i, psec->get_name(), psec->get_size());
        for (auto reloc_sec: relocElfSections) {
            if (!psec->get_name().compare(reloc_sec->get_name())) {
                section_in_reloc = true;
                break;
            }
        }
        if (!section_in_reloc)
            elfIo.sections.add(psec->get_name());
    }//fmt::print("\n");
    // fmt::println("Final ELF has {} Sections", elfIo.sections.size());
    // for (int i = 0; i < elfIo.sections.size(); i++) {
    //     const ELFIO::section* psec = elfIo.sections[i];
    //     fmt::println(" [{}] {}\t{}", i, psec->get_name(), psec->get_size());
    // }
    return elfIo;
}



}// namespace luthier::co_manip
