/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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

#ifndef AMDGPU_CODE_OBJECT_MANIPULATION
#define AMDGPU_CODE_OBJECT_MANIPULATION

#include "luthier_types.hpp"
#include <amd_comgr/amd_comgr.h>
#include <elfio/elfio.hpp>
#include <map>

namespace luthier::co_manip {

// Taken from the hipamd project
struct CudaFatBinaryWrapper {
    unsigned int magic;
    unsigned int version;
    void *binary;
    void *dummy1;
};

// Taken from the hipamd project
constexpr unsigned hipFatMAGIC2 = 0x48495046;

typedef struct {
    luthier_address_t data;
    size_t size;
} code_object_region_t;

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

struct SymbolInfo {
    std::string sec_name;//!   section name
    const char *sec_addr;//!   section address
    uint64_t sec_size;   //!   section size
    std::string sym_name;//!   symbol name
    const char *address; //!   address of corresponding to symbol data
    uint64_t size;       //!   size of data corresponding to symbol
    size_t value;        //!   value of the symbol
    SymbolInfo() : sec_name(), sec_addr(nullptr), sec_size(0), sym_name(), address(nullptr), size(0), value(0) {}

    SymbolInfo(const char *sename, const char *seaddr, uint64_t sesize, const char *syname,
               const char *syaddr, uint64_t sysize, size_t syvalue) : sec_name(sename), sec_addr(seaddr),
                                                                      sec_size(sesize), sym_name(syname), address(syaddr), size(sysize), value(syvalue) {}
};

unsigned int getSymbolNum(const ELFIO::elfio &io);

/* Return SymbolInfo of the index-th symbol in SYMTAB section */
bool getSymbolInfo(const ELFIO::elfio &io, unsigned int index, SymbolInfo &symInfo);


std::string getDemangledName(const char *mangledName);

amd_comgr_status_t getCodeObjectElfsFromFatBinary(const void *data, std::vector<ELFIO::elfio>& fatBinaryElfs);

code_object_region_t getFunctionFromSymbol(ELFIO::elfio &elfio, const std::string &functionName);

std::vector<luthier::co_manip::code_object_region_t> getDeviceLoadedCodeObjectOfExecutable(hsa_executable_t executable, hsa_agent_t agent);

std::vector<luthier::co_manip::code_object_region_t> getHostLoadedCodeObjectOfExecutable(hsa_executable_t executable, hsa_agent_t agent);

}// namespace luthier::elf

#endif
