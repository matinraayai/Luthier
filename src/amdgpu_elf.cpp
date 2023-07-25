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
#include "error_check.hpp"

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

/////////////////////////////////////////////////////////////////
//////////////////////// elf initializers ///////////////////////
/////////////////////////////////////////////////////////////////
//
//ElfAmd::ElfAmd(
//    unsigned char eclass,
//    const char*   rawElfBytes,
//    uint64_t      rawElfSize,
//    const char*   elfFileName,
//    ElfCmd        elfcmd
//    )
//: fname_(elfFileName ? elfFileName : ""),
//      eclass_(eclass),
//      rawElfBytes_(rawElfBytes),
//      rawElfSize_(rawElfSize),
//  _elfCmd (elfcmd),
//      elfMemory_(),
//  _shstrtab_ndx (SHN_UNDEF),
//  _strtab_ndx (SHN_UNDEF),
//  _symtab_ndx (SHN_UNDEF),
//      successful_(false)
//{
//  SibirInfoFmt("fname=%s, rawElfSize=%lu, elfcmd=%d, %s",
//               fname_.c_str(), rawElfSize_, _elfCmd, _elfCmd == ELF_C_WRITE ? "writer" : "reader");
//
//  if (rawElfBytes != nullptr) {
//    /*
//       In general, 'eclass' should be the same as rawElfBytes's. 'eclass' is what the runtime
//       will use for generating an ELF, and therefore it expects the input ELF to have this 'eclass'.
//       However, GPU needs to accept both 32-bit and 64-bit ELF for compatibility (we used to
//       generate 64-bit ELF, which is the bad design in the first place). Here we just uses eclass
//       from rawElfBytes, and overrides the input 'eclass'.
//       */
//    eclass_ = (unsigned char)rawElfBytes[EI_CLASS];
//  }
//  (void)Init();
//}
//
//ElfAmd::~ElfAmd()
//{
//  SibirInfoFmt("fname=%s, rawElfSize=%lu, elfcmd=%d",
//               fname_.c_str(), rawElfSize_, _elfCmd);
//  elfMemoryRelease();
//}
//
///*
// Initialize Elf object
// */
//bool ElfAmd::Init()
//{
//  successful_ = false;
//
//  switch (_elfCmd) {
//    case ELF_C_WRITE:
//        elfio_.create(eclass_, ELFDATA2LSB);
//      break;
//
//    case ELF_C_READ:
//      if(rawElfBytes_ == nullptr || rawElfSize_ == 0) {
//        SibirErrorMsg("failed: _rawElfBytes = nullptr or _rawElfSize = 0");
//        return false;
//      }
//      {
//        std::istringstream is { std::string(rawElfBytes_, rawElfSize_) };
//        if (!elfio_.load(is)) {
//            SibirErrorFmt("failed in _elfio.load(%p, %lu)", rawElfBytes_, rawElfSize_);
//          return false;
//        }
//      }
//      break;
//
//    default:
//      SibirErrorFmt("failed: unexpected cmd %d", _elfCmd);
//      return false; // Don't support other mode
//  }
//
//  if (!InitElf()) {
//    return false;
//  }
//
//  // Success
//  successful_ = true;
//
//  return true;
//}
//
//bool ElfAmd::InitElf ()
//{
//  if (_elfCmd == ELF_C_READ) {
//    assert(elfio_.sections.size() > 0 && "elfio object should have been created already");
//
//    // Set up _shstrtab_ndx
//    _shstrtab_ndx = elfio_.get_section_name_str_index();
//    if(_shstrtab_ndx == SHN_UNDEF) {
//      SibirErrorMsg("failed: _shstrtab_ndx = SHN_UNDEF");
//      return false;
//    }
//
//    // Set up _strtab_ndx
//    section* strtab_sec = elfio_.sections[ElfSecDesc[STRTAB].name];
//    if (strtab_sec == nullptr) {
//      SibirErrorMsg("failed: null sections(STRTAB)");
//      return false;
//    }
//
//    _strtab_ndx = strtab_sec->get_index();
//
//    section* symtab_sec = elfio_.sections[ElfSecDesc[SYMTAB].name];
//
//    if (symtab_sec != nullptr) {
//      _symtab_ndx = symtab_sec->get_index();
//    }
//    // It's ok for empty SYMTAB
//  } else if(_elfCmd == ELF_C_WRITE) {
//    /*********************************/
//    /******** ELF_C_WRITE ************/
//    /*********************************/
//
//    //
//    // 1. Create ELF header
//    //
//    elfio_.create(eclass_, ELFDATA2LSB);
//
//    //
//    // 2. Check created ELF shstrtab
//    //
//    section* shstrtab_sec = elfio_.sections[ElfSecDesc[SHSTRTAB].name];
//    if (shstrtab_sec == nullptr) {
//      SibirErrorMsg("failed: shstrtab_sec = nullptr");
//      return false;
//    }
//
//    if (!setupShdr(SHSTRTAB, shstrtab_sec)) {
//      return false;
//    }
//
//    // Save shstrtab section index
//    _shstrtab_ndx = shstrtab_sec->get_index();
//
//    //
//    // 3. Create .strtab section
//    //
//    auto *strtab_sec = elfio_.sections.add(ElfSecDesc[STRTAB].name);
//    if (strtab_sec == nullptr) {
//      SibirErrorMsg("failed to add section STRTAB");
//      return false;
//    }
//
//    // adding null string data associated with section
//    // index 0 is reserved and must be there (NULL name)
//    constexpr char strtab[] = {
//      /* index 0 */ '\0'
//    };
//    strtab_sec->set_data(const_cast<char*>(strtab), sizeof(strtab));
//
//    if (!setupShdr(STRTAB, strtab_sec)) {
//      return false;
//    }
//
//    // Save strtab section index
//    _strtab_ndx = strtab_sec->get_index();
//
//    //
//    // 4. Create the symbol table
//    //
//
//    // Create the first reserved dummy symbol (undefined symbol)
//    size_t sym_sz = (eclass_ == ELFCLASS32) ? sizeof(Elf32_Sym) : sizeof(Elf64_Sym);
//    char* sym = static_cast<char *>(::calloc(1, sym_sz));
//    if (sym == nullptr) {
//      SibirErrorMsg("failed to calloc memory for SYMTAB section");
//      return false;
//    }
//
//    auto* symtab_sec = newSection(SYMTAB, sym, sym_sz);
//    free(sym);
//
//    if (symtab_sec == nullptr) {
//      SibirErrorMsg("failed to create SYMTAB");
//      return false;
//    }
//
//    _symtab_ndx = symtab_sec->get_index();
//  } else {
//    SibirErrorFmt("failed: wrong cmd %d", _elfCmd);
//    return false;
//  }
//
//  SibirInfoFmt("succeeded: secs=%d, segs=%d, _shstrtab_ndx=%u, _strtab_ndx=%u, _symtab_ndx=%u",
//             elfio_.sections.size(), elfio_.segments.size(), _shstrtab_ndx, _strtab_ndx, _symtab_ndx);
//  return true;
//}
//
//bool ElfAmd::createElfData(
//    section*&   sec,
//    ElfSections id,
//    const char* d_buf,
//    size_t      d_size
//    )
//{
//  assert((ElfSecDesc[id].id == id) &&
//      "ElfSecDesc[] should be in the same order as enum ElfSections");
//
//  sec = elfio_.sections[ElfSecDesc[id].name];
//  if (sec == nullptr) {
//    SibirErrorFmt("failed: null sections(%s)", ElfSecDesc[id].name);
//    return false;
//  }
//
//  sec->set_data(d_buf, d_size);
//  return true;
//}
//
//bool ElfAmd::setupShdr (
//    ElfSections id,
//    section* section,
//    Elf64_Word shlink
//    ) const
//{
//  section->set_addr_align(ElfSecDesc[id].d_align);
//  section->set_type(ElfSecDesc[id].sh_type);
//  section->set_flags(ElfSecDesc[id].sh_flags);
//  section->set_link(shlink);
//
//  auto class_num = elfio_.get_class();
//  size_t entry_size = 0;
//  switch(id) {
//    case SYMTAB:
//      if (class_num == ELFCLASS32) {
//        entry_size = sizeof(Elf32_Sym);
//      }
//      else {
//        entry_size = sizeof(Elf64_Sym);
//      }
//      break;
//    default:
//      // .dynsym and .relaNAME also have table entries
//      break;
//  }
//  if(entry_size > 0) {
//    section->set_entry_size(entry_size);
//  }
//  return true;
//}
//
//bool ElfAmd::getTarget(uint16_t& machine, ElfPlatform& platform) const
//{
//  Elf64_Half mach = elfio_.get_machine();
//  if ((mach >= CPU_FIRST) && (mach <= CPU_LAST)) {
//    platform = CPU_PLATFORM;
//    machine = mach - CPU_BASE;
//  }
//  else if (mach == EM_386
//      || mach == EM_HSAIL
//      || mach == EM_HSAIL_64
//      || mach == EM_AMDIL
//      || mach == EM_AMDIL_64
//      || mach == EM_X86_64) {
//    platform = COMPLIB_PLATFORM;
//    machine = mach;
//  } else {
//    // Invalid machine
//    SibirErrorFmt("failed: Invalid machine=0x%04x(%d)", mach, mach);
//    return false;
//  }
//  SibirInfoFmt("succeeded: machine=0x%04x, platform=%d", machine, platform);
//  return true;
//}
//
//bool ElfAmd::setTarget(uint16_t machine, ElfPlatform platform)
//{
//  Elf64_Half mach;
//  if (platform == CPU_PLATFORM)
//    mach = machine + CPU_BASE;
//  else if (platform == CAL_PLATFORM)
//    mach = machine + CAL_BASE;
//  else
//    mach = machine;
//
//  elfio_.set_machine(mach);
//  SibirInfoFmt("succeeded: machine=0x%04x(%d), platform=%d", machine, machine, platform);
//
//  return true;
//}
//
//bool ElfAmd::getType(uint16_t &type) const {
//  type = elfio_.get_type();
//  return true;
//}
//
//bool ElfAmd::setType(uint16_t  type) {
//  elfio_.set_type(type);
//  return true;
//}
//
//bool ElfAmd::getFlags(uint32_t &flag) const {
//  flag = elfio_.get_flags();
//  return true;
//}
//
//bool ElfAmd::setFlags(uint32_t  flag) {
//  elfio_.set_flags(flag);
//  return true;
//}
//
//bool ElfAmd::getSection(ElfAmd::ElfSections id, char** dst, size_t* sz) const
//{
//  assert((ElfSecDesc[id].id == id) &&
//      "ElfSecDesc[] should be in the same order as enum ElfSections");
//
//  section* sec = elfio_.sections[ElfSecDesc[id].name];
//  if (sec == nullptr) {
//    SibirErrorFmt("failed: null sections(%s)", ElfSecDesc[id].name);
//    return false;
//  }
//
//  // There is only one data descriptor (we are reading!)
//  *dst = const_cast<char*>(sec->get_data());
//  *sz = sec->get_size();
//
//  SibirInfoFmt("succeeded: *dst=%p, *sz=%zu", *dst, *sz);
//  return true;
//}
//
unsigned int getSymbolNum(const elfio &io) {
    symbol_section_accessor symbol_reader(io, io.sections[ElfSecDesc[SYMTAB].name]);
    auto num = symbol_reader.get_symbols_num() - 1;// Exclude the first dummy symbol
    return num;
}
//
//unsigned int ElfAmd::getSegmentNum() const {
//  return elfio_.segments.size();
//}
//
//bool ElfAmd::getSegment(const unsigned int index, segment*& seg) const {
//  bool ret = false;
//  if (index < elfio_.segments.size()) {
//    seg = elfio_.segments[index];
//    ret = true;
//  }
//  return ret;
//}

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

//bool ElfAmd::addSectionData (
//    Elf_Xword&  outOffset,
//    ElfSections id,
//    const void* buffer,
//    size_t      size
//    )
//{
//  assert(ElfSecDesc[id].id == id &&
//      "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");
//
//  outOffset = 0;
//  section* sec = elfio_.sections[ElfSecDesc[id].name];
//  if (sec == nullptr) {
//    SibirErrorFmt("failed: null sections(%s)", ElfSecDesc[id].name);
//    return false;
//  }
//
//  outOffset = sec->get_size();
//
//  sec->append_data(static_cast<const char *>(buffer), size);
//  SibirInfoFmt("succeeded: buffer=%p, size=%zu", buffer, size);
//
//  return true;
//}
//
//bool ElfAmd::getShstrtabNdx(Elf64_Word& outNdx, const char* name)
//{
//  outNdx = 0;
//  auto *section = elfio_.sections[name];
//  if (section == nullptr) {
//    SibirErrorFmt("failed: sections[%s] = nullptr", name);
//    return false;
//  }
//
//  // .shstrtab must be created already
//  auto idx = section->get_name_string_offset();
//
//  if (idx <= 0) {
//    SibirErrorFmt("failed: idx=%d", idx);
//    return false;
//  }
//  outNdx = idx;
//  SibirDebug("Succeeded: name=%s, idx=%d", name, idx);
//  return true;
//}
//
//section*ElfAmd::newSection (
//    ElfAmd::ElfSections id,
//    const char*            d_buf,
//    size_t                 d_size
//    )
//{
//  assert(ElfSecDesc[id].id == id &&
//      "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");
//
//  section* sec = elfio_.sections[ElfSecDesc[id].name];
//  if (sec == nullptr) {
//    sec = elfio_.sections.add(ElfSecDesc[id].name);
//  }
//  if (sec == nullptr) {
//    SibirErrorFmt("failed: sections.add(%s) = nullptr", ElfSecDesc[id].name);
//    return sec;
//  }
//
//  if (d_buf != nullptr && d_size > 0) {
//    sec->set_data(d_buf, d_size);
//  }
//
//  if (!setupShdr(id, sec, (id == SYMTAB) ? _strtab_ndx : 0)) {
//    return nullptr;
//  }
//
//  SibirDebug("succeeded: name=%s, d_buf=%p, d_size=%zu",
//              ElfSecDesc[id].name, d_buf, d_size);
//  return sec;
//}
//
//bool ElfAmd::addSection (
//    ElfSections id,
//    const void*    d_buf,
//    size_t         d_size
//    )
//{
//  assert(ElfSecDesc[id].id == id &&
//      "struct ElfSecDesc should be ordered by id same as enum Elf::ElfSections");
//
//  section* sec = elfio_.sections[ElfSecDesc[id].name];
//
//  if (sec != nullptr) {
//    Elf_Xword sec_offset = 0;
//    if (!addSectionData(sec_offset, id, d_buf, d_size)) {
//      SibirErrorFmt("failed in addSectionData(name=%s, d_buf=%p, d_size=%zu)",
//                  ElfSecDesc[id].name, d_buf, d_size);
//      return false;
//    }
//  }
//  else {
//    sec = newSection(id, static_cast<const char*>(d_buf), d_size);
//    if (sec == nullptr) {
//      SibirErrorFmt("failed in newSection(name=%s, d_buf=%p, d_size=%zu)",
//                  ElfSecDesc[id].name, d_buf, d_size);
//      return false;
//    }
//  }
//
//  SibirDebug("succeeded: name=%s, d_buf=%p, d_size=%zu", ElfSecDesc[id].name, d_buf, d_size);
//  return true;
//}
//
//bool ElfAmd::addSymbol(
//    ElfSections id,
//    const char* symbolName,
//    const void* buffer,
//    size_t size
//    )
//{
//  assert(ElfSecDesc[id].id == id &&
//      "The order of ElfSecDesc[] and Elf::ElfSections mismatches.");
//
//  if(_symtab_ndx == SHN_UNDEF) {
//    SibirErrorMsg("failed: _symtab_ndx = SHN_UNDEF");
//    return false; // No SYMTAB
//  }
//
//  const char* sectionName = ElfSecDesc[id].name;
//
//  bool isFunction = ((id == ElfAmd::CAL) || (id == ElfAmd::DLL) || (id == ElfAmd::JITBINARY)) ? true : false;
//
//  // Get section index
//  section* sec = elfio_.sections[sectionName];
//  if (sec == nullptr) {
//    // Create a new section.
//    if ((sec = newSection(id, nullptr, 0)) == NULL) {
//      SibirErrorFmt("failed in newSection(name=%s)", sectionName);
//      return false;
//    }
//  }
//  size_t sec_ndx = sec->get_index();
//  if (sec_ndx == SHN_UNDEF) {
//    SibirErrorMsg("failed: sec->get_index() = SHN_UNDEF");
//    return false;
//  }
//
//  // Put symbolName into .strtab section
//  Elf_Xword strtab_offset = 0;
//  if (!addSectionData(strtab_offset, STRTAB, symbolName,
//        strlen(symbolName)+1)) {
//    SibirErrorFmt("failed in addSectionData(name=%s, symbolName=%s, length=%zu)",
//                ElfSecDesc[STRTAB].name, symbolName, strlen(symbolName)+1);
//    return false;
//  }
//
//  // Put buffer into section
//  Elf_Xword sec_offset = 0;
//  if ( (buffer != nullptr) && (size != 0) ) {
//    if (!addSectionData(sec_offset, id, buffer, size)) {
//      SibirErrorFmt("failed in addSectionData(name=%s, buffer=%p, size=%zu)",
//                  sectionName, buffer, size);
//      return false;
//    }
//  }
//
//  symbol_section_accessor symbol_writter(elfio_, elfio_.sections[_symtab_ndx]);
//
//  auto ret = symbol_writter.add_symbol(strtab_offset, sec_offset, size, 0,
//                     (isFunction)? STT_FUNC : STT_OBJECT, 0, sec_ndx);
//
//  SibirDebug("%s: sectionName=%s symbolName=%s strtab_offset=%lu, sec_offset=%lu, "
//      "size=%zu, sec_ndx=%zu, ret=%d", ret >= 1 ? "succeeded" : "failed",
//          sectionName, symbolName, strtab_offset, sec_offset, size, sec_ndx, ret);
//  return ret >= 1;
//}
//
//bool ElfAmd::getSymbol(
//    ElfSections id,
//    const char* symbolName,
//    char** buffer,
//    size_t* size
//    ) const
//{
//  assert(ElfSecDesc[id].id == id &&
//      "The order of ElfSecDesc[] and Elf::ElfSections mismatches.");
//
//  if (!size || !buffer || !symbolName) {
//    SibirErrorMsg("failed: invalid parameters");
//    return false;
//  }
//  if (_symtab_ndx == SHN_UNDEF) {
//    SibirErrorMsg("failed: _symtab_ndx = SHN_UNDEF");
//    return false; // No SYMTAB
//  }
//
//  *size = 0;
//  *buffer = nullptr;
//  symbol_section_accessor symbol_reader(elfio_, elfio_.sections[_symtab_ndx]);
//
//  Elf64_Addr value = 0;
//  Elf_Xword  size0 = 0;
//  unsigned char bind = 0;
//  unsigned char type = 0;
//  unsigned char other = 0;
//  Elf_Half sec_ndx = SHN_UNDEF;
//
//  // Search by symbolName, sectionName
//  bool ret = symbol_reader.get_symbol(symbolName, value, size0,
//                    bind, type, sec_ndx, other);
//
//  if (ret) {
//    *buffer = const_cast<char*>(elfio_.sections[sec_ndx]->get_data() + value);
//    *size = static_cast<size_t>(size0);
//  }
//#if 0
//  // For debug purpose
//  LogElfDebug("%s: sectionName=%s symbolName=%s value=%lu, buffer=%p, size=%zu, sec_ndx=%u",
//              ret ? "succeeded" : "failed",
//              ElfSecDesc[id].name, symbolName, value, *buffer, *size, sec_ndx);
//#endif
//  return ret;
//}
//
//bool ElfAmd::addNote(
//    const char* noteName,
//    const char* noteDesc,
//    size_t descSize
//    )
//{
//  if (descSize == 0
//      || noteName == nullptr
//      || (descSize != 0 && noteDesc == nullptr)) {
//    SibirErrorMsg("failed: empty note");
//    return false;
//  }
//
//  // Get section
//  section* sec = elfio_.sections[ElfSecDesc[NOTES].name];
//  if (sec == nullptr) {
//    // Create a new section.
//    if ((sec = newSection(NOTES, nullptr, 0)) == nullptr) {
//      SibirErrorMsg("failed in newSection(NOTES)");
//      return false;
//    }
//  }
//
//  note_section_accessor note_writer(elfio_, sec);
//  // noteName is null terminated
//  note_writer.add_note(0, noteName, noteDesc, descSize);
//
//  SibirDebug("Succeed: add_note(%s, %s)", noteName, std::string(noteDesc, descSize).c_str());
//
//  return true;
//}
//
//bool ElfAmd::getNote(
//    const char* noteName,
//    char** noteDesc,
//    size_t *descSize
//    )
//{
//  if (!descSize || !noteDesc || !noteName) {
//    SibirErrorMsg("failed: empty note");
//    return false;
//  }
//
//  // Get section
//  section* sec = elfio_.sections[ElfSecDesc[NOTES].name];
//  if (sec == nullptr) {
//    SibirErrorMsg("failed: null sections(NOTES)");
//    return false;
//  }
//
//  // Initialize the size and buffer to invalid data points.
//  *descSize = 0;
//  *noteDesc = nullptr;
//
//  note_section_accessor note_reader(elfio_, sec);
//
//  auto num = note_reader.get_notes_num();
//  Elf_Word type = 0;
//  char* desc = nullptr;
//  Elf_Word descSize1 = 0;
//
//  for (unsigned int i = 0; i < num; i++) {
//    std::string name;
//    if(note_reader.get_note(i, type, name, desc, descSize1)) {
//      if(name == noteName) {
//        *noteDesc = static_cast<char *>(desc);
//        *descSize = descSize1;
//        SibirDebug("Succeed: get_note(%s, %s)", name.c_str(),
//                    std::string(*noteDesc, *descSize).c_str());
//        return true;
//      }
//    }
//  }
//
//  return false;
//}
//
//std::string ElfAmd::generateUUIDV4() {
//  static std::random_device rd;
//  static std::mt19937 gen(rd());
//  static std::uniform_int_distribution<> dis(0, 15);
//  static std::uniform_int_distribution<> dis2(8, 11);
//  std::stringstream ss;
//  int i;
//  ss << std::hex;
//  for (i = 0; i < 8; i++) {
//    ss << dis(gen);
//  }
//  ss << "-";
//  for (i = 0; i < 4; i++) {
//    ss << dis(gen);
//  }
//  ss << "-4";
//  for (i = 0; i < 3; i++) {
//    ss << dis(gen);
//  }
//  ss << "-";
//  ss << dis2(gen);
//  for (i = 0; i < 3; i++) {
//    ss << dis(gen);
//  }
//  ss << "-";
//  for (i = 0; i < 12; i++) {
//    ss << dis(gen);
//  };
//  return ss.str();
//}
//
//bool ElfAmd::dumpImage(char** buff, size_t* len)
//{
//  bool ret = false;
//  std::string dumpFile = fname_;
//  if (fname_.empty()) {
//    dumpFile = generateUUIDV4();
//    dumpFile += ".bin";
//    SibirInfoFmt("Generated temporary dump file: %s", dumpFile.c_str());
//  }
//
//  if (!elfio_.save(dumpFile)) {
//    SibirErrorFmt("failed in _elfio.save(%s)", dumpFile.c_str());
//    return false;
//  }
//
//  if (buff != nullptr && len != nullptr) {
//    std::ifstream is;
//    is.open(dumpFile, std::ifstream::in | std::ifstream::binary); // open input file
//    if (!is.good()) {
//      SibirErrorFmt("failed in is.open(%s)", dumpFile.c_str());
//      return false;
//    }
//    ret = dumpImage(is, buff, len);
//    is.close();  // close file
//  }
//
//  if (fname_.empty()) {
//    std::remove(dumpFile.c_str());
//  }
//  SibirInfoFmt("%s: buff=%p, len=%zu\n", ret ? "Succeed" : "failed", *buff, *len);
//  return ret;
//}
//
//bool ElfAmd::dumpImage(std::istream &is, char **buff, size_t *len) const {
//  if (buff == nullptr || len == nullptr) {
//    return false;
//  }
//  is.seekg(0, std::ios::end);  // go to the end
//  *len = is.tellg();           // report location (this is the length)
//  is.seekg(0, std::ios::beg);  // go back to the beginning
//  *buff = new char[*len];      // allocate memory which should be deleted by caller
//  is.read(*buff, *len);        // read the whole file into the buffer
//  return true;
//}
//
//uint64_t ElfAmd::getElfSize(const void *emi) {
//  const unsigned char eclass = static_cast<const unsigned char*>(emi)[EI_CLASS];
//  uint64_t total_size = 0;
//  if (eclass == ELFCLASS32) {
//    auto ehdr = static_cast<const Elf32_Ehdr*>(emi);
//    auto shdr = reinterpret_cast<const Elf32_Shdr*>(static_cast<const char*>(emi) + ehdr->e_shoff);
//
//    auto max_offset = ehdr->e_shoff;
//    total_size = max_offset + ehdr->e_shentsize * ehdr->e_shnum;
//
//    for (decltype(ehdr->e_shnum) i = 0; i < ehdr->e_shnum; ++i) {
//      auto cur_offset = shdr[i].sh_offset;
//      if (max_offset < cur_offset) {
//        max_offset = cur_offset;
//        total_size = max_offset;
//        if (SHT_NOBITS != shdr[i].sh_type) {
//          total_size += shdr[i].sh_size;
//        }
//      }
//    }
//  } else if (eclass == ELFCLASS64) {
//    auto ehdr = static_cast<const Elf64_Ehdr*>(emi);
//    auto shdr = reinterpret_cast<const Elf64_Shdr*>(static_cast<const char*>(emi) + ehdr->e_shoff);
//
//    auto max_offset = ehdr->e_shoff;
//    total_size = max_offset + ehdr->e_shentsize * ehdr->e_shnum;
//
//    for (decltype(ehdr->e_shnum) i = 0; i < ehdr->e_shnum; ++i) {
//      auto cur_offset = shdr[i].sh_offset;
//      if (max_offset < cur_offset) {
//        max_offset = cur_offset;
//        total_size = max_offset;
//        if (SHT_NOBITS != shdr[i].sh_type) {
//          total_size += shdr[i].sh_size;
//        }
//      }
//    }
//  }
//  return total_size;
//}
//
//bool ElfAmd::isElfMagic(const char* p)
//{
//  if (p == nullptr || strncmp(p, ELFMAG, SELFMAG) != 0) {
//    return false;
//  }
//  return true;
//}
//
//bool ElfAmd::isCALTarget(const char* p, signed char ec)
//{
//  if (!isElfMagic(p)) {
//    return false;
//  }
//
//  return false;
//}
//
//void*ElfAmd::xmalloc(const size_t len) {
//  void *retval = ::calloc(1, len);
//  if (retval == nullptr) {
//    SibirErrorMsg("failed: out of memory");
//    return nullptr;
//  }
//  return retval;
//}
//
//void*ElfAmd::allocAndCopy(void* p, size_t sz)
//{
//  if (p == 0 || sz == 0) return p;
//
//  void* buf = xmalloc(sz);
//  if (buf == nullptr) {
//    SibirErrorMsg("failed: out of memory");
//    return 0;
//  }
//
//  memcpy(buf, p, sz);
//  elfMemory_.insert( std::make_pair(buf, sz));
//  return buf;
//}
//
//void*ElfAmd::calloc(size_t sz)
//{
//  void* buf = xmalloc(sz);
//  if (buf == nullptr) {
//    SibirErrorMsg("failed: out of memory");
//    return 0;
//  }
//  elfMemory_.insert( std::make_pair(buf, sz));
//  return buf;
//}
//
//  void ElfAmd::elfMemoryRelease()
//{
//  for(EMemory::iterator it = elfMemory_.begin(); it != elfMemory_.end(); ++it) {
//    free(it->first);
//  }
//  elfMemory_.clear();
//}

}// namespace sibir::elf
