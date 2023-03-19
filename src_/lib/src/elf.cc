#include "elf.hpp"

#include <elf.h>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

namespace elfio {

File File::FromMem(char *blob) {
  File f;

  f.header = (Elf64_Ehdr *)blob;
  f.ParseSections();
  f.ParseSymbols();

  return f;
}

void File::ParseSections() {
  auto *shdr =
      reinterpret_cast<Elf64_Shdr *>(this->Blob() + this->header->e_shoff);

  Elf64_Shdr *sh_strtab = &shdr[this->header->e_shstrndx];
  const char *const sh_strtab_p = this->Blob() + sh_strtab->sh_offset;

  for (int i = 0; i < this->header->e_shnum; i++) {
    auto section = std::make_unique<Section>(this, &shdr[i]);

    section->name = std::string(sh_strtab_p + shdr[i].sh_name);
    section->type = Section::SectionTypeStr(shdr[i].sh_type);
    section->offset = shdr[i].sh_offset;
    section->size = shdr[i].sh_size;
    section->entsize = shdr[i].sh_entsize;
    section->align = shdr[i].sh_addralign;

    this->sections.push_back(std::move(section));
  }
}

void File::ParseSymbols() {
  Section *str_table_section = this->GetSectionByName(".strtab");
  if (!str_table_section) {
    throw std::runtime_error("string table section not found.");
  }

  Section *sym_table_section = this->GetSectionByType("SHT_SYMTAB");
  if (!sym_table_section) {
    throw std::runtime_error("symboL table section not found.");
  }

  const char *const strtab_p = str_table_section->Blob();

  auto total_syms = sym_table_section->size / sizeof(Elf64_Sym);
  auto syms = reinterpret_cast<Elf64_Sym *>(sym_table_section->Blob());

  for (int i = 0; i < total_syms; i++) {
    auto symbol = std::make_unique<Symbol>(this, &syms[i]);

    symbol->name = std::string(strtab_p + syms[i].st_name);
    symbol->value = syms[i].st_value;
    symbol->size = syms[i].st_size;

    this->symbols.push_back(std::move(symbol));
  }
}

char *File::Blob() { return reinterpret_cast<char *>(this->header); }

Section *File::GetSectionByName(const std::string &name) {
  for (auto &section : this->sections) {
    if (section->name == name) {
      return section.get();
    }
  }
  return nullptr;
}

Section *File::GetSectionByType(const std::string &type) {
  for (auto &section : this->sections) {
    if (section->type == type) {
      return section.get();
    }
  }
  return nullptr;
}

Symbol *File::GetSymbolByName(const std::string &name) {
  for (auto &symbol : this->symbols) {
    if (symbol->name == name) {
      return symbol.get();
    }
  }
  return nullptr;
}

void File::PrintSymbolsForSection(const std::string &name) {
  Section *sec = GetSectionByName(name);
  uint64_t start = sec->offset;
  uint64_t end = start + sec->size;
  for (auto &symbol : this->symbols) {
    if (symbol->value < end && symbol->value >= start) {
      std::cout << "{Name: " << symbol->name << ", Value: " << symbol->value
                << ", Size: " << symbol->size << "}" << std::endl;
    }
  }
}

std::vector<Symbol *> File::GetSymbols() {
  std::vector<Symbol *> symbols_out;
  for (auto &symbol : this->symbols) {
    symbols_out.push_back(symbol.get());
  }
  return symbols_out;
}

Section::Section(File *file, Elf64_Shdr *header) : file(file), header(header) {}

char *Section::Blob() {
  return reinterpret_cast<char *>(this->file->Blob() + this->offset);
}

std::string Section::SectionTypeStr(int tt) {
  if (tt < 0)
    return "UNKNOWN";

  switch (tt) {
  case 0:
    return "SHT_NULL"; /* Section header table entry unused */
  case 1:
    return "SHT_PROGBITS"; /* Program data */
  case 2:
    return "SHT_SYMTAB"; /* Symbol table */
  case 3:
    return "SHT_STRTAB"; /* String table */
  case 4:
    return "SHT_RELA"; /* Relocation entries with addends */
  case 5:
    return "SHT_HASH"; /* Symbol hash table */
  case 6:
    return "SHT_DYNAMIC"; /* Dynamic linking information */
  case 7:
    return "SHT_NOTE"; /* Notes */
  case 8:
    return "SHT_NOBITS"; /* Program space with no data (bss) */
  case 9:
    return "SHT_REL"; /* Relocation entries, no addends */
  case 11:
    return "SHT_DYNSYM"; /* Dynamic linker symbol table */
  default:
    return "UNKNOWN";
  }
  return "UNKNOWN";
}

Symbol::Symbol(File *file, Elf64_Sym *header) : file(file), header(header) {}

char *Symbol::Blob() {
  return reinterpret_cast<char *>(this->file->Blob() + this->value);
}

Note::Note(File *file, char *header)
    : file(file), header(reinterpret_cast<Elf64_Nhdr *>(header)) {
  this->type = this->header->n_type;
  this->desc_size = this->header->n_descsz;

  auto name_ptr = this->Blob() + sizeof(Elf64_Nhdr);
  this->name = std::string{name_ptr, this->header->n_namesz};

  int desc_ptr_align = 4;
  int desc_offset = sizeof(Elf64_Nhdr) + this->header->n_namesz;
  if (desc_offset % desc_ptr_align != 0) {
    desc_offset += desc_ptr_align - desc_offset % desc_ptr_align;
  }
  auto desc_ptr = this->Blob() + desc_offset;
  this->desc = std::string{desc_ptr, this->header->n_descsz};
}

char *Note::Blob() { return reinterpret_cast<char *>(this->header); }

uint64_t Note::TotalSize() {
  return sizeof(Elf64_Nhdr) + this->header->n_descsz + this->header->n_namesz;
}

} // namespace elfio
