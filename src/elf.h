#ifndef SRC_ELF_H_
#define SRC_ELF_H_

#include <elf.h>
#include <memory>
#include <string>
#include <vector>

namespace elfio
{

  class Section;
  class Symbol;

  class File
  {
  public:
    static File FromMem(char *blob);

    Section *GetSectionByName(const std::string &name);
    Section *GetSectionByType(const std::string &type);
    Symbol *GetSymbolByName(const std::string &name);
    char *Blob();
    void PrintSymbolsForSection(const std::string &name);

  private:
    Elf64_Ehdr *header;
    std::vector<std::unique_ptr<Section>> sections;
    std::vector<std::unique_ptr<Symbol>> symbols;

    void ParseSections();
    void ParseSymbols();
  };

  class Section
  {
  public:
    static std::string SectionTypeStr(int type);

    Section(File *file, Elf64_Shdr *header);
    char *Blob();

    std::string name;
    std::string type;
    uint64_t offset;
    uint64_t size;

  private:
    File *file;
    Elf64_Shdr *header;
  };

  class Symbol
  {
  public:
    Symbol(File *file, Elf64_Sym *header);

    char *Blob();

    std::string name;
    uint64_t value;
    uint64_t size;

  private:
    File *file;
    Elf64_Sym *header;
  };

  class Note
  {
  public:
    Note(File *file, char *header);

    uint64_t TotalSize();
    char *Blob();

    std::string name;
    std::string desc;
    uint32_t desc_size;
    uint32_t type;

  private:
    File *file;
    Elf64_Nhdr *header;
  };

} // namespace elfio

#endif // SRC_ELF_H_
