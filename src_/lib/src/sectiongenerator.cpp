#include "sectiongenerator.h"
#include <cstring>
void getDynsymSecBinary(char *newBinary, elfio::Section *pSec,
                        elfio::Section *iSec) {
  std::memcpy(newBinary, pSec->Blob(), pSec->size);
  int offset = pSec->size;
  char *blob = iSec->Blob();
  Elf64_Sym *symTable = reinterpret_cast<Elf64_Sym *>(blob);
  // modify section idx new .bss is at 13
  symTable[1].st_shndx = 13;
  symTable[4].st_shndx = 13;

  std::memcpy(newBinary + offset, &symTable[1], iSec->entsize);
  offset += iSec->entsize;
  std::memcpy(newBinary + offset, &symTable[4], iSec->entsize);
}
void getSymtabSecBinary(char *newBinary, elfio::Section *pSec,
                        elfio::Section *iSec) {
  std::memcpy(newBinary, pSec->Blob(), pSec->size);
  int offset = pSec->size;
  char *blob = iSec->Blob();
  Elf64_Sym *symTable = reinterpret_cast<Elf64_Sym *>(blob);
  // modify section idx new .bss is at 13
  symTable[3].st_shndx = 13;
  symTable[6].st_shndx = 13;
  // modify symbol incr_counter's value
  symTable[1].st_value = 0x10c0;
  std::memcpy(newBinary + offset, &symTable[1], iSec->entsize);
  offset += iSec->entsize;
  std::memcpy(newBinary + offset, &symTable[3], iSec->entsize);
  offset += iSec->entsize;
  std::memcpy(newBinary + offset, &symTable[6], iSec->entsize);
  offset += iSec->entsize;

  // create a symbol entry for trampoline
  Elf64_Sym *tSym = new Elf64_Sym;
  tSym->st_name = symTable[1].st_name; // MISSING symbol name in strtab
  tSym->st_info = symTable[1].st_info;
  tSym->st_other = symTable[1].st_other;
  tSym->st_value = 0x1100;
  tSym->st_size = 32;
  std::memcpy(newBinary + offset, tSym, iSec->entsize);
}

void getShstrtabSecBinary(char *newBinary, elfio::Section *pSec) {
  std::memcpy(newBinary, pSec->Blob(), pSec->size);
  int offset = pSec->size;
  char *p = ".bss";
  std::memcpy(newBinary + offset, p, strlen(".bss"));
}

void getStrtabSecBinary(char *newBinary, elfio::Section *pSec,
                        elfio::Section *iSec) {
  std::memcpy(newBinary, pSec->Blob(), pSec->size);
  int offset = pSec->size;
  std::string str1 = std::string(iSec->Blob() + 1);    //"incr_counter"
  std::string str2 = std::string(iSec->Blob() + 14);   //"counter"
  std::string str3 = std::string(iSec->Blob() + 0x27); //"counter.managed"
  std::string str4 = "trampoline";

  std::memcpy(newBinary + offset, iSec->Blob() + 1, str1.size() + 1);
  offset += str1.size() + 1;
  std::memcpy(newBinary + offset, iSec->Blob() + 14, str2.size() + 1);
  offset += str2.size() + 1;
  std::memcpy(newBinary + offset, iSec->Blob() + 0x27, str3.size() + 1);
  offset += str3.size() + 1;
  std::memcpy(newBinary + offset, str4.c_str(), str4.size() + 1);
}

void getDynstrSecBinary(char *newBinary, elfio::Section *pSec,
                        elfio::Section *iSec) {
  std::memcpy(newBinary, pSec->Blob(), pSec->size);
  int offset = pSec->size;
  std::string str1 = std::string(iSec->Blob() + 1);    //"counter"
  std::string str2 = std::string(iSec->Blob() + 0x1a); //"counter.managed"

  std::memcpy(newBinary + offset, iSec->Blob() + 1, str1.size() + 1);
  offset += str1.size() + 1;
  std::memcpy(newBinary + offset, iSec->Blob() + 0x1a, str2.size() + 1);
}

void getHashSecBinary(char *newBinary, char *dynstr, int num) {
  int entrySize = sizeof(Elf64_Word);
  std::memcpy(newBinary, &num, entrySize);
  int offset = entrySize;
  std::memcpy(newBinary + offset, &num, entrySize);
  offset += entrySize;
  std::string str0 = "";
  std::string str1 = std::string(dynstr + 1); //"_Z15vectoradd_floatPfPKfS1_ii"
  std::string str2 =
      std::string(dynstr + 0x1f); //"_Z15vectoradd_floatPfPKfS1_ii.kd"
  std::string str3 = std::string(dynstr + 0x40); //"counter"
  std::string str4 = std::string(dynstr + 0x48); //"counter.managed"
  std::string strArr[num] = {str0, str1, str2, str3, str4};
  // fill bucket
  for (int i = 0; i < num; i++) {
    unsigned int idx = elf_Hash(strArr[i].c_str());
    idx = idx % num;
    std::memcpy(newBinary + offset, &idx, entrySize);
    offset += entrySize;
  }
  // fill chain
  for (int i = 0; i < num; i++) {
    unsigned int idx = 0;
    std::memcpy(newBinary + offset, &idx, entrySize);
    offset += entrySize;
  }
}

unsigned int elf_Hash(const char *name) {
  unsigned int h = 0, g;

  while (*name) {
    h = (h << 4) + *name++;
    if (g = h & 0xf0000000)
      h ^= g >> 24;
    h &= ~g;
  }
  return h;
}

void getShdrBinary(char *newBinary, elfio::File elfFilep, elfio::File elfFilei,
                   std::vector<int> offsets, std::vector<int> sizes) {
  Elf64_Ehdr *header = elfFilep.GetHeader();
  Elf64_Shdr *shr =
      reinterpret_cast<Elf64_Shdr *>(elfFilep.Blob() + header->e_shoff);
  // modify current section header table: offset, size and addr
  for (int i = 1; i < header->e_shnum; i++) {
    shr[i].sh_offset = offsets[i - 1];
    shr[i].sh_size = sizes[i - 1];
  }
  for (int i = 1; i < 9; i++) {
    shr[i].sh_addr = shr[i].sh_offset;
  }
  std::memcpy(newBinary, shr, header->e_shnum * header->e_shentsize);
  int offset = header->e_shnum * header->e_shentsize;
  int bssidx = 9;
  Elf64_Shdr *bssshr = elfFilei.ExtractShr(bssidx);
  std::memcpy(newBinary + offset, bssshr, header->e_shentsize);
}
