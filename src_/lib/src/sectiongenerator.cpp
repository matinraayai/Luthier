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
  std::string str1 = std::string(iSec->Blob() + 1);
  std::string str2 = std::string(iSec->Blob() + 14);
  std::string str3 = std::string(iSec->Blob() + 27);
  std::string str4 = "trampoline";

  std::memcpy(newBinary + offset, iSec->Blob() + 1, str1.size() + 1);
  offset += str1.size() + 1;
  std::memcpy(newBinary + offset, iSec->Blob() + 14, str2.size() + 1);
  offset += str2.size() + 1;
  std::memcpy(newBinary + offset, iSec->Blob() + 27, str3.size() + 1);
  offset += str3.size() + 1;
  std::memcpy(newBinary + offset, str4.c_str(), str4.size() + 1);
}