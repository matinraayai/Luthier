#include "sectiongenerator.h"
#include <cstring>
void getDynsymSecBinary(char *newBinary, elfio::Section *pSec,
                        elfio::Section *iSec) {
  std::memcpy(newBinary, pSec->Blob(), pSec->size);
  char *blob = iSec->Blob();
  Elf64_Sym *symTable = reinterpret_cast<Elf64_Sym *>(blob);
  // modify section idx new .bss is at 13
  symTable[1].st_shndx = 13;
  symTable[4].st_shndx = 13;

  std::memcpy(newBinary + pSec->size, &symTable[1], iSec->entsize);
}