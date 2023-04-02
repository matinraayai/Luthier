#include "../../src/elf.h"
#include "sectiongenerator.h"
#include "trampoline.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>

int main(int argc, char **argv) {
  if (argc != 3) {
    printf(
        "vectoradd_kernel and instrumentation input elf files are required\n");
    return 1;
  }
  std::string filenamep = argv[1];
  std::string filenamei = argv[2];

  std::streampos size;
  char *blobp, *blobi;
  std::ifstream filep(filenamep,
                      std::ios::in | std::ios::binary | std::ios::ate);
  if (filep.is_open()) {
    size = filep.tellg();
    blobp = new char[size];
    filep.seekg(0, std::ios::beg);
    filep.read(blobp, size);
    filep.close();
  } else {
    printf("unable to open vectoradd_kernel elf in main\n");
    return 1;
  }
  std::ifstream filei(filenamei,
                      std::ios::in | std::ios::binary | std::ios::ate);
  if (filei.is_open()) {
    size = filei.tellg();
    blobi = new char[size];
    filei.seekg(0, std::ios::beg);
    filei.read(blobi, size);
    filei.close();
  } else {
    printf("unable to open instrumentation elf in main\n");
    return 1;
  }

  elfio::File elfFilep, elfFilei;
  elfFilep = elfFilep.FromMem(blobp);
  elfFilei = elfFilei.FromMem(blobi);

  int newSize = 0x3680;
  char *newELFBinary = new char[newSize];

  std::vector<int> offsets, sizes;

  int offset = 0x200; // .note's offset

  // copy .note section
  int copySize = elfFilep.GetSectionByName(".note")->size;
  std::memcpy(newELFBinary + offset, elfFilep.GetSectionByName(".note")->Blob(),
              copySize);
  offsets.push_back(offset);
  sizes.push_back(copySize);
  offset += copySize;

  // generate new .dynsym section
  int newSecSize =
      elfFilep.GetSectionByName(".dynsym")->size +
      elfFilep.GetSectionByName(".dynsym")->entsize * 2; // hex num * dec num
  char *newSecBinary = new char[newSecSize];
  getDynsymSecBinary(newSecBinary, elfFilep.GetSectionByName(".dynsym"),
                     elfFilei.GetSectionByName(".dynsym"));
  // copy new .dynsym section
  int align_req = elfFilep.GetSectionByName(".dynsym")->align;
  if (offset % align_req != 0) {
    offset += align_req - offset % align_req;
  }
  std::memcpy(newELFBinary + offset, newSecBinary, newSecSize);
  offsets.push_back(offset);
  sizes.push_back(newSecSize);
  free(newSecBinary);
  offset += newSecSize;

  // copy .gnu.hash sections
  copySize = elfFilep.GetSectionByName(".gnu.hash")->size;
  align_req = elfFilep.GetSectionByName(".gnu.hash")->align;
  if (offset % align_req != 0) {
    offset += align_req - offset % align_req;
  }
  std::memcpy(newSecBinary + offset,
              elfFilep.GetSectionByName(".gnu.hash")->Blob(), copySize);
  offsets.push_back(offset);
  sizes.push_back(copySize);
  offset += copySize;

  // find size for new .hash section
  int numEntry = elfFilep.GetSectionByName(".dynsym")->size /
                     elfFilep.GetSectionByName(".dynsym")->entsize +
                 2;
  int newHashSize =
      elfFilep.GetSectionByName(".hash")->entsize * (1 + 1 + 2 * numEntry);

  // generate new .dynstr section
  newSecSize = elfFilep.GetSectionByName(".dynstr")->size + strlen("counter") +
               strlen("counter.managed") + 2; //\0 null character problem
  newSecBinary = new char[newSecSize];
  char *newHashBinary = new char[newHashSize];
  getDynstrSecBinary(newSecBinary, elfFilep.GetSectionByName(".dynstr"),
                     elfFilei.GetSectionByName(".dynstr"));
  // generate new .hash section
  getHashSecBinary(newHashBinary, newSecBinary, numEntry);
  // copy new .hash section
  align_req = elfFilep.GetSectionByName(".hash")->align;
  if (offset % align_req != 0) {
    offset += align_req - offset % align_req;
  }
  std::memcpy(newELFBinary + offset, newHashBinary, newHashSize);
  offsets.push_back(offset);
  sizes.push_back(newHashSize);
  free(newHashBinary);
  offset += newHashSize; // find the begining of .dynstr section

  // copy new .dynstr section
  std::memcpy(newELFBinary + offset, newSecBinary, newSecSize);
  offsets.push_back(offset);
  sizes.push_back(newSecSize);
  free(newSecBinary);
  offset += newSecSize;

  // copy .rodata
  align_req = elfFilep.GetSectionByName(".rodata")->align;
  if (offset % align_req != 0) {
    offset += align_req - offset % align_req;
  }
  copySize = elfFilep.GetSectionByName(".rodata")->size;
  std::memcpy(newELFBinary + offset,
              elfFilep.GetSectionByName(".rodata")->Blob(), copySize);
  offsets.push_back(offset);
  sizes.push_back(copySize);
  offset += copySize;

  // generate new .text section
  int newTextSize = elfFilep.GetSectionByName(".text")->size +
                    elfFilei.GetSymbolByName("incr_counter")->size +
                    32; // trampoline size :32 bytes;
  char *newTextBinary = new char[newTextSize];
  getNewTextBinary(newTextBinary, blobp, (char *)(filenamei.c_str()));

  // copy new .text section
  offset = 0x1000;
  std::memcpy(newELFBinary + offset, newTextBinary, newTextSize);
  offsets.push_back(offset);
  sizes.push_back(newTextSize);
  free(newTextBinary);

  // copy .dynamic and .comment sections
  copySize = elfFilep.GetSectionByName(".dynamic")->size;

  offset = 0x2000;
  std::memcpy(newELFBinary + offset,
              elfFilep.GetSectionByName(".dynamic")->Blob(), copySize);
  offsets.push_back(offset);
  sizes.push_back(copySize);
  offset += copySize;

  copySize = elfFilep.GetSectionByName(".comment")->size;
  std::memcpy(newELFBinary + offset,
              elfFilep.GetSectionByName(".comment")->Blob(), copySize);
  offsets.push_back(offset);
  sizes.push_back(copySize);
  offset += copySize;

  // generate new .symtab section
  newSecSize = elfFilep.GetSectionByName(".symtab")->size +
               elfFilep.GetSectionByName(".symtab")->entsize * 4;
  newSecBinary = new char[newSecSize];
  getSymtabSecBinary(newSecBinary, elfFilep.GetSectionByName(".symtab"),
                     elfFilei.GetSectionByName(".symtab"));

  // copy new .symtab
  align_req = elfFilep.GetSectionByName(".symtab")->align;
  if (offset % align_req != 0) {
    offset += align_req - offset % align_req;
  }
  std::memcpy(newELFBinary + offset, newSecBinary, newSecSize);
  offsets.push_back(offset);
  sizes.push_back(newSecSize);
  free(newSecBinary);
  offset += newSecSize;

  // generate new .shstrtab section
  newSecSize = elfFilep.GetSectionByName(".shstrtab")->size + strlen(".bss") +
               1; //\0 null character problem
  newSecBinary = new char[newSecSize];
  getShstrtabSecBinary(newSecBinary, elfFilep.GetSectionByName(".shstrtab"));

  // copy new .shstrtab
  std::memcpy(newELFBinary + offset, newSecBinary, newSecSize);
  offsets.push_back(offset);
  sizes.push_back(newSecSize);
  free(newSecBinary);
  offset += newSecSize;

  // generate new .strtab section
  newSecSize = elfFilep.GetSectionByName(".strtab")->size +
               strlen("incr_counter") + strlen("counter") +
               strlen("counter.managed") + strlen("trampoline") +
               4; //\0 null character problem
  newSecBinary = new char[newSecSize];
  getStrtabSecBinary(newSecBinary, elfFilep.GetSectionByName(".strtab"),
                     elfFilei.GetSectionByName(".strtab"));

  // copy new .strtab
  std::memcpy(newELFBinary + offset, newSecBinary, newSecSize);
  offsets.push_back(offset);
  sizes.push_back(newSecSize);
  free(newSecBinary);
  offset += newSecSize;

  // generate new section header table
  Elf64_Ehdr *Eheader = elfFilep.GetHeader();
  // add .bss section header into section header table
  int newShdrSize = (Eheader->e_shnum + 1) * Eheader->e_shentsize;
  char *newShdrBinary = new char[newShdrSize];
  Elf64_Shdr *shr =
      reinterpret_cast<Elf64_Shdr *>(elfFilep.Blob() + Eheader->e_shoff);
  // modify current section header table: offset, size and addr
  for (int i = 1; i < Eheader->e_shnum; i++) {
    shr[i].sh_offset = offsets[i - 1];
    shr[i].sh_size = sizes[i - 1];
  }
  for (int i = 1; i < 9; i++) {
    shr[i].sh_addr = shr[i].sh_offset;
  }
  std::memcpy(newShdrBinary, shr, Eheader->e_shnum * Eheader->e_shentsize);
  int endOld = Eheader->e_shnum * Eheader->e_shentsize;
  int bssidx = 9;
  Elf64_Shdr *bssshr = elfFilei.ExtractShr(bssidx);
  bssshr->sh_name = elfFilep.GetSectionByName(".shstrtab")->size;
  std::memcpy(newShdrBinary + endOld, bssshr, Eheader->e_shentsize);

  offset = 0x3000;
  std::memcpy(newELFBinary + offset, newShdrBinary, newShdrSize);
  free(newShdrBinary);

  // copy ELF header
  copySize = Eheader->e_ehsize;
  // modify ELF header before copy
  Eheader->e_shoff = offset;
  Eheader->e_shnum += 1;
  std::memcpy(newELFBinary, Eheader, copySize);

  elfio::File newELF;
  newELF = newELF.FromMem(newELFBinary);

  std::ofstream outfile("newfile.exe", std::ios::out | std::ios::binary);
  outfile.write(newELFBinary, newSize);
}