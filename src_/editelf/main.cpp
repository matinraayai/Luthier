#include "../../src/elf.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string.h>
#include <string>

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("src and dst input elf files are required\n");
    return 1;
  }
  std::string filename_src = argv[1];
  std::string filename_dst = argv[2];

  std::streampos size;
  char *blob_s, *blob_d;
  std::ifstream file_src(filename_src,
                         std::ios::in | std::ios::binary | std::ios::ate);
  if (file_src.is_open()) {
    size = file_src.tellg();
    blob_s = new char[size];
    file_src.seekg(0, std::ios::beg);
    file_src.read(blob_s, size);
    file_src.close();
  } else {
    printf("unable to open src elf in main\n");
    return 1;
  }
  std::ifstream file_dst(filename_dst,
                         std::ios::in | std::ios::binary | std::ios::ate);
  if (file_dst.is_open()) {
    size = file_dst.tellg();
    blob_d = new char[size];
    file_dst.seekg(0, std::ios::beg);
    file_dst.read(blob_d, size);
    file_dst.close();
  } else {
    printf("unable to open dst elf in main\n");
    return 1;
  }

  elfio::File elfFile_s, elfFile_d;
  elfFile_s = elfFile_s.FromMem(blob_s);
  elfFile_d = elfFile_d.FromMem(blob_d);

  Elf64_Ehdr *header = elfFile_d.GetHeader();
  Elf64_Shdr *shr =
      reinterpret_cast<Elf64_Shdr *>(elfFile_d.Blob() + header->e_shoff);
  int bssidx = 9;
  Elf64_Shdr *bssshr = elfFile_s.ExtractShr(bssidx);

  bssshr->sh_offset = reinterpret_cast<Elf64_Off>(
      shr[header->e_shoff - 1].sh_offset + shr[header->e_shoff - 1].sh_size);

  int bss_section_align = bssshr->sh_addralign;
  if (bssshr->sh_offset % bss_section_align != 0) {
    bssshr->sh_offset +=
        bss_section_align - bssshr->sh_offset % bss_section_align;
  }

  std::memcpy(&shr[header->e_shnum], bssshr, sizeof(Elf64_Shdr));

  elfio::Section *str_table_section_s = elfFile_s.GetSectionByName(".strtab");
  const char *const strtab_p_s = str_table_section_s->Blob();
  std::string str1 = std::string(strtab_p_s + 1);
  std::string str2 = std::string(strtab_p_s + 14);
  std::string str3 = std::string(strtab_p_s + 27);

  elfio::Section *str_table_section_d = elfFile_d.GetSectionByName(".strtab");
  char *const strtab_p_d = str_table_section_d->Blob();

  std::memcpy(strtab_p_d + 55, strtab_p_s + 1, str1.size() + 1);
  std::memcpy(strtab_p_d + 55 + str1.size() + 1, strtab_p_s + 14,
              str2.size() + 1);
  std::memcpy(strtab_p_d + 55 + str1.size() + 1 + str2.size() + 1,
              strtab_p_s + 27, str3.size() + 1);
}