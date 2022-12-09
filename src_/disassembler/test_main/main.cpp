#include "bitops.h"
#include "disassembler.h"
#include "elf.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

elfio::File getELF(elfio::File* elf, std::string fname, size_t blobsize) {
  std::fstream file;
  char *blob;

  file.open(fname, std::ios_base::in | std::ios_base::binary);
  if(file.is_open()){
    blob = new char[blobsize];
    file.read(blob, blobsize);
    file.close();
  }
  return elf->FromMem(blob);
}

void disassembleInstruKernel(
  Disassembler *d, std::string prgmfile, std::string instrufile) {
  elfio::File *prgmelf   = new elfio::File;
  elfio::File *instruelf = new elfio::File;

  *prgmelf   = getELF(prgmelf, prgmfile, 50000);
  *instruelf = getELF(instruelf, instrufile, 50000);

  auto prgmtexsec    = prgmelf->GetSectionByName(".text");
  auto instrutexsec  = instruelf->GetSectionByName(".text");

  printf("Program size:\t%X\nInstru size:\t%X\n\n", 
          prgmtexsec->size, instrutexsec->size);

  char *newkernel = new char[prgmtexsec->size + instrutexsec->size];
  std::memcpy(newkernel, 
              prgmtexsec->Blob(), prgmtexsec->size);
  std::memcpy(newkernel + prgmtexsec->size, 
              instrutexsec->Blob(), instrutexsec->size);

  d->Disassemble(
    charToByteArray(newkernel, prgmtexsec->size + instrutexsec->size), std::cout);
}

int main(int argc, char **argv) {
  // if (argc != 3) {
  //   std::cout << "Expected 2 inputs: Program File, Instrumentation Function\n";
  //   return 1;
  // }
  elfio::File *prgmelf   = new elfio::File;
  // elfio::File *instruelf = new elfio::File;

  *prgmelf   = getELF(prgmelf, argv[1], 50000);
  // *instruelf = getELF(instruelf, argv[2], 50000);

  auto prgmtexsec    = prgmelf->GetSectionByName(".text");
  // auto instrutexsec  = instruelf->GetSectionByName(".text");

  // printf("Program size:\t%X\nInstru size:\t%X\n\n", 
  //         prgmtexsec->size, instrutexsec->size);

  // char *newkernel = new char[prgmtexsec->size + instrutexsec->size];
  char *newkernel = new char[prgmtexsec->size];
  std::memcpy(newkernel, 
              prgmtexsec->Blob(), prgmtexsec->size);
  // std::memcpy(newkernel + prgmtexsec->size, 
  //             instrutexsec->Blob(), instrutexsec->size);

  Disassembler d;
  // disassembleInstruKernel(&d, argv[1], argv[2]);
  // d.Disassemble(
  //   charToByteArray(newkernel, prgmtexsec->size + instrutexsec->size));

  auto prgmbytes = charToByteArray(newkernel, prgmtexsec->size);
  
  printf("Program size:\t%X\nByte Array size:\t%X\n\n", 
          prgmtexsec->size, prgmbytes.size());

  d.Disassemble(
    charToByteArray(newkernel, prgmtexsec->size), std::cout);

  return 0;
}



/*
int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("a input elf file is required\n");
    return 1;
  }
  std::string filename = argv[1];
  // std::string filename_elf;

  initRegs();
  initFormatTable();

  // if (filename.find("csv") != std::string::npos) {
  //   filename_elf = filename.substr(0, filename.length() - 6);
  // } else {
  //   filename_elf = filename;
  // }
  // std::streampos size;

  char *blob;
  // std::ifstream file(filename_elf,
  std::ifstream file(filename,
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open()) {
    // size = file.tellg();

    // std::cout << std::hex << size << std::endl;

    blob = new char[size];
    // file.seekg(0, std::ios::beg);

    file.read(blob, size);
    file.close();
  } else {
    printf("unable to open executable file in main\n");
    return 1;
  }


  // elfio::File elfFile;
  // elfFile = elfFile.FromMem(blob);
  // elfFile.PrintSymbolsForSection(".text");

  // Disassembler d(&elfFile);
  // if (filename.find("csv") != std::string::npos) {
  //   d.Disassemble(&elfFile, filename);
  // } else {
  //   d.Disassemble(&elfFile, filename, std::cout);
  // }

  auto textsec = charToByteArray(blob, size);

  Disassembler d;
  d.Disassemble(textsec);
  return 0;
}
*/

