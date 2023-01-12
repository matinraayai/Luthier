#include "assembler.h"
#include "disassembler.h"
#include "elf.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Expected 2 inputs: Program File, Instrumentation Function\n";
    return 1;
  }
  std::string pFilename = argv[1];
  std::string iFilename = argv[2];
  std::streampos size;
  char *blobP, *blobI;

  std::ifstream fileP(pFilename,
                      std::ios::in | std::ios::binary | std::ios::ate);
  if (fileP.is_open()) {
    size = fileP.tellg();
    blobP = new char[size];
    fileP.seekg(0, std::ios::beg);
    fileP.read(blobP, size);
    fileP.close();
  } else {
    printf("unable to open program file in main\n");
    return 1;
  }
  elfio::File elfFileP;
  elfFileP = elfFileP.FromMem(blobP);
  elfFileP.PrintSymbolsForSection(".text");

  Disassembler d(&elfFileP);
  d.Disassemble(&elfFileP, pFilename, std::cout);
  return 0;
}
