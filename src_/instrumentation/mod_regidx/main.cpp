#include "assembler.h"
#include "disassembler.h"
#include "elf.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
char *getELF(std::string filename);
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Expected 2 inputs: Program File, Instrumentation Function\n";
    return 1;
  }
  std::string pFilename = argv[1];
  std::string iFilename = argv[2];

  elfio::File elfFileP;
  char *blob = getELF(pFilename);
  elfFileP = elfFileP.FromMem(blob);
  elfFileP.PrintSymbolsForSection(".text");

  Disassembler d(&elfFileP);
  int sRegMax, vRegMax;
  d.getMaxRegIdx(&elfFileP, &sRegMax, &vRegMax);
  std::cout << "The maximum number of sReg is " << sRegMax << "\n";
  std::cout << "The maximum number of vReg is " << vRegMax << "\n";

  elfio::File elfFileI;
  char *blob1 = getELF(iFilename);
  elfFileI = elfFileI.FromMem(blob1);

  Disassembler d1(&elfFileI);
  d1.SetModVal(vRegMax, sRegMax);
  std::vector<std::unique_ptr<Inst>> insts = d1.GetInsts(&elfFileI);
  std::cout << insts.size() << "\n";

  return 0;
}
char *getELF(std::string filename) {
  std::streampos size;
  char *blob;

  std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    blob = new char[size];
    file.seekg(0, std::ios::beg);
    file.read(blob, size);
    file.close();
  } else {
    throw std::runtime_error("can't open file");
  }

  return blob;
}