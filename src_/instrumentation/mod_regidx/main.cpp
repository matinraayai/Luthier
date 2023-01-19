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
  /*
    elfio::File elfFileP;
    char *blob = getELF(pFilename);
    elfFileP = elfFileP.FromMem(blob);
    elfFileP.PrintSymbolsForSection(".text");

    Disassembler d(&elfFileP);
    int sRegMax, vRegMax;
    d.getMaxRegIdx(&elfFileP, &sRegMax, &vRegMax);
    std::cout << "The maximum number of sReg is " << sRegMax << "\n";
    std::cout << "The maximum number of vReg is " << vRegMax << "\n";
    std::vector<std::unique_ptr<Inst>> instsP = d.GetOrigInsts(&elfFileP);

    elfio::File elfFileI;
    char *blob1 = getELF(iFilename);
    elfFileI = elfFileI.FromMem(blob1);

    Disassembler d1(&elfFileI);
    d1.SetModVal(vRegMax, sRegMax + 2);
    std::vector<std::unique_ptr<Inst>> instsI = d1.GetInstruInsts(&elfFileI);
    std::cout << instsI.size() << "\n";
  */
  // handwritten instructions
  std::string hwInsts[6] = {"s_branch 0x39",
                            "s_getpc_b64 s[10:11]",
                            "s_add_u32 s10, s10, 0xffffffbc",
                            "s_addc_u32 s11, s11, -1",
                            "s_swappc_b64 s[30:31], s[10:11]",
                            "s_branch 0x3fbf"};
  std::vector<uint32_t> hwEncoding = {0xbf820039, 0xbe8a1c00, 0x800aff0a,
                                      0XFFFFFEB4, 0x820bff0b, 0XFFFFFFFF,
                                      0XBE9E1E0A, 0xbf823fbf};
  std::vector<unsigned char> codeBytes = instcodeToByteArray(hwEncoding);

  std::cout << codeBytes.size() << "\n";
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