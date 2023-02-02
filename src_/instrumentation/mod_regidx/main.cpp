#include "assembler.h"
#include "disassembler.h"
#include "elf.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
char *getELF(std::string filename);
std::unique_ptr<Inst>
replaceTargetInst(std::vector<std::unique_ptr<Inst>> &insts, int idx,
                  std::unique_ptr<Inst> replace);
void modifyInstruInst(std::vector<std::unique_ptr<Inst>> &insts, int idx);
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
  std::vector<std::unique_ptr<Inst>> instsP = d.GetOrigInsts(&elfFileP);

  elfio::File elfFileI;
  char *blob1 = getELF(iFilename);
  elfFileI = elfFileI.FromMem(blob1);

  Disassembler d1(&elfFileI);
  d1.SetModVal(vRegMax, sRegMax + 2);
  std::vector<std::unique_ptr<Inst>> instsI = d1.GetInstruInsts(&elfFileI);
  std::cout << instsI.size() << "\n";

  // manually written instructions
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
  Disassembler d2;
  std::vector<std::unique_ptr<Inst>> instsMW = d2.GetManualWrInsts(codeBytes);

  std::unique_ptr<Inst> fromOrigInst =
      replaceTargetInst(instsP, 4, std::move(instsMW.at(0)));
  modifyInstruInst(instsI, 3);
  std::vector<std::unique_ptr<Inst>> instsT;
  for (int i = 1; i < 5; i++) {
    instsT.push_back(std::move(instsMW.at(i)));
  }
  instsT.push_back(std::move(fromOrigInst));
  instsT.push_back(std::move(instsMW.at(5)));

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
std::unique_ptr<Inst>
replaceTargetInst(std::vector<std::unique_ptr<Inst>> &insts, int idx,
                  std::unique_ptr<Inst> replace) {
  std::unique_ptr<Inst> target = std::move(insts.at(idx));
  insts.at(idx) = std::move(replace);
  return std::move(target);
}

void modifyInstruInst(std::vector<std::unique_ptr<Inst>> &insts, int idx) {
  std::vector<unsigned char> low4(insts.at(idx)->bytes.begin(),
                                  insts.at(idx)->bytes.begin() + 4);
  uint32_t num = 0x00001f3c;
  std::vector<unsigned char> high4 = u32ToByteArray(num);
  low4.insert(low4.end(), high4.begin(), high4.end());
  insts.at(idx)->bytes = low4;
  insts.at(idx)->second = num;
  insts.at(idx)->src1.literalConstant = num;
}