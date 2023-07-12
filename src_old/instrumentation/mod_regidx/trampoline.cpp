#include "trampoline.h"
#include "amdgpu_elf.hpp"
#include "disassembler.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
void getNewTextBinary(char *newtextbinary, char *codeobj, char *ipath) {
  char *iblob = getELF(std::string(ipath));
  elfio::File elfFileP, elfFileI;
  elfFileP = elfFileP.FromMem(codeobj);
  elfFileI = elfFileI.FromMem(iblob);

  Disassembler d(&elfFileP);
  int sRegMax, vRegMax;
  d.getMaxRegIdx(&elfFileP, &sRegMax, &vRegMax);
  std::cout << "The maximum number of sReg is " << sRegMax << "\n";
  std::cout << "The maximum number of vReg is " << vRegMax << "\n";
  std::vector<std::unique_ptr<Inst>> instsP = d.GetOrigInsts(&elfFileP);

  // 2 more scalar registers used in trampoline to store
  // the address of the instrumentation function
  d1.SetModVal(vRegMax, sRegMax + 2);
  std::vector<std::unique_ptr<Inst>> instsI = d1.GetInstruInsts(&elfFileI);
  std::cout << instsI.size() << "\n";

  // manually written instructions
  std::string mwInsts[6] = {"s_branch 0x39",
                            "s_getpc_b64 s[10:11]",
                            "s_add_u32 s10, s10, 0xffffffbc",
                            "s_addc_u32 s11, s11, -1",
                            "s_swappc_b64 s[30:31], s[10:11]",
                            "s_branch 0x3fbf"};
  std::vector<uint32_t> mwEncoding = {0xbf820039, 0xbe8a1c00, 0x800aff0a,
                                      0XFFFFFFBC, 0x820bff0b, 0XFFFFFFFF,
                                      0XBE9E1E0A, 0xbf823fbf};
  std::vector<unsigned char> codeBytes = instcodeToByteArray(mwEncoding);

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

  int size = getNewTextSize(instsP, instsI, instsT);
  printf("new text size is %d\n", size);
  unsigned char *blobText = new unsigned char[size];
  getNewTextBinary(blobText, instsP, instsI, instsT);
  elfio::Section *text = elfFileP.GetSectionByName(".text");
  char *blobTextold = text->Blob();
  text->size = size;
  std::memcpy(blobTextold, (char *)blobText, size);

  d.Disassemble(&elfFileP, "new elf", std::cout);

  // copy new text binary back
  // std::memcpy(newtextbinary, (char *)blobText, size);

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

int getNewTextSize(std::vector<std::unique_ptr<Inst>> &instsP,
                   std::vector<std::unique_ptr<Inst>> &instsI,
                   std::vector<std::unique_ptr<Inst>> &instsT) {
  int bytes = 0;
  for (int i = 0; i < instsP.size(); i++) {
    bytes += instsP.at(i)->byteSize;
  }
  for (int i = 0; i < instsI.size(); i++) {
    bytes += instsI.at(i)->byteSize;
  }
  for (int i = 0; i < instsT.size(); i++) {
    bytes += instsT.at(i)->byteSize;
  }

  return bytes;
}

void getNewTextBinary(unsigned char *blob,
                      std::vector<std::unique_ptr<Inst>> &instsP,
                      std::vector<std::unique_ptr<Inst>> &instsI,
                      std::vector<std::unique_ptr<Inst>> &instsT) {
  int offset = 0;
  for (int i = 0; i < instsP.size(); i++) {
    printf("address: %x\n", offset);
    std::unique_ptr<Inst> inst = std::move(instsP.at(i));
    int size = inst->bytes.size();
    for (int i = 0; i < size; i++) {
      blob[offset + i] = inst->bytes[i];
    }
    offset += size;
  }
  for (int i = 0; i < instsI.size(); i++) {
    printf("address: %x\n", offset);
    std::unique_ptr<Inst> inst = std::move(instsI.at(i));
    int size = inst->bytes.size();
    std::vector<unsigned char> low4 = u32ToByteArray(inst->first);
    for (int i = 0; i < 4; i++) {
      blob[offset + i] = low4[i];
    }
    if (inst->byteSize == 8) {
      std::vector<unsigned char> high4 = u32ToByteArray(inst->second);
      for (int i = 0; i < 4; i++) {
        blob[offset + 4 + i] = high4[i];
      }
    }
    offset += size;
  }
  for (int i = 0; i < instsT.size(); i++) {
    printf("address: %x\n", offset);
    std::unique_ptr<Inst> inst = std::move(instsT.at(i));
    int size = inst->bytes.size();
    for (int i = 0; i < size; i++) {
      blob[offset + i] = inst->bytes[i];
    }
    offset += size;
  }
  return;
}