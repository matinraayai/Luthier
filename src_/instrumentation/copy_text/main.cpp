#include "assembler.h"
#include "bitops.h"
#include "disassembler.h"
#include "elf.hpp"
#include "instprinter.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

char *getELF(std::string filename);
void printInstruFn(std::vector<std::shared_ptr<Inst>> instList);
void makeTrampoline(std::vector<std::shared_ptr<Inst>> &instList, 
                    Assembler a, uint64_t inum);

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

  elfio::File elfFileI;
  char *blob1 = getELF(iFilename);
  elfFileI = elfFileI.FromMem(blob1);

  Assembler a;
  Disassembler d(&elfFileP);

  auto ptexsec = elfFileP.GetSectionByName(".text");
  uint64_t poff = ptexsec->offset;
  uint64_t psize = ptexsec->size;

  auto itexsec = elfFileI.GetSectionByName(".text");
  uint64_t ioff = itexsec->offset;
  uint64_t isize = itexsec->size;

  int sRegMax, vRegMax;
  d.getMaxRegIdx(&elfFileP, &sRegMax, &vRegMax);

  std::cout << "Program Offset:\t" << poff    << std::endl
            << "Program Size:\t"   << psize   << std::endl;
  std::cout << "Instru Offset:\t"  << ioff    << std::endl
            << "Instru Size:\t"    << isize   << std::endl << std::endl;

  std::cout << "Max S reg:\t" << sRegMax << std::endl
            << "Max V reg:\t" << vRegMax << std::endl    << std::endl;
  std::cout << "---------------------------------------" << std::endl;

  char *newkernel = new char[psize + isize];

  std::memcpy(newkernel, ptexsec->Blob(), psize);
  std::memcpy(newkernel + psize, itexsec->Blob(), isize);

  auto kernelbytes = charToByteArray(newkernel, psize + isize);
  std::vector<std::shared_ptr<Inst>> instList = d.GetInsts(kernelbytes, poff);

  int i, j;
  for (i = 0; i < instList.size(); i++) {
    if (instList.at(i)->instType.instName == "s_endpgm") {
      j = i++;
      break;
    }
  }
  for (j = i; j < instList.size(); j++) {
    if (instList.at(j)->instType.instName == "s_nop") {
      break;
    }
    a.offsetRegs(instList.at(j),  sRegMax, vRegMax);
  }

  makeTrampoline(instList, a, 0);

  // a.Assemble("s_branch 0x3fb1", instList.at(0));
  

  d.Disassemble(a.ilstbuf(instList), std::cout);

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

void printInstruFn(std::vector<std::shared_ptr<Inst>> instList) {
  InstPrinter printer;
  std::string istr;

  uint64_t i, j;
  Inst *inst;

  for (i = 0; i < instList.size(); i++) {
    if (instList.at(i)->instType.instName == "s_endpgm") {
      j = i++;
      break;
    }
  }
  for (j = i; j < instList.size(); j++) {
    inst = instList.at(j).get();
    istr = printer.print(inst);

    std::cout << istr;
    
    for (int k = istr.size(); k < 59; k++) {
      std::cout << " ";
    }
    std::cout << std::setw(8) << std::setbase(16) << std::setfill('0')
              << inst->first << " ";
    if (inst->byteSize == 8)
      std::cout << std::setw(8) << std::setbase(16) << std::setfill('0')
                << inst->second;
    std::cout << std::endl;
  }
}

void makeTrampoline(std::vector<std::shared_ptr<Inst>> &instList, 
                    Assembler a, uint64_t inum) {
  // manually written instructions -- trampoline
  std::string hwInsts[7] = {"s_branch ",
                            "s_getpc_b64 s[10:11]",
                            "s_add_u32 s10, s10, 0xffffffbc",
                            "s_addc_u32 s11, s11, -1",
                            "s_swappc_b64 s[30:31], s[10:11]",
                            "s_nop", // replace this with the original instruction
                            "s_branch "};

  std::shared_ptr<Inst> originalInst = instList.at(inum);
  uint64_t trmpPC = instList.at(instList.size() - 1)->PC + 4;

  //SIGNED immediate values for jump!
  short trmpBranchAddr = (trmpPC - (originalInst->PC + 4))/4; 
  short origBranchAddr = ((originalInst->PC + 4) - (trmpPC + 0x14));

  std::stringstream t_branch;
  std::stringstream o_branch;

  t_branch << "0x" << std::hex << trmpBranchAddr;
  o_branch << "0x" << std::hex << origBranchAddr;

  hwInsts[0].append(t_branch.str());
  hwInsts[6].append(o_branch.str());

  /* 
   * Pad end of intrumentation function with NOP
   * I feel like we don't need this
   */
  for (uint64_t i = 0; i < 8; i++) {
    uint64_t new_nop;
    if (instList.at(instList.size() - 1)->byteSize == 8) {
      new_nop = instList.at(instList.size() - 1)->PC + 8;
    } else {
      new_nop = instList.at(instList.size() - 1)->PC + 4;
    }
    instList.push_back(a.Assemble("s_nop", new_nop));
  }

  uint64_t newpc;
  for (int i = 1; i < 7; i++) {
    if (i == 5) {
      originalInst->PC = trmpPC + i*4;
      instList.push_back(originalInst);
    } else {
      instList.push_back(a.Assemble(hwInsts[i], trmpPC + i*4));
    }
    // For some reason the s_addc_u32 is not being printed by the disassembler
    // However, you can see that the instruction is still assembled
    // std::cout << trampoline.at(i)->instType.instName << std::endl;
  }
  a.Assemble(hwInsts[0], instList.at(inum));
}

