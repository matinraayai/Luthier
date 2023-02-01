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
void offsetInstruRegs(std::vector<std::shared_ptr<Inst>> instList,
                      Assembler a, int smax, int vmax);

void editSALUinst(std::shared_ptr<Inst> i, Assembler a, int smax);
void editVALUinst(std::shared_ptr<Inst> i, Assembler a, int smax, int vmax);
void editFLATinst(std::shared_ptr<Inst> i, Assembler a, int smax, int vmax);

std::vector<unsigned char> extractIlistBuf(
                               std::vector<std::shared_ptr<Inst>> instList);

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

  // printInstruFn(instList);
  // offsetInstruRegs(instList, a, sRegMax, vRegMax);
  // printInstruFn(instList);

  for (int i = 0; i < instList.size(); i++) {
    a.Assemble("s_branch 0x3fb1", instList.at(i));
  } std::cout << "Assembling done\n";

  // auto bufFromiList = extractIlistBuf(instList);
  d.Disassemble(extractIlistBuf(instList), std::cout);

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
  std::cout << "---------------------------------------" << std::endl;
}

void offsetInstruRegs(std::vector<std::shared_ptr<Inst>> instList,
                      Assembler a, int smax, int vmax) {
  uint64_t i, j;
  int newcode;

  for (i = 0; i < instList.size(); i++) {
    if (instList.at(i)->instType.instName == "s_endpgm") {
      j = i++;
      break;
    }
  }
  for (j = i; j < instList.size(); j++) {
    switch(instList.at(j)->format.formatType) {
      case SOP2:
        editSALUinst(instList.at(j), a, smax);
        break;
      case SOP1:
        editSALUinst(instList.at(j), a, smax);
        break;
      case VOP1:
        editVALUinst(instList.at(j), a, smax, vmax);
        break;
      case FLAT:
        editFLATinst(instList.at(j), a, smax, vmax);
        break;
      default:
        break;
    }
  }
  std::cout << "---------------------------------------" << std::endl;
}

void editSALUinst(std::shared_ptr<Inst> i, Assembler a, int smax) {
  int newcode;

  if (i->instType.DSTWidth != 0) {
    if (i->dst.operandType == RegOperand) {
      newcode = i->dst.code + smax;
      a.editDSTreg(i, newcode);
    }
  }

  if (i->instType.SRC0Width != 0) {
    if (i->src0.operandType == RegOperand) {
      newcode = i->src0.code + smax;
      a.editSRC0reg(i, newcode);
    }
  }

  if (i->instType.SRC1Width != 0) {
    if (i->src1.operandType == RegOperand) {
      newcode = i->src1.code + smax;
      a.editSRC1reg(i, newcode);
    }
  }
}

void editVALUinst(std::shared_ptr<Inst> i, Assembler a, int smax, int vmax) {
  int newcode;

  if (i->instType.DSTWidth != 0) {
    if (i->dst.operandType == RegOperand) {
      if (i->dst.code >= 256) {
        newcode = i->dst.code + vmax;
      } else {
        newcode = i->dst.code + smax;
      }
      a.editDSTreg(i, newcode);
    }
  }

  if (i->instType.SRC0Width != 0) {
    if (i->src0.operandType == RegOperand) {
      if (i->src0.code >= 256) {
        newcode = i->src0.code + vmax;
      } else {
        newcode = i->src0.code + smax;
      }
      a.editSRC0reg(i, newcode);
    }
  }

  if (i->instType.SRC1Width != 0) {
    if (i->src1.operandType == RegOperand) {
      if (i->src1.code >= 256) {
        newcode = i->src1.code + vmax;
      } else {
        newcode = i->src1.code + smax;
      }
      a.editSRC1reg(i, newcode);
    }
  }
}

void editFLATinst(std::shared_ptr<Inst> i, Assembler a, int smax, int vmax) {
  int newcode;

  if (i->instType.DSTWidth != 0) {
    newcode = i->dst.code + vmax;
    a.editDSTreg(i, newcode);
  }

  if ((i->second & 0x007F0000)>>16 != 0x7F) {
    newcode = i->addr.code + vmax;
    a.editSRC0flat(i, newcode, 0);

    newcode = i->sAddr.code + smax;
    a.editSRC1reg(i, newcode);
  } else {
    newcode = i->data.code + vmax;
    a.editSRC0flat(i, newcode, 1);
  }
}

std::vector<unsigned char> extractIlistBuf(std::vector<std::shared_ptr<Inst>> instList) {
  std::vector<unsigned char> buf;
  Inst *inst;
  for (int i = 0; i < instList.size(); i++) {
    inst = instList.at(i).get();
    
    for (int j = 0; j < inst->bytes.size(); j++) {
      buf.push_back(inst->bytes.at(j));
    }
  }   
  std::cout << "---------------------------------------" << std::endl;
  return buf;
}



