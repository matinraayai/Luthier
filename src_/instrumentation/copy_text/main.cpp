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
void printInstList(std::vector<std::shared_ptr<Inst>> instList);
void printInstruFn(std::vector<std::shared_ptr<Inst>> instList);
void offsetInstruRegs(std::vector<std::shared_ptr<Inst>> instList,
                      Assembler a, int smax, int vmax);
std::vector<unsigned char> extractIlistBuf(std::vector<std::shared_ptr<Inst>> instList);

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
  // printInstList(instList);

  printInstruFn(instList);
  offsetInstruRegs(instList, a, sRegMax, vRegMax);

  printInstruFn(instList);
  // d.Disassemble(extractIlistBuf(instList), std::cout);

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


void printInstList(std::vector<std::shared_ptr<Inst>> instList) {
  for (int i = 0; i < instList.size(); i++) {
    std::cout << i << "\t\t" << instList.at(i)->instType.instName << "\t\t"
              << instList.at(i)->PC << std::endl;
  }

  /* This code causes seg fault */
  // InstPrinter printer;
  // std::string istr;
  // for (int i = 0; i < instList.size(); i++) {
  //   istr = printer.print(instList.at(i).get());
  //   std::cout << istr << std::endl;
  // }
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
    std::cout << "SRC0: " << inst->src0.code << std::endl
              << "SRC1: " << inst->src1.code << std::endl
              << "DST:  " << inst->dst.code  << std::endl;
  }
  std::cout << "---------------------------------------" << std::endl;
}

void offsetInstruRegs(std::vector<std::shared_ptr<Inst>> instList,
                      Assembler a, int smax, int vmax) {
  uint64_t i, j;
  int newcode;

  std::string istr; // delete this later

  for (i = 0; i < instList.size(); i++) {
    if (instList.at(i)->instType.instName == "s_endpgm") {
      j = i++;
      break;
    }
  }
  for (j = i; j < instList.size(); j++) {
    if (instList.at(j)->format.formatType == SOPP)
      continue;
    
    
    /***debugging***/
    // istr = instList.at(j)->instType.instName;
    // std::cout << istr << "\t\t";
    // std::cout << std::setw(8) << std::setbase(16) << std::setfill('0')
    //           << instList.at(j)->first << " ";
    // if (instList.at(j)->byteSize == 8)
    //   std::cout << std::setw(8) << std::setbase(16) << std::setfill('0')
    //             << instList.at(j)->second;
    // std::cout << std::endl;
    /**************/


    if (instList.at(j)->instType.DSTWidth != 0) {
      if (instList.at(j)->dst.operandType == RegOperand) {
        if (instList.at(j)->dst.code >= 256) {
          newcode = instList.at(j)->dst.code + vmax;
        } else {
          newcode = instList.at(j)->dst.code + smax;
        }
        a.editDSTreg(instList.at(j), newcode);
      }
    }

    if (instList.at(j)->instType.SRC0Width != 0) {
      if (instList.at(j)->src0.operandType == RegOperand) {
        if (instList.at(j)->src0.code >= 256) {
          newcode = instList.at(j)->src0.code + vmax;
        } else {
          newcode = instList.at(j)->src0.code + smax;
        }
        a.editSRC0reg(instList.at(j), newcode);
      }
    }

    if (instList.at(j)->instType.SRC1Width != 0) {
      if (instList.at(j)->src1.operandType == RegOperand) {
        if (instList.at(j)->src1.code >= 256) {
          newcode = instList.at(j)->src1.code + vmax;
        } else {
          newcode = instList.at(j)->src1.code + smax;
        }
        a.editSRC1reg(instList.at(j), newcode);
      }
    }

    /***debugging***/
    // for (int k = 0; k < istr.size(); k++) 
    //   std::cout << " ";
    // std::cout << "\t\t";
    // std::cout << std::setw(8) << std::setbase(16) << std::setfill('0')
    //           << instList.at(j)->first << " ";
    // if (instList.at(j)->byteSize == 8)
    //   std::cout << std::setw(8) << std::setbase(16) << std::setfill('0')
    //             << instList.at(j)->second;
    // std::cout << std::endl;
    /**************/

    // if (inst->instType.SRC2Width != 0)
  }
  std::cout << "---------------------------------------" << std::endl;
}


std::vector<unsigned char> extractIlistBuf(std::vector<std::shared_ptr<Inst>> instList) {
  std::vector<unsigned char> buf;
  Inst *inst;
  for (int i = 0; i < instList.size(); i++) {
    inst = instList.at(i).get();
    
    for (int j = 0; j < inst->byteSize; j++) {
      buf.push_back(inst->bytes.at(j));
    }
  }   
  std::cout << "---------------------------------------" << std::endl;
  return buf;
}






