#include "assembler.h"
#include "bitops.h"
#include "disassembler.h"
#include "elf.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct prgminfo{
  uint64_t offset;
  uint64_t size;
};

elfio::File getELF(elfio::File* elf, std::string fname, size_t blobsize);

// prgminfo initInstruKernel(instnode *head, Disassembler *d, 
//                       std::string program, std::string instru);
void printInstList(instnode *head);

instnode* getInstFromPC(instnode *head, uint64_t pc);

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Expected 2 inputs: Program File, Instrumentation Function\n";
    return 1;
  }

  std::string prgmfile   = argv[1];
  std::string instrufile = argv[2];

  elfio::File *prgmelf   = new elfio::File;
  elfio::File *instruelf = new elfio::File;

  *prgmelf   = getELF(prgmelf, prgmfile, 50000);
  *instruelf = getELF(instruelf, instrufile, 50000);

  auto prgmtexsec   = prgmelf->GetSectionByName(".text");
  auto instrutexsec = instruelf->GetSectionByName(".text");

  std::cout << "---------------------------------------" << std::endl;

  uint64_t prgmoff    = prgmtexsec->offset;
  uint64_t prgmsize   = prgmtexsec->size;
  uint64_t instruoff  = instrutexsec->offset;
  uint64_t instrusize = instrutexsec->size;  

  std::cout << "Program Offset:\t" << prgmoff    << std::endl
            << "Program Size:\t"   << prgmsize   << std::endl;
  std::cout << "Instru Offset:\t"  << instruoff  << std::endl
            << "Instru Size:\t"    << instrusize << std::endl;


  char *newkernel = new char[prgmsize + instrusize];
  std::memcpy(newkernel, prgmtexsec->Blob(), prgmsize);
  std::memcpy(newkernel + prgmsize, instrutexsec->Blob(), instrusize);


  std::cout << "---------------------------------------" << std::endl;

  auto oldkernelbytes = charToByteArray(prgmtexsec->Blob(), prgmsize);
  auto newkernelbytes = charToByteArray(newkernel, prgmsize + instrusize);

  instnode *instrukernel = new instnode;
  Disassembler d(prgmelf);
  d.Disassemble(oldkernelbytes);
  printf(
    "Max S reg:\t%d\nMax V reg:\t%d\n",
    d.maxNumSReg(),
    d.maxNumVReg()
  );
  d.Disassemble(newkernelbytes, instrukernel, prgmtexsec->offset);

  std::cout << "---------------------------------------" << std::endl;

  printInstList(instrukernel);

  std::cout << "---------------------------------------" << std::endl;
  

  return 0;
}

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

// prgminfo initInstruKernel(instnode *head, Disassembler *d, 
//                       std::string program, std::string instru) {                          
//   std::string prgmfile  = program;
//   std::string instrufile = instru;

//   elfio::File *prgmelf   = new elfio::File;
//   elfio::File *instruelf = new elfio::File;

//   *prgmelf   = getELF(prgmelf, prgmfile, 50000);
//   *instruelf = getELF(instruelf, instrufile, 50000);

//   auto prgmtexsec    = prgmelf->GetSectionByName(".text");
//   auto instrutexsec  = instruelf->GetSectionByName(".text");

//   char *newkernel = new char[prgmtexsec->size + instrutexsec->size];
//   std::memcpy(newkernel, 
//               prgmtexsec->Blob(), prgmtexsec->size);
//   std::memcpy(newkernel + prgmtexsec->size, 
//               instrutexsec->Blob(), instrutexsec->size);

//   auto kernelbytes = charToByteArray(newkernel, 
//                         prgmtexsec->size + instrutexsec->size);

//   d->Disassemble(kernelbytes, head, prgmtexsec->offset);
//   return {prgmtexsec->offset, prgmtexsec->size};
// }

void printInstList(instnode *head) {
  std::string instStr;
  instnode *curr = head;

  while(curr->next != NULL) {
    instStr = curr->instStr;
    std::cout << "\t" << instStr;
    for (int i = instStr.size(); i < 59; i++) {
      std::cout << " ";
    }

    std::cout << std::hex << "//" << std::setw(12) << std::setfill('0') 
              << curr->pc << ": ";
    std::cout << std::setw(8) << std::hex << convertLE(curr->bytes);
    if (curr->byteSize == 8) {
      std::vector<unsigned char> sfb(curr->bytes.begin() + 4, 
                                 curr->bytes.begin() + 8);
      std::cout << std::setw(8) << std::hex << convertLE(sfb) << std::endl;
    } else {
      std::cout << std::endl;
    }
    curr = curr->next;
  }
}

instnode* getInstFromPC(instnode *head, uint64_t pc) {
  instnode *curr = head;
  while (curr->pc != pc) {
    curr = curr->next;
  }
  return curr;
}
