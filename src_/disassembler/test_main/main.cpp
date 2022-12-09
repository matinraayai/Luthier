#include "bitops.h"
#include "disassembler.h"
#include "elf.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

elfio::File getELF(elfio::File* elf, std::string fname, size_t blobsize);

void initInstruKernel(instnode *head, Disassembler *d, 
                      std::string program, std::string instru);
void printInstList(instnode *head);

instnode* getInstFromPC(instnode *head, uint64_t pc);

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Expected 2 inputs: Program File, Instrumentation Function\n";
    return 1;
  }

  Disassembler d;

  instnode *instrukernel = new instnode;
  initInstruKernel(instrukernel, &d, argv[1], argv[2]);

  // instnode *v_add = getInstFromPC(instrukernel, 0x101c);
  // std::cout << v_add->instStr << std::endl;
  printInstList(instrukernel);

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

void initInstruKernel(instnode *head, Disassembler *d, 
                      std::string program, std::string instru) {                          
  std::string prgmfile  = program;
  std::string instrufile = instru;

  elfio::File *prgmelf   = new elfio::File;
  elfio::File *instruelf = new elfio::File;

  *prgmelf   = getELF(prgmelf, prgmfile, 50000);
  *instruelf = getELF(instruelf, instrufile, 50000);

  auto prgmtexsec    = prgmelf->GetSectionByName(".text");
  auto instrutexsec  = instruelf->GetSectionByName(".text");

  char *newkernel = new char[prgmtexsec->size + instrutexsec->size];
  std::memcpy(newkernel, 
              prgmtexsec->Blob(), prgmtexsec->size);
  std::memcpy(newkernel + prgmtexsec->size, 
              instrutexsec->Blob(), instrutexsec->size);

  auto kernelbytes = charToByteArray(newkernel, 
                        prgmtexsec->size + instrutexsec->size);

  d->Disassemble(kernelbytes, head, prgmtexsec->offset);
}

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

