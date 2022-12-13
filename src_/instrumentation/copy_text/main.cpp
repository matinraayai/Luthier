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

prgminfo initInstruKernel(instnode *head, Disassembler *d, 
                      std::string program, std::string instru);
void printInstList(instnode *head);

instnode* getInstFromPC(instnode *head, uint64_t pc);

void getRegMax(instnode *head, prgminfo info);

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Expected 2 inputs: Program File, Instrumentation Function\n";
    return 1;
  }
  Disassembler d;

  instnode *instrukernel = new instnode;
  prgminfo prgm = initInstruKernel(instrukernel, &d, argv[1], argv[2]);

  std::cout << std::hex  << prgm.offset << std::endl 
            << prgm.size << std::endl;
  
  std::cout << "---------------------------------------" << std::endl;

  // instnode *v_add = getInstFromPC(instrukernel, 0x101c);
  // std::cout << v_add->instStr << std::endl;
  printInstList(instrukernel);

  std::cout << "---------------------------------------" << std::endl;
  getRegMax(instrukernel, prgm);

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

prgminfo initInstruKernel(instnode *head, Disassembler *d, 
                      std::string program, std::string instru) {                          
  // WOULD LOVE TO COPY THE CODE TO GET TEXT SECTIONS INTO A STANDALONE FUNCTION!
  // BUT! Whenever I try to do that, I break everything ;-;

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
  return {prgmtexsec->offset, prgmtexsec->size};
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

// Get the largest sreg and vreg used by the kernel 
void getRegMax(instnode *head, prgminfo info) {
    instnode *curr = head;
    std::string currInstStr;
    std::string reg;

  int sregmax;
  int vregmax;

  while(curr->pc != info.offset + info.size){
    currInstStr = curr->instStr;
    std::cout << currInstStr << "\t" << curr->pc << std::endl;

    currInstStr.erase(0, currInstStr.find(" ") + 1);

    //NEXT STEP: Seperate the inst into a list of operands and check each operand!!!!
    //In other words...
    //Use the functions I already wrote in Assembler!
    //Wait
    //NEED TO INCORPORATE ASSEMBLER INTO THIS EVENTUALLY ANYWAYS!!!!
    //TIME TO FINALLY LEARN CMAKE!!!

    while(currInstStr.find(" ") != std::string::npos) {
      if(currInstStr.at(0) == 's'){
        printf("You have an S reg!\n");
      } else if (currInstStr.at(0) == 'v') {
        printf("You have a V reg!\n");
      } else {
        printf("Not a reg\n");
      }
      currInstStr.erase(0, currInstStr.find(" ") + 1);
    }
    curr = curr->next;
  }
}

/*
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Expected 2 inputs: Program File; Instrumentation Function\n";
    return 1;
  }
  std::string prgmfile = argv[1];
  std::string instfile = argv[2];
  std::string outname = "output.text";

  std::fstream file1;
  std::fstream file2;
  std::fstream texout;

  char *blob = new char[50000];	// implement smarter mem allocation later
  elfio::File *elfFile1 = new elfio::File;
  elfio::File *elfFile2 = new elfio::File;

  file1.open(prgmfile, std::ios_base::in | std::ios_base::binary);
  if(file1.is_open()){
    file1.read(blob, 50000);
    *elfFile1 = elfFile1->FromMem(blob);
    file1.close();
  } else{	
    printf("Could not create elfio::File for %s\n", prgmfile.c_str());
    return 1;
  }
  
  file2.open(instfile, std::ios_base::in | std::ios_base::binary);
  if(file2.is_open()){
    delete blob;
    blob = new char[50000];

    file2.read(blob, 50000);
    *elfFile2 = elfFile2->FromMem(blob);
    file2.close();
  } else{	
    printf("Could not create elfio::File for %s\n", instfile.c_str());
    return 1;
  }

  auto prgmtexsec  = elfFile1->GetSectionByName(".text");
  auto prgmtexsize = uint32_t(prgmtexsec->size);

  auto insttexsec  = elfFile2->GetSectionByName(".text");
  auto insttexsize = uint32_t(insttexsec->size);

  std::cout << std::hex << prgmtexsize << std::endl;
  std::cout << std::hex << insttexsize << std::endl;

  char *outtexhead = new char[prgmtexsize];
  char *prgmtexblob = new char[prgmtexsize];
  char *insttexblob = new char[insttexsize];

  prgmtexblob = prgmtexsec->Blob();
  insttexblob = insttexsec->Blob();

  // std::memcpy(outtexhead, prgmtexblob, prgmtexsize);
  std::memcpy(outtexhead, insttexblob, insttexsize);

  texout.open(outname, std::ios_base::out | std::ios_base::binary);
  if(texout.is_open()){
    texout.write(outtexhead, prgmtexsize);
    // texout.write(insttexsec->Blob(), insttexsize);

    texout.close();
  } else{
    for(int i = 0; i < 64; i++){
      std::cout << std::hex << prgmtexsec->Blob();
      std::cout << " | ";
      std::cout << std::hex << insttexsec->Blob();
      std::cout << std::endl;
    }    
  }





  return 0;
}
*/
