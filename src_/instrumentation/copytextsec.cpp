#include <cstring>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

#include "../../src/elf.h"

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
