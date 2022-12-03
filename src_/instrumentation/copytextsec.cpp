#include <cstring>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

#include "../../src/elf.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "File name is required.\n";
    return 1;
  }
  std::string filename = argv[1];
  std::fstream file;

  char *blob = new char[50000];	// implement smarter mem allocation later
  elfio::File *elfFile = new elfio::File;

  file.open(filename, std::ios_base::in | std::ios_base::binary);
  
  if(file.is_open())
  {
    file.read(blob, 50000);
    // std::cout<<std::hex<<blob<<std::endl;

    *elfFile = elfFile->FromMem(blob);

    file.close();
  }
  else
  {	
    printf("F\n");
    return 1;
  }


  auto texsec = elfFile->GetSectionByName(".text");
  auto texsiz = uint32_t(texsec->size);
  auto texoff = uint32_t(texsec->offset);

  // std::cout << std::endl;
  // std::cout << std::hex << "text section size is " << textsiz << std::endl;
  // std::cout << std::hex << "text section offset is " << textoff << std::endl;
  // std::cout << std::endl;

  char *texhead = new char[texsiz];
  char *texblob = new char;

  std::memcpy(texhead, texsec->Blob(), texsiz);
  texblob = texsec->Blob();

  for(int i = 0; i < 512; i++)
  {
    std::cout << std::hex << texhead[i] << " | " << texblob[i] << std::endl;
  }    

  return 0;
}
