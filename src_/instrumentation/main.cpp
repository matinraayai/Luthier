#include "disassembler.h"
#include "elf.hpp"
#include "parser.h"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "File name is required.\n";
  }
  std::string filename = argv[1];
  // processBundle(filename);
  elfio::File elfFile = extractFromBundle(filename);
  Disassembler d(&elfFile);
  d.Disassemble(&elfFile, filename, std::cout);
  std::cout << "The maximum number of sReg is " << std::setbase(10)
            << d.maxNumSReg() << "\n";
  std::cout << "The maximum number of vReg is " << d.maxNumVReg() << "\n";
}