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

std::vector<unsigned char> 
  newKernel(elfio::Section *pgrm, elfio::Section *instru);

void printInstruFn(std::vector<std::shared_ptr<Inst>> instList);
void offsetInstruRegs(std::vector<std::shared_ptr<Inst>> instList,
                      Assembler a, int sRegMax, int vRegMax);
void makeTrampoline(std::vector<std::shared_ptr<Inst>> &instList, 
                    Assembler a, uint64_t inum);

