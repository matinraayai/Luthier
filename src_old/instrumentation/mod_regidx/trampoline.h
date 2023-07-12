#include "disassembler.hpp"
#include <memory>
#include <string>
#include <vector>
char *getELF(std::string filename);
std::unique_ptr<Inst>
replaceTargetInst(std::vector<std::unique_ptr<Inst>> &insts, int idx,
                  std::unique_ptr<Inst> replace);
void modifyInstruInst(std::vector<std::unique_ptr<Inst>> &insts, int idx);
int getNewTextSize(std::vector<std::unique_ptr<Inst>> &instsP,
                   std::vector<std::unique_ptr<Inst>> &instsI,
                   std::vector<std::unique_ptr<Inst>> &instsT);
void getNewTextBinary(unsigned char *blob,
                      std::vector<std::unique_ptr<Inst>> &instsP,
                      std::vector<std::unique_ptr<Inst>> &instsI,
                      std::vector<std::unique_ptr<Inst>> &instsT);