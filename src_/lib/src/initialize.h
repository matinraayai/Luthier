#ifndef INITIALIZE_H
#define INTIIALIZE_H
#include "reg.h"
#include "inst.h"
#include <map>
#include <memory>
extern std::map<FormatType, Format> FormatTable;
extern std::map<RegType, Reg> Regs;
void initFormatTable();
void initRegs();
Reg VReg(int index);
Reg SReg(int index);

#endif