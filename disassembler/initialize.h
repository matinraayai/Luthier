#ifndef INITIALIZE_H
#define INTIIALIZE_H
#include "reg.h"
void initFormatTable();
void initRegs();
Reg *VReg(int index);
Reg *SReg(int index);
#endif