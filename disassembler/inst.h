#ifndef INST_H
#define INST_H
#include <string>
#include "format.h"
typedef uint16_t Opcode;
enum ExeUnit
{
	ExeUnitVALU,
	ExeUnitScalar,
	ExeUnitVMem,
	ExeUnitBranch,
	ExeUnitLDS,
	ExeUnitGDS,
	ExeUnitSpecial
};
struct InstType
{
	std::string instName;
	Opcode opcode;
	Format *format;
	int ID;
	ExeUnit exeUnit;
	int DSTWidth;
	int SRC0Width;
	int SRC1Width;
	int SRC2Width;
	int SDSTWidth;
};
#endif