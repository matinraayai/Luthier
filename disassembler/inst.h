#ifndef INST_H
#define INST_H
#include <string>
#include "format.h"
#include "operand.h"
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
struct Inst
{
	Format *format;
	InstType *instType;
	int byteSize;
	uint64_t PC;
	Operand *src0;
	Operand *src1;
	Operand *src2;
	Operand *dst;
	Operand *sdst;
	Operand *addr;
	Operand *sAddr;
	Operand *data;
	Operand *data1;
	Operand *base;
	Operand *offset;
	Operand *simm16;

	int Abs;
	int Omod;
	int Neg;
	int Offset0;
	int Offset1;
	bool SystemLevelCoherent;
	bool GlobalLevelCoherent;
	bool TextureFailEnable;
	bool Imm;
	bool Clamp;
	bool GDS;
	int Seg;
	int VMCNT;
	int VSCNT;
	int LKGMCNT;
};
class InstPrinter
{
};
#endif