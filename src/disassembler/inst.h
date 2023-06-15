#ifndef INST_H
#define INST_H

#include <string>
#include "operand.h"
#include "elf.h"
#include "bitops.h"

typedef uint16_t Opcode;

enum FormatType
{
	SOP2,
	SOPK,
	SOP1,
	SOPC,
	SOPP,
	SMEM,
	VOP2,
	VOP1,
	VOP3a,
	VOP3b,
	VOP3P,
	VOPC,
	VINTRP,
	DS,
	MUBUF,
	MTBUF,
	MIMG,
	FLAT
};
struct Format
{
	std::string formatName;
	uint32_t encoding;
	uint32_t mask;
	int byteSizeExLiteral;
	uint8_t opcodeLow;
	uint8_t opcodeHigh;

	Opcode retrieveOpcode(uint32_t firstFourBytes)
	{
		uint32_t opcode = extractBitsFromU32(firstFourBytes, int(opcodeLow), int(opcodeHigh));
		return Opcode(opcode);
	}
};
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
	Format format;
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
	Format format;
	InstType instType;
	int byteSize{};
	uint64_t PC{};
	Operand src0;
	Operand src1;
	Operand src2;
	Operand dst;
	Operand sdst;
	Operand addr;
	Operand sAddr;
	Operand data;
	Operand data1;
	Operand base;
	Operand offset;
	Operand simm16;

	int Abs{};
	int Omod{};
	int Neg{};
	int Offset0{};
	int Offset1{};
	bool SystemLevelCoherent{};
	bool GlobalLevelCoherent{};
	bool TextureFailEnable{};
	bool Imm{};
	bool Clamp{};
	bool GDS{};
	int Seg{};
	int VMCNT{};
	int VSCNT{};
	int LKGMCNT{};
	int EXPCNT{};

	bool IsSdwa{};
	int DstSel{};
	int DstUnused{};
	int Src0Sel{};
	bool Src0Sext{};
	bool Src0Abs{};
	bool Src0Neg{};
	int Src1Sel{};
	bool Src1Sext{};
	bool Src1Abs{};
	bool Src1Neg{};
	bool Src2Abs{};
	bool Src2Neg{};
};




struct instnode {
  instnode *prev;
  instnode *next;

  std::string instStr;
  std::vector<unsigned char> bytes;

  int byteSize;
  uint64_t pc;
};

#endif