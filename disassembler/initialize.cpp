#include "inst.h"
#include "reg.h"
#include <map>

std::map<FormatType, Format> FormatTable;
std::map<RegType, Reg> Regs;

void initFormatTable()
{
	FormatTable[SOP1] = Format{SOP1, "sop1", 0xBE800000, 0xFF800000, 4, 8, 15};
	FormatTable[SOPC] = Format{SOPC, "sopc", 0xBF000000, 0xFF800000, 4, 16, 22};
	FormatTable[SOPP] = Format{SOPP, "sopp", 0xBF800000, 0xFF800000, 4, 16, 22};
	FormatTable[VOP1] = Format{VOP1, "vop1", 0x7E000000, 0xFE000000, 4, 9, 16};
	FormatTable[VOPC] = Format{VOPC, "vopc", 0x7C000000, 0xFE000000, 4, 17, 24};
	FormatTable[SMEM] = Format{SMEM, "smem", 0xC0000000, 0xFC000000, 8, 18, 25};
	FormatTable[VOP3a] = Format{VOP3a, "vop3a", 0xD0000000, 0xFC000000, 8, 16, 25};
	FormatTable[VOP3b] = Format{VOP3b, "vop3b", 0xD0000000, 0xFC000000, 8, 16, 25};
	FormatTable[VINTRP] = Format{VINTRP, "vintrp", 0xC8000000, 0xFC000000, 4, 16, 17};
	FormatTable[DS] = Format{DS, "ds", 0xD8000000, 0xFC000000, 8, 17, 24};
	FormatTable[MUBUF] = Format{MUBUF, "mubuf", 0xE0000000, 0xFC000000, 8, 18, 24};
	FormatTable[MTBUF] = Format{MTBUF, "mtbuf", 0xE8000000, 0xFC000000, 8, 15, 18};
	FormatTable[MIMG] = Format{MIMG, "mimg", 0xF0000000, 0xFC000000, 8, 18, 24};
	FormatTable[FLAT] = Format{FLAT, "flat", 0xDC000000, 0xFC000000, 8, 18, 24};
	FormatTable[SOPK] = Format{SOPK, "sopk", 0xB0000000, 0xF0000000, 4, 23, 27};
	FormatTable[SOP2] = Format{SOP2, "sop2", 0x80000000, 0xC0000000, 4, 23, 29};
	FormatTable[VOP2] = Format{VOP2, "vop2", 0x00000000, 0x80000000, 4, 25, 30};
	FormatTable[VOP3P] = Format{VOP3P, "vop3p", 0xCC000000, 0xFC000000, 8, 16, 22};
}
void initRegs()
{
	Regs.insert({InvalidRegType, {InvalidRegType, "invalidregtype", 0, false}});
	Regs.insert({PC, {PC, "pc", 8, false}});
	Regs.insert({V0, {V0, "v0", 4, false}});
}

Reg VReg(int index)
{
	return Regs[RegType(index + (int)V0)];
}
Reg SReg(int index)
{
	return Regs[RegType(index + (int)S0)];
}
