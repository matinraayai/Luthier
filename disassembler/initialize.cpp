#include "format.h"
#include "initialize.h"
#include <map>

std::map<FormatType, Format *> FormatTable;
std::map<RegType, Reg> Regs;

void initFormatTable()
{
	Format f = {SOP1, "sop1", 0xBE800000, 0xFF800000, 4, 8, 15};
	FormatTable[SOP1] = &f;
	f = {SOPC, "sopc", 0xBF000000, 0xFF800000, 4, 16, 22};
	FormatTable[SOPC] = &f;
	f = {SOPP, "sopp", 0xBF800000, 0xFF800000, 4, 16, 22};
	FormatTable[SOPP] = &f;
	f = {VOP1, "vop1", 0x7E000000, 0xFE000000, 4, 9, 16};
	FormatTable[VOP1] = &f;
	f = {VOPC, "vopc", 0x7C000000, 0xFE000000, 4, 17, 24};
	FormatTable[VOPC] = &f;
	f = {SMEM, "smem", 0xC0000000, 0xFC000000, 8, 18, 25};
	FormatTable[SMEM] = &f;
	f = {VOP3a, "vop3a", 0xD0000000, 0xFC000000, 8, 16, 25};
	FormatTable[VOP3a] = &f;
	f = {VOP3b, "vop3b", 0xD0000000, 0xFC000000, 8, 16, 25};
	FormatTable[VOP3b] = &f;
	f = {VINTRP, "vintrp", 0xC8000000, 0xFC000000, 4, 16, 17};
	FormatTable[VINTRP] = &f;
	f = {DS, "ds", 0xD8000000, 0xFC000000, 8, 17, 24};
	FormatTable[DS] = &f;
	f = {MUBUF, "mubuf", 0xE0000000, 0xFC000000, 8, 18, 24};
	FormatTable[MUBUF] = &f;
	f = {MTBUF, "mtbuf", 0xE8000000, 0xFC000000, 8, 15, 18};
	FormatTable[MTBUF] = &f;
	f = {MIMG, "mimg", 0xF0000000, 0xFC000000, 8, 18, 24};
	FormatTable[MIMG] = &f;
	f = {EXP, "exp", 0xC4000000, 0xFC000000, 8, 0, 0};
	FormatTable[EXP] = &f;
	f = {FLAT, "flat", 0xDC000000, 0xFC000000, 8, 18, 24};
	FormatTable[FLAT] = &f;
	f = {SOPK, "sopk", 0xB0000000, 0xF0000000, 4, 23, 27};
	FormatTable[SOPK] = &f;
	f = {SOP2, "sop2", 0x80000000, 0xC0000000, 4, 23, 29};
	FormatTable[SOP2] = &f;
	f = {VOP2, "vop2", 0x00000000, 0x80000000, 4, 25, 30};
	FormatTable[VOP2] = &f;
	f = {VOP2, "vop3p", 0xCC000000, 0xFC000000, 8, 16, 22};
	FormatTable[VOP3P] = &f;
}
void initRegs()
{
	Regs.insert({InvalidRegType, {InvalidRegType, "invalidregtype", 0, false}});
	Regs.insert({PC, {PC, "pc", 8, false}});
	Regs.insert({V0, {V0, "v0", 4, false}});
}

Reg *VReg(int index)
{
	return &Regs[RegType(index + (int)V0)];
}
Reg *SReg(int index)
{
	return &Regs[RegType(index + (int)S0)];
}