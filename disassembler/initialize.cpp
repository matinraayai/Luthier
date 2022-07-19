#include "format.h"
#include "reg.h"
#include <map>
#include <memory>

std::map<FormatType, std::unique_ptr<Format>> FormatTable;
std::map<RegType, Reg> Regs;

void initFormatTable()
{
	auto f = std::make_unique<Format>(SOP1, "sop1", 0xBE800000, 0xFF800000, 4, 8, 15);
	FormatTable[SOP1] = std::move(f);
	auto f = std::make_unique<Format>(SOPC, "sopc", 0xBF000000, 0xFF800000, 4, 16, 22);
	FormatTable[SOPC] = std::move(f);
	auto f = std::make_unique<Format>(SOPP, "sopp", 0xBF800000, 0xFF800000, 4, 16, 22);
	FormatTable[SOPP] = std::move(f);
	auto f = std::make_unique<Format>(VOP1, "vop1", 0x7E000000, 0xFE000000, 4, 9, 16);
	FormatTable[VOP1] = std::move(f);
	auto f = std::make_unique<Format>(VOPC, "vopc", 0x7C000000, 0xFE000000, 4, 17, 24);
	FormatTable[VOPC] = std::move(f);
	auto f = std::make_unique<Format>(SMEM, "smem", 0xC0000000, 0xFC000000, 8, 18, 25);
	FormatTable[SMEM] = std::move(f);
	auto f = std::make_unique<Format>(VOP3a, "vop3a", 0xD0000000, 0xFC000000, 8, 16, 25);
	FormatTable[VOP3a] = std::move(f);
	auto f = std::make_unique<Format>(VOP3b, "vop3b", 0xD0000000, 0xFC000000, 8, 16, 25);
	FormatTable[VOP3b] = std::move(f);
	auto f = std::make_unique<Format>(VINTRP, "vintrp", 0xC8000000, 0xFC000000, 4, 16, 17);
	FormatTable[VINTRP] = std::move(f);
	auto f = std::make_unique<Format>(DS, "ds", 0xD8000000, 0xFC000000, 8, 17, 24);
	FormatTable[DS] = std::move(f);
	auto f = std::make_unique<Format>(MUBUF, "mubuf", 0xE0000000, 0xFC000000, 8, 18, 24);
	FormatTable[MUBUF] = std::move(f);
	auto f = std::make_unique<Format>(MTBUF, "mtbuf", 0xE8000000, 0xFC000000, 8, 15, 18);
	FormatTable[MTBUF] = std::move(f);
	auto f = std::make_unique<Format>(MIMG, "mimg", 0xF0000000, 0xFC000000, 8, 18, 24);
	FormatTable[MIMG] = std::move(f);
	auto f = std::make_unique<Format>(EXP, "exp", 0xC4000000, 0xFC000000, 8, 0, 0);
	FormatTable[EXP] = std::move(f);
	auto f = std::make_unique<Format>(FLAT, "flat", 0xDC000000, 0xFC000000, 8, 18, 24);
	FormatTable[FLAT] = std::move(f);
	auto f = std::make_unique<Format>(SOPK, "sopk", 0xB0000000, 0xF0000000, 4, 23, 27);
	FormatTable[SOPK] = std::move(f);
	auto f = std::make_unique<Format>(SOP2, "sop2", 0x80000000, 0xC0000000, 4, 23, 29);
	FormatTable[SOP2] = std::move(f);
	auto f = std::make_unique<Format>(VOP2, "vop2", 0x00000000, 0x80000000, 4, 25, 30);
	FormatTable[VOP2] = std::move(f);
	auto f = std::make_unique<Format>(VOP3P, "vop3p", 0xCC000000, 0xFC000000, 8, 16, 22);
	FormatTable[VOP3P] = std::move(f);
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
