#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H
#include "format.h"
#include "inst.h"
#include <map>
#include "initialize.h"
class Disassembler
{
public:
	void Disassemble();

private:
	void initializeDecodeTable();
	void initFormatList();
	void addInstType(InstType *info);
	Format matchFormat(uint32_t firstFourBytes);
	Format **formatList;
	std::map<FormatType, DecodeTable *> decodeTables;
	int nextInstID;
	InstPrinter printer;
};
#endif