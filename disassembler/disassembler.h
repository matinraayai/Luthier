#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H
#include "format.h"
#include "inst.h"
#include <map>
#include "initialize.h"
#include <memory>
#include <vector>
class Disassembler
{
public:
	void Disassemble();

	Disassembler();

private:
	void initializeDecodeTable();
	void initFormatList();
	void addInstType(std::unique_ptr<InstType> info);
	Format matchFormat(uint32_t firstFourBytes);

	std::vector<Format *> formatList;
	std::map<FormatType, std::unique_ptr<DecodeTable>> decodeTables;
	int nextInstID;
	InstPrinter printer;
};
#endif