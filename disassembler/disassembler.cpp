#include "disassembler.h"
#include <memory>
#include "initialize.h"

Disassembler::Disassembler()
{
	nextInstID = 0;
	initFormatList();
	initializeDecodeTable();
}

void Disassembler::addInstType(std::unique_ptr<InstType> info)
{
	if (decodeTables.find(info->format->formatType) == decodeTables.end())
	{
		decodeTables[info->format->formatType] = std::unique_ptr<DecodeTable>(new DecodeTable);
	}
	(decodeTables[info->format->formatType])->insts[info->opcode] = std::move(info);
	info->ID = this->nextInstID;
	this->nextInstID++;
}

void Disassembler::initFormatList()
{
	for (auto &item : FormatTable)
	{
		formatList.push_back(item.second);
	}
}
