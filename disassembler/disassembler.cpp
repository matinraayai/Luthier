#include "disassembler.h"
#include <memory>
#include <iostream>
#include "initialize.h"

Disassembler::Disassembler()
{
	nextInstID = 0;
	initFormatList();
	initializeDecodeTable();
}

void Disassembler::addInstType(InstType *info)
{
	if (decodeTables.find(info->format->formatType) == decodeTables.end())
	{
		decodeTables[info->format->formatType] = std::unique_ptr<DecodeTable>(new DecodeTable);
	}
	(decodeTables[info->format->formatType])->insts[info->opcode] = std::move(info);
	info->ID = nextInstID;
	nextInstID++;
}

void Disassembler::initFormatList()
{
	for (auto &item : FormatTable)
	{
		formatList.push_back(item.second);
	}
}
void Disassembler::Disassemble(elfio::File *file, std::string filename)
{
	printer.file = file;
}

Format *Disassembler::matchFormat(uint32_t firstFourBytes)
{
	for (auto &f : formatList)
	{
		if (f->formatType == VOP3b)
		{
			continue;
		}
		if ((firstFourBytes ^ f->encoding) & f->mask == 0)
		{
			if (f->formatType == VOP3a)
			{
				auto opcode = f->retrieveOpcode(firstFourBytes);
				if (isVOP3bOpcode(opcode))
				{
					return FormatTable[VOP3b];
				}
			}
			return f;
		}
	}
	printf("cannot find the instruction format, first two bytes are %04x", firstFourBytes);
}
bool Disassembler::isVOP3bOpcode(Opcode opcode)
{
	switch (opcode)
	{
	case 281:
		return true;
	case 282:
		return true;
	case 283:
		return true;
	case 284:
		return true;
	case 285:
		return true;
	case 286:
		return true;
	case 480:
		return true;
	case 481:
		return true;
	}
	return false;
}
InstType *Disassembler::lookUp(Format *format, Opcode opcode)
{
	if (decodeTables.find(format->formatType) != decodeTables.end() && decodeTables[format->formatType]->insts.find(opcode) != decodeTables[format->formatType]->insts.end())
	{
		return decodeTables[format->formatType]->insts[opcode];
	}
	std::cerr << "instruction format " << format->formatType << ", opcode " << opcode << " not found\n";
}
Inst *Disassembler::decode(char *blob)
{
	Format *format = matchFormat(convertLE(blob));
}