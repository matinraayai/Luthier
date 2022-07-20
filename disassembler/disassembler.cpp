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

void Disassembler::addInstType(InstType info)
{
	if (decodeTables.find(info.format.formatType) == decodeTables.end())
	{
		decodeTables[info.format.formatType] = std::unique_ptr<DecodeTable>(new DecodeTable);
	}
	(decodeTables[info.format.formatType])->insts[info.opcode] = info;
	info.ID = nextInstID;
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

Format Disassembler::matchFormat(uint32_t firstFourBytes)
{
	for (auto &f : formatList)
	{
		if (f.formatType == VOP3b)
		{
			continue;
		}
		if ((firstFourBytes ^ f.encoding) & f.mask == 0)
		{
			if (f.formatType == VOP3a)
			{
				auto opcode = f.retrieveOpcode(firstFourBytes);
				if (isVOP3bOpcode(opcode))
				{
					return FormatTable[VOP3b];
				}
			}
			return f;
		}
	}
	printf("cannot find the instruction format, first two bytes are %04x", firstFourBytes);
	return nullptr;
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
InstType Disassembler::lookUp(Format format, Opcode opcode)
{
	if (decodeTables.find(format.formatType) != decodeTables.end() && decodeTables[format.formatType]->insts.find(opcode) != decodeTables[format.formatType]->insts.end())
	{
		return decodeTables[format.formatType]->insts[opcode];
	}
	std::cerr << "instruction format " << format.formatType << ", opcode " << opcode << " not found\n";
	return nullptr;
}

std::unique_ptr<Inst> Disassembler::decode(std::vector<char> buf)
{
	int err = 0;
	Format format = matchFormat(convertLE(buf));

	Opcode opcode = format.retrieveOpcode(convertLE(buf));
	InstType instType = lookUp(format, opcode);

	auto inst = std::make_unique<Inst>();
	inst->format = format;
	inst->instType = instType;
	inst->byteSize = format.byteSizeExLiteral;

	if (inst->byteSize > buf.size())
	{
		std::cerr << "no enough buffer\n";
		return nullptr;
	}

	switch (format.formatType)
	{
	case SOP2:
		err = decodeSOP2(std::move(inst), buf);
		break;
	case SOP1:
		err = decodeSOP1(std::move(inst), buf);
		break;
	}
	if (err != 0)
	{
		std::cerr << "unable to decode instruction type " << format.formatName;
		return nullptr;
	}
	return inst;
}
int Disassembler::decodeSOP2(std::unique_ptr<Inst> inst, std::vector<char> buf)
{
	uint32_t bytes = convertLE(buf);
	uint32_t src0Value = extractBitsFromU32(bytes, 0, 7);
}