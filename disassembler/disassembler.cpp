#include "disassembler.h"
#include <memory>
#include <iostream>
#include "initialize.h"
#include "notFoundException.h"
#include "operand.h"

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
	char buffer[100];
	sprintf(buffer, "cannot find the instruction format, first two bytes are %04x", firstFourBytes);
	throw NotFoundException(buffer);
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
	char buffer[100];
	sprintf(buffer, "instruction format %s, opcode %d not found\n", format.formatType, opcode);
	throw NotFoundException(buffer);
}

std::unique_ptr<Inst> Disassembler::decode(std::vector<char> buf)
{
	Format format;
	InstType instType;
	try
	{
		format = matchFormat(convertLE(buf));
	}
	catch (NotFoundException &notFoundException)
	{
		std::cout << notFoundException.what() << std::endl;
		throw;
	}
	Opcode opcode = format.retrieveOpcode(convertLE(buf));
	try
	{
		instType = lookUp(format, opcode);
	}
	catch (NotFoundException &notFoundException)
	{
		std::cout << notFoundException.what() << std::endl;
		throw;
	}
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
		decodeSOP2(std::move(inst), buf);
		break;
	case SOP1:
		decodeSOP1(std::move(inst), buf);
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
	inst->src0 = getOperandByCode(uint16_t(src0Value));
	if (inst->src0.operandType == LiteralConstant)
	{
		inst->byteSize += 4;
		if (buf.size() < 8)
		{
			throw;
		}
		std::vector<char> sub(&buf[4], &buf[8]);
		inst->src0.literalConstant = convertLE(sub);
	}
	uint32_t src1Value = extractBitsFromU32(bytes, 8, 15);
	inst->src1 = getOperandByCode(uint16_t(src1Value));
	if (inst->src1.operandType == LiteralConstant)
	{
		inst->byteSize += 4;
		if (buf.size() < 8)
		{
			throw;
		}
		std::vector<char> sub(&buf[4], &buf[8]);
		inst->src1.literalConstant = convertLE(sub);
	}
}