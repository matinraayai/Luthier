#include "disassembler.h"
#include <memory>
#include <iostream>
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
	auto text_section = file->GetSectionByName(".text");
	if (!text_section)
	{
		throw std::runtime_error("text section is not found");
	}
	std::vector<char> buf(text_section->Blob(), text_section->Blob() + text_section->size);
	auto pc = (uint64_t)0x100;
	if (!buf.empty())
	{
		std::unique_ptr<Inst> inst = decode(buf);
		inst->PC = pc + text_section->offset;
		std::string instStr = printer.print(inst.get());
		std::cout << "\t" << instStr;
		for (int i = instStr.size(); i < 59; i++)
		{
			std::cout << " ";
		}
		std::cout << "//" << std::setw(12) << std::setfill('0') << text_section->offset + pc << ": ";
		std::cout << std::setw(8) << std::hex << convertLE(buf);
		if (inst->byteSize == 8)
		{
			std::vector<char> sfb(buf.begin() + 4, buf.begin() + 8);
			std::cout << std::setw(8) << std::hex << convertLE(sfb) << std::endl;
		}
		buf.erase(buf.begin(), buf.begin() + inst->byteSize);
		pc += uint64_t(inst->byteSize);
	}
}

Format Disassembler::matchFormat(uint32_t firstFourBytes)
{
	for (auto &f : formatList)
	{
		if (f.formatType == VOP3b)
		{
			continue;
		}
		if (((firstFourBytes ^ f.encoding) & f.mask) == 0)
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
	throw std::runtime_error(buffer);
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
	throw std::runtime_error(buffer);
}

std::unique_ptr<Inst> Disassembler::decode(std::vector<char> buf)
{
	Format format = matchFormat(convertLE(buf));

	Opcode opcode = format.retrieveOpcode(convertLE(buf));

	InstType instType = lookUp(format, opcode);

	auto inst = std::make_unique<Inst>();
	inst->format = format;
	inst->instType = instType;
	inst->byteSize = format.byteSizeExLiteral;

	if (inst->byteSize > buf.size())
	{
		throw std::runtime_error("no enough buffer");
	}

	switch (format.formatType)
	{
	case SOP2:
		decodeSOP2(inst.get(), buf);
		break;
	case SMEM:
		decodeSMEM(inst.get(), buf);
		break;
	}
	// if (err != 0)
	// {
	// 	std::cerr << "unable to decode instruction type " << format.formatName;
	// 	return nullptr;
	// }
	return std::move(inst);
}
void Disassembler::decodeSOP2(Inst *inst, std::vector<char> buf)
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
void Disassembler::decodeSMEM(Inst *inst, std::vector<char> buf)
{
	auto bytesLo = convertLE(buf);
	std::vector<char> sfb(buf.begin() + 4, buf.end());
	auto bytesHi = convertLE(sfb);

	if (extractBitsFromU32(bytesLo, 16, 16) != 0)
	{
		inst->GlobalLevelCoherent = true;
	}

	if (extractBitsFromU32(bytesLo, 17, 17) != 0)
	{
		inst->Imm = true;
	}

	auto sbaseValue = extractBitsFromU32(bytesLo, 0, 5);
	auto bits = int(sbaseValue << 1);
	inst->base = newSRegOperand(bits, bits, 2);

	auto sdataValue = extractBitsFromU32(bytesLo, 6, 12);
	inst->data = getOperandByCode(uint16_t(sdataValue));
	if (inst->data.operandType == LiteralConstant)
	{
		inst->byteSize += 4;
		if (buf.size() < 8)
		{
			throw std::runtime_error("no enough bytes");
		}
		std::vector<char> nfb(buf.begin() + 8, buf.begin() + 12);
		inst->data.literalConstant = convertLE(nfb);
	}

	if (inst->instType.opcode == 0)
	{
		inst->data.regCount = 1;
	}
	else if (inst->instType.opcode == 1 || inst->instType.opcode == 9 || inst->instType.opcode == 17 || inst->instType.opcode == 25)
	{
		inst->data.regCount = 2;
	}
	else if (inst->instType.opcode == 2 || inst->instType.opcode == 10 || inst->instType.opcode == 18 || inst->instType.opcode == 26)
	{
		inst->data.regCount = 4;
	}
	else if (inst->instType.opcode == 3 || inst->instType.opcode == 11 || inst->instType.opcode == 19 || inst->instType.opcode == 27)
	{
		inst->data.regCount = 8;
	}
	else
	{
		inst->data.regCount = 16;
	}

	if (inst->Imm)
	{
		uint64_t bits64 = (uint64_t)extractBitsFromU32(bytesHi, 0, 19);
		inst->offset = newIntOperand(0, bits64);
	}
	else
	{
		int bits = (int)extractBitsFromU32(bytesHi, 0, 19);
		inst->offset = newSRegOperand(bits, bits, 1);
	}
}