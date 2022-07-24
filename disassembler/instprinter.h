#ifndef INSTPRINTER_H
#define INSTPRINTER_H
#include "../src/elf.h"
#include <string>
#include <iomanip>
#include <sstream>
#include "inst.h"
#include "operand.h"
struct InstPrinter
{
	elfio::File *file;
	std::string operandString(Operand o)
	{
		std::stringstream stream;
		switch (o.operandType)
		{
		case RegOperand:
			return regOperandToString(o);
		case IntOperand:
			return std::to_string(o.intValue);
		case FloatOperand:
			return std::to_string(o.floatValue);
		case LiteralConstant:
			stream << "0x" << std::hex << o.literalConstant;
			return stream.str();
		default:
			return "";
		}
	}
	std::string regOperandToString(Operand o)
	{
		std::stringstream stream;
		if (o.regCount > 1)
		{
			if (o.reg.IsSReg())
			{
				stream << "s[" << o.reg.RegIndex() << ":" << o.reg.RegIndex() + o.regCount - 1 << "]";
				return stream.str();
			}
			else if (o.reg.IsVReg())
			{
				stream << "v[" << o.reg.RegIndex() << ":" << o.reg.RegIndex() + o.regCount - 1 << "]";
				return stream.str();
			}
			else if (o.reg.name.find("lo") != std::string::npos)
			{
				return o.reg.name.substr(0, o.reg.name.length() - 2);
			}
			return "unknown register";
		}
		return o.reg.name;
	}
	std::string sop2String(Inst *i)
	{
	}
	std::string smemString(Inst *i)
	{
		std::stringstream stream;
		stream << i->instType.instName << " " << operandString(i->data) << ", " << operandString(i->base) << ", " << std::hex << uint16_t(i->offset.intValue);
		return stream.str();
	}
	std::string print(Inst *i)
	{
		switch (i->format.formatType)
		{
		case SOP2:
			return sop2String(i);
		case SMEM:
			return smemString(i);
		}
	}
};
#endif