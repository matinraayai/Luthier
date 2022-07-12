#ifndef OPERAND_H
#define OPERAND_H
#include <string>
#include <iomanip>
#include <sstream>
#include "reg.h"

enum OperandType
{
	InvalidOperantType,
	RegOperand,
	FloatOperand,
	IntOperand,
	LiteralConstant
};
struct Operand
{
	int code;
	OperandType operandType;
	Reg *reg;
	int regCount;
	double floatValue;
	long int intValue;
	uint32_t literalConstant;
	std::string regOperandToString()
	{
		char buffer[50];
		if (regCount > 1)
		{
			if (reg->IsSReg())
			{
				std::sprintf(buffer, "s[%d:%d]", reg->RegIndex(), reg->RegIndex() + regCount - 1);
				return buffer;
			}
			else if (reg->IsVReg())
			{
				std::sprintf(buffer, "v[%d:%d]", reg->RegIndex(), reg->RegIndex() + regCount - 1);
				return buffer;
			}
			else if (reg->name.find("lo") != std::string::npos)
			{
				return reg->name.substr(0, reg->name.length() - 2);
			}
			return "unknown register";
		}
		return reg->name;
	}
	std::string String()
	{
		std::stringstream stream;
		switch (operandType)
		{
		case RegOperand:
			return regOperandToString();
		case IntOperand:
			return std::to_string(intValue);
		case FloatOperand:
			return std::to_string(floatValue);
		case LiteralConstant:
			stream << "0x" << std::hex << literalConstant;
			return stream.str();
		default:
			return "";
		}
	}
};
void newRegOperand(Operand *o, int code, RegType reg, int count);
void newSRegOperand(Operand *o, int code, int index, int count);
void newVRegOperand(Operand *o, int code, int index, int count);
void newIntOperand(Operand *o, int code, long int value);
void newFloatOperand(Operand *o, int code, double value);
#endif