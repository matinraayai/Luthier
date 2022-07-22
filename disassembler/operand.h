#ifndef OPERAND_H
#define OPERAND_H
#include <string>

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
	Reg reg;
	int regCount;
	double floatValue;
	long int intValue;
	uint32_t literalConstant;
};
Operand newRegOperand(int code, RegType reg, int count);
Operand newSRegOperand(int code, int index, int count);
Operand newVRegOperand(int code, int index, int count);
Operand newIntOperand(int code, long int value);
Operand newFloatOperand(int code, double value);
Operand getOperandByCode(uint16_t num);
#endif