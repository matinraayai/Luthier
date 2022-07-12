#include "operand.h"
#include "initialize.h"

Operand *newRegOperand(int code, RegType reg, int count)
{
	Operand *o;
	o->code = code;
	o->operandType = RegOperand;
	o->reg = &Regs[reg];
	o->regCount = count;
	return o;
}
Operand *newSRegOperand(int code, int index, int count)
{
	Operand *o;
	o->code = code;
	o->operandType = RegOperand;
	o->reg = SReg(index);
	o->regCount = count;
	return o;
}
Operand *newVRegOperand(int code, int index, int count)
{
	Operand *o;
	o->code = code;
	o->operandType = RegOperand;
	o->reg = VReg(index);
	o->regCount = count;
	return o;
}
Operand *newIntOperand(int code, long int value)
{
	Operand *o;
	o->code = code;
	o->operandType = IntOperand;
	o->intValue = value;
	return o;
}
Operand *newFloatOperand(int code, double value)
{
	Operand *o;
	o->code = code;
	o->operandType = FloatOperand;
	o->floatValue = value;
	return o;
}