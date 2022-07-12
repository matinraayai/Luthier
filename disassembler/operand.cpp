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
void newSRegOperand(Operand *o, int code, int index, int count)
{
	o->code = code;
	o->operandType = RegOperand;
	o->reg = SReg(index);
	o->regCount = count;
}
void newVRegOperand(Operand *o, int code, int index, int count)
{
	o->code = code;
	o->operandType = RegOperand;
	o->reg = VReg(index);
	o->regCount = count;
}
void newIntOperand(Operand *o, int code, long int value)
{
	o->code = code;
	o->operandType = IntOperand;
	o->intValue = value;
}
void newFloatOperand(Operand *o, int code, double value)
{
	o->code = code;
	o->operandType = FloatOperand;
	o->floatValue = value;
}