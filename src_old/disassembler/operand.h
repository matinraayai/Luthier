#ifndef OPERAND_H
#define OPERAND_H
#include <string>
#include <vector>
#include "reg.h"
extern std::vector<int> sRegNum;
extern std::vector<int> vRegNum;

struct Operand
{
 public:
    enum OperandType
    {
        InvalidOperandType,
        RegOperand,
        FloatOperand,
        IntOperand,
        LiteralConstant
    };

    static Operand makeRegOperand(int code, RegType reg, int count);
    static Operand makeSRegOperand(int code, int index, int count);
    static Operand makeVRegOperand(int code, int index, int count);
    static Operand makeIntOperand(int code, long int value);
    static Operand makeFloatOperand(int code, double value);
    static Operand makeOperandByCode(uint16_t num);
    std::string toString() const;
 private:
    std::string regOperandToString() const;


    Operand() = default;
	int code_{};
	OperandType operandType_{};
	RegType reg_{};
	int regCount_{};
	double floatValue_{};
	long int intValue_{};
	uint32_t literalConstant_{};
};



#endif