#ifndef SIBIR_INSTR_H
#define SIBIR_INSTR_H

#include <string>
#include <vector>

#include "sibir_types.h"

typedef uint64_t sibir_address_t;
typedef uint64_t inst_size_t;

enum memOpType {
    NONE,
    LOCAL,
    GENERIC,
    GLOBAL,
    SHARED,
    TEXTURE,
    CONSTANT
};

enum OperandType {      // I ported these over from our old Inst.h
    InvalidOperandType,
	RegOperand,
    SpecialRegOperand,
	ImmOperand,
	LiteralConstant
};

struct operand {
    std::string op_str;
    int code;
    OperandType type;
    unsigned long val;
};

class sibir_instr {
private:
    sibir_address_t addr;
    inst_size_t size;
    std::string str;

    memOpType memtype;
    
    std::vector<operand> operands;
    std::vector<operand> GetOperandsFromString(std::string inst_string);
    operand EncodeOperand(std::string op);

public:
    sibir_instr(sibir_address_t inst_addr, inst_size_t inst_size, std::string inst_string);
    sibir_instr(Instr inst);

    std::string getInstString();
    int getNumOperands();
    operand getOperand(int num);
    OperandType getOperandType(int num);

    std::vector<operand> getAllOperands();
    std::vector<operand> getImmOperands();
    std::vector<operand> getRegOperands();
};



#endif