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

enum OperandType {
    InvalidOperantType,
	RegOperand,
	ImmOperand,
	LiteralConstant
};

struct operand {
    int code;
    OperandType optype;
    long val[2];
};

class sibir_instr {
private:
    sibir_address_t addr;
    inst_size_t size;
    std::string str;
    
    std::vector<operand> operands;
    // std::vector<operand> GetOperandsFromString(std::string inst_string);
    void GetOperandsFromString(std::string inst_string);

public:
    sibir_instr(sibir_address_t inst_addr, inst_size_t inst_size, std::string inst_string);
    sibir_instr(Instr inst);

    std::vector<operand> getOperands();
    std::vector<operand> getImmOperands();
    std::vector<operand> getRegOperands();

    OperandType getOperandType(unsigned int operand_num);
};



#endif