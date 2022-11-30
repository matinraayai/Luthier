#ifndef ASSEMBLER_H
#define ASSEMBLER_H

#include <cstring>
#include <iostream>
#include <sstream>
#include "inst.h"
#include "operand.h"
#include "encodetable.h"

class Assembler
{
public:
    void Assemble(std::string instruction);

private:
    void getInstData(Inst* inst, std::string inststr);
    std::vector<std::string> getInstParams(std::string inststr);

    uint32_t getCodeByOperand(Operand op);
    std::string extractreg(std::string reg);
    Operand getOperandInfo(std::string opstring);

    uint32_t* assembleSOP2(Inst *inst);
    uint32_t* assembleSOP1(Inst *inst);
};



#endif