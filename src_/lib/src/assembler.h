#ifndef ASSEMBLER_H
#define ASSEMBLER_H

#include <cstring>
#include <iostream>
#include <sstream>
#include "initialize.h"
#include "inst.h"
#include "operand.h"
#include "encodetable.h"

class Assembler
{
public:
    Assembler();
    std::vector<unsigned char> Assemble(std::string instruction);
    
    std::vector<std::string> getInstParams(std::string inststr);
    std::string extractreg(std::string reg);

private:
    void initEncodeTable();
    void getInstData(Inst* inst, std::string inststr);

    uint32_t getCodeByOperand(Operand op);
    Operand getOperandInfo(std::string opstring);

    std::vector<uint32_t> assembleSOP1(Inst *inst);
    std::vector<uint32_t> assembleVOP1(Inst *inst);
    std::vector<uint32_t> assembleSMEM(Inst *inst);
    std::vector<uint32_t> assembleSOP2(Inst *inst);
};



#endif