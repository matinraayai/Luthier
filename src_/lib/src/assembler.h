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
    void Assemble(std::string inststr, std::ostream &o);

    void editDSTreg(instnode *inst, std::string reg);
    void editSRC0reg(instnode *inst, std::string reg);
    void editSRC1reg(instnode *inst, std::string reg);
    void editSIMM(instnode *inst, short simm);

private:
    void initEncodeTable();
    void  getInstData(std::string inststr, Inst* inst);

    std::vector<std::string> getInstParams(std::string inststr);
    std::string extractGPRstr(std::string reg);
    uint32_t extractGPRbyte(std::string reg);
    uint32_t getCodeByOperand(Operand op);
    Operand getOperandInfo(std::string opstring);

    std::vector<uint32_t> assembleSOP1(Inst *inst);
    std::vector<uint32_t> assembleVOP1(Inst *inst);
    std::vector<uint32_t> assembleSMEM(Inst *inst);
    std::vector<uint32_t> assembleSOP2(Inst *inst);
    std::vector<uint32_t> assembleSOPP(Inst *inst);
};



#endif
