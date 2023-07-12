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
    void Assemble(std::string inststr, std::shared_ptr<Inst> &inst);
    char* Assemble(std::string inststr);
    std::shared_ptr<Inst> Assemble(std::string inststr, uint64_t pc);
    void offsetRegs(std::shared_ptr<Inst> i, int smax, int vmax);
    std::vector<unsigned char> ilstbuf(std::vector<std::shared_ptr<Inst>> ilst);

private:
    void initEncodeTable();
    void getInstData(std::string inststr, Inst* inst);

    std::vector<std::string> getInstParams(std::string inststr);
    std::string extractGPRstr(std::string reg);
    uint32_t extractGPRbyte(std::string reg);
    uint32_t getCodeByOperand(Operand op);
    Operand getOperandInfo(std::string opstring);

    void editSALUinst(std::shared_ptr<Inst> i, int smax);
    void editVALUinst(std::shared_ptr<Inst> i, int smax, int vmax);
    void editFLATinst(std::shared_ptr<Inst> i, int smax, int vmax);

    void editSRC0reg(std::shared_ptr<Inst> inst, int code);
    void editSRC0flat(std::shared_ptr<Inst> inst, int code, bool data);
    void editSRC1reg(std::shared_ptr<Inst> inst, int code);
    void editDSTreg(std::shared_ptr<Inst> inst, int code);
    void editSIMM(Inst *inst, short simm);

    uint32_t assembleSOP2(Inst *inst);
    uint32_t assembleSOP1(Inst *inst);
    uint32_t assembleSOPP(Inst *inst);
};

#endif
