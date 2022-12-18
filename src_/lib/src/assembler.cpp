#include <iostream>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include "bitops.h"
#include "assembler.h"

Assembler::Assembler()
{
    initFormatTable();
    initRegs();
    initEncodeTable();
}

std::vector<unsigned char> Assembler::Assemble(std::string instruction)
{
    Inst *inst = new Inst;
    std::vector<uint32_t> assembly;

    getInstData(inst, instruction);

    switch (inst->instType.format.formatType)
    {
    case SOP1:
        assembly = assembleSOP1(inst);
        break;
    case VOP1:
        assembly = assembleVOP1(inst);
        break;
    case SMEM:
        assembly = assembleSMEM(inst);
        break;
    case SOP2:
        assembly = assembleSOP2(inst);
        break;
    default:
        break;
    }

    return instcodeToByteArray(assembly);
}

void Assembler::getInstData(Inst* inst, std::string inststr)
{
    std::vector<std::string> params = getInstParams(inststr);

    std::string iname = params.at(0);
    auto index = iname.find("_e32");
    if(index != std::string::npos)
    {
        iname.erase(index, index+3);
    }

    inst->instType = EncodeTable[iname];
    inst->dst = getOperandInfo(params.at(1));

    if(inst->instType.SRC0Width != 0)
        inst->src0 = getOperandInfo(params.at(2));
    if(inst->instType.SRC1Width != 0)
        inst->src1 = getOperandInfo(params.at(3));
    if(inst->instType.SRC2Width != 0)
        inst->src2 = getOperandInfo(params.at(4));
}

std::vector<std::string> Assembler::getInstParams(std::string inststr)
{
    std::vector<std::string> params;
    std::string delim = " ";
    
    size_t i;
    while(i != std::string::npos)
    {
        i = inststr.find(delim);
        params.push_back(inststr.substr(0, i));
        inststr.erase(0, i+1);
    }

    return params;
}

uint32_t Assembler::getCodeByOperand(Operand op)
{
    if(op.operandType == RegOperand)
    {
        return uint32_t(op.code);
    }
    else if(op.operandType == IntOperand)
    {
        return uint32_t(op.code);
    }
}

std::string Assembler::extractreg(std::string reg)
{
    if(reg.find("v") != std::string::npos) 
    {
        reg.erase(reg.find("v"), 1);
    }
    else if(reg.find("s") != std::string::npos) 
    {
        reg.erase(reg.find("s"), 1);
    }

    if(reg.find("[") != std::string::npos) 
    {
        reg.erase(reg.find("["),1);
        reg.erase(reg.find(":"),reg.length());
        return reg;
    }
    else
    {
        reg.erase(1, reg.length());
        return reg;
    }
}

Operand Assembler::getOperandInfo(std::string opstring){
	Operand op;
    std::stringstream opstream;
    uint32_t operandcode;

	if(opstring.find("v") != std::string::npos ||
        opstring.find("s") != std::string::npos) 
    {
        opstring = extractreg(opstring);
        opstream << opstring;
        opstream >> operandcode;

		op.operandType = RegOperand;
        op.code = operandcode;
    }
    else 
    {
        if(opstring.find("0x") != std::string::npos)
        {
            opstring.erase(0,2);
        }
        opstream << std::hex << opstring;
        opstream >> operandcode;

		op.operandType = IntOperand;
        op.code = operandcode;
	}
	return op;
}

/* take decimal values, cast them to 32-bit unsigned values,  *
 * then and shift them to line up with the instruction format */

std::vector<uint32_t> Assembler::assembleSOP1(Inst *inst)
{
    std::vector<uint32_t> newasm;
    uint32_t instcode = 0xBE800000;

    uint32_t opcode  = uint32_t(inst->instType.opcode);
    opcode = opcode << 8;
    instcode = instcode | opcode;

    uint32_t dst = getCodeByOperand(inst->dst);
    dst = dst << 16;
    instcode = instcode | dst;

    uint32_t src0 = getCodeByOperand(inst->src0);
    instcode = instcode | src0;

    newasm.push_back(instcode);

    return newasm;
}

std::vector<uint32_t>  Assembler::assembleVOP1(Inst *inst)
{
    std::vector<uint32_t> newasm;
    uint32_t instcode = 0x7E000000;

    uint32_t opcode = uint32_t(inst->instType.opcode);
    opcode = opcode << 9;
    instcode = instcode | opcode;

    uint32_t dst = getCodeByOperand(inst->dst);
    dst = dst << 17;
    instcode = instcode | dst;

    uint32_t src0 = getCodeByOperand(inst->src0);
    instcode = instcode | src0;

    newasm.push_back(instcode);

    return newasm;
}

std::vector<uint32_t>  Assembler::assembleSMEM(Inst *inst)
{
    std::vector<uint32_t> newasm;
    uint32_t insthigh = 0xC0000000;
    uint32_t instlow  = 0x00000000;

    uint32_t opcode = uint32_t(inst->instType.opcode);
    opcode = opcode << 17;
    insthigh = insthigh | opcode;

    if(inst->instType.SRC2Width != 0)
    {
        insthigh = insthigh | 0x00040000;
    }

    newasm.push_back(insthigh);
    newasm.push_back(instlow);

    return newasm;
}


std::vector<uint32_t>  Assembler::assembleSOP2(Inst *inst)
{
    std::vector<uint32_t> newasm;
    uint32_t instcode = 0x80000000;
    uint32_t imm      = 0x00000000;

    uint32_t opcode = uint32_t(inst->instType.opcode);
    opcode = opcode << 23;
    instcode = instcode | opcode;

    uint32_t dst = getCodeByOperand(inst->dst);
    dst = dst << 16;
    instcode = instcode | dst;

    if(inst->instType.SRC0Width != 0)
    {   // don't shift this one
        uint32_t src0 = getCodeByOperand(inst->src0);
        instcode = instcode | src0;
    }
    if(inst->instType.SRC1Width != 0)
    { 
        uint32_t src1;
        //Sign extending immediate
        if (inst->src1.operandType == IntOperand)
        {
            imm = getCodeByOperand(inst->src1);
            src1 = 0x000000FF;
            // src1 = imm >> 24; // want last byte of imm
        }
        else
        {
            src1 = getCodeByOperand(inst->src1);
        }
        src1 = src1 << 8;
        instcode = instcode | src1;
    }
    // if(inst->instType.SRC2Width != 0)
    //     inst->src2 = getOperandInfo(params.at(4));

    newasm.push_back(instcode);
    newasm.push_back(imm);

    return newasm;
}

