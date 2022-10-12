#include "inst.h"
#include "operand.h"
#include <cstring>
#include <iostream>
#include <sstream>

// For now, I'm just creating the actual logic of the assembler
// Eventually, the assembler will mirror the disasembler - it'll
// be a standalone class, with its own header file, etc.

uint32_t getCodeByOperand(Operand op)
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

std::string extractreg(std::string reg)
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
        return reg;
    }
}

Operand getOperandInfo(std::string opstring){
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

std::vector<std::string> getInstParams(std::string inststr)
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

void getInstData(Inst* inst, std::vector<std::string> params)
{
    inst->instType.instName = params.at(0);

    if(inst->instType.instName == "s_swappc_b64")
    {
        inst->format.formatType = SOP1;
        inst->instType.opcode = 30;

        inst->dst  = getOperandInfo(params.at(1));
        inst->src0 = getOperandInfo(params.at(2));
    }
    else if(inst->instType.instName == "s_getpc_b64")
    {
        inst->format.formatType = SOP1;
        inst->instType.opcode = 28;
        inst->dst = getOperandInfo(params.at(1));
    }
    else if(inst->instType.instName == "s_add_u32")
    {
        inst->format.formatType = SOP2;
        inst->instType.opcode = 0;
        inst->dst  = getOperandInfo(params.at(1));
        inst->src0 = getOperandInfo(params.at(2));
        inst->src1 = getOperandInfo(params.at(3));
    }
    else if(inst->instType.instName == "s_addc_u32")
    {
        inst->format.formatType = SOP2;
        inst->instType.opcode = 4;
        inst->dst  = getOperandInfo(params.at(1));
        inst->src0 = getOperandInfo(params.at(2));
        inst->src1 = getOperandInfo(params.at(3));
    }
}

void printAssembly(Inst inst, int instlen)
{
    uint32_t instCode;
    if(inst.format.formatType == SOP2)
    {
        uint32_t imm;
        instCode = 0x80000000;

        //take decimal values, cast them to 32-bit unsigned values,
        //then and shift them to line up with the instruction format 
        uint32_t opcode = uint32_t(inst.instType.opcode);
        opcode = opcode << 23;
        instCode = instCode | opcode;

        uint32_t dst = getCodeByOperand(inst.dst);
        dst = dst << 16;
        instCode = instCode | dst;

        if(instlen >= 3) 
        {   // don't shift this one
            uint32_t src0 = getCodeByOperand(inst.src0);
            instCode = instCode | src0;
        }
        if(instlen >= 4)
        { 
            uint32_t src1;
            //Sign extending immediate
            if (inst.src1.operandType == IntOperand)
            {
                imm = getCodeByOperand(inst.src1);
                src1 = imm >> 24; // want last byte of imm
            }
            else
            {
                src1 = getCodeByOperand(inst.src1);
            }
            src1 = src1 << 8;
            instCode = instCode | src1;
        }
        std::cout << std::hex << instCode << " " << imm << std::endl;
    } 
    else if(inst.format.formatType == SOP1)
    {
        instCode = 0xBE800000;

        //take decimal values, cast them to 32-bit unsigned values,
        //then and shift them to line up with the instruction format 
        uint32_t opcode  = uint32_t(inst.instType.opcode);
        opcode = opcode << 8;
        instCode = instCode | opcode;

        uint32_t dst = getCodeByOperand(inst.dst);
        dst = dst << 16;
        instCode = instCode | dst;

        if(instlen >= 3) 
        {   // don't shift this one
            uint32_t src0 = getCodeByOperand(inst.src0);
            instCode = instCode | src0;
        }
        std::cout << std::hex << instCode << std::endl;
    }
}

int main(int argc, char *argv[])
{
	if (argc != 2) 
    {
		printf("An input assembly file is required\n");
		return 1;
	}

    Inst inst;
    std::string newInst = std::string(argv[1]);
    std::vector<std::string> params = getInstParams(newInst);

    getInstData(&inst, params);
    printAssembly(inst, params.size());
    /*
    inst->instType.instName = std::string(argv[1]);


    if(inst->instType.instName == "s_swappc_b64")
    {
        inst->format.formatType = SOP1;
        inst->instType.opcode = 30;
        inst->dst  = getOperandInfo(std::string(argv[2]));
        inst->src0 = getOperandInfo(std::string(argv[3]));
    }
    else if(inst->instType.instName == "s_getpc_b64")
    {
        inst->format.formatType = SOP1;
        inst->instType.opcode = 28;
        inst->dst = getOperandInfo(std::string(argv[2]));
    }
    else if(inst->instType.instName == "s_add_u32")
    {
        inst->format.formatType = SOP2;
        inst->instType.opcode = 0;
        inst->dst  = getOperandInfo(std::string(argv[2]));
        inst->src0 = getOperandInfo(std::string(argv[3]));
        inst->src1 = getOperandInfo(std::string(argv[4]));
    }
    else if(inst->instType.instName == "s_addc_u32")
    {
        inst->format.formatType = SOP2;
        inst->instType.opcode = 4;
        inst->dst  = getOperandInfo(std::string(argv[2]));
        inst->src0 = getOperandInfo(std::string(argv[3]));
        inst->src1 = getOperandInfo(std::string(argv[4]));
    }
    */
   /*
    uint32_t instCode;
    if(inst.format.formatType == SOP2)
    {
        uint32_t imm;
        instCode = 0x80000000;

        //take decimal values, cast them to 32-bit unsigned values,
        //then and shift them to line up with the instruction format 
        uint32_t opcode = uint32_t(inst.instType.opcode);
        opcode = opcode << 23;
        instCode = instCode | opcode;

        uint32_t dst = getCodeByOperand(inst.dst);
        dst = dst << 16;
        instCode = instCode | dst;

        if(argc >= 4) 
        {   // don't shift this one
            uint32_t src0 = getCodeByOperand(inst.src0);
            instCode = instCode | src0;
        }
        if(argc >= 5)
        { 
            uint32_t src1;
            //Sign extending immediate
            if (inst.src1.operandType == IntOperand)
            {
                imm = getCodeByOperand(inst.src1);
                src1 = imm >> 24; // want last byte of imm
            }
            else
            {
                src1 = getCodeByOperand(inst.src1);
            }
            src1 = src1 << 8;
            instCode = instCode | src1;
        }
        std::cout << std::hex << instCode << " " << imm << std::endl;
    } 
    else if(inst.format.formatType == SOP1)
    {
        instCode = 0xBE800000;

        //take decimal values, cast them to 32-bit unsigned values,
        //then and shift them to line up with the instruction format 
        uint32_t opcode  = uint32_t(inst.instType.opcode);
        opcode = opcode << 8;
        instCode = instCode | opcode;

        uint32_t dst = getCodeByOperand(inst.dst);
        std::cout <<"dst"<< dst << std::endl;
        dst = dst << 16;
        instCode = instCode | dst;

        if(argc >= 4) 
        {   // don't shift this one
            uint32_t src0 = getCodeByOperand(inst.src0);
            std::cout <<"src0"<< src0 << std::endl;

            instCode = instCode | src0;
        }
        std::cout << std::hex << instCode << std::endl;
    }
    */
    return 0;
}
