#include "inst.h"
#include "initialize.h"
#include "operand.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <map>

// For now, I'm just creating the actual logic of the assembler
// Eventually, the assembler will mirror the disasembler - it'll
// be a standalone class, with its own header file, etc.


/*
Operand getOperandCode(std::string op)
{
    Operand newOp;

    //casting "v" and "s" to char bc they're const char by default
    if(op[0] == char("v"))
    {
        newOp.operandType = RegOperand;
        newOp.reg.name = op;
        newOp.code = atoi(reinterpret_cast<const char*>(op[1]));
    } else if(op[0] == char("s"))
    {
        newOp.operandType = RegOperand;
        newOp.reg.name = op;
        newOp.code = atoi(reinterpret_cast<const char*>(op[1]));
    }
}
*/

unsigned int getCodeByOperand(char* op)
{
    std::string opStr(op);
    
    for (int i = 0; i < opStr.length(); i++)
    {
        if(int(op[i]) >= 0x30 && int(op[i]) <= 0x39)
        {
            // printf("%c | %X \n", op[i], int(op[i]));
            return int(op[i]) - 0x30;
        }
        else if(int(op[i]) == 0x2D)
        {
            int immval = int(op[i+1]) - 0x30;
            int immSigned = immval * -1;
            // printf("%X \n", uint32_t(immSigned));

            return uint32_t(immSigned) & 0x000000FF;
        }
    }
}


int main(int argc, char **argv)
{
    if(argc < 3 || argc > 5) 
    {    
        printf("Expect 2-4 args:\n a single asssembly instruction and its operands\n");
        return -1;
    }

    // // initFormatTable(); // causes undefinded referance error

    Inst inst;

    std::string namestr(argv[1]);
    inst.instType.instName = namestr;
    // std::string dststr(argv[2]);
    // std::string src0str(argv[3]);
    // std::string src1str(argv[4]);


    if(inst.instType.instName == "s_swappc_b64")
    {
        inst.format.formatType = SOP1;
        inst.instType.opcode = 30;
    }
    else if(inst.instType.instName == "s_getpc_b64")
    {
        inst.format.formatType = SOP1;
        inst.instType.opcode = 28;
    }
    else if(inst.instType.instName == "s_add_u32")
    {
        inst.format.formatType = SOP2;
        inst.instType.opcode = 0;
    }
    else if(inst.instType.instName == "s_addc_u32")
    {
        inst.format.formatType = SOP2;
        inst.instType.opcode = 4;
    }

    unsigned int instCode;
    if(inst.format.formatType == SOP2)
    {
        unsigned int imm;
        instCode = 0x80000000;

        //take decimal values, cast them to 32-bit unsigned values,
        //then and shift them to line up with the instruction format 
        unsigned int opcode  = uint32_t(inst.instType.opcode);
        opcode = opcode << 23;
        instCode = instCode | opcode;

        unsigned int dst = getCodeByOperand(argv[2]);
        dst = dst << 16;
        instCode = instCode | dst;

        if(argc >= 4) 
        {   // don't shift this one
            unsigned int src0 = getCodeByOperand(argv[3]);
            instCode = instCode | src0;
        }
        if(argc >= 5)
        { 
            unsigned int src1 = getCodeByOperand(argv[4]);

            //Sign extending immediate
            for (int i = 0; i < 32; i++)
            {            
                if((0x80000000 & (src1 << i)) == 0x80000000)
                {
                    unsigned int mask = 0xFFFFFFFF;
                    mask = mask << i*4;
                    imm = src1 | mask;
                    break;
                }
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
        unsigned int opcode  = uint32_t(inst.instType.opcode);
        opcode = opcode << 8;
        instCode = instCode | opcode;

        unsigned int dst = getCodeByOperand(argv[2]);
        dst = dst << 16;
        instCode = instCode | dst;

        if(argc >= 4) 
        {   // don't shift this one
            unsigned int src0 = getCodeByOperand(argv[3]);
            instCode = instCode | src0;
        }
        std::cout << std::hex << instCode << std::endl;
    }
    return 0;
}