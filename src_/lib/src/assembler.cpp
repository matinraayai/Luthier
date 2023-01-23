#include "assembler.h"
#include "bitops.h"
#include "operand.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

Assembler::Assembler() {
  initFormatTable();
  initRegs();
  initEncodeTable();
}

void Assembler::Assemble(std::string inststr, std::ostream &o) {
  Inst *inst = new Inst;
  uint32_t assembly;

  getInstData(inststr, inst);

  switch (inst->instType.format.formatType) {
    case SOP2:
      assembly = assembleSOP2(inst);
      break;
    case SOP1:
      assembly = assembleSOP1(inst);
      break;
    case SOPP:
      assembly = assembleSOPP(inst);
      break;
    default:
      o << "Format for instruction " << inststr
        << " not supported"          << std::endl;
      return;
  }

  o << std::hex << assembly << std::endl;
}

void Assembler::editDSTreg(Inst *inst, std::string reg) {
  uint32_t code;
  uint32_t mask;
  uint32_t newReg;
  std::vector<uint32_t> assembly;
  
  assembly.push_back(convertLE(inst->bytes));
  code = extractGPRbyte(reg);
  inst->dst.code = (int)code;

  if (inst->format.formatType <= SOP1) {
    mask = 0xFF80FFFF;
    newReg = code<<16; }
  else if (inst->format.formatType >= VOP2 ||
           inst->format.formatType <= VOP1) {
    if (reg.at(0) == 'v') {
      code += 256;
    }
    mask = 0xFE01FFFF;
    newReg = code<<17;
  }

  assembly.at(0) = assembly.at(0) & mask;
  assembly.at(0) = assembly.at(0) | newReg;
  inst->bytes = instcodeToByteArray(assembly);
}

void Assembler::editSRC0reg(Inst *inst, std::string reg) {
  uint32_t code;
  uint32_t mask;
  uint32_t newReg;
  std::vector<uint32_t> assembly;
  
  assembly.push_back(convertLE(inst->bytes));
  code = extractGPRbyte(reg);
  inst->src0.code = (int)code;

  if (inst->format.formatType <= SOP1) {
    mask = 0xFFFFFF00;
  }
  else if (inst->format.formatType >= VOP2 ||
           inst->format.formatType <= VOP1) {
    mask = 0xFFFFFE00;
  }

  if (reg.at(0) == 'v') {
    code += 256;
  }
  newReg = code;

  assembly.at(0) = assembly.at(0) & mask;
  assembly.at(0) = assembly.at(0) | newReg;
  inst->bytes = instcodeToByteArray(assembly);
}

void Assembler::editSRC1reg(Inst *inst, std::string reg) {
  uint32_t code;
  uint32_t mask;
  uint32_t newReg;
  std::vector<uint32_t> assembly;
  
  assembly.push_back(convertLE(inst->bytes));
  code = extractGPRbyte(reg);
  inst->src1.code = (int)code;

  if (inst->format.formatType <= SOP1) {
    mask = 0xFFFF00FF;
    newReg = code<<8;
  }
  else if (inst->format.formatType >= VOP2 ||
           inst->format.formatType <= VOP1) {
    mask = 0xFFFE01FF;
    newReg = code<<9;
  }

  assembly.at(0) = assembly.at(0) & mask;
  assembly.at(0) = assembly.at(0) | newReg;
  inst->bytes = instcodeToByteArray(assembly);
}


void Assembler::editSIMM(Inst *inst, short simm) {
  uint32_t code;
  uint32_t mask = 0xFFFF0000;
  std::vector<uint32_t> assembly;
  
  assembly.push_back(convertLE(inst->bytes));

  code = uint32_t(simm);
  inst->simm16.code = (int)code;

  assembly.at(0) = assembly.at(0) & mask;

  mask = 0x0000FFFF;
  code = code & mask;

  assembly.at(0) = assembly.at(0) | code;
  inst->bytes = instcodeToByteArray(assembly);
}

void Assembler::getInstData(std::string inststr, Inst *inst) {
  std::vector<std::string> params = getInstParams(inststr);

  std::string iname = params.at(0);
  auto index = iname.find("_e32");
  if (index != std::string::npos) {
    iname.erase(index, index + 3);
  }

  inst->instType = EncodeTable[iname];
  uint16_t op_val;

  if (inst->instType.DSTWidth != 0)
    inst->dst = getOperandInfo(params.at(1));

  if (inst->instType.SRC0Width != 0)
    inst->src0 = getOperandInfo(params.at(2));

  if (inst->instType.SRC1Width != 0)
    inst->src1 = getOperandInfo(params.at(3));

  if (inst->instType.SRC2Width != 0)
    inst->src2 = getOperandInfo(params.at(4));
  
  if (inst->instType.format.formatType == SOPP)
    if (inst->instType.DSTWidth != 0)
      inst->simm16 = getOperandInfo(params.at(1));
}

std::vector<std::string> Assembler::getInstParams(std::string inststr) {
  std::vector<std::string> params;
  std::string delim = " ";

  size_t i;
  while (i != std::string::npos) {
    i = inststr.find(delim);
    params.push_back(inststr.substr(0, i));
    inststr.erase(0, i + 1);
  }

  return params;
}

std::string Assembler::extractGPRstr(std::string reg) {
  if (reg.find("v") != std::string::npos) {
    reg.erase(reg.find("v"), 1);
  } else if (reg.find("s") != std::string::npos) {
    reg.erase(reg.find("s"), 1);
  }

  if (reg.find("[") != std::string::npos) {
    reg.erase(reg.find("["), 1);
    reg.erase(reg.find(":"), reg.length());
  }

  if (reg.find(",") != std::string::npos) {
    reg.erase(reg.find(","), 1);
  }
  if (reg.find(" ") != std::string::npos) {
    reg.erase(reg.find(" "), 1);
  }
  return reg;
}

uint32_t Assembler::extractGPRbyte(std::string reg) {
  int regval;

  if (reg.find("v") != std::string::npos) {
    if (reg == "vmnct") {
      return 0;
    } else {
    reg.erase(reg.find("v"), 1);
    }
  } else if (reg.find("s") != std::string::npos) {
    reg.erase(reg.find("s"), 1);
  }

  if (reg.find("[") != std::string::npos) {
    reg.erase(reg.find("["), 1);
    reg.erase(reg.find(":"), reg.length());
  }

  if (reg.find(",") != std::string::npos) {
    reg.erase(reg.find(","), 1);
  }
  if (reg.find(" ") != std::string::npos) {
    reg.erase(reg.find(" "), 1);
  }
  
  regval = stoi(reg);

  return uint32_t(regval);
}

uint32_t Assembler::getCodeByOperand(Operand op) {
  switch (op.operandType) {
    case RegOperand:
      return uint32_t(op.code);

    case IntOperand:
      if (op.code >= 1 && op.code <= 64)
        return uint32_t(op.code + 128);
      else
        return uint32_t(op.code) + 128;

    case LiteralConstant:
      return 0x000000FF;
  }
}

Operand Assembler::getOperandInfo(std::string opstring) {
  Operand op;
  std::stringstream opstream;
  uint32_t operandcode;

  if (opstring.find("v") != std::string::npos ||
      opstring.find("s") != std::string::npos) {
    opstring = extractGPRstr(opstring);

    opstream << opstring;
    opstream >> operandcode;

    op.operandType = RegOperand;
    op.code = operandcode;
  }
  else if (opstring.find("0x") != std::string::npos) {
    opstring.erase(0, 2);
    
    opstream << std::hex << opstring;
    opstream >> operandcode;

    op.operandType = LiteralConstant;
    op.code = operandcode;
  }

  return op;
}

/* take decimal values, cast them to 32-bit unsigned values,  *
 * then and shift them to line up with the instruction format */

uint32_t Assembler::assembleSOP2(Inst *inst) {
  uint32_t instcode = 0x80000000;

  uint32_t opcode = uint32_t(inst->instType.opcode);
  opcode = opcode << 23;
  instcode = instcode | opcode;

  uint32_t dst = getCodeByOperand(inst->dst);
  dst = dst << 16;
  instcode = instcode | dst;

  uint32_t src1 = getCodeByOperand(inst->src1);
  src1 = src1 << 8;
  instcode = instcode | src1;

  uint32_t src0 = getCodeByOperand(inst->src0);
  instcode = instcode | src0;

  return instcode;
}

uint32_t Assembler::assembleSOP1(Inst *inst) {
  uint32_t instcode = 0xBE800000;

  uint32_t opcode = uint32_t(inst->instType.opcode);
  opcode = opcode << 8;
  instcode = instcode | opcode;

  uint32_t dst = getCodeByOperand(inst->dst);
  dst = dst << 16;
  instcode = instcode | dst;

  uint32_t src0 = getCodeByOperand(inst->src0);
  instcode = instcode | src0;

  return instcode;
}

uint32_t Assembler::assembleSOPP(Inst *inst) {
  uint32_t instcode = 0xBF800000;

  uint32_t opcode = uint32_t(inst->instType.opcode);
  opcode = opcode << 16;
  instcode = instcode | opcode;

  instcode = instcode | uint16_t(inst->simm16.code);

  return instcode;
}
