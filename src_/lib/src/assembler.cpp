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
  std::vector<uint32_t> assembly;

  getInstData(inststr, inst);

  switch (inst->instType.format.formatType) {
    case SOP2:
      assembly.push_back(assembleSOP2(inst));
      break;
    case SOP1:
      assembly.push_back(assembleSOP1(inst));
      break;
    case SOPP:
      assembly.push_back(assembleSOPP(inst));
      break;
    default:
      // maybe throw an exception here
      o << "Format for instruction "     << inststr
        << " not supported by assembler" << std::endl;
  }

  o << "\t" << inststr;
  for (int i = inststr.size(); i < 59; i++) {
    o << " ";
  } 
  o << std::hex << "//" << inst->PC << ": ";
  for (int i = 0; i < assembly.size(); i++) {
    o << std::hex << assembly.at(i) << " ";
  }
  o << std::endl;

  delete inst;
}

void Assembler::Assemble(std::string inststr, std::shared_ptr<Inst> &inst) {
  std::unique_ptr<Inst> new_inst = std::make_unique<Inst>();
  std::vector<uint32_t> assembly;

  getInstData(inststr, new_inst.get());

  switch (new_inst->instType.format.formatType) {
    case SOP2:
      assembly.push_back(assembleSOP2(new_inst.get()));
      break;
    case SOP1:
      assembly.push_back(assembleSOP1(new_inst.get()));
      break;
    case SOPP:
      assembly.push_back(assembleSOPP(new_inst.get()));
      break;
    default:
      // maybe throw an exception here
      std::cout << "Format for instruction "     << inststr
                << " not supported by assembler" << std::endl;
  }
  new_inst->first = assembly.at(0);
  new_inst->bytes = instcodeToByteArray(assembly);
  new_inst->PC = inst->PC;

  inst = std::move(new_inst);
}

std::shared_ptr<Inst> Assembler::Assemble(std::string inststr, uint64_t pc) {
  std::unique_ptr<Inst> inst = std::make_unique<Inst>();
  std::vector<uint32_t> assembly;

  getInstData(inststr, inst.get());

  switch (inst->instType.format.formatType) {
    case SOP2:
      assembly.push_back(assembleSOP2(inst.get()));
      break;
    case SOP1:
      assembly.push_back(assembleSOP1(inst.get()));
      break;
    case SOPP:
      assembly.push_back(assembleSOPP(inst.get()));
      break;
    default:
      // maybe throw an exception here
      std::cout << "Format for instruction "     << inststr
                << " not supported by assembler" << std::endl;
  }
  inst->first = assembly.at(0);
  inst->bytes = instcodeToByteArray(assembly);
  inst->PC = PC;

  return inst;
}

void Assembler::offsetRegs(std::shared_ptr<Inst> i, int smax, int vmax) {
  switch(i->format.formatType) {
    case SOP2:
      editSALUinst(i, smax);
      break;
    case SOP1:
      editSALUinst(i, smax);
      break;
    case VOP1:
      editVALUinst(i, smax, vmax);
      break;
    case FLAT:
      editFLATinst(i, smax, vmax);
      break;
    default:
      break;
  }
}

void Assembler::editSALUinst(std::shared_ptr<Inst> i, int smax) {
  int newcode;

  if (i->instType.DSTWidth != 0) {
    if (i->dst.operandType == RegOperand) {
      newcode = i->dst.code + smax;
      editDSTreg(i, newcode);
    }
  }

  if (i->instType.SRC0Width != 0) {
    if (i->src0.operandType == RegOperand) {
      newcode = i->src0.code + smax;
      editSRC0reg(i, newcode);
    }
  }

  if (i->instType.SRC1Width != 0) {
    if (i->src1.operandType == RegOperand) {
      newcode = i->src1.code + smax;
      editSRC1reg(i, newcode);
    }
  }
}

void Assembler::editVALUinst(std::shared_ptr<Inst> i, int smax, int vmax) {
  int newcode;

  if (i->instType.DSTWidth != 0) {
    if (i->dst.operandType == RegOperand) {
      if (i->dst.code >= 256) {
        newcode = i->dst.code + vmax;
      } else {
        newcode = i->dst.code + smax;
      }
      editDSTreg(i, newcode);
    }
  }

  if (i->instType.SRC0Width != 0) {
    if (i->src0.operandType == RegOperand) {
      if (i->src0.code >= 256) {
        newcode = i->src0.code + vmax;
      } else {
        newcode = i->src0.code + smax;
      }
      editSRC0reg(i, newcode);
    }
  }

  if (i->instType.SRC1Width != 0) {
    if (i->src1.operandType == RegOperand) {
      if (i->src1.code >= 256) {
        newcode = i->src1.code + vmax;
      } else {
        newcode = i->src1.code + smax;
      }
      editSRC1reg(i, newcode);
    }
  }
}

void Assembler::editFLATinst(std::shared_ptr<Inst> i, int smax, int vmax) {
  int newcode;

  if (i->instType.DSTWidth != 0) {
    newcode = i->dst.code + vmax;
    editDSTreg(i, newcode);
  }

  if ((i->second & 0x007F0000)>>16 != 0x7F) {
    newcode = i->addr.code + vmax;
    editSRC0flat(i, newcode, 0);

    newcode = i->sAddr.code + smax;
    editSRC1reg(i, newcode);
  } else {
    newcode = i->data.code + vmax;
    editSRC0flat(i, newcode, 1);
  }
}

void Assembler::editSRC0reg(std::shared_ptr<Inst> inst, int code) {
  uint32_t mask;
  std::vector<uint32_t> assembly;
  
  if (inst->format.formatType <= SOP1) {
    inst->src0.code = code;
    mask = 0xFFFFFF00;

    inst->first = inst->first & mask;
    inst->first = inst->first | code;
  }
  else if (inst->format.formatType == VOP2 ||
           inst->format.formatType == VOP1) {
    inst->src0.code = code;
    mask = 0xFFFFFE00;

    inst->first = inst->first & mask;
    inst->first = inst->first | code;
  }
  else if (inst->format.formatType == FLAT) {
    inst->addr.code = code;
    mask = 0xFFFFFF00;

    inst->second = inst->second & mask;
    inst->second = inst->second | code;
  }

  assembly.push_back(inst->first);
  if (inst->byteSize == 8) {
    assembly.push_back(inst->second);
  }
  inst->bytes = instcodeToByteArray(assembly);
}

void Assembler::editSRC0flat(std::shared_ptr<Inst> inst, int code, bool data) {
  uint32_t mask;
  uint32_t newReg;
  std::vector<uint32_t> assembly;
  
  if (data) {
    inst->data.code = code;
    mask = 0xFFFF00FF;
    newReg = code<<8;
  } else {
    inst->addr.code = code;
    mask = 0xFFFFFF00;
    newReg = code;
  }

  inst->second = inst->second & mask;
  inst->second = inst->second | newReg;

  assembly.push_back(inst->first);
  assembly.push_back(inst->second);

  inst->bytes = instcodeToByteArray(assembly);
}

void Assembler::editSRC1reg(std::shared_ptr<Inst> inst, int code) {
  uint32_t mask;
  uint32_t newReg;
  std::vector<uint32_t> assembly;
  
  if (inst->format.formatType <= SOP1) {
    inst->src1.code = code;
    mask = 0xFFFF00FF;
    newReg = code<<8;

    inst->first = inst->first & mask;
    inst->first = inst->first | newReg;
  }
  else if (inst->format.formatType == VOP2 ||
           inst->format.formatType == VOP1) {
    inst->src1.code = code;
    mask = 0xFFFE01FF;
    newReg = code<<9;

    inst->first = inst->first & mask;
    inst->first = inst->first | newReg;
  }
  else if (inst->format.formatType == FLAT) {
    inst->sAddr.code = code;
    mask = 0xFF80FFFF;
    newReg = code<<16;

    inst->second = inst->second & mask;
    inst->second = inst->second | newReg;
  }

  assembly.push_back(inst->first);
  if (inst->byteSize == 8) {
    assembly.push_back(inst->second);
  }
  inst->bytes = instcodeToByteArray(assembly);
}

void Assembler::editDSTreg(std::shared_ptr<Inst> inst, int code) {
  uint32_t mask;
  uint32_t newReg;
  std::vector<uint32_t> assembly;
  
  inst->dst.code = code;

  if (inst->format.formatType <= SOP1) {
    mask = 0xFF80FFFF;
    newReg = code<<16; 

    inst->first = inst->first & mask;
    inst->first = inst->first | newReg;
  }
  else if (inst->format.formatType == VOP2 ||
           inst->format.formatType == VOP1) {
    mask = 0xFE01FFFF;
    newReg = code<<17;

    inst->first = inst->first & mask;
    inst->first = inst->first | newReg;
  }
  else if (inst->format.formatType == FLAT) {
    mask = 0x00FFFFFF;
    newReg = code<<24;

    inst->second = inst->second & mask;
    inst->second = inst->second | newReg;
  }

  assembly.push_back(inst->first);
  if (inst->byteSize == 8) {
    assembly.push_back(inst->second);
  }
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

std::vector<unsigned char> 
Assembler::ilstbuf(std::vector<std::shared_ptr<Inst>> ilst) {
  std::vector<unsigned char> buf;
  Inst *inst;
  for (int i = 0; i < ilst.size(); i++) {
    inst = ilst.at(i).get();
    for (int j = 0; j < inst->bytes.size(); j++) {
      buf.push_back(inst->bytes.at(j));
    }
  }   
  return buf;
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
