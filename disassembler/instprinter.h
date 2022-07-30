#ifndef INSTPRINTER_H
#define INSTPRINTER_H
#include "../src/elf.h"
#include "inst.h"
#include "operand.h"
#include <iomanip>
#include <sstream>
#include <string>
struct InstPrinter {
  elfio::File *file;
  std::string operandString(Operand o) {
    std::stringstream stream;
    switch (o.operandType) {
    case RegOperand:
      return regOperandToString(o);
    case IntOperand:
      return std::to_string(o.intValue);
    case FloatOperand:
      return std::to_string(o.floatValue);
    case LiteralConstant:
      stream << "0x" << std::hex << o.literalConstant;
      return stream.str();
    default:
      return "";
    }
  }
  std::string regOperandToString(Operand o) {
    Reg reg = o.reg;
    std::stringstream stream;
    if (o.regCount > 1) {
      if (reg.IsSReg()) {
        stream << "s[" << reg.RegIndex() << ":"
               << reg.RegIndex() + o.regCount - 1 << "]";
        return stream.str();
      } else if (reg.IsVReg()) {
        stream << "v[" << reg.RegIndex() << ":"
               << reg.RegIndex() + o.regCount - 1 << "]";
        return stream.str();
      } else if (reg.name.find("lo") != std::string::npos) {
        return reg.name.substr(0, reg.name.length() - 2);
      }
      return "unknown register";
    }
    return reg.name;
  }
  std::string sop2String(Inst *i) {}
  std::string smemString(Inst *i) {
    std::stringstream stream;
    stream << i->instType.instName << " " << operandString(i->data) << ", "
           << operandString(i->base) << ", 0x" << std::hex
           << uint16_t(i->offset.intValue);
    return stream.str();
  }
  std::string print(Inst *i) {
    switch (i->format.formatType) {
    case SOP2:
      return sop2String(i);
    case SMEM:
      return smemString(i);
    default:
      return std::string("");
      // throw std::runtime_error("unknown instruction format type");
    }
  }
};
#endif