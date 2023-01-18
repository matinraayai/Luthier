#ifndef INSTMODIFIER_H
#define INSTMODIFIER_H
#include "elf.hpp"
#include "inst.h"
#include "operand.h"
#include <iomanip>
#include <sstream>
#include <string>

struct InstModifier {
  int v_offset;
  int s_offset;

  void setOffset(int voff, int soff) {
    v_offset = voff;
    s_offset = soff;
  }

  void vop1Mod(Inst *i) {
    uint32_t bytes = i->first;
    if (i->src0.operandType == RegOperand) {
      uint32_t src0Value = extractBitsFromU32(bytes, 0, 8);
      src0Value =
          i->src0.reg.IsSReg() ? src0Value + s_offset : src0Value + v_offset;
      bytes = zerooutBitsFromU32(bytes, 0, 8);
      bytes = bytes | src0Value;
    }
    if (i->dst.operandType == RegOperand) {
      uint32_t dstValue = extractBitsFromU32(bytes, 17, 24);
      dstValue = dstValue + v_offset;
      bytes = zerooutBitsFromU32(bytes, 17, 24);
      bytes = bytes | dstValue << 17;
    }
    i->first = bytes;
  }
};
#endif