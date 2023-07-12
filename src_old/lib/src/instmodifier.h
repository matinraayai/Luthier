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

  void sop1Mod(Inst *i) {
    uint32_t bytes = i->first;
    if (i->src0.operandType == RegOperand &&
        i->instType.opcode != 28) { // s_setpc
      uint32_t src0Value = extractBitsFromU32(bytes, 0, 7);
      if (i->instType.opcode != 29 &&
          src0Value != 30) { // rule out s_setpc_b64 s[30:31]
        src0Value += s_offset;
        bytes = zerooutBitsFromU32(bytes, 0, 7);
        bytes = bytes | src0Value;
      }
    }
    if (i->dst.operandType == RegOperand &&
        i->instType.opcode != 29) { // s_getpc
      uint32_t dstValue = extractBitsFromU32(bytes, 16, 22);
      dstValue = dstValue + s_offset;
      bytes = zerooutBitsFromU32(bytes, 16, 22);
      bytes = bytes | dstValue << 16;
    }
    i->first = bytes;
  }

  void sop2Mod(Inst *i) {
    uint32_t bytes = i->first;
    if (i->src0.operandType == RegOperand) {
      uint32_t src0Value = extractBitsFromU32(bytes, 0, 7);
      src0Value += s_offset;
      bytes = zerooutBitsFromU32(bytes, 0, 7);
      bytes = bytes | src0Value;
    }
    if (i->src1.operandType == RegOperand) {
      uint32_t src1Value = extractBitsFromU32(bytes, 8, 15);
      src1Value += s_offset;
      bytes = zerooutBitsFromU32(bytes, 8, 15);
      bytes = bytes | src1Value << 8;
    }
    if (i->dst.operandType == RegOperand) {
      uint32_t dstValue = extractBitsFromU32(bytes, 16, 22);
      dstValue += s_offset;
      bytes = zerooutBitsFromU32(bytes, 16, 22);
      bytes = bytes | dstValue << 16;
    }
    i->first = bytes;
  }

  void flatMod(Inst *i) {
    uint32_t bytes = i->second;
    auto opcode = i->instType.opcode;
    uint32_t addrbits = extractBitsFromU32(bytes, 0, 7);
    addrbits += v_offset;
    bytes = zerooutBitsFromU32(bytes, 0, 7);
    bytes = bytes | addrbits;

    if (opcode >= 16 && opcode <= 23) { // flat_load
      uint32_t dstbits = extractBitsFromU32(bytes, 24, 31);
      dstbits += v_offset;
      bytes = zerooutBitsFromU32(bytes, 24, 31);
      bytes = bytes | dstbits << 24;
    }
    if (opcode >= 24 && opcode <= 31 ||
        opcode >= 66 && opcode <= 108) { // flat_store
      uint32_t databits = extractBitsFromU32(bytes, 8, 15);
      databits += v_offset;
      bytes = zerooutBitsFromU32(bytes, 8, 15);
      bytes = bytes | databits << 8;
    }

    uint32_t bits = extractBitsFromU32(bytes, 16, 22);
    if (bits != 0x7f) {
      bits += s_offset;
      bytes = zerooutBitsFromU32(bytes, 16, 22);
      bytes = bytes | bits << 16;
    }
    i->second = bytes;
  }

  void modify(Inst *i) {
    switch (i->format.formatType) {
    case SOP2:
      sop2Mod(i);
      break;
    // case SOPK:
    //   return sopkString(i);
    case SOP1:
      sop1Mod(i);
      break;
    // case SOPC:
    //   return sopcString(i);
    // case SOPP:
    //   return soppString(i);
    // case SMEM:
    //   return smemString(i);
    case VOP1:
      vop1Mod(i);
      break;
      // case VOP2:
      //   return vop2String(i);
      // case VOPC:
      //   return vopcString(i);
      // case VOP3a:
      //   return vop3aString(i);
      // case VOP3b:
      //   return vop3bString(i);
      // case VOP3P:
      // 	return vop3pString(i);
    case FLAT:
      flatMod(i);
      break;
      // case DS:
      //   return dsString(i);
    default:
      break;
      // throw std::runtime_error("unknown instruction format type");
    }
  }
};
#endif