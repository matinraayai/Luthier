#include "initialize.h"
#include "operand.h"
#include "reg.h"
#define _USE_MATH_DEFINES
#include <iostream>
#include <math.h>
#include <vector>

std::vector<int> sRegNum;
std::vector<int> vRegNum;

Operand newRegOperand(int code, RegType reg, int count)
{
  Operand o;
  o.code = code;
  o.operandType = RegOperand;
  o.reg = Regs[reg];
  o.regCount = count;
  return o;
}
Operand newSRegOperand(int code, int index, int count)
{
  Operand o;
  o.code = code;
  o.operandType = RegOperand;
  o.reg = SReg(index);
  o.regCount = count;

  sRegNum.push_back(index);
  return o;
}
Operand newVRegOperand(int code, int index, int count)
{
  Operand o;
  o.code = code;
  o.operandType = RegOperand;
  o.reg = VReg(index);
  o.regCount = count;

  vRegNum.push_back(index);
  return o;
}
Operand newIntOperand(int code, long int value)
{
  Operand o;
  o.code = code;
  o.operandType = IntOperand;
  o.intValue = value;
  return o;
}
Operand newFloatOperand(int code, double value)
{
  Operand o;
  o.code = code;
  o.operandType = FloatOperand;
  o.floatValue = value;
  return o;
}

Operand getOperandByCode(uint16_t num)
{
  int code = (int)num;
  if (num >= 0 && num <= 101)
  {
    return newSRegOperand(code, code, 0);
  }
  else if (num == 102)
  {
    return newRegOperand(code, FlatScratchLo, 0);
  }
  else if (num == 103)
  {
    return newRegOperand(code, FlatScratchHi, 0);
  }
  else if (num == 104)
  {
    return newRegOperand(code, XnackMaskLo, 0);
  }
  else if (num == 105)
  {
    return newRegOperand(code, XnackMaskHi, 0);
  }
  else if (num == 106)
  {
    return newRegOperand(code, VCCLO, 0);
  }
  else if (num == 107)
  {
    return newRegOperand(code, VCCHI, 0);
  }
  else if (num == 108)
  {
    return newRegOperand(code, TbaLo, 0);
  }

  else if (num == 109)
  {
    return newRegOperand(code, TbaHi, 0);
  }

  else if (num == 110)
  {
    return newRegOperand(code, TmaLo, 0);
  }

  else if (num == 111)
  {
    return newRegOperand(code, TmaHi, 0);
  }

  else if (num >= 112 && num < 123)
  {
    return newRegOperand(code, RegType(int(Timp0) + num - 112), 0);
  }

  else if (num == 124)
  {
    return newRegOperand(code, M0, 0);
  }

  else if (num == 126)
  {
    return newRegOperand(code, EXECLO, 0);
  }

  else if (num == 127)
  {
    return newRegOperand(code, EXECHI, 0);
  }
  else if (num >= 128 && num <= 192)
  {
    return newIntOperand(code, (long)num - 128);
  }
  else if (num >= 193 && num <= 208)
  {
    return newIntOperand(code, 192 - (long)num);
  }
  else if (num == 240)
  {
    return newFloatOperand(code, 0.5);
  }

  else if (num == 241)
  {
    return newFloatOperand(code, -0.5);
  }

  else if (num == 242)
  {
    return newFloatOperand(code, 1.0);
  }

  else if (num == 243)
  {
    return newFloatOperand(code, -1.0);
  }

  else if (num == 244)
  {
    return newFloatOperand(code, 2.0);
  }

  else if (num == 245)
  {
    return newFloatOperand(code, -2.0);
  }

  else if (num == 246)
  {
    return newFloatOperand(code, 4.0);
  }

  else if (num == 247)
  {
    return newFloatOperand(code, -4.0);
  }

  else if (num == 248)
  {
    return newFloatOperand(code, 1.0 / (2.0 * M_PI));
  }
  else if (num == 251)
  {
    return newRegOperand(code, VCCZ, 0);
  }

  else if (num == 252)
  {
    return newRegOperand(code, EXECZ, 0);
  }

  else if (num == 253)
  {
    return newRegOperand(code, SCC, 0);
  }

  else if (num == 255)
  {
    Operand literalConstantO;
    literalConstantO.code = code;
    literalConstantO.operandType = LiteralConstant;
    return literalConstantO;
  }

  else if (num >= 256 && num <= 511)
  {
    return newVRegOperand(code, int(num) - 256, 0);
  }
  else
  {
    std::cerr << "cannot find Operand\n";
    return {};
  }
}
