#include "operand.h"
#include "initialize.h"
#include "reg.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

std::vector<int> sRegNum;
std::vector<int> vRegNum;

Operand Operand::makeRegOperand(int code, RegType reg, int count) {
    Operand o;
    o.code_ = code;
    o.operandType_ = RegOperand;
    o.reg_ = reg;
    o.regCount_ = count;
    return o;
}

Operand Operand::makeSRegOperand(int code, int index, int count) {
    Operand o;
    o.code_ = code;
    o.operandType_ = RegOperand;
    o.reg_ = SReg(index);
    o.regCount_ = count;

    sRegNum.push_back(index);
    return o;
}
Operand Operand::makeVRegOperand(int code, int index, int count) {
    Operand o;
    o.code_ = code;
    o.operandType_ = RegOperand;
    o.reg_ = VReg(index);
    o.regCount_ = count;

    vRegNum.push_back(index);
    return o;
}
Operand Operand::makeIntOperand(int code, long int value) {
    Operand o;
    o.code_ = code;
    o.operandType_ = IntOperand;
    o.intValue_ = value;
    return o;
}
Operand Operand::makeFloatOperand(int code, double value) {
    Operand o;
    o.code_ = code;
    o.operandType_ = FloatOperand;
    o.floatValue_ = value;
    return o;
}

Operand Operand::makeOperandByCode(uint16_t num) {
    int code = (int) num;
    if (num >= 0 && num <= 101) {
        return makeSRegOperand(code, code, 0);
    } else if (num == 102) {
        return makeRegOperand(code, FlatScratchLo, 0);
    } else if (num == 103) {
        return makeRegOperand(code, FlatScratchHi, 0);
    } else if (num == 104) {
        return makeRegOperand(code, XnackMaskLo, 0);
    } else if (num == 105) {
        return makeRegOperand(code, XnackMaskHi, 0);
    } else if (num == 106) {
        return makeRegOperand(code, VCCLO, 0);
    } else if (num == 107) {
        return makeRegOperand(code, VCCHI, 0);
    } else if (num == 108) {
        return makeRegOperand(code, TbaLo, 0);
    }

    else if (num == 109) {
        return makeRegOperand(code, TbaHi, 0);
    }

    else if (num == 110) {
        return makeRegOperand(code, TmaLo, 0);
    }

    else if (num == 111) {
        return makeRegOperand(code, TmaHi, 0);
    }

    else if (num >= 112 && num < 123) {
        return makeRegOperand(code, RegType(int(Timp0) + num - 112), 0);
    }

    else if (num == 124) {
        return makeRegOperand(code, M0, 0);
    }

    else if (num == 126) {
        return makeRegOperand(code, EXECLO, 0);
    }

    else if (num == 127) {
        return makeRegOperand(code, EXECHI, 0);
    } else if (num >= 128 && num <= 192) {
        return makeIntOperand(code, (long) num - 128);
    } else if (num >= 193 && num <= 208) {
        return makeIntOperand(code, 192 - (long) num);
    } else if (num == 240) {
        return makeFloatOperand(code, 0.5);
    }

    else if (num == 241) {
        return makeFloatOperand(code, -0.5);
    }

    else if (num == 242) {
        return makeFloatOperand(code, 1.0);
    }

    else if (num == 243) {
        return makeFloatOperand(code, -1.0);
    }

    else if (num == 244) {
        return makeFloatOperand(code, 2.0);
    }

    else if (num == 245) {
        return makeFloatOperand(code, -2.0);
    }

    else if (num == 246) {
        return makeFloatOperand(code, 4.0);
    }

    else if (num == 247) {
        return makeFloatOperand(code, -4.0);
    }

    else if (num == 248) {
        return makeFloatOperand(code, 1.0 / (2.0 * M_PI));
    } else if (num == 251) {
        return makeRegOperand(code, VCCZ, 0);
    }

    else if (num == 252) {
        return makeRegOperand(code, EXECZ, 0);
    }

    else if (num == 253) {
        return makeRegOperand(code, SCC, 0);
    }

    else if (num == 255) {
        Operand literalConstantO;
        literalConstantO.code_ = code;
        literalConstantO.operandType_ = LiteralConstant;
        return literalConstantO;
    }

    else if (num >= 256 && num <= 511) {
        return makeVRegOperand(code, int(num) - 256, 0);
    } else {
        std::cerr << "cannot find Operand\n";
        return {};
    }
}




std::string Operand::toString() const {
    std::stringstream stream;
    switch (operandType_) {
        case RegOperand:
            return regOperandToString();
        case IntOperand:
            return std::to_string(intValue_);
        case FloatOperand:
            return std::to_string(floatValue_);
        case LiteralConstant:
            if (literalConstant_ == 0xffffffff) {
                return "-1";
            }
            stream << "0x" << std::hex << literalConstant_;
            return stream.str();
        default:
            return "";
    }
}


std::string Operand::regOperandToString() const {
    std::stringstream stream;
    if (regCount_ > 1) {
        if (isSReg(reg_)) {
            stream << "s[" << regIndex(reg_) << ":"
                   << regIndex(reg_) + regCount_ - 1 << "]";
            sRegNum.push_back(regIndex(reg_) + regCount_ - 1);
            return stream.str();
        } else if (isVReg(reg_)) {
            stream << "v[" << regIndex(reg_) << ":"
                   << regIndex(reg_) + regCount_ - 1 << "]";
            vRegNum.push_back(regIndex(reg_) + regCount_ - 1);
            return stream.str();
        } else if (regName(reg_).find("lo") != std::string::npos) {
            return regName(reg_).substr(0, regName(reg_).length() - 3);
        }
        return "unknown register";
    }
    return regName(reg_);
}