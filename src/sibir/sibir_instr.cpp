#include "sibir_instr.h"

sibir_instr::sibir_instr(sibir_address_t inst_addr, inst_size_t inst_size, std::string inst_string) {
    addr     = inst_addr;
    size     = inst_size;
    str      = inst_string;
    // operands = GetOperandsFromString(inst_string);
}

sibir_instr::sibir_instr(Instr inst) {
    addr     = inst.addr;
    size     = inst.size;
    str      = inst.instr;
    // operands = GetOperandsFromString(inst.instr);

    GetOperandsFromString(inst.instr);
}

std::vector<operand> sibir_instr::GetOperandsFromString(std::string inst_string) {
    std::string instrstr_cpy(inst_string);
    std::string delim = " ";
    std::vector<std::string> op_str;
    std::vector<operand> op_vec;
    
    size_t i = instrstr_cpy.find(delim);
    while (i != std::string::npos) {
        i = instrstr_cpy.find(delim);
        op_str.push_back(instrstr_cpy.substr(0, i));
        instrstr_cpy.erase(0, i + 1);
    }
    
    op_str.erase(op_str.begin());
    if (op_str.size() > 0) {
        for (int i = 0; i < op_str.size(); i++) {
            if (op_str.at(i).find(",") != std::string::npos) {
                op_str.at(i).erase(op_str.at(i).find(","), 1);
            }
            if (op_str.at(i).find(" ") != std::string::npos) {
                op_str.at(i).erase(op_str.at(i).find(" "), 1);
            }

            printf("(%s)\t", op_str.at(i).c_str());

            op_vec.push_back(EncodeOperand(op_str.at(i)));
        }   printf("\n");
    }
    return op_vec;
}

operand sibir_instr::EncodeOperand(std::string op) {
    if (op.find("vmcnt")   != std::string::npos ||
        op.find("lgkmcnt") != std::string::npos) {
            return {-1, SpecialRegOperand}; // Can't find the code to use here in the ISA, need to fix this
    } else if (!op.compare("vcc")) {
        return {251, SpecialRegOperand};
    } else if (op.find("0x") != std::string::npos) { // these are immediates, want to implement operand val
        op.erase(0, 2);
        return {255, ImmOperand};
    } else { // If we get here, assume that we're dealing with a gpr
        int opnd_code;

        if (op.find("[") != std::string::npos) {
            op.erase(op.find("["), 1);
            op.erase(op.find(":"), op.length());
        }
        if (op.find("s") != std::string::npos) {
            op.erase(op.find("s"), 1);
            opnd_code = stoi(op);
        } else if (op.find("v") != std::string::npos) {
            op.erase(op.find("s"), 1);
            opnd_code = stoi(op) + 256;
        }
        return {opnd_code, RegOperand};
    }
}

std::string sibir_instr::getInstString() { return str; }

int sibir_instr::getNumOperands() { return operands.size(); }

operand sibir_instr::getOperand(int num) { return operands.at(num); }

OperandType 
sibir_instr::getOperandType(int num) { return operands.at(num).type; }

std::vector<operand> sibir_instr::getAllOperands() { return operands; }

std::vector<operand> sibir_instr::getImmOperands() {
    std::vector<operand> imm_ops;
    for (int i = 0; i < operands.size(); i++) {
        if (operands.at(i).type == ImmOperand) {
            imm_ops.push_back(operands.at(i));
        }
    }
    return imm_ops;
}

std::vector<operand> sibir_instr::getRegOperands() {
    std::vector<operand> reg_ops;
    for (int i = 0; i < operands.size(); i++) {
        if (operands.at(i).type == RegOperand) {
            reg_ops.push_back(operands.at(i));
        }
    }    return reg_ops;
}

