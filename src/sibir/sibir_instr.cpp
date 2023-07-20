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

// std::vector<operand> sibir_instr::GetOperandsFromString(std::string inst_string) {
void sibir_instr::GetOperandsFromString(std::string inst_string) {    
    std::string instrstr_cpy(inst_string);
    std::string delim = " ";
    std::vector<std::string> op_str_list;
    
    size_t i = instrstr_cpy.find(delim);
    while (i != std::string::npos) {
        i = instrstr_cpy.find(delim);
        op_str_list.push_back(instrstr_cpy.substr(0, i));
        instrstr_cpy.erase(0, i + 1);
    }
    
    op_str_list.erase(op_str_list.begin());

    for (int i = 0; i < op_str_list.size(); i++) {
        if (op_str_list.at(i).find(",") != std::string::npos) {
            op_str_list.at(i).erase(op_str_list.at(i).find(","), 1);
        }
        if (op_str_list.at(i).find(" ") != std::string::npos) {
            op_str_list.at(i).erase(op_str_list.at(i).find(" "), 1);
        }

        printf("(%s)\t", op_str_list.at(i).c_str());
    }   printf("\n");
    
}

std::vector<operand> sibir_instr::getOperands() { return operands; }

std::vector<operand> sibir_instr::getImmOperands() {
    std::vector<operand> imm_ops;
    for (int i = 0; i < operands.size(); i++) {
        if (operands.at(i).optype == ImmOperand) {
            imm_ops.push_back(operands.at(i));
        }
    }
    return imm_ops;
}

std::vector<operand> sibir_instr::getRegOperands() {
    std::vector<operand> reg_ops;
    for (int i = 0; i < operands.size(); i++) {
        if (operands.at(i).optype == RegOperand) {
            reg_ops.push_back(operands.at(i));
        }
    }    return reg_ops;
}

