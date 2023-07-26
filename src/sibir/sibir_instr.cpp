#include "sibir_instr.h"

sibir_instr::sibir_instr(sibir_address_t inst_addr, inst_size_t inst_size, std::string inst_string) {
    addr     = inst_addr;
    size     = inst_size;
    str      = inst_string;
    GetOperandsFromString(inst.instr);
}

sibir_instr::sibir_instr(Instr inst) {
    addr     = inst.addr;
    size     = inst.size;
    str      = inst.instr;
    GetOperandsFromString(inst.instr);
}

std::vector<operand> sibir_instr::GetOperandsFromString(std::string inst_string) {
    std::string instrstr_cpy(inst_string);
    std::string delim = " ";
    std::vector<std::string> op_str_vec;
    std::vector<operand> op_vec;
    
    size_t i = instrstr_cpy.find(delim);
    while (i != std::string::npos) {
        i = instrstr_cpy.find(delim);
        op_str_vec.push_back(instrstr_cpy.substr(0, i));
        instrstr_cpy.erase(0, i + 1);
    }
    
    if (op_str_vec.size() > 0) {
        op_str_vec.erase(op_str_vec.begin());
        for (int i = 0; i < op_str_vec.size(); i++) {
            if (op_str_vec.at(i).find(",") != std::string::npos) {
                op_str_vec.at(i).erase(op_str_vec.at(i).find(","), 1);
            }
            if (op_str_vec.at(i).find(" ") != std::string::npos) {
                op_str_vec.at(i).erase(op_str_vec.at(i).find(" "), 1);
            }
            op_vec.push_back(EncodeOperand(op_str_vec.at(i)));
            // auto new_op = EncodeOperand(op_str_vec.at(i));
            // printf("Operand: %s\t", new_op.op_str.c_str());
            // printf("code: %d\t", new_op.code);
            // printf("type: %d\t", new_op.type);
            // printf("value: %lu\n\n", new_op.val);
            // op_vec.push_back(new_op);
        }
    }
    return op_vec;
}

// IMPORTANT NOTE -- I use -1 in the opnd_code when I'm not sure what the ISA expects for encoding
// This is fine for now while I'm developing the instruction class because this isn't being passed
// into anything like an assembler. However, I hope to eventually have correct values for everything
// to prevent any problems down the road.
operand sibir_instr::EncodeOperand(std::string op) {
    int opnd_code;
    OperandType type;
    unsigned long val;

    // Special Register Operands
    if (op.find("vmcnt")   != std::string::npos || // For now, operands like VMCNT and LGKMCNT will be counted as special regs
        op.find("lgkmcnt") != std::string::npos) {
        opnd_code = -1; // Can't find the code to use here in the ISA, need to fix this
        type = SpecialRegOperand;
        return {op, opnd_code, type, val};
    } else if (!op.compare("vcc")) {
        opnd_code = 251;
        type = SpecialRegOperand;
        return {op, opnd_code, type, val};
    }
   
    // Register Operands (gpr)
    size_t sgpr_label_at = op.find("s");
    size_t vgpr_label_at = op.find("v");
    if (sgpr_label_at != std::string::npos) {
        op.erase(sgpr_label_at, 1);
        if (op.find("[") != std::string::npos) {
            op.erase(op.find("["), 1);
            op.erase(op.find(":"), op.length());
        }
        opnd_code = stoi(op, 0, 10);
        type = RegOperand;
        return {op, opnd_code, type, val};
    } else if (vgpr_label_at != std::string::npos) {
        op.erase(vgpr_label_at, 1);
        if (op.find("[") != std::string::npos) {
            op.erase(op.find("["), 1);
            op.erase(op.find(":"), op.length());
        }
        opnd_code = stoi(op, 0, 10) +256;
        type = RegOperand;    
        return {op, opnd_code, type, val};
    }   

    // Immediates
    if (op.find("0x") != std::string::npos) {
        op.erase(0, 2);
        opnd_code = -1;
        type = ImmOperand;
        val = stoi(op, 0, 16);
        return {op, opnd_code, type, val};
    }
    // Inline Constants 
    else {
        opnd_code = -1;
        type = LiteralConstant;
        if (op.compare("off"))  // This operand is used for global mem instructions -- the ISA doc doesn't mention it anywhere
            val = stoi(op, 0, 10);
        return {op, opnd_code, type, val};
    }
    // return {op, opnd_code, type, val};
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
        if (operands.at(i).type == RegOperand || 
            operands.at(i).type == SpecialRegOperand) {
            reg_ops.push_back(operands.at(i));
        }
    }    return reg_ops;
}

