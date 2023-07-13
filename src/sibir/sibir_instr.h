#ifndef SIBIR_INSTR_H
#define SIBIR_INSTR_H

#include <string>
#include <vector>

typedef uint64_t sibir_address_t;
typedef uint64_t inst_size_t;

class sibir_instr {
public:
    sibir_instr(sibir_address_t inst_addr, std::string inst_string, inst_size_t inst_size);

    void printInstruction();

    void getOperandType(unsigned int operand_num);

    std::vector<std::string> getOperands();
    std::vector<std::string> getImmOperands();
    std::vector<std::string> getRegOperands();

private:
    sibir_address_t addr;
    inst_size_t size;
    std::string inst_name;
    std::vector<std::string> operands;
};

sibir_instr::sibir_instr(sibir_address_t inst_addr, std::string inst_string, inst_size_t inst_size) {
    addr     = inst_addr;
    size     = inst_size;

    std::string instrstr_cpy(inst_string);
    std::string delim = " ";
    
    size_t i = instrstr_cpy.find(delim);
    while (i != std::string::npos) {
        i = instrstr_cpy.find(delim);
        operands.push_back(instrstr_cpy.substr(0, i));
        instrstr_cpy.erase(0, i + 1);
    }
    if (operands.size() == 0)
        inst_name = inst_string;
    else {
        for (int i = 1; i < operands.size(); i++) {
            if (operands.at(i).find(",") != std::string::npos) {
                operands.at(i).erase(operands.at(i).find(","), 1);
            }
            if (operands.at(i).find(" ") != std::string::npos) {
                operands.at(i).erase(operands.at(i).find(" "), 1);
            }
        }
        inst_name = operands.at(0);
        operands.erase(operands.begin());
    }
}

void sibir_instr::printInstruction() {
    printf("%lX:%s ", addr, inst_name.c_str()); 
    for (int i = 0; i < operands.size(); i++) 
        printf("%s, ", operands.at(i).c_str()); 
    printf("\n");
}

std::vector<std::string> sibir_instr::getOperands() { return operands; }

std::vector<std::string> sibir_instr::getImmOperands() {
    std::vector<std::string> imm_ops;
    for (int i = 0; i < operands.size(); i++) {
        if (operands.at(i).find("v") == std::string::npos && 
            operands.at(i).find("s") == std::string::npos) {
            imm_ops.push_back(operands.at(i));
        }
    }
    return imm_ops;
}

std::vector<std::string> sibir_instr::getRegOperands() {
    std::vector<std::string> reg_ops;
    for (int i = 0; i < operands.size(); i++) {
        if (operands.at(i).find("v") != std::string::npos || 
            operands.at(i).find("s") != std::string::npos) {
            reg_ops.push_back(operands.at(i));
        }
    }
    return reg_ops;
}


#endif