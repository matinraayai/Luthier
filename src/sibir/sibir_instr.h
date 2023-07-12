#ifndef SIBIR_INSTR_H
#define SIBIR_INSTR_H

// #include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

typedef uint64_t sibir_address_t;
typedef uint64_t inst_size_t;

class sibir_instr {
public:
    sibir_instr(std::string inststr, sibir_address_t addr, inst_size_t size);

    void operandType(unsigned int operand_num);

    void printInstruction(std::ostream &o);
    void printImmOperands(std::ostream &o);
    void printOperands(std::ostream &o);
    void printUsedRegisters(std::ostream &o);

private:
    sibir_address_t addr;
    inst_size_t size;
    std::string inststr;
    std::string inst_name;
    std::vector<std::string> operands;

    // std::vector<std::string> operandsFromInstString(std::string inststr);
};

sibir_instr::sibir_instr(std::string inststr, sibir_address_t addr, inst_size_t size) {
    addr    = addr;
    size    = size;
    inststr = inststr;
    
    std::string delim = " ";
    size_t i;
    
    while (i != std::string::npos) {
        i = inststr.find(delim);
        operands.push_back(inststr.substr(0, i));
        inststr.erase(0, i + 1);
    }
    for (int i = 1; i < operands.size(); i++) {
        if (operands.at(i).find(",") != std::string::npos) {
            operands.at(i).erase(operands.at(i).find(","), 1);
        }
        if (operands.at(i).find(" ") != std::string::npos) {
            operands.at(i).erase(operands.at(i).find(" "), 1);
        }
    }

    inst_name = operands.at(0);
    operands.erase(0);
}

// std::vector<std::string> sibir_instr::operandsFromInstString(std::string inststr) {
//     std::vector<std::string> operands;
//     std::string delim = " ";
//     size_t i;
    
//     while (i != std::string::npos) {
//         i = inststr.find(delim);
//         operands.push_back(inststr.substr(0, i));
//         inststr.erase(0, i + 1);
//     }
//     for (int i = 1; i < operands.size(); i++) {
//         if (operands.at(i).find(",") != std::string::npos) {
//             operands.at(i).erase(operands.at(i).find(","), 1);
//         }
//         if (operands.at(i).find(" ") != std::string::npos) {
//             operands.at(i).erase(operands.at(i).find(" "), 1);
//         }
//     }
//     return operands;
// }

void sibir_instr::printInstruction(std::ostream &o) {
    o << addr << "\t" << inststr << std::endl;
}

void sibit_instr::printOperands(std::ostream &o) {
    
}

#endif