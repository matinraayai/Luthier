#ifndef SIBIR_INSTR_H
#define SIBIR_INSTR_H

#include "operand.h"
// #include "sibir_types.h"

// struct Operand
// {
//     /* data */
// };

typedef uint64_t sibir_address_t;
typedef uint64_t inst_size_t;

class sibir_instr {
private:
    sibir_address_t addr;

    std::string inst_name;

    Operand src0;
    Operand src1;
    Operand src2;
    Operand dst;
    Operand sdst;
    Operand addr;
    Operand sAddr;
    Operand data;
    Operand data1;
    Operand base;
    Operand offset;
    Operand simm16;

    std::vector<std::string> getInstParams(std::string inststr);

public:
    sibir_instr();
    sibir_instr(Instr inst);
    ~sibir_instr();
};

sibir_instr::sibir_instr() {
// addr   = NULL;
// src0   = NULL;
// src1   = NULL;
// src2   = NULL;
// dst    = NULL;
// sdst   = NULL;
// addr   = NULL;
// sAddr  = NULL;
// data   = NULL;
// data1  = NULL;
// base   = NULL;
// offset = NULL;
// simm16 = NULL;
}

sibir_instr::sibir_instr(Instr inst) {
    addr = inst.addr;

    std::vector<std::string> args;
    
    size_t pos = 0;
    std::string inst_str = inst.instr;


    while ((pos = inst_str.find(" ")) != std::string::npos) {
        args.push_back(inst_str.substr(0, pos));
        inst_str.erase(0, pos + 1);
    }
}

sibir_instr::~sibir_instr() {

}


std::vector<std::string> sibir_instr::getInstParams(std::string inststr) {
    std::vector<std::string> params;
    std::string delim = " ";

    size_t i;
    while (i != std::string::npos) {
        i = inststr.find(delim);
        params.push_back(inststr.substr(0, i));
        inststr.erase(0, i + 1);
    }

    for (int i = 1; i < params.size(); i++) {
        if (params.at(i).find("v") != std::string::npos) {
            if (params.at(i) == "vmnct") {
            continue;;
            } else {
            params.at(i).erase(params.at(i).find("v"), 1);
            }
        } else if (params.at(i).find("s") != std::string::npos) {
            params.at(i).erase(params.at(i).find("s"), 1);
        }

        if (params.at(i).find("[") != std::string::npos) {
            params.at(i).erase(params.at(i).find("["), 1);
            params.at(i).erase(params.at(i).find(":"), params.at(i).length());
        }

        if (params.at(i).find(",") != std::string::npos) {
            params.at(i).erase(params.at(i).find(","), 1);
        }
        if (params.at(i).find(" ") != std::string::npos) {
            params.at(i).erase(params.at(i).find(" "), 1);
        }
    }

    return params;
}


#endif