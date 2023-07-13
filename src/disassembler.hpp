#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H
#include <hsa/hsa.h>
#include <unordered_map>
#include <vector>

#include "sibir_types.h"
namespace sibir {

class Disassembler {
 private:


    Disassembler() {}
    ~Disassembler() {}

 public:
    Disassembler(const Disassembler &) = delete;
    Disassembler &operator=(const Disassembler &) = delete;

    static inline Disassembler &Instance() {
        static Disassembler instance;
        return instance;
    }

    static std::vector<Instr> disassemble(sibir_address_t kernelObject);

    static std::vector<Instr> disassemble(hsa_agent_t agent, sibir_address_t address, size_t size);
};

}// namespace sibir

#endif
