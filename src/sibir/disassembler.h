#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H
#include <hsa/hsa.h>
#include <vector>

#include "sibir_types.h"

class Disassembler {
 private:
    std::vector<hsa_agent_t> agents_{};

    /**
     * Initializes the GPU agents the first time a disassembly request is sent
     * @return status of the agent query from HSA
     */
    hsa_status_t initGpuAgents();

    Disassembler() {}
    ~Disassembler() {}
 public:
    Disassembler(const Disassembler &) = delete;
    Disassembler & operator=(const Disassembler &) = delete;


    static inline Disassembler& Instance() {
        static Disassembler instance;
        return instance;
    }

    std::vector<Instr*> disassemble(sibir_address_t kernelObject);
};





#endif
