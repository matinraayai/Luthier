#ifndef DISASSEMBLER_H
#define DISASSEMBLER_H
#include <hsa/hsa.h>
#include <unordered_map>
#include <vector>

#include "sibir_types.h"
namespace sibir {

class Disassembler {
 private:
    // Cache the agent information needed for disassembly
    // Maybe we need more than just a single entry
    // Maybe this is not the right place to do it
    typedef struct hsa_agent_entry_s {
        std::string isa;
    } hsa_agent_entry_t;

    // Use the agent handle for hashing
    std::unordered_map<decltype(hsa_agent_t::handle), hsa_agent_entry_t> agents_{};

    /**
     * Initializes the GPU agents the first time a disassembly request is sent
     * @return status of the agent query from HSA
     */
    hsa_status_t initGpuAgentsMap();

    static hsa_status_t populateAgentInfo(hsa_agent_t agent, hsa_agent_entry_t &entry);

    Disassembler() {}
    ~Disassembler() {}

 public:
    Disassembler(const Disassembler &) = delete;
    Disassembler &operator=(const Disassembler &) = delete;

    static inline Disassembler &Instance() {
        static Disassembler instance;
        return instance;
    }

    std::vector<Instr> disassemble(sibir_address_t kernelObject);
};

}// namespace sibir

#endif
