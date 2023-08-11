#ifndef DISASSEMBLER_HPP
#define DISASSEMBLER_HPP
#include "instr.hpp"
#include "luthier_types.hpp"
#include <amd_comgr/amd_comgr.h>
#include <functional>
#include <hsa/hsa.h>
#include <unordered_map>
#include <vector>

namespace luthier {

class Disassembler {
 public:
    Disassembler(const Disassembler &) = delete;
    Disassembler &operator=(const Disassembler &) = delete;

    static inline Disassembler &Instance() {
        static Disassembler instance;
        return instance;
    }

    std::vector<Instr> disassemble(hsa_executable_symbol_t symbol, size_t size);

    std::vector<Instr> disassemble(hsa_executable_symbol_t symbol);

    std::vector<Instr> disassemble(luthier_address_t kernelObject);

    std::vector<Instr> disassemble(luthier_address_t kernelObject, size_t size);

    std::vector<Instr> disassemble(hsa_agent_t agent, luthier_address_t address);

    std::vector<Instr> disassemble(hsa_agent_t agent, luthier_address_t address, size_t size);

 private:
    Disassembler() = default;
    ~Disassembler() {
        for (const auto& i: sizeDisassemblyInfoMap_)
            amd_comgr_destroy_disassembly_info(i.second);
        for (const auto& i: endPgmDisassemblyInfoMap_)
            amd_comgr_destroy_disassembly_info(i.second);
    };

    amd_comgr_disassembly_info_t getEndPgmDisassemblyInfo(const std::string& isa);

    amd_comgr_disassembly_info_t getSizeDisassemblyInfo(const std::string& isa);

    std::unordered_map<std::string, amd_comgr_disassembly_info_t> sizeDisassemblyInfoMap_;
    std::unordered_map<std::string, amd_comgr_disassembly_info_t> endPgmDisassemblyInfoMap_;

};

}// namespace luthier

#endif
