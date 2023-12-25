#ifndef DISASSEMBLER_HPP
#define DISASSEMBLER_HPP
#include "code_view.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_agent.hpp"
#include "instr.hpp"
#include "luthier_types.h"
#include <amd_comgr/amd_comgr.h>
#include <functional>
#include <hsa/hsa.h>
#include <unordered_map>
#include <vector>

namespace luthier {

/**
 * \brief a singleton class in charge of disassembling device instructions and returning them as an std::vector of \class Instr
 * Uses the AMD COMGR library internally
 */
class Disassembler {
 public:
    Disassembler(const Disassembler &) = delete;
    Disassembler &operator=(const Disassembler &) = delete;

    static inline Disassembler &instance() {
        static Disassembler instance;
        return instance;
    }

    std::vector<Instr> disassemble(hsa::ExecutableSymbol symbol, size_t size);

    /**
     *
     * @param symbol
     * @return
     */
    std::vector<Instr> disassemble(hsa_executable_symbol_t symbol);


    std::vector<Instr> disassemble(luthier_address_t kernelObject);

    std::vector<Instr> disassemble(luthier_address_t kernelObject, size_t size);

    std::vector<Instr> disassemble(const hsa::GpuAgent& agent, luthier_address_t address);

    std::vector<Instr> disassemble(const hsa::GpuAgent& agent, byte_string_view code);

    std::vector<Instr> disassemble(code::SymbolView symbolView);

 private:
    Disassembler() = default;
    ~Disassembler() {
        for (const auto &i: sizeDisassemblyInfoMap_)
            amd_comgr_destroy_disassembly_info(i.second);
        for (const auto &i: endPgmDisassemblyInfoMap_)
            amd_comgr_destroy_disassembly_info(i.second);
    };

    amd_comgr_disassembly_info_t getEndPgmDisassemblyInfo(const std::string &isa);

    amd_comgr_disassembly_info_t getSizeDisassemblyInfo(const std::string &isa);

    std::unordered_map<std::string, amd_comgr_disassembly_info_t> sizeDisassemblyInfoMap_;
    std::unordered_map<std::string, amd_comgr_disassembly_info_t> endPgmDisassemblyInfoMap_;
};

}// namespace luthier

#endif
