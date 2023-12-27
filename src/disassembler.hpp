#ifndef DISASSEMBLER_HPP
#define DISASSEMBLER_HPP
#include "code_view.hpp"
#include "hsa_executable_symbol.hpp"
#include "hsa_isa.hpp"
#include "instr.hpp"
#include "luthier_types.h"
#include <amd_comgr/amd_comgr.h>
#include <functional>
#include <unordered_map>
#include <vector>

namespace luthier {

/**
 * \brief a singleton class in charge of disassembling device instructions and returning them as an std::vector of \class Instr
 * Uses the AMD COMGR library internally
 */
class Disassembler {
 private:
    Disassembler() = default;
    ~Disassembler() {
        for (const auto &i: sizeDisassemblyInfoMap_)
            amd_comgr_destroy_disassembly_info(i.second);
        for (const auto &i: endPgmDisassemblyInfoMap_)
            amd_comgr_destroy_disassembly_info(i.second);
    };

    amd_comgr_disassembly_info_t getEndPgmDisassemblyInfo(const hsa::Isa &isa);

    amd_comgr_disassembly_info_t getSizeDisassemblyInfo(const hsa::Isa &isa);

    std::unordered_map<hsa::Isa, amd_comgr_disassembly_info_t> sizeDisassemblyInfoMap_;
    std::unordered_map<hsa::Isa, amd_comgr_disassembly_info_t> endPgmDisassemblyInfoMap_;

 public:
    Disassembler(const Disassembler &) = delete;
    Disassembler &operator=(const Disassembler &) = delete;

    static inline Disassembler &instance() {
        static Disassembler instance;
        return instance;
    }

    std::vector<Instr> disassemble(const hsa::ExecutableSymbol &symbol, size_t size);

    std::vector<Instr> disassemble(const hsa::ExecutableSymbol &symbol);

    std::vector<Instr> disassemble(const code::SymbolView &symbol);

    std::vector<Instr> disassemble(const code::SymbolView &symbol, size_t size);

    std::vector<Instr> disassemble(const hsa::Isa &isa, luthier_address_t address);

    std::vector<Instr> disassemble(const hsa::Isa &isa, byte_string_view code);
};

}// namespace luthier

#endif
