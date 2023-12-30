#ifndef HSA_INSTR_HPP
#define HSA_INSTR_HPP
#include <llvm/MC/MCInst.h>

#include <unordered_set>

#include "hsa_executable_symbol.hpp"
#include "luthier_types.h"

namespace luthier {
class Disassembler;

namespace hsa {

class Instr {
 private:
    friend class luthier::Disassembler;
    const llvm::MCInst inst_;
    const ExecutableSymbol symbol_;

    Instr(llvm::MCInst inst, hsa::ExecutableSymbol symbol);

 public:
    Instr() = delete;

    static luthier_instruction_t toHandle(Instr* instr) {
        return {reinterpret_cast<decltype(luthier_instruction_t::handle)>(instr)};
    }

    static luthier_instruction_t toHandle(const Instr* instr) {
        return {reinterpret_cast<const decltype(luthier_instruction_t::handle)>(instr)};
    }

    static Instr* fromHandle(luthier_instruction_t instr) { return reinterpret_cast<Instr*>(instr.handle);}

    [[nodiscard]] hsa::GpuAgent getAgent() const;

    [[nodiscard]] hsa::Executable getExecutable() const;

    [[nodiscard]] hsa::ExecutableSymbol getExecutableSymbol() const;

    [[nodiscard]] llvm::MCInst getInstr() const;
};
}// namespace hsa
}// namespace luthier

#endif