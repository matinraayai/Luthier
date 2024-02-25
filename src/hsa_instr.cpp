#include "hsa_instr.hpp"

#include <utility>

#include "hsa_agent.hpp"
#include "hsa_executable.hpp"

namespace luthier::hsa {

luthier::hsa::Instr::Instr(llvm::MCInst Inst,
                           luthier::hsa::ExecutableSymbol Symbol,
                           luthier_address_t Address)
    : Inst(std::move(Inst)), Symbol(std::move(Symbol)), Address(Address) {}

hsa::Executable Instr::getExecutable() const { return Symbol.getExecutable(); }

hsa::GpuAgent Instr::getAgent() const { return Symbol.getAgent(); }

hsa::ExecutableSymbol Instr::getExecutableSymbol() const { return Symbol; }

llvm::MCInst Instr::getInstr() const { return Inst; }

luthier_address_t Instr::getAddress() const { return Address; }
} // namespace luthier::hsa
