#include "hsa_instr.hpp"

#include <utility>

#include "hsa_agent.hpp"
#include "hsa_executable.hpp"

namespace luthier::hsa {

luthier::hsa::Instr::Instr(llvm::MCInst Inst,
                           const luthier::hsa::ExecutableSymbol& Symbol,
                           luthier_address_t Address, size_t Size)
    : Inst(std::move(Inst)), Symbol(Symbol), Address(Address),
      Size(Size) {}

hsa::Executable Instr::getExecutable() const { return Symbol.getExecutable(); }

hsa::GpuAgent Instr::getAgent() const { return Symbol.getAgent(); }

hsa::ExecutableSymbol Instr::getExecutableSymbol() const { return Symbol; }

llvm::MCInst Instr::getInstr() const { return Inst; }

luthier_address_t Instr::getAddress() const { return Address; }

size_t Instr::getSize() const { return Size; }

} // namespace luthier::hsa
