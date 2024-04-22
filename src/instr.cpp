#include <luthier/instr.h>

#include <utility>

#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"

namespace luthier {

Instr::Instr(llvm::MCInst Inst, hsa_loaded_code_object_t LCO,
             hsa_executable_symbol_t Symbol, luthier::address_t Address,
             size_t Size)
    : Inst(std::move(Inst)), LCO(LCO), Symbol(Symbol),
      LoadedDeviceAddress(Address), Size(Size) {}

hsa_executable_t Instr::getExecutable() const {
  return hsa::ExecutableSymbol::fromHandle(Symbol).getExecutable()->asHsaType();
}

hsa_loaded_code_object_t Instr::getLoadedCodeObject() const { return LCO; }

hsa_agent_t Instr::getAgent() const {
  return hsa::ExecutableSymbol::fromHandle(Symbol).getAgent()->asHsaType();
}

hsa_executable_symbol_t Instr::getExecutableSymbol() const { return Symbol; }

llvm::MCInst Instr::getMCInst() const { return Inst; }

luthier::address_t Instr::getLoadedDeviceAddress() const {
  return LoadedDeviceAddress;
}

size_t Instr::getSize() const { return Size; }

} // namespace luthier
