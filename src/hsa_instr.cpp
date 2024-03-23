#include "hsa_instr.hpp"

#include <utility>

#include "hsa_agent.hpp"
#include "hsa_executable.hpp"
#include "hsa_loaded_code_object.hpp"

namespace luthier::hsa {

luthier::hsa::Instr::Instr(llvm::MCInst Inst,
                           hsa::LoadedCodeObject  LCO,
                           const hsa::ExecutableSymbol & Symbol,
                           luthier_address_t Address, size_t Size)
    : Inst(std::move(Inst)), LCO(std::move(LCO)),
      Symbol(Symbol), LoadedDeviceAddress(Address),
      Size(Size) {}

hsa::Executable Instr::getExecutable() const {
  return Symbol.getExecutable(); }

hsa::LoadedCodeObject Instr::getLoadedCodeObject() const {
  return LCO;
}

hsa::GpuAgent Instr::getAgent() const {
  return Symbol.getAgent(); }

hsa::ExecutableSymbol Instr::getExecutableSymbol() const {
  return Symbol; }

llvm::MCInst Instr::getInstr() const {
  return Inst; }

luthier_address_t Instr::getLoadedDeviceAddress() const {
  return LoadedDeviceAddress;
}

size_t Instr::getSize() const {
  return Size; }

} // namespace luthier::hsa
