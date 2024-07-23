//===-- instr.cpp - HSA Instruction  --------------------------------------===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the HSA instruction.
//===----------------------------------------------------------------------===//
#include <luthier/instr.h>

namespace luthier::hsa {

Instr::Instr(llvm::MCInst Inst, hsa_executable_symbol_t Symbol,
             luthier::address_t Address, size_t Size)
    : Inst(std::move(Inst)), Symbol(Symbol), LoadedDeviceAddress(Address),
      Size(Size) {}

hsa_executable_symbol_t Instr::getExecutableSymbol() const { return Symbol; }

llvm::MCInst Instr::getMCInst() const { return Inst; }

luthier::address_t Instr::getLoadedDeviceAddress() const {
  return LoadedDeviceAddress;
}

size_t Instr::getSize() const { return Size; }

/**
 * Returns this Instr's DWARFDie (a debug info entry for some executable symbol)
*/
// std::optional<llvm::DWARFDie> Instr::getDie() const { return DWARFDebugInfoEntry; }
} // namespace luthier::hsa
