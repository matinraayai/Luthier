//===-- instr.h - HSA Instruction  ------------------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the HSA instruction, which represents an \c MCInst
/// loaded onto the device
//===----------------------------------------------------------------------===//
#ifndef LUTHIER_HSA_INSTR_H
#define LUTHIER_HSA_INSTR_H
#include <llvm/MC/MCInst.h>
#include <llvm/Support/Error.h>

#include "types.h"

namespace luthier {

class CodeLifter;

namespace hsa {

/// \brief represents an AMDGPU instruction loaded by HSA onto a device
/// \details each \c Instr is backed by an \c hsa_executable_symbol_t of type
/// \c KERNEL or \c DEVICE_FUNCTION
class Instr {
private:
  /// \c CodeLifter is the only class allowed to construct an Instr
  /// \c CodeLifter is internal to Luthier and cannot be accessed directly
  friend class luthier::CodeLifter;
  /// The MC representation of the instruction
  const llvm::MCInst Inst;
  /// The address on the GPU Agent this instruction is loaded at
  const address_t LoadedDeviceAddress;
  /// The symbol this instruction belongs to
  const hsa_executable_symbol_t Symbol;
  /// Size of the instruction
  const size_t Size;

  // TODO: add DWARF information

  /// Constructor
  /// \param Inst \c MCInst of the instruction
  /// \param Symbol an \c hsa_executable_symbol_t of type \c KERNEL or type
  /// \c DEVICE_FUNCTION that the instruction belongs to
  /// \param Address the device address this instruction is loaded on
  /// \param Size size of the instruction in bytes
  Instr(llvm::MCInst Inst, hsa_executable_symbol_t Symbol, address_t Address,
        size_t Size);

public:
  /// Deleted default constructor
  Instr() = delete;

  /// \return the symbol of the instruction
  [[nodiscard]] hsa_executable_symbol_t getExecutableSymbol() const;

  /// \return the MC representation of the instruction
  [[nodiscard]] llvm::MCInst getMCInst() const;

  /// \return the address
  /// to get the agent for this
  [[nodiscard]] address_t getLoadedDeviceAddress() const;

  /// \return the size of the instruction in bytes
  [[nodiscard]] size_t getSize() const;
};

} // namespace hsa

} // namespace luthier

#endif