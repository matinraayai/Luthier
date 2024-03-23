#ifndef HSA_INSTR_HPP
#define HSA_INSTR_HPP
#include <llvm/MC/MCInst.h>

#include <unordered_set>

#include "hsa_executable_symbol.hpp"
#include "hsa_loaded_code_object.hpp"
#include "luthier_types.h"

namespace luthier {

class CodeLifter;

namespace hsa {

/**
 * \brief represents an AMDGPU instruction loaded by HSA
 * It is backed by an \p hsa::ExecutableSymbol of type
 * \p HSA_SYMBOL_KIND_INDIRECT_FUNCTION and \p HSA_SYMBOL_KIND_KERNEL
 */
class Instr {
private:
  friend class luthier::CodeLifter; // CodeLifter is the only class allowed to
                                    // construct an hsa::Instr
  const llvm::MCInst Inst; // < The MC representation of the instruction
  const luthier_address_t LoadedDeviceAddress; // < The address on the device
                                               // this instruction is loaded at
  const ExecutableSymbol Symbol; // < The symbol this instruction belongs to
  const LoadedCodeObject LCO;    // < The Loaded Code Object this instruction
                                 // belongs to
                                 // This is cached to avoid looking it up using
                                 // the symbol

  const size_t Size; // < Size of the instruction

  Instr(llvm::MCInst Inst, hsa::LoadedCodeObject LCO,
        const hsa::ExecutableSymbol &Symbol, luthier_address_t Address,
        size_t Size);

public:
  Instr() = delete;

  static luthier_instruction_t toHandle(Instr *Instr) {
    return {reinterpret_cast<decltype(luthier_instruction_t::handle)>(Instr)};
  }

  static luthier_instruction_t toHandle(const Instr *Instr) {
    return {
        reinterpret_cast<const decltype(luthier_instruction_t::handle)>(Instr)};
  }

  static Instr *fromHandle(luthier_instruction_t Instr) {
    return reinterpret_cast<hsa::Instr *>(Instr.handle);
  }

  [[nodiscard]] hsa::GpuAgent getAgent() const;

  [[nodiscard]] hsa::Executable getExecutable() const;

  [[nodiscard]] hsa::LoadedCodeObject getLoadedCodeObject() const;

  [[nodiscard]] hsa::ExecutableSymbol getExecutableSymbol() const;

  [[nodiscard]] llvm::MCInst getInstr() const;

  [[nodiscard]] luthier_address_t getLoadedDeviceAddress() const;

  [[nodiscard]] size_t getSize() const;
};
} // namespace hsa
} // namespace luthier

#endif