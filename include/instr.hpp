#ifndef INSTR_HPP
#define INSTR_HPP
#include <llvm/MC/MCInst.h>

#include "luthier_types.h"

namespace luthier {

class CodeLifter;

/**
 * \brief represents an AMDGPU instruction loaded by ROCr
 * It is backed by an \p hsa::ExecutableSymbol of type
 * \p HSA_SYMBOL_KIND_INDIRECT_FUNCTION and \p HSA_SYMBOL_KIND_KERNEL
 */
class Instr {
private:
  friend class luthier::CodeLifter; // CodeLifter is the only class allowed to
                                    // construct a luthier::Instr
                                    // CodeLifter is an internal component and
                                    // cannot be accessed externally
  const llvm::MCInst Inst; // < The MC representation of the instruction
  const address_t LoadedDeviceAddress; // < The address on the device
                                               // this instruction is loaded at
  const hsa_executable_symbol_t
      Symbol; // < The symbol this instruction belongs to
  const hsa_loaded_code_object_t LCO; // < The Loaded Code Object this
                                      // instruction belongs to This is cached
                                      // to avoid looking it up using the symbol

  const size_t Size; // < Size of the instruction

  Instr(llvm::MCInst Inst, hsa_loaded_code_object_t LCO,
        hsa_executable_symbol_t Symbol, address_t Address,
        size_t Size);

public:
  Instr() = delete;

  [[nodiscard]] hsa_agent_t getAgent() const;

  [[nodiscard]] hsa_executable_t getExecutable() const;

  [[nodiscard]] hsa_loaded_code_object_t getLoadedCodeObject() const;

  [[nodiscard]] hsa_executable_symbol_t getExecutableSymbol() const;

  [[nodiscard]] llvm::MCInst getInstr() const;

  [[nodiscard]] address_t getLoadedDeviceAddress() const;

  [[nodiscard]] size_t getSize() const;
};
} // namespace luthier

#endif