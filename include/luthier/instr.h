#ifndef INSTR_HPP
#define INSTR_HPP
#include <llvm/MC/MCInst.h>
#include "llvm/IR/DebugLoc.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "types.h"

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

  const llvm::DWARFDie DWARFDebugInfoEntry; // debug info parsed from ELF's dwarf section

  /**
  TO BE DELETED: once all instances of this constructor have been replaced with the below
  */
  Instr(llvm::MCInst Inst, hsa_loaded_code_object_t LCO,
        hsa_executable_symbol_t Symbol, address_t Address, size_t Size);


  Instr(llvm::MCInst Inst, hsa_loaded_code_object_t LCO,
        hsa_executable_symbol_t Symbol, address_t Address, size_t Size, llvm::DWARFDie &Die);

public:
  Instr() = delete;

  [[nodiscard]] llvm::Expected<hsa_agent_t> getAgent() const;

  [[nodiscard]] hsa_executable_t getExecutable() const;

  [[nodiscard]] hsa_loaded_code_object_t getLoadedCodeObject() const;

  [[nodiscard]] hsa_executable_symbol_t getExecutableSymbol() const;

  [[nodiscard]] llvm::MCInst getMCInst() const;

  [[nodiscard]] address_t getLoadedDeviceAddress() const;

  [[nodiscard]] size_t getSize() const;

  [[nodiscard]] llvm::DWARFDie getDWARFDie() const;
};

} // namespace luthier

#endif
