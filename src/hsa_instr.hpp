#ifndef HSA_INSTR_HPP
#define HSA_INSTR_HPP
#include <llvm/MC/MCInst.h>

#include <unordered_set>

#include "hsa_executable_symbol.hpp"
#include "luthier_types.h"

namespace luthier {
class CodeLifter;

namespace hsa {

class Instr {
private:
  friend class luthier::CodeLifter;
  const llvm::MCInst Inst;
  const luthier_address_t Address;
  const ExecutableSymbol Symbol;

  Instr(llvm::MCInst Inst, hsa::ExecutableSymbol Symbol,
        luthier_address_t Address);

public:
  Instr() = delete;

  static luthier_instruction_t toHandle(Instr *instr) {
    return {reinterpret_cast<decltype(luthier_instruction_t::handle)>(instr)};
  }

  static luthier_instruction_t toHandle(const Instr *instr) {
    return {
        reinterpret_cast<const decltype(luthier_instruction_t::handle)>(instr)};
  }

  static Instr *fromHandle(luthier_instruction_t instr) {
    return reinterpret_cast<Instr *>(instr.handle);
  }

  [[nodiscard]] hsa::GpuAgent getAgent() const;

  [[nodiscard]] hsa::Executable getExecutable() const;

  [[nodiscard]] hsa::ExecutableSymbol getExecutableSymbol() const;

  [[nodiscard]] llvm::MCInst getInstr() const;

  [[nodiscard]] luthier_address_t getAddress() const;
};
} // namespace hsa
} // namespace luthier

#endif