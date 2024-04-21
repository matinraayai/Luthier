#ifndef LUTHIER_PASS_H
#define LUTHIER_PASS_H
#include <hsa/hsa.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <luthier/instr.hpp>

namespace luthier {

class CodeLifter;

/**
 * \brief contains information regarding a lifted \ref hsa_executable_symbol_t
 * of type \ref HSA_SYMBOL_KIND_INDIRECT_FUNCTION or
 * \ref HSA_SYMBOL_KIND_KERNEL, plus:
 * 1. All functions it can possibly call (i.e. related functions)
 * 2. All variables it can possibly address (i.e. related variables)
 * If the related variables or functions can't be statically determined, all
 * the functions and variables of the parent \ref hsa_loaded_code_object_t
 * will be considered related
 */
class LiftedSymbolInfo {
  friend luthier::CodeLifter;

private:
  hsa_executable_symbol_t Symbol;
  llvm::MachineFunction *SymbolMF{nullptr};
  llvm::DenseMap<decltype(hsa_executable_symbol_t::handle),
                 llvm::MachineFunction *>
      RelatedFunctions{};
  llvm::DenseMap<decltype(hsa_executable_symbol_t::handle),
                 llvm::GlobalVariable *>
      RelatedGlobalVariables{};
  llvm::DenseMap<Instr *, llvm::MachineInstr *> MCToMachineInstrMap{};
  llvm::DenseMap<llvm::MachineInstr *, Instr *> MachineInstrToMCMap{};

public:
  hsa_executable_symbol_t getSymbol() const { return Symbol; };

  llvm::MachineFunction &getMFofSymbol() const { return *SymbolMF; };

  std::optional<llvm::MachineFunction *>
  getMFofSymbol(hsa_executable_symbol_t S) const {
    if (RelatedFunctions.contains(S.handle))
      return RelatedFunctions.at(S.handle);
    else
      return std::nullopt;
  }

  std::optional<llvm::GlobalVariable *>
  getGlobalVariableOfSymbol(hsa_executable_symbol_t S) const {
    if (RelatedGlobalVariables.contains(S.handle))
      return RelatedGlobalVariables.at(S.handle);
    else
      return std::nullopt;
  }

  std::vector<hsa_executable_symbol_t> getRelatedFunctions() const {
    std::vector<hsa_executable_symbol_t> Out;
    Out.reserve(RelatedFunctions.size());
    for (const auto &[Handle, MF] : RelatedFunctions) {
      Out.reserve(Handle);
    }
    return Out;
  }

  std::vector<hsa_executable_symbol_t> getRelatedVariables() const {
    std::vector<hsa_executable_symbol_t> Out;
    Out.reserve(RelatedFunctions.size());
    for (const auto &[Handle, GV] : RelatedGlobalVariables) {
      Out.reserve(Handle);
    }
    return Out;
  }

  std::optional<Instr *>
  getHSAInstrOfMachineInstr(const llvm::MachineInstr &MI) const {
    if (MachineInstrToMCMap.contains(&MI))
      return MachineInstrToMCMap.at(const_cast<llvm::MachineInstr *>(&MI));
    else
      return std::nullopt;
  }
};

class InstrumentationPass : public llvm::ModulePass {

public:
  static char ID;

  InstrumentationPass() : llvm::ModulePass(ID){};


  llvm::Error insertCallTo(llvm::MachineInstr &MI, const void *DevFunc,
                           InstrPoint IPoint, llvm::SmallVectorImpl<llvm::Constant);
};

} // namespace luthier

#endif