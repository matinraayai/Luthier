#ifndef CODE_GENERATOR_HPP
#define CODE_GENERATOR_HPP

#include "hsa_executable.hpp"
#include "hsa_executable_symbol.hpp"
#include "luthier/instr.h"
#include "luthier/types.h"
#include "object_utils.hpp"
#include <luthier/pass.h>

#include <llvm/IR/Type.h>

#include <llvm/Pass.h>
#include <queue>

namespace luthier {

namespace hsa {

class GpuAgent;

class ISA;

} // namespace hsa

class LiftedSymbolInfoWrapperPass : public llvm::ImmutablePass {
private:
  const LiftedSymbolInfo &LSI;

public:
  static char ID;

  explicit LiftedSymbolInfoWrapperPass(const LiftedSymbolInfo &LSI);

  virtual void anchor();

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  const LiftedSymbolInfo &getLSI() { return LSI; }
};

class CodeGenerator {
public:
  CodeGenerator(const CodeGenerator &) = delete;
  CodeGenerator &operator=(const CodeGenerator &) = delete;

  static inline CodeGenerator &instance() {
    static CodeGenerator Instance;
    return Instance;
  }

  static llvm::Error
  compileRelocatableToExecutable(const llvm::ArrayRef<uint8_t> &Code,
                                 const hsa::GpuAgent &Agent,
                                 llvm::SmallVectorImpl<uint8_t> &Out);

  static llvm::Error
  compileRelocatableToExecutable(const llvm::ArrayRef<uint8_t> &Code,
                                 const hsa::ISA &ISA,
                                 llvm::SmallVectorImpl<uint8_t> &Out);

  llvm::Error
  instrument(std::unique_ptr<llvm::Module> Module,
             std::unique_ptr<llvm::MachineModuleInfoWrapperPass> MMIWP,
             const LiftedSymbolInfo &LSO,
             std::unique_ptr<luthier::InstrumentationPass> IPass);

private:
  CodeGenerator() = default;
  ~CodeGenerator() = default;
};
} // namespace luthier

#endif