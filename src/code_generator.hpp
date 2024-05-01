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
             const LiftedSymbolInfo &LSO, luthier::InstrumentationTask &ITask,
             int *Addrs);

private:
  CodeGenerator() = default;
  ~CodeGenerator() = default;

  llvm::Expected<std::vector<LiftedSymbolInfo>>
  applyInstrumentation(llvm::Module &Module, llvm::MachineModuleInfo &MMI,
                       const LiftedSymbolInfo &LSO,
                       const InstrumentationTask &ITask);

  llvm::Expected<std::vector<LiftedSymbolInfo>>
  insertFunctionCalls(llvm::Module &Module, llvm::MachineModuleInfo &MMI,
                      const LiftedSymbolInfo &LSI,
                      const InstrumentationTask::insert_call_tasks &Tasks);
};
} // namespace luthier

#endif