#include "hsa_instr.hpp"

#include <utility>

#include "hsa_executable.hpp"

namespace luthier::hsa {

luthier::hsa::Instr::Instr(llvm::MCInst inst, luthier::hsa::ExecutableSymbol symbol)
    : inst_(std::move(inst)),
      symbol_(std::move(symbol)) {}

hsa::Executable Instr::getExecutable() const { return symbol_.getExecutable(); }

hsa::GpuAgent Instr::getAgent() const { return symbol_.getAgent(); }

hsa::ExecutableSymbol Instr::getExecutableSymbol() const { return symbol_; }

llvm::MCInst Instr::getInstr() const { return inst_; }
}// namespace luthier::hsa
