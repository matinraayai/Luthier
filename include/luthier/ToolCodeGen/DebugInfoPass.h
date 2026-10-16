#ifndef LUTHIER_DEBUG_INFO_PASS_H
#define LUTHIER_DEBUG_INFO_PASS_H

#include "luthier/ToolCodeGen/Prototype.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace luthier {

class DebugInfoPass : public llvm::PassInfoMixin<DebugInfoPass> {
public:
  DebugInfoPass() = default;

  llvm::PreservedAnalyses run(Prototype &IP, PrototypeAnalysisManager &IPAM);
};

class DebugInfoPrinterPass : public llvm::PassInfoMixin<DebugInfoPrinterPass> {
public:
  explicit DebugInfoPrinterPass(llvm::raw_ostream &OS) : OS(OS) {}

  llvm::PreservedAnalyses run(Prototype &IP, PrototypeAnalysisManager &IPAM);

private:
  llvm::raw_ostream &OS;
};

} // namespace luthier

#endif
