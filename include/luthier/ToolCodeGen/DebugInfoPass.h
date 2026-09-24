#ifndef LUTHIER_DEBUG_INFO_PASS_H
#define LUTHIER_DEBUG_INFO_PASS_H

#include "luthier/Object/AMDGCNObjectFile.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace luthier {

struct MITraceEntry {
  uint64_t TraceAddr;
  uint64_t DWARFOffset;
};

using MIToTraceMapping =
    llvm::DenseMap<const llvm::MachineInstr *, MITraceEntry>;

using CodeObjectFunctionToDISP = llvm::DenseMap<
    luthier::object::AMDGCNObjectFile *,
    llvm::DenseMap<const char* , llvm::DISubprogram *>>;

class DebugInfoPass : public llvm::PassInfoMixin<DebugInfoPass> {
public:
  DebugInfoPass() = default;

  llvm::PreservedAnalyses run(Prototype &IP, PrototypeAnalysisManager &IPAM);

  [[nodiscard]] const MIToTraceMapping &getMIToTraceMapping() const {
    return MIToTrace;
  }

private:
  MIToTraceMapping MIToTrace;
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
