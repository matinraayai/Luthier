#ifndef LUTHIER_DEBUG_INFO_PASS_H
#define LUTHIER_DEBUG_INFO_PASS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace luthier {

// Per MI Trace Address and Offset
struct MITraceEntry {
  uint64_t TraceAddr;  
  uint64_t DWARFOffset;
};

struct MIToTraceMapping {
  llvm::DenseMap<const llvm::MachineInstr *, MITraceEntry> Map;
};

class DebugInfoPass : public llvm::PassInfoMixin<DebugInfoPass> {
public:
  DebugInfoPass() = default;

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  /// Access the MI-to-trace mapping populated during run().
  // ! TODO: why const and no const versions?
  const MIToTraceMapping &getMIToTraceMapping() const { return Mapping; }
  MIToTraceMapping &getMIToTraceMapping() { return Mapping; }

private:
  MIToTraceMapping Mapping;
};

class DebugInfoPrinterPass
    : public llvm::PassInfoMixin<DebugInfoPrinterPass> {
public:
  explicit DebugInfoPrinterPass(llvm::raw_ostream &OS) : OS(OS) {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

private:
  llvm::raw_ostream &OS;
};

} // namespace luthier

#endif
