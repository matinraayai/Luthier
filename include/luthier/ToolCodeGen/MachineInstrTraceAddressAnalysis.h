#ifndef LUTHIER_TOOL_CODE_GEN_MACHINE_INSTR_TRACE_ADDRESS_ANALYSIS_H
#define LUTHIER_TOOL_CODE_GEN_MACHINE_INSTR_TRACE_ADDRESS_ANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/PassManager.h"
#include <cstdint>

namespace luthier {

class MachineInstrTraceAddressAnalysis
    : public llvm::AnalysisInfoMixin<MachineInstrTraceAddressAnalysis> {

  friend AnalysisInfoMixin;

  static llvm::AnalysisKey Key;

public:
  using MIToTraceAddr = llvm::DenseMap<llvm::MachineInstr *, uint64_t>;

  class Result {

    friend MachineInstrTraceAddressAnalysis;

    MIToTraceAddr MIToTraceAddrMap;

    explicit Result(MIToTraceAddr MIToTraceAddrMap)
        : MIToTraceAddrMap{std::move(MIToTraceAddrMap)} {}

  public:
    bool invalidate(llvm::MachineFunction &MF,
                    const llvm::PreservedAnalyses &PA,
                    llvm::MachineFunctionAnalysisManager::Invalidator &Inv);

    [[nodiscard]] const MIToTraceAddr &getMIToTraceMap() const {
      return MIToTraceAddrMap;
    }
  };

  Result run(llvm::MachineFunction &TargetMF,
             llvm::MachineFunctionAnalysisManager &TargetMFAM);
};

} // namespace luthier
#endif
