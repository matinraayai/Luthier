#include "luthier/ToolCodeGen/MachineInstrTraceAddressAnalysis.h"
#include "luthier/ToolCodeGen/DebugInfoPass.h"
#include "luthier/ToolCodeGen/MemoryAllocationAccessor.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"
#include <cstdlib>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-mir-trace-address"

namespace luthier {

llvm::AnalysisKey MachineInstrTraceAddressAnalysis::Key;

bool MachineInstrTraceAddressAnalysis::Result::invalidate(
    llvm::MachineFunction &MF, const llvm::PreservedAnalyses &PA,
    llvm::MachineFunctionAnalysisManager::Invalidator &Inv) {
  auto PAC = PA.getChecker<MachineInstrTraceAddressAnalysis>();
  return !PAC.preserved() && !PAC.preservedSet<llvm::AllAnalysesOn<llvm::MachineFunction>>();
}

MachineInstrTraceAddressAnalysis::Result MachineInstrTraceAddressAnalysis::run(
    llvm::MachineFunction &MF, llvm::MachineFunctionAnalysisManager &MFAM) {
  LLVM_DEBUG(
      llvm::dbgs() << "[MachineInstrTraceAddressAnalysis] Running analysis for "
                   << MF.getName() << "\n";);

  MIToTraceAddr MIToTraceAddrMap;

  for (llvm::MachineBasicBlock &MBB : MF) {
    for (llvm::MachineInstr &MI : MBB) {
      auto *MDNode = TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI);
      if (!MDNode)
        continue;

      std::optional<uint64_t> TraceAddrOpt = MDNode->getTraceInstrAddress();
      if (!TraceAddrOpt)
        continue;

      uint64_t TraceAddr = *TraceAddrOpt;

      MIToTraceAddrMap[&MI] = TraceAddr;

      LLVM_DEBUG(llvm::dbgs()
                 << llvm::formatv("[MachineInstrTraceAddressAnalysis] Cached "
                                  "MI at Trace Addr {0:x}\n",
                                  TraceAddr));
    }
  }

  return Result(std::move(MIToTraceAddrMap));
}

} // namespace luthier
