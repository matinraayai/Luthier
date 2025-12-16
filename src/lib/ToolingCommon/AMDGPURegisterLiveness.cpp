//===-- AMDGPURegisterLiveness.cpp ----------------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the \c AMDGPURegisterLiveness class and its pass.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/AMDGPURegisterLiveness.h"
#include "luthier/LLVM/streams.h"
#include "luthier/Tooling/LiftedRepresentation.h"
#include "luthier/Tooling/VectorCFG.h"
#include <llvm/Support/TimeProfiler.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lr-register-liveness"

namespace luthier {

static void computeLiveIns(
    llvm::LivePhysRegs &LiveRegs, const VectorMBB &MBB,
    llvm::DenseMap<const llvm::MachineInstr *,
                   std::unique_ptr<llvm::LivePhysRegs>> &PerMILiveIns) {
  auto &TRI = *MBB.getParent().getMF().getSubtarget().getRegisterInfo();
  LiveRegs.init(TRI);
  luthier::addLiveOutsNoPristines(LiveRegs, MBB);
  for (const llvm::MachineInstr &MI : llvm::reverse(MBB)) {
    LiveRegs.stepBackward(MI);
    // Update the MI LiveIns map
    if (!PerMILiveIns.contains(&MI))
      (void)PerMILiveIns.insert(
          {const_cast<llvm::MachineInstr *>(&MI),
           std::move(std::make_unique<llvm::LivePhysRegs>())});
    auto &MILivePhysRegs = PerMILiveIns[&MI];
    MILivePhysRegs->init(TRI);
    for (const auto &LivePhysReg : LiveRegs) {
      MILivePhysRegs->addReg(LivePhysReg);
    }
  }
}

std::vector<llvm::MachineBasicBlock::RegisterMaskPair> computeAndAddLiveIns(
    llvm::LivePhysRegs &LiveRegs, VectorMBB &MBB,
    llvm::DenseMap<const llvm::MachineInstr *,
                   std::unique_ptr<llvm::LivePhysRegs>> &PerMILiveIns) {
  luthier::computeLiveIns(LiveRegs, MBB, PerMILiveIns);
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair> OldLiveIns;
  // Clear out the live-ins before adding the new ones
  // This ensures correct live-out information calculations in loops i.e.
  // where the MBB is a successor/predecessor of itself
  MBB.clearLiveIns(OldLiveIns);
  luthier::addLiveIns(MBB, LiveRegs);
  return OldLiveIns;
}

///// Convenience function for recomputing live-in's for a MBB
///// \return \c true if any changes were made.
static bool recomputeLiveIns(
    VectorMBB &VectorMBB,
    llvm::DenseMap<const llvm::MachineInstr *,
                   std::unique_ptr<llvm::LivePhysRegs>> &PerMILiveIns) {
  llvm::LivePhysRegs LPR;

  auto OldLiveIns = luthier::computeAndAddLiveIns(LPR, VectorMBB, PerMILiveIns);
  VectorMBB.sortUniqueLiveIns();

  const std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &NewLiveIns =
      VectorMBB.getLiveIns();
  return OldLiveIns != NewLiveIns;
}

static void recomputeLiveIns(
    VectorCFG &CFG,
    llvm::DenseMap<const llvm::MachineInstr *,
                   std::unique_ptr<llvm::LivePhysRegs>> &PerMILiveIns) {
  while (true) {
    bool AnyChange = false;
    for (auto &[MBB, SMBB] : CFG) {
      for (auto &VectorMBB : *SMBB)
        if (luthier::recomputeLiveIns(*VectorMBB, PerMILiveIns))
          AnyChange = true;
    }
    if (luthier::recomputeLiveIns(CFG.getEntryBlock(), PerMILiveIns))
      AnyChange = true;
    if (luthier::recomputeLiveIns(CFG.getExitBlock(), PerMILiveIns))
      AnyChange = true;
    if (!AnyChange)
      return;
  }
}

AMDGPURegisterLiveness::AMDGPURegisterLiveness(
    const llvm::Module &M, const llvm::MachineModuleInfo &MMI,
    const LRCallGraph &CG)
    : CG(CG) {
  llvm::TimeTraceScope Scope("Liveness Analysis Computation");
  for (const auto &F : M) {
    auto *MF = MMI.getMachineFunction(F);
    if (!MF)
      continue;
    auto VecCFG = luthier::VectorCFG::getVectorCFG(*MF);
    // TODO: remove liveness vectors from VectorMBBs and use a
    // map inside this pass instead
    luthier::recomputeLiveIns(*VecCFG, MachineInstrLivenessMap);

    LLVM_DEBUG(VecCFG->print(llvm::dbgs()););
  }
}

llvm::AnalysisKey AMDGPURegLivenessAnalysis::Key;

AMDGPURegLivenessAnalysis::Result
AMDGPURegLivenessAnalysis::run(llvm::Module &M,
                               llvm::ModuleAnalysisManager &MAM) {
  return {M, MAM.getResult<llvm::MachineModuleAnalysis>(M).getMMI(),
          MAM.getResult<LRCallGraphAnalysis>(M)};
}

} // namespace luthier
