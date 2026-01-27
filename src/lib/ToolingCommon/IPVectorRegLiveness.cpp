//===-- IPVectorRegLiveness.cpp -------------------------------------------===//
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
/// \file IPVectorRegLiveness.cpp
/// Implements the \c IPVectorRegLiveness class and its analysis pass.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/IPVectorRegLiveness.h"
#include "luthier/Tooling/IPVectorCFG.h"
#include "luthier/Tooling/LiftedRepresentation.h"
#include <llvm/Support/TimeProfiler.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lr-register-liveness"

namespace luthier {

static void
addLiveIn(std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &VecMBBLiveIns,
          llvm::MCRegister PhysReg,
          llvm::LaneBitmask LaneMask = llvm::LaneBitmask::getAll()) {
  VecMBBLiveIns.emplace_back(PhysReg, LaneMask);
}

static void clearLiveIns(
    std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &CurrentLiveIns,
    std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &OldLiveIns) {
  std::swap(CurrentLiveIns, OldLiveIns);
}

static void sortUniqueLiveIns(
    std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &LiveIns) {
  llvm::sort(LiveIns, [](const llvm::MachineBasicBlock::RegisterMaskPair &LI0,
                         const llvm::MachineBasicBlock::RegisterMaskPair &LI1) {
    return LI0.PhysReg < LI1.PhysReg;
  });
  // Liveins are sorted by physreg now we can merge their lanemasks.
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair>::const_iterator I =
      LiveIns.begin();
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair>::const_iterator J;
  auto Out = LiveIns.begin();
  for (; I != LiveIns.end(); ++Out, I = J) {
    llvm::MCRegister PhysReg = I->PhysReg;
    llvm::LaneBitmask LaneMask = I->LaneMask;
    for (J = std::next(I); J != LiveIns.end() && J->PhysReg == PhysReg; ++J)
      LaneMask |= J->LaneMask;
    Out->PhysReg = PhysReg;
    Out->LaneMask = LaneMask;
  }
  LiveIns.erase(Out, LiveIns.end());
}

static void
addBlockLiveIns(llvm::LivePhysRegs &LPR, const VectorMBB &VecMBB,
                const std::vector<llvm::MachineBasicBlock::RegisterMaskPair>
                    &VecMBBLiveIns) {
  for (const auto &LI : VecMBBLiveIns) {
    llvm::MCPhysReg Reg = LI.PhysReg;
    llvm::LaneBitmask Mask = LI.LaneMask;
    auto *TRI =
        VecMBB.getParent().getParent().getMF().getSubtarget().getRegisterInfo();
    llvm::MCSubRegIndexIterator S(Reg, TRI);
    assert(Mask.any() && "Invalid livein mask");
    if (Mask.all() || !S.isValid()) {
      LPR.addReg(Reg);
      continue;
    }
    for (; S.isValid(); ++S) {
      unsigned SI = S.getSubRegIndex();
      if ((Mask & TRI->getSubRegIndexLaneMask(SI)).any())
        LPR.addReg(S.getSubReg());
    }
  }
}

static void addLiveOutsNoPristines(
    llvm::LivePhysRegs &LPR, const VectorMBB &MBB,
    const std::vector<llvm::MachineBasicBlock::RegisterMaskPair>
        &VecMBBLiveIns) {
  // To get the live-outs we simply merge the live-ins of all successors.
  for (const VectorMBB *Succ : MBB.successors())
    addBlockLiveIns(LPR, *Succ, VecMBBLiveIns);
}

static void addLiveIns(
    std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &VecMBBLiveIns,
    VectorMBB &MBB, const llvm::LivePhysRegs &LiveRegs) {
  const auto &MF = MBB.getParent().getParent().getMF();
  const auto &MRI = MF.getRegInfo();
  const auto &TRI = *MRI.getTargetRegisterInfo();
  for (llvm::MCPhysReg Reg : LiveRegs) {
    if (MRI.isReserved(Reg))
      continue;
    // Skip the register if we are about to add one of its super registers.
    if (llvm::any_of(TRI.superregs(Reg), [&](llvm::MCPhysReg SReg) {
          return LiveRegs.contains(SReg) && !MRI.isReserved(SReg);
        }))
      continue;
    addLiveIn(VecMBBLiveIns, Reg);
  }
}

static void computeLiveIns(
    llvm::LivePhysRegs &LiveRegs, const VectorMBB &MBB,
    std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &PerMILiveIns) {
  auto &TRI =
      *MBB.getParent().getParent().getMF().getSubtarget().getRegisterInfo();
  LiveRegs.init(TRI);
  luthier::addLiveOutsNoPristines(LiveRegs, MBB, PerMILiveIns);
  for (const llvm::MachineInstr &MI : llvm::reverse(MBB)) {
    LiveRegs.stepBackward(MI);
  }
}

std::vector<llvm::MachineBasicBlock::RegisterMaskPair> computeAndAddLiveIns(
    llvm::LivePhysRegs &LiveRegs, VectorMBB &MBB,
    std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &PerVecMBBLiveIns) {
  luthier::computeLiveIns(LiveRegs, MBB, PerVecMBBLiveIns);
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair> OldLiveIns;
  // Clear out the live-ins before adding the new ones
  // This ensures correct live-out information calculations in loops i.e.
  // where the MBB is a successor/predecessor of itself
  clearLiveIns(PerVecMBBLiveIns, OldLiveIns);
  luthier::addLiveIns(PerVecMBBLiveIns, MBB, LiveRegs);
  return OldLiveIns;
}

/// Convenience function for recomputing live-in's for a MBB
/// \return \c true if any changes were made.
static bool recomputeLiveIns(
    VectorMBB &VectorMBB,
    std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &PerVecMBBLiveIns) {
  llvm::LivePhysRegs LPR;

  auto OldLiveIns =
      luthier::computeAndAddLiveIns(LPR, VectorMBB, PerVecMBBLiveIns);
  sortUniqueLiveIns(PerVecMBBLiveIns);

  return OldLiveIns != PerVecMBBLiveIns;
}

static void recomputeLiveIns(
    const IPVectorCFG &CFG,
    llvm::DenseMap<const VectorMBB *,
                   std::vector<llvm::MachineBasicBlock::RegisterMaskPair>>
        &PerVecMBBLiveIns) {
  while (true) {
    bool AnyChange = false;
    for (auto &VectorCFG : CFG) {
      for (auto &SMBB : VectorCFG) {
        for (auto &VectorMBB : SMBB)
          if (PerVecMBBLiveIns.try_emplace(
                  &VectorMBB,
                  std::move(std::vector<
                            llvm::MachineBasicBlock::RegisterMaskPair>{}));
              luthier::recomputeLiveIns(VectorMBB,
                                        PerVecMBBLiveIns.at(&VectorMBB)))
            AnyChange = true;
      }
    }
    if (!AnyChange)
      return;
  }
}

IPVectorRegLiveness::IPVectorRegLiveness(llvm::Module &M,
                                         llvm::ModuleAnalysisManager &MAM) {
  llvm::TimeTraceScope Scope("Liveness Analysis Computation");
  const IPVectorCFG &IPVecGFG =
      MAM.getResult<IPVectorCFGAnalysis>(M).getVecCFG();

  luthier::recomputeLiveIns(IPVecGFG, VectorMBBLivenessMap);
}

bool IPVectorRegLiveness::invalidate(
    llvm::Module &M, const llvm::PreservedAnalyses &PA,
    llvm::ModuleAnalysisManager::Invalidator &Inv) {
  // Check whether the analysis, all analyses on machine functions, or the
  // machine function's CFG have been preserved.
  auto PAC = PA.getChecker<IPVectorRegLivenessAnalysis>();
  return !PAC.preserved() &&
         !PAC.preservedSet<llvm::AllAnalysesOn<
             llvm::MachineFunctionAnalysisManagerModuleProxy>>() &&
         !PAC.preservedSet<llvm::CFGAnalyses>();
}

llvm::AnalysisKey IPVectorRegLivenessAnalysis::Key;

IPVectorRegLivenessAnalysis::Result
IPVectorRegLivenessAnalysis::run(llvm::Module &TargetModule,
                                 llvm::ModuleAnalysisManager &TargetMAM) {
  return {TargetModule, TargetMAM};
}

} // namespace luthier
