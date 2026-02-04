//===-- IPVectorRegLiveness.cpp -------------------------------------------===//
// Copyright 2022-2026 @ Northeastern University Computer Architecture Lab
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
#include "luthier/Tooling/IPPredicatedCFG.h"
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

void IPVectorRegLiveness::addBlockLiveOuts(
    const PredicatedMachineBasicBlock &MBB,
    llvm::LiveRegUnits &LiveUnits) const {
  for (const auto &LO : getPredMBBLiveOuts(MBB))
    LiveUnits.addRegMasked(LO.PhysReg, LO.LaneMask);
}

void IPVectorRegLiveness::addBlockLiveIns(
    const PredicatedMachineBasicBlock &MBB,
    llvm::LiveRegUnits &LiveUnits) const {
  for (const auto &LI : getPredMBBLiveIns(MBB))
    LiveUnits.addRegMasked(LI.PhysReg, LI.LaneMask);
}

void IPVectorRegLiveness::addBlockLiveIns(
    llvm::LivePhysRegs &LPR, const PredicatedMachineBasicBlock &PredMBB) const {
  for (const auto &LI : PredMBBLivenessMap.at(PredMBB)) {
    llvm::MCPhysReg Reg = LI.PhysReg;
    llvm::LaneBitmask Mask = LI.LaneMask;
    auto *TRI = PredMBB.getParent()
                    .getParent()
                    .getMF()
                    .getSubtarget()
                    .getRegisterInfo();
    llvm::MCSubRegIndexIterator S(Reg, TRI);
    assert(Mask.any() && "Invalid live in mask");
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

void IPVectorRegLiveness::addLiveOutsNoPristines(
    llvm::LivePhysRegs &LPR, const PredicatedMachineBasicBlock &MBB) const {
  // To get the live-outs we simply merge the live-ins of all successors.
  for (const auto &Succ : MBB.successors())
    addBlockLiveIns(LPR, Succ);
}

static void addLiveIns(
    std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &VecMBBLiveIns,
    const PredicatedMachineBasicBlock &MBB,
    const llvm::LivePhysRegs &LiveRegs) {
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

void IPVectorRegLiveness::computeLiveIns(
    llvm::LivePhysRegs &LiveRegs, const PredicatedMachineBasicBlock &MBB) {
  addLiveOutsNoPristines(LiveRegs, MBB);
  for (const llvm::MachineInstr &MI : llvm::reverse(MBB)) {
    LiveRegs.stepBackward(MI);
  }
}

std::vector<llvm::MachineBasicBlock::RegisterMaskPair>
IPVectorRegLiveness::computeAndAddLiveIns(
    llvm::LivePhysRegs &LiveRegs, const PredicatedMachineBasicBlock &MBB) {
  auto &TRI =
      *MBB.getParent().getParent().getMF().getSubtarget().getRegisterInfo();
  LiveRegs.init(TRI);
  computeLiveIns(LiveRegs, MBB);
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &MBBLiveIns =
      PredMBBLivenessMap.at(MBB);
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair> OldLiveIns{};
  // Clear out the live-ins before adding the new ones
  // This ensures correct live-out information calculations in loops i.e.
  // where the MBB is a successor/predecessor of itself
  std::swap(MBBLiveIns, OldLiveIns);
  luthier::addLiveIns(MBBLiveIns, MBB, LiveRegs);
  return OldLiveIns;
}

/// Convenience function for recomputing live-in's for a MBB
/// \return \c true if any changes were made.
bool IPVectorRegLiveness::recomputeLiveIns(
    const PredicatedMachineBasicBlock &PredMBB) {
  llvm::LivePhysRegs LPR;
  auto OldLiveIns = computeAndAddLiveIns(LPR, PredMBB);
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &NewLiveIns =
      PredMBBLivenessMap.at(PredMBB);
  sortUniqueLiveIns(NewLiveIns);

  return OldLiveIns != NewLiveIns;
}

void IPVectorRegLiveness::recomputePredMBBLiveIns(
    const IPPredicatedCFG &IPPredCFG) {
  while (true) {
    bool AnyChange = false;
    for (auto &PredMF : IPPredCFG) {
      for (auto &LinearMBB : PredMF) {
        for (auto &PredMBB : LinearMBB) {
          if (recomputeLiveIns(PredMBB))
            AnyChange = true;
        }
      }
    }
    if (!AnyChange)
      return;
  }
}

bool IPVectorRegLiveness::isLiveIn(const PredicatedMachineBasicBlock &PredMBB,
                                   llvm::MCRegister Reg,
                                   llvm::LaneBitmask LaneMask) const {
  auto LiveIns = getPredMBBLiveIns(PredMBB);
  auto I = llvm::find_if(
      LiveIns, [Reg](const llvm::MachineBasicBlock::RegisterMaskPair &LI) {
        return LI.PhysReg == Reg;
      });
  return I != LiveIns.end() && (I->LaneMask & LaneMask).any();
}

void IPVectorRegLiveness::addLiveIns(const PredicatedMachineBasicBlock &PredMBB,
                                     llvm::LiveRegUnits &LRU) const {
  addBlockLiveIns(PredMBB, LRU);
}

void IPVectorRegLiveness::addLiveOuts(
    const PredicatedMachineBasicBlock &PredMBB, llvm::LiveRegUnits &LRU) const {
  addBlockLiveOuts(PredMBB, LRU);
}

std::vector<llvm::MachineBasicBlock::RegisterMaskPair>
IPVectorRegLiveness::getPredMBBLiveOuts(
    const PredicatedMachineBasicBlock &PredMBB) const {
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair> LiveOuts;
  llvm::LivePhysRegs LPR;
  LPR.init(*PredMBB.getParent()
                .getParent()
                .getMF()
                .getSubtarget()
                .getRegisterInfo());
  addLiveOutsNoPristines(LPR, PredMBB);
  luthier::addLiveIns(LiveOuts, PredMBB, LPR);
  return LiveOuts;
}

IPVectorRegLiveness::IPVectorRegLiveness(llvm::Module &M,
                                         llvm::ModuleAnalysisManager &MAM) {
  llvm::TimeTraceScope Scope("Liveness Analysis Computation");
  const IPPredicatedCFG &IPPredGFG =
      MAM.getResult<IPPredCFGAnalysis>(M).getVecCFG();
  for (auto &PredMF : IPPredGFG) {
    for (auto &LinearMBB : PredMF) {
      for (auto &PredMBB : LinearMBB) {
        PredMBBLivenessMap.insert({PredMBB, {}});
      }
    }
  }
  recomputePredMBBLiveIns(IPPredGFG);
}

bool IPVectorRegLiveness::invalidate(
    llvm::Module &M, const llvm::PreservedAnalyses &PA,
    llvm::ModuleAnalysisManager::Invalidator &Inv) {
  // Check whether the analysis, all analyses on machine functions, or the
  // machine function's CFG have been preserved.
  auto PAC = PA.getChecker<IPVectorRegLivenessAnalysis>();
  return !PAC.preserved() &&
         !PAC.preservedSet<llvm::AllAnalysesOn<llvm::MachineFunction>>() &&
         !PAC.preservedSet<llvm::CFGAnalyses>();
}

llvm::AnalysisKey IPVectorRegLivenessAnalysis::Key;

IPVectorRegLivenessAnalysis::Result
IPVectorRegLivenessAnalysis::run(llvm::Module &TargetModule,
                                 llvm::ModuleAnalysisManager &TargetMAM) {
  return {TargetModule, TargetMAM};
}

llvm::PreservedAnalyses
IPVectorRegLivenessPrinter::run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &MAM) {
  const IPPredicatedCFG &IPPredCFG =
      MAM.getResult<IPPredCFGAnalysis>(M).getVecCFG();
  const IPVectorRegLiveness &IPRegLiveness =
      MAM.getResult<IPVectorRegLivenessAnalysis>(M);
  for (const PredicatedMachineFunction &PredMF : IPPredCFG) {
    const llvm::MachineFunction &MF = PredMF.getMF();
    const llvm::TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    OS << "Liveness info for Predicated MF " << MF.getName() << "\n";
    for (const LinearMachineBasicBlock &LinearMBB : PredMF) {
      for (const PredicatedMachineBasicBlock &PredMBB : LinearMBB) {
        OS.indent(2) << "Live ins: [";
        llvm::interleave(
            IPRegLiveness.getPredMBBLiveIns(PredMBB),
            [&](const auto &LiveIn) {
              OS << llvm::printReg(LiveIn.PhysReg, TRI);
            },
            [&]() { OS << ", "; });
        OS << "]\n";
        PredMBB.print(OS, 2);
        OS.indent(2) << "Live outs: [";
        llvm::interleave(
            IPRegLiveness.getPredMBBLiveOuts(PredMBB),
            [&](const auto &LiveIn) {
              OS << llvm::printReg(LiveIn.PhysReg, TRI);
            },
            [&]() { OS << ", "; });
        OS << "]\n\n";
      }
    }
  }
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier
