//===-- LRRegisterLiveness.cpp --------------------------------------------===//
// Copyright 2022-2024 @ Northeastern University Computer Architecture Lab
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
/// This file implements the \c LRRegisterLiveness class.
//===----------------------------------------------------------------------===//
#include "common/Error.hpp"
#include <SIRegisterInfo.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/Support/TimeProfiler.h>
#include <luthier/LRRegisterLiveness.h>
#include <luthier/LiftedRepresentation.h>
#include <luthier/VectorCFG.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lr-register-liveness"

namespace luthier {

static void computeLiveIns(
    llvm::LivePhysRegs &LiveRegs, const llvm::MachineBasicBlock &MBB,
    llvm::DenseMap<const llvm::MachineInstr *,
                   std::unique_ptr<llvm::LivePhysRegs>> &PerMILiveIns) {
  const llvm::MachineFunction &MF = *MBB.getParent();
  const llvm::TargetRegisterInfo &TRI =
      *MF.getRegInfo().getTargetRegisterInfo();
  LiveRegs.init(TRI);
  LiveRegs.addLiveOutsNoPristines(MBB);
  LLVM_DEBUG(llvm::dbgs() << "Live out registers of MBB " << MBB.getFullName()
                          << ": " << LiveRegs << "\n");
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
    llvm::LivePhysRegs &LiveRegs, llvm::MachineBasicBlock &MBB,
    llvm::DenseMap<const llvm::MachineInstr *,
                   std::unique_ptr<llvm::LivePhysRegs>> &PerMILiveIns) {
  luthier::computeLiveIns(LiveRegs, MBB, PerMILiveIns);
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair> OldLiveIns;
  // Clear out the live-ins before adding the new ones
  // This ensures correct live-out information calculations in loops i.e.
  // where the MBB is a successor/predecessor of itself
  MBB.clearLiveIns(OldLiveIns);
  llvm::addLiveIns(MBB, LiveRegs);
  return OldLiveIns;
}

/// Convenience function for recomputing live-in's for a MBB
/// \return \c true if any changes were made.
static bool recomputeLiveIns(
    llvm::MachineBasicBlock &MBB,
    llvm::DenseMap<const llvm::MachineInstr *,
                   std::unique_ptr<llvm::LivePhysRegs>> &PerMILiveIns) {
  llvm::LivePhysRegs LPR;
  LLVM_DEBUG(llvm::dbgs() << "Recalculating Live-in registers for MBB "
                          << MBB.getFullName() << "\n";);

  auto OldLiveIns = luthier::computeAndAddLiveIns(LPR, MBB, PerMILiveIns);
  MBB.sortUniqueLiveIns();

  const std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &NewLiveIns =
      MBB.getLiveIns();

  LLVM_DEBUG(auto TRI = MBB.getParent()->getSubtarget().getRegisterInfo();
             llvm::dbgs() << "Old Live-ins: ["; llvm::interleave(
                 OldLiveIns.begin(), OldLiveIns.end(),
                 [&](const llvm::MachineBasicBlock::RegisterMaskPair &Vec) {
                   llvm::dbgs() << printReg(llvm::Register(Vec.PhysReg), TRI);
                 },
                 [&]() { llvm::dbgs() << ", "; });
             llvm::dbgs() << "]\n";

             llvm::dbgs() << "New Live-ins: ["; llvm::interleave(
                 NewLiveIns.begin(), NewLiveIns.end(),
                 [&](const llvm::MachineBasicBlock::RegisterMaskPair &Vec) {
                   llvm::dbgs() << printReg(llvm::Register(Vec.PhysReg), TRI);
                 },
                 [&]() { llvm::dbgs() << ", "; });
             llvm::dbgs() << "]\n";

  );

  return OldLiveIns != NewLiveIns;
}

/// Re-implementation of \c llvm::fullyRecomputeLiveIns to save
/// the Live-ins at each \c MI
/// \param MF the lifted \c llvm::MachineFunction
/// \param PerMILiveIns a map which keeps track of the \c llvm::LivePhysRegs at
/// each \c llvm::MachineInstr
static void recomputeLiveIns(
    llvm::MachineFunction &MF,
    llvm::DenseMap<const llvm::MachineInstr *,
                   std::unique_ptr<llvm::LivePhysRegs>> &PerMILiveIns) {
  while (true) {
    bool AnyChange = false;
    for (auto &MBB : MF)
      if (luthier::recomputeLiveIns(MBB, PerMILiveIns))
        AnyChange = true;
    if (!AnyChange)
      return;
  }
}

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

   auto OldLiveIns = luthier::computeAndAddLiveIns(LPR, VectorMBB,
   PerMILiveIns); VectorMBB.sortUniqueLiveIns();

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
     for (auto &MBB : CFG)
       if (luthier::recomputeLiveIns(*MBB, PerMILiveIns))
         AnyChange = true;
     if (!AnyChange)
       return;
   }
 }

void LRRegisterLiveness::recomputeLiveIns(const llvm::Module &M,
                                          const llvm::MachineModuleInfo &MMI,
                                          const LRCallGraph &CG) {
  llvm::TimeTraceScope Scope("Liveness Analysis Computation");
  LLVM_DEBUG(llvm::dbgs() << "Recomputing LR Register Liveness analysis.\n");
  //  llvm::DenseMap<const llvm::MachineInstr *,
  //                 std::unique_ptr<llvm::LivePhysRegs>>
  //      MFLiveIns;
  //  llvm::DenseMap<const llvm::MachineInstr *,
  //                 std::unique_ptr<llvm::LivePhysRegs>>
  //      VCFGLiveIns;
  for (const auto &F : M) {
    auto *MF = MMI.getMachineFunction(F);
    if (!MF)
      continue;
//    VecCFG.insert({MF, VectorCFG::getVectorCFG(*MF)});
    luthier::recomputeLiveIns(*MF, MachineInstrLivenessMap);
    //    luthier::recomputeLiveIns(*VecCFG[MF], VCFGLiveIns);
    //    const auto &TRI =
    //    *MF->getSubtarget<llvm::GCNSubtarget>().getRegisterInfo(); const auto
    //    &MRI = MF->getRegInfo(); for (const auto &MBB : *MF) {
    //      for (const auto &MI : MBB) {
    //        (void)MachineInstrLivenessMap.insert(
    //            {const_cast<llvm::MachineInstr *>(&MI),
    //             std::move(std::make_unique<llvm::LivePhysRegs>())});
    //        auto &MILivePhysRegs =
    //            MachineInstrLivenessMap[const_cast<llvm::MachineInstr
    //            *>(&MI)];
    //        MILivePhysRegs->init(TRI);
    //        for (const auto &LivePhysReg : *MFLiveIns[&MI]) {
    //          //          if (!TRI.isVGPR(MRI, LivePhysReg) &&
    //          !TRI.isAGPR(MRI,
    //          //          LivePhysReg))
    //          MILivePhysRegs->addReg(LivePhysReg);
    //        }
    //        //        for (const auto &LivePhysReg : *VCFGLiveIns[&MI]) {
    //        //          if (TRI.isVGPR(MRI, LivePhysReg) || TRI.isAGPR(MRI,
    //        //          LivePhysReg))
    //        //            MILivePhysRegs->addReg(LivePhysReg);
    //      }
    //    }
  }
  // After computing the per-function live-ins, we need to update the liveness
  // information of the call sites
}

llvm::AnalysisKey LRRegLivenessAnalysis::Key;

LRRegLivenessAnalysis::Result
LRRegLivenessAnalysis::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  auto &MMI = MAM.getResult<llvm::MachineModuleAnalysis>(M).getMMI();
  LRRegisterLiveness RegLiveness;
  RegLiveness.recomputeLiveIns(M, MMI, MAM.getResult<LRCallGraphAnalysis>(M));
  return RegLiveness;
}

} // namespace luthier
