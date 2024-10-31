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
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineRegisterInfo.h>
#include <llvm/Support/TimeProfiler.h>
#include <luthier/LRRegisterLiveness.h>
#include <luthier/LiftedRepresentation.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-lr-register-liveness"

namespace luthier {

LRRegisterLiveness::LRRegisterLiveness(const luthier::LiftedRepresentation &LR)
    : LR(LR) {}

static void computeLiveIns(
    llvm::LivePhysRegs &LiveRegs, const llvm::MachineBasicBlock &MBB,
    llvm::DenseMap<llvm::MachineInstr *, std::unique_ptr<llvm::LivePhysRegs>>
        &PerMILiveIns) {
  const llvm::MachineFunction &MF = *MBB.getParent();
  const llvm::MachineRegisterInfo &MRI = MF.getRegInfo();
  const llvm::TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();
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
    auto &MILivePhysRegs = PerMILiveIns[const_cast<llvm::MachineInstr *>(&MI)];
    MILivePhysRegs->init(TRI);
    for (const auto &LivePhysReg : LiveRegs) {
      MILivePhysRegs->addReg(LivePhysReg);
    }
  }
}

std::vector<llvm::MachineBasicBlock::RegisterMaskPair> computeAndAddLiveIns(
    llvm::LivePhysRegs &LiveRegs, llvm::MachineBasicBlock &MBB,
    llvm::DenseMap<llvm::MachineInstr *, std::unique_ptr<llvm::LivePhysRegs>>
        &PerMILiveIns) {
  luthier::computeLiveIns(LiveRegs, MBB, PerMILiveIns);
  std::vector<llvm::MachineBasicBlock::RegisterMaskPair> OldLiveIns;
  // Clear out the live-ins before adding the new ones
  // This ensures correct live-out information calculations in loops i.e.
  // where the MBB is a successor/predecessor of itself
  MBB.clearLiveIns(OldLiveIns);
  llvm::addLiveIns(MBB, LiveRegs);
  return OldLiveIns;
}

/// Convenience function for recomputing live-in's for a MBB. Returns true if
/// any changes were made.
static bool recomputeLiveIns(
    llvm::MachineBasicBlock &MBB,
    llvm::DenseMap<llvm::MachineInstr *, std::unique_ptr<llvm::LivePhysRegs>>
        &PerMILiveIns) {
  llvm::LivePhysRegs LPR;
  LLVM_DEBUG(auto TRI = MBB.getParent()->getSubtarget().getRegisterInfo();
             llvm::dbgs() << "Old live-in registers for MBB "
                          << MBB.getFullName() << "\n";
             for (auto &LiveInPhysReg : MBB.getLiveIns()) {
               llvm::dbgs()
                   << printReg(llvm::Register(LiveInPhysReg.PhysReg), TRI)
                   << "\n";
             }

  );

  auto OldLiveIns = luthier::computeAndAddLiveIns(LPR, MBB, PerMILiveIns);
  MBB.sortUniqueLiveIns();

  const std::vector<llvm::MachineBasicBlock::RegisterMaskPair> &NewLiveIns =
      MBB.getLiveIns();
  return OldLiveIns != NewLiveIns;
}

/// Re-implementation of \c llvm::fullyRecomputeLiveIns to save
/// the Live-ins at each MI as well
/// \param MF the lifted \c llvm::MachineFunction
/// \param PerMILiveIns a map which keeps track of the \c llvm::LivePhysRegs at
/// each \c llvm::MachineInstr
static void fullyComputeLiveInsOfLiftedMF(
    llvm::MachineFunction &MF,
    llvm::DenseMap<llvm::MachineInstr *, std::unique_ptr<llvm::LivePhysRegs>>
        &PerMILiveIns) {
  while (true) {
    bool AnyChange = false;
    for (auto &MBB : MF)
      if (luthier::recomputeLiveIns(MBB, PerMILiveIns))
        AnyChange = true;
    if (!AnyChange)
      return;
  }
}

void LRRegisterLiveness::recomputeLiveIns() {
  llvm::TimeTraceScope Scope("Liveness Analysis Computation");
  for (const auto &[FuncSymbol, MF] : LR.functions()) {
    luthier::fullyComputeLiveInsOfLiftedMF(*MF, MachineInstrLivenessMap);
  }
}

} // namespace luthier
