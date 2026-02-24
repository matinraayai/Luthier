//===---- ReachingDefAnalysis.cpp - Reaching Def Analysis ---*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/IPPredicatedLoopTraversal.h"
#include "luthier/Tooling/IPReachingDefAnalysis.h"
#include "luthier/Tooling/IPVectorRegLiveness.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "luthier-reaching-defs-analysis"

namespace luthier {
llvm::AnalysisKey IPReachingDefAnalysis::Key;

IPReachingDefAnalysis::Result
IPReachingDefAnalysis::run(llvm::Module &TargetModule,
                         llvm::ModuleAnalysisManager &TargetMAM) {
  IPReachingDefInfo RDI;
  RDI.run(TargetModule, TargetMAM);
  return RDI;
}

llvm::PreservedAnalyses
ReachingDefPrinterPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  auto &RDI = MAM.getResult<IPReachingDefAnalysis>(M);
  RDI.print(OS);
  return llvm::PreservedAnalyses::all();
}

IPReachingDefInfo::IPReachingDefInfo() = default;
IPReachingDefInfo::IPReachingDefInfo(IPReachingDefInfo &&) noexcept = default;
IPReachingDefInfo::~IPReachingDefInfo() = default;

bool IPReachingDefInfo::invalidate(llvm::Module &Module,
                                 const llvm::PreservedAnalyses &PA,
                                 llvm::ModuleAnalysisManager::Invalidator &) {
  // Check whether the analysis, all analyses on machine functions, or the
  // machine function's CFG have been preserved.
  auto PAC = PA.getChecker<IPReachingDefAnalysis>();
  return !PAC.preserved() &&
         !PAC.preservedSet<llvm::AllAnalysesOn<llvm::MachineFunction>>() &&
         !PAC.preservedSet<llvm::CFGAnalyses>();
}

static bool isValidReg(const llvm::MachineOperand &MO) {
  return MO.isReg() && MO.getReg();
}

static bool isValidRegUse(const llvm::MachineOperand &MO) {
  return isValidReg(MO) && MO.isUse();
}

static bool isValidRegUseOf(const llvm::MachineOperand &MO, llvm::Register Reg,
                            const llvm::TargetRegisterInfo *TRI) {
  if (!isValidRegUse(MO))
    return false;
  return TRI->regsOverlap(MO.getReg(), Reg);
}

static bool isValidRegDef(const llvm::MachineOperand &MO) {
  return isValidReg(MO) && MO.isDef();
}

static bool isValidRegDefOf(const llvm::MachineOperand &MO, llvm::Register Reg,
                            const llvm::TargetRegisterInfo *TRI) {
  if (!isValidRegDef(MO))
    return false;
  return TRI->regsOverlap(MO.getReg(), Reg);
}

static bool isFIDef(const llvm::MachineInstr &MI, int FrameIndex,
                    const llvm::TargetInstrInfo *TII) {
  int DefFrameIndex = 0;
  int SrcFrameIndex = 0;
  if (TII->isStoreToStackSlot(MI, DefFrameIndex) ||
      TII->isStackSlotCopy(MI, DefFrameIndex, SrcFrameIndex))
    return DefFrameIndex == FrameIndex;
  return false;
}

void IPReachingDefInfo::enterBasicBlock(const PredicatedMachineBasicBlock &MBB) {
  unsigned MBBNumber = MBB.getGlobalNumber();
  assert(MBBNumber < MBBReachingDefs.numBlockIDs() &&
         "Unexpected basic block number.");
  MBBReachingDefs.startBasicBlock(MBBNumber, NumRegUnits);

  // Reset instruction counter in each basic block.
  CurInstr = 0;

  // Set up LiveRegs to represent registers entering MBB.
  // Default values are 'nothing happened a long time ago'.
  if (LiveRegs.empty())
    LiveRegs.assign(NumRegUnits, ReachingDefDefaultVal);

  // This is the entry block.
  if (MBB.preds_empty()) {
    const llvm::TargetRegisterInfo *TRI =
        MBB.getParent().getParent().getMF().getSubtarget().getRegisterInfo();
    for (const auto &LI : this->IPRegLiveness->getPredMBBLiveIns(MBB)) {
      for (llvm::MCRegUnit Unit : TRI->regunits(LI.PhysReg)) {
        // Treat function live-ins as if they were defined just before the first
        // instruction.  Usually, function arguments are set up immediately
        // before the call.
        if (LiveRegs[static_cast<unsigned>(Unit)] != -1) {
          LiveRegs[static_cast<unsigned>(Unit)] = -1;
          MBBReachingDefs.append(MBBNumber, Unit, -1);
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Pred MBB " << MBB.getGlobalNumber() << ": entry\n");
    return;
  }

  // Try to coalesce live-out registers from predecessors.
  for (const PredicatedMachineBasicBlock &Pred : MBB.predecessors()) {
    assert(Pred.getGlobalNumber() < MBBOutRegsInfos.size() &&
           "Should have pre-allocated MBBInfos for all MBBs");
    const LiveRegsDefInfo &Incoming = MBBOutRegsInfos[Pred.getGlobalNumber()];
    // Incoming is null if this is a backedge from a BB
    // we haven't processed yet
    if (Incoming.empty())
      continue;

    // Find the most recent reaching definition from a predecessor.
    for (unsigned Unit = 0; Unit != NumRegUnits; ++Unit)
      LiveRegs[Unit] = std::max(LiveRegs[Unit], Incoming[Unit]);
  }

  // Insert the most recent reaching definition we found.
  for (unsigned Unit = 0; Unit != NumRegUnits; ++Unit)
    if (LiveRegs[Unit] != ReachingDefDefaultVal)
      MBBReachingDefs.append(MBBNumber, static_cast<llvm::MCRegUnit>(Unit),
                             LiveRegs[Unit]);
}

void IPReachingDefInfo::leaveBasicBlock(const PredicatedMachineBasicBlock &MBB) {
  assert(!LiveRegs.empty() && "Must enter basic block first.");
  unsigned MBBNumber = MBB.getGlobalNumber();
  assert(MBBNumber < MBBOutRegsInfos.size() &&
         "Unexpected basic block number.");
  // Save register clearances at end of MBB - used by enterBasicBlock().
  MBBOutRegsInfos[MBBNumber] = LiveRegs;

  // While processing the basic block, we kept `Def` relative to the start
  // of the basic block for convenience. However, future use of this information
  // only cares about the clearance from the end of the block, so adjust
  // everything to be relative to the end of the basic block.
  for (int &OutLiveReg : MBBOutRegsInfos[MBBNumber])
    if (OutLiveReg != ReachingDefDefaultVal)
      OutLiveReg -= CurInstr;
  LiveRegs.clear();
}

void IPReachingDefInfo::processDefs(const llvm::MachineInstr &MI,
                                  unsigned int VectorMBBIdx) {
  assert(!MI.isDebugInstr() && "Won't process debug instructions");

  unsigned MBBNumber = VectorMBBIdx;
  assert(MBBNumber < MBBReachingDefs.numBlockIDs() &&
         "Unexpected basic block number.");

  const llvm::TargetRegisterInfo *TRI =
      MI.getParent()->getParent()->getSubtarget().getRegisterInfo();
  const llvm::TargetInstrInfo *TII =
      MI.getParent()->getParent()->getSubtarget().getInstrInfo();

  for (auto &MO : MI.operands()) {
    if (MO.isFI()) {
      int FrameIndex = MO.getIndex();
      if (!isFIDef(MI, FrameIndex, TII))
        continue;
      MBBFrameObjsReachingDefs[{MBBNumber, FrameIndex}].push_back(CurInstr);
    }
    if (!isValidRegDef(MO))
      continue;
    for (llvm::MCRegUnit Unit : TRI->regunits(MO.getReg().asMCReg())) {
      // This instruction explicitly defines the current reg unit.
      LLVM_DEBUG(llvm::dbgs()
                 << printRegUnit(Unit, TRI) << ":\t" << CurInstr << '\t' << MI);

      // How many instructions since this reg unit was last written?
      if (LiveRegs[static_cast<unsigned>(Unit)] != CurInstr) {
        LiveRegs[static_cast<unsigned>(Unit)] = CurInstr;
        MBBReachingDefs.append(MBBNumber, Unit, CurInstr);
      }
    }
  }
  InstIds[(&MI)] = CurInstr;
  ++CurInstr;
}

void IPReachingDefInfo::reprocessBasicBlock(
    const PredicatedMachineBasicBlock &MBB) {
  unsigned MBBNumber = MBB.getGlobalNumber();
  assert(MBBNumber < MBBReachingDefs.numBlockIDs() &&
         "Unexpected basic block number.");

  // Count number of non-debug instructions for end of block adjustment.
  auto NonDbgInsts = llvm::instructionsWithoutDebug(MBB.begin(), MBB.end());
  unsigned int NumInsts = std::distance(NonDbgInsts.begin(), NonDbgInsts.end());

  // When reprocessing a block, the only thing we need to do is check whether
  // there is now a more recent incoming reaching definition from a predecessor.
  for (const PredicatedMachineBasicBlock &Pred : MBB.predecessors()) {
    assert(Pred.getGlobalNumber() < MBBOutRegsInfos.size() &&
           "Should have pre-allocated MBBInfos for all MBBs");
    const LiveRegsDefInfo &Incoming = MBBOutRegsInfos[Pred.getGlobalNumber()];
    // Incoming may be empty for dead predecessors.
    if (Incoming.empty())
      continue;

    for (unsigned Unit = 0; Unit != NumRegUnits; ++Unit) {
      int Def = Incoming[Unit];
      if (Def == ReachingDefDefaultVal)
        continue;

      auto Defs =
          MBBReachingDefs.defs(MBBNumber, static_cast<llvm::MCRegUnit>(Unit));
      if (!Defs.empty() && Defs.front() < 0) {
        if (Defs.front() >= Def)
          continue;

        // Update existing reaching def from predecessor to a more recent one.
        MBBReachingDefs.replaceFront(MBBNumber,
                                     static_cast<llvm::MCRegUnit>(Unit), Def);
      } else {
        // Insert new reaching def from predecessor.
        MBBReachingDefs.prepend(MBBNumber, static_cast<llvm::MCRegUnit>(Unit),
                                Def);
      }

      // Update reaching def at end of BB. Keep in mind that these are
      // adjusted relative to the end of the basic block.
      if (MBBOutRegsInfos[MBBNumber][Unit] < Def - NumInsts)
        MBBOutRegsInfos[MBBNumber][Unit] = Def - NumInsts;
    }
  }
}

void IPReachingDefInfo::processBasicBlock(
    const IPPredicatedLoopTraversal::TraversedPredMBBInfo &TraversedMBB) {
  const PredicatedMachineBasicBlock *MBB = TraversedMBB.MBB;
  LLVM_DEBUG(llvm::dbgs() << "MBB " << MBB->getGlobalNumber()
                          << (!TraversedMBB.IsDone ? ": incomplete\n"
                                                   : ": all preds known\n"));

  if (!TraversedMBB.PrimaryPass) {
    // Reprocess MBB that is part of a loop.
    reprocessBasicBlock(*MBB);
    return;
  }
  enterBasicBlock(*MBB);
  for (const llvm::MachineInstr &MI :
       llvm::instructionsWithoutDebug(MBB->begin(), MBB->end()))
    processDefs(MI, MBB->getGlobalNumber());
  leaveBasicBlock(*MBB);
}

void IPReachingDefInfo::run(llvm::Module &TargetModule,
                          llvm::ModuleAnalysisManager &TargetMAM) {
  LLVM_DEBUG(llvm::dbgs() << "********** INTER-PROCEDURAL REACHING DEFINITION "
                             "ANALYSIS **********\n");
  const IPPredicatedCFG &IPPredCFG =
      TargetMAM.getResult<IPPredCFGAnalysis>(TargetModule).getVecCFG();
  const IPPredRegLiveness &IPRegLiveness =
      TargetMAM.getResult<IPVectorRegLivenessAnalysis>(TargetModule);
  init(IPPredCFG, IPRegLiveness);
  traverse();
}

void IPReachingDefInfo::print(llvm::raw_ostream &OS) {

  llvm::DenseMap<std::reference_wrapper<const llvm::MachineInstr>, int>
      InstToNumMap{};
  int Num = 0;
  for (const PredicatedMachineFunction &VecCFG : *IPVecCFG) {
    for (const llvm::MachineBasicBlock &MBB : VecCFG.getMF()) {
      for (const llvm::MachineInstr &MI : MBB) {
        InstToNumMap[MI] = Num;
        Num++;
      }
    }
  }

  for (const auto &VecCFG : *IPVecCFG) {
    const llvm::MachineFunction &MF = VecCFG.getMF();
    const llvm::TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    OS << "IP Vector RDA results for " << MF.getName() << "\n";

    llvm::SmallPtrSet<const llvm::MachineInstr *, 2> Defs;
    for (const llvm::MachineBasicBlock &MBB : MF) {
      for (const llvm::MachineInstr &MI : MBB) {
        for (const llvm::MachineOperand &MO : MI.operands()) {
          llvm::Register Reg;
          if (MO.isFI()) {
            int FrameIndex = MO.getIndex();
            Reg = llvm::Register::index2StackSlot(FrameIndex);
          } else if (MO.isReg()) {
            if (MO.isDef())
              continue;
            Reg = MO.getReg();
            if (!Reg.isValid())
              continue;
          } else
            continue;
          Defs.clear();
          getGlobalReachingDefs(MI, Reg, Defs);
          MO.print(OS, TRI);
          llvm::SmallVector<int, 0> Nums;
          for (const llvm::MachineInstr *Def : Defs)
            Nums.push_back(InstToNumMap[*Def]);
          llvm::sort(Nums);
          OS << ":{ ";
          for (const int N : Nums)
            OS << N << " ";
          OS << "}\n";
        }
        OS << InstToNumMap[MI] << ": " << MI << "\n";
      }
    }
  }
}

void IPReachingDefInfo::releaseMemory() {
  // Clear the internal vectors.
  MBBOutRegsInfos.clear();
  MBBReachingDefs.clear();
  MBBFrameObjsReachingDefs.clear();
  InstIds.clear();
  LiveRegs.clear();
}

void IPReachingDefInfo::init(const IPPredicatedCFG &IPPredCFG,
                           const IPPredRegLiveness &IPRegLiveness) {
  this->IPVecCFG = &IPPredCFG;
  this->IPRegLiveness = &IPRegLiveness;
  NumRegUnits = 0;
  for (const auto &VecCFG : IPPredCFG) {
    const llvm::TargetRegisterInfo *TRI =
        VecCFG.getMF().getSubtarget().getRegisterInfo();
    NumRegUnits = std::max(TRI->getNumRegUnits(), NumRegUnits);
  }
  MBBReachingDefs.init(IPPredCFG.getNumVecMBBs());
  // Initialize the MBBOutRegsInfos
  MBBOutRegsInfos.resize(IPPredCFG.getNumVecMBBs());
  luthier::IPPredicatedLoopTraversal Traversal;
  TraversedMBBOrder = Traversal.traverse(IPPredCFG);
}

void IPReachingDefInfo::traverse() {
  // Traverse the basic blocks.
  for (IPPredicatedLoopTraversal::TraversedPredMBBInfo TraversedMBB :
       TraversedMBBOrder)
    processBasicBlock(TraversedMBB);
#ifndef NDEBUG
  // Make sure reaching defs are sorted and unique.
  for (unsigned MBBNumber = 0, NumBlockIDs = IPVecCFG->getNumVecMBBs();
       MBBNumber != NumBlockIDs; ++MBBNumber) {
    for (unsigned Unit = 0; Unit != NumRegUnits; ++Unit) {
      int LastDef = ReachingDefDefaultVal;
      for (int Def : MBBReachingDefs.defs(MBBNumber,
                                          static_cast<llvm::MCRegUnit>(Unit))) {
        assert(Def > LastDef && "Defs must be sorted and unique");
        LastDef = Def;
      }
    }
  }
#endif
}

int IPReachingDefInfo::getReachingDef(const llvm::MachineInstr &MI,
                                    llvm::Register Reg) const {
  assert(InstIds.count(&MI) && "Unexpected machine instruction.");
  int InstId = InstIds.lookup(&MI);
  int DefRes = ReachingDefDefaultVal;
  const PredicatedMachineBasicBlock &PredMBB = IPVecCFG->getPredMBB(MI);
  unsigned MBBNumber = PredMBB.getGlobalNumber();
  assert(MBBNumber < MBBReachingDefs.numBlockIDs() &&
         "Unexpected basic block number.");
  int LatestDef = ReachingDefDefaultVal;

  if (Reg.isStack()) {
    // Check that there was a reaching def.
    int FrameIndex = Reg.stackSlotIndex();
    auto Lookup = MBBFrameObjsReachingDefs.find({MBBNumber, FrameIndex});
    if (Lookup == MBBFrameObjsReachingDefs.end())
      return LatestDef;
    auto &Defs = Lookup->second;
    for (int Def : Defs) {
      if (Def >= InstId)
        break;
      DefRes = Def;
    }
    LatestDef = std::max(LatestDef, DefRes);
    return LatestDef;
  }

  const llvm::TargetRegisterInfo *TRI =
      MI.getParent()->getParent()->getSubtarget().getRegisterInfo();

  for (llvm::MCRegUnit Unit : TRI->regunits(Reg)) {
    for (int Def : MBBReachingDefs.defs(MBBNumber, Unit)) {
      if (Def >= InstId)
        break;
      DefRes = Def;
    }
    LatestDef = std::max(LatestDef, DefRes);
  }
  return LatestDef;
}

const llvm::MachineInstr *
IPReachingDefInfo::getReachingLocalMIDef(const llvm::MachineInstr &MI,
                                       llvm::Register Reg) const {
  return hasLocalDefBefore(MI, Reg)
             ? getInstFromId(IPVecCFG->getPredMBB(MI), getReachingDef(MI, Reg))
             : nullptr;
}

bool IPReachingDefInfo::hasSameReachingDef(const llvm::MachineInstr &A,
                                         const llvm::MachineInstr &B,
                                         llvm::Register Reg) const {
  const PredicatedMachineBasicBlock &ParentA = IPVecCFG->getPredMBB(A);
  const PredicatedMachineBasicBlock &ParentB = IPVecCFG->getPredMBB(B);
  if (&ParentA != &ParentB)
    return false;

  return getReachingDef(A, Reg) == getReachingDef(B, Reg);
}

const llvm::MachineInstr *
IPReachingDefInfo::getInstFromId(const PredicatedMachineBasicBlock &MBB,
                               int InstId) const {
  assert(static_cast<size_t>(MBB.getGlobalNumber()) <
             MBBReachingDefs.numBlockIDs() &&
         "Unexpected basic block number.");
  assert(InstId < static_cast<int>(MBB.size()) && "Unexpected instruction id.");

  if (InstId < 0)
    return nullptr;

  for (auto &MI : MBB) {
    if (auto F = InstIds.find(&MI); F != InstIds.end() && F->second == InstId)
      return &MI;
  }

  return nullptr;
}

int IPReachingDefInfo::getClearance(const llvm::MachineInstr &MI,
                                  llvm::Register Reg) const {
  assert(InstIds.count(&MI) && "Unexpected machine instruction.");
  return InstIds.lookup(&MI) - getReachingDef(MI, Reg);
}

bool IPReachingDefInfo::hasLocalDefBefore(const llvm::MachineInstr &MI,
                                        llvm::Register Reg) const {
  return getReachingDef(MI, Reg) >= 0;
}

void IPReachingDefInfo::getReachingLocalUses(const llvm::MachineInstr &Def,
                                           llvm::Register Reg,
                                           InstSet &Uses) const {
  const PredicatedMachineBasicBlock &MBB = IPVecCFG->getPredMBB(Def);
  auto MI = PredicatedMachineBasicBlock::const_iterator(Def);
  const llvm::TargetRegisterInfo *TRI =
      MBB.getParent().getParent().getMF().getSubtarget().getRegisterInfo();
  while (++MI != MBB.end()) {

    // If/when we find a new reaching def, we know that there's no more uses
    // of 'Def'.
    if (auto LocalDef = getReachingLocalMIDef(*MI, Reg);
        LocalDef && LocalDef != &Def)
      return;

    for (auto &MO : MI->operands()) {
      if (!isValidRegUseOf(MO, Reg, TRI))
        continue;

      Uses.insert(&*MI);
      if (MO.isKill())
        return;
    }
  }
}

bool IPReachingDefInfo::getLiveInUses(const PredicatedMachineBasicBlock &MBB,
                                    llvm::Register Reg, InstSet &Uses) const {
  const llvm::TargetRegisterInfo *TRI =
      MBB.getParent().getParent().getMF().getSubtarget().getRegisterInfo();
  for (const llvm::MachineInstr &MI :
       llvm::instructionsWithoutDebug(MBB.begin(), MBB.end())) {
    for (auto &MO : MI.operands()) {
      if (!isValidRegUseOf(MO, Reg, TRI))
        continue;
      if (getReachingDef(MI, Reg) >= 0)
        return false;
      Uses.insert(&MI);
    }
  }
  auto Last = MBB.getLastNonDebugInstr();
  if (Last == MBB.end())
    return true;
  return isReachingDefLiveOut(*Last, Reg);
}

void IPReachingDefInfo::getGlobalUses(const llvm::MachineInstr &MI,
                                    llvm::Register Reg, InstSet &Uses) const {
  const PredicatedMachineBasicBlock &ParentMBB = IPVecCFG->getPredMBB(MI);

  // Collect the uses that each def touches within the block.
  getReachingLocalUses(MI, Reg, Uses);

  // Handle live-out values.
  if (auto *LiveOut = getLocalLiveOutMIDef(ParentMBB, Reg)) {
    if (LiveOut != &MI)
      return;

    llvm::SmallVector<std::reference_wrapper<const PredicatedMachineBasicBlock>,
                      4>
        ToVisit(ParentMBB.successors());
    llvm::SmallPtrSet<const PredicatedMachineBasicBlock *, 4> Visited;
    while (!ToVisit.empty()) {
      const PredicatedMachineBasicBlock &MBB = ToVisit.pop_back_val();
      if (Visited.count(&MBB) || !IPRegLiveness->isLiveIn(MBB, Reg))
        continue;
      if (getLiveInUses(MBB, Reg, Uses))
        llvm::append_range(ToVisit, MBB.successors());
      Visited.insert(&MBB);
    }
  }
}

void IPReachingDefInfo::getGlobalReachingDefs(const llvm::MachineInstr &MI,
                                            llvm::Register Reg,
                                            InstSet &Defs) const {
  LLVM_DEBUG(
      llvm::dbgs()
      << "Starting to find global reaching defs for reg "
      << llvm::printReg(
             Reg, MI.getParent()->getParent()->getSubtarget().getRegisterInfo())
      << " in instruction " << MI << ".\n");
  if (auto *Def = getUniqueReachingMIDef(MI, Reg)) {
    LLVM_DEBUG(llvm::dbgs() << "Found def at :" << MI << "\n";);
    Defs.insert(Def);
    return;
  }
  LLVM_DEBUG(
      llvm::dbgs()
          << "Didn't find any defs for reg "
          << llvm::printReg(
                 Reg,
                 MI.getParent()->getParent()->getSubtarget().getRegisterInfo())
          << ". Searching in preds.\n";);
  for (auto &MBB : IPVecCFG->getPredMBB(MI).predecessors())
    getLiveOuts(MBB, Reg, Defs);
}

void IPReachingDefInfo::getLiveOuts(const PredicatedMachineBasicBlock &MBB,
                                  llvm::Register Reg, InstSet &Defs) const {
  llvm::SmallPtrSet<const PredicatedMachineBasicBlock *, 2> VisitedBBs;
  getLiveOuts(MBB, Reg, Defs, VisitedBBs);
}

void IPReachingDefInfo::getLiveOuts(const PredicatedMachineBasicBlock &MBB,
                                  llvm::Register Reg, InstSet &Defs,
                                  BlockSet &VisitedBBs) const {
  const llvm::TargetRegisterInfo *TRI =
      MBB.getParent().getParent().getMF().getSubtarget().getRegisterInfo();
  LLVM_DEBUG(llvm::dbgs() << "Looking at MBB: " << MBB.getName()
                          << " to find definition of "
                          << llvm::printReg(Reg, TRI) << ".\n";);

  if (VisitedBBs.count(&MBB)) {
    LLVM_DEBUG(llvm::dbgs() << "Already visited\n";);
    return;
  }

  VisitedBBs.insert(&MBB);
  llvm::LiveRegUnits LR(*TRI);
  IPRegLiveness->addLiveOuts(MBB, LR);
  if (Reg.isPhysical() && LR.available(Reg)) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Reg is available and not defined in this MBB\n";);
    return;
  }

  if (auto *Def = getLocalLiveOutMIDef(MBB, Reg)) {
    LLVM_DEBUG(llvm::dbgs() << "Found def at " << *Def << "\n";);
    Defs.insert(Def);
  } else {
    LLVM_DEBUG(
        llvm::dbgs()
            << "Didn't find anything in the current block; searching preds\n";);
    for (auto &Pred : MBB.predecessors()) {
      getLiveOuts(Pred, Reg, Defs, VisitedBBs);
    }
  }
}

const llvm::MachineInstr *
IPReachingDefInfo::getUniqueReachingMIDef(const llvm::MachineInstr &MI,
                                        llvm::Register Reg) const {
  // If there's a local def before MI, return it.
  const llvm::MachineInstr *LocalDef = getReachingLocalMIDef(MI, Reg);
  if (LocalDef && InstIds.lookup(LocalDef) < InstIds.lookup(&MI))
    return LocalDef;

  llvm::SmallPtrSet<const llvm::MachineInstr *, 2> Incoming;
  const PredicatedMachineBasicBlock &Parent = IPVecCFG->getPredMBB(MI);
  for (auto &Pred : Parent.predecessors())
    getLiveOuts(Pred, Reg, Incoming);

  // Check that we have a single incoming value and that it does not
  // come from the same block as MI - since it would mean that the def
  // is executed after MI.
  if (Incoming.size() == 1 &&
      &IPVecCFG->getPredMBB(**Incoming.begin()) != &Parent)
    return *Incoming.begin();
  return nullptr;
}

const llvm::MachineInstr *
IPReachingDefInfo::getMIOperand(const llvm::MachineInstr &MI,
                              unsigned Idx) const {
  assert(MI.getOperand(Idx).isReg() && "Expected register operand");
  return getUniqueReachingMIDef(MI, MI.getOperand(Idx).getReg());
}

const llvm::MachineInstr *
IPReachingDefInfo::getMIOperand(const llvm::MachineInstr &MI,
                              const llvm::MachineOperand &MO) const {
  assert(MO.isReg() && "Expected register operand");
  return getUniqueReachingMIDef(MI, MO.getReg());
}

bool IPReachingDefInfo::isRegUsedAfter(const llvm::MachineInstr &MI,
                                     llvm::Register Reg) const {
  const PredicatedMachineBasicBlock &MBB = IPVecCFG->getPredMBB(MI);
  const llvm::TargetRegisterInfo *TRI =
      MBB.getParent().getParent().getMF().getSubtarget().getRegisterInfo();
  llvm::LiveRegUnits LR(*TRI);
  IPRegLiveness->addLiveOuts(MBB, LR);

  // Yes if the register is live out of the basic block.
  if (!LR.available(Reg))
    return true;

  // Walk backwards through the block to see if the register is live at some
  // point.
  for (const llvm::MachineInstr &Last :
       llvm::instructionsWithoutDebug(MBB.begin(), MBB.end())) {
    LR.stepBackward(Last);
    if (!LR.available(Reg))
      return InstIds.lookup(&Last) > InstIds.lookup(&MI);
  }
  return false;
}

bool IPReachingDefInfo::isRegDefinedAfter(const llvm::MachineInstr &MI,
                                        llvm::Register Reg) const {
  const PredicatedMachineBasicBlock &MBB = IPVecCFG->getPredMBB(MI);
  auto Last = MBB.getLastNonDebugInstr();
  if (Last != MBB.end() &&
      getReachingDef(MI, Reg) != getReachingDef(*Last, Reg))
    return true;

  if (auto *Def = getLocalLiveOutMIDef(MBB, Reg))
    return Def == getReachingLocalMIDef(MI, Reg);

  return false;
}

bool IPReachingDefInfo::isReachingDefLiveOut(const llvm::MachineInstr &MI,
                                           llvm::Register Reg) const {
  const PredicatedMachineBasicBlock &MBB = IPVecCFG->getPredMBB(MI);
  const llvm::TargetRegisterInfo *TRI =
      MBB.getParent().getParent().getMF().getSubtarget().getRegisterInfo();
  llvm::LiveRegUnits LR(*TRI);
  IPRegLiveness->addLiveOuts(MBB, LR);
  if (Reg.isPhysical() && LR.available(Reg))
    return false;

  auto Last = MBB.getLastNonDebugInstr();
  int Def = getReachingDef(MI, Reg);
  if (Last != MBB.end() && getReachingDef(*Last, Reg) != Def)
    return false;

  // Finally check that the last instruction doesn't redefine the register.
  for (auto &MO : Last->operands())
    if (isValidRegDefOf(MO, Reg, TRI))
      return false;

  return true;
}

const llvm::MachineInstr *
IPReachingDefInfo::getLocalLiveOutMIDef(const PredicatedMachineBasicBlock &MBB,
                                      llvm::Register Reg) const {
  const llvm::TargetRegisterInfo *TRI =
      MBB.getParent().getParent().getMF().getSubtarget().getRegisterInfo();
  const llvm::TargetInstrInfo *TII =
      MBB.getParent().getParent().getMF().getSubtarget().getInstrInfo();
  llvm::LiveRegUnits LRU(*TRI);
  IPRegLiveness->addLiveOuts(MBB, LRU);
  if (Reg.isPhysical() && LRU.available(Reg))
    return nullptr;

  auto Last = MBB.getLastNonDebugInstr();
  if (Last == MBB.end())
    return nullptr;

  if (Reg.isStack()) {
    int FrameIndex = Reg.stackSlotIndex();
    if (isFIDef(*Last, FrameIndex, TII))
      return &*Last;
  }

  int Def = getReachingDef(*Last, Reg);

  for (auto &MO : Last->operands())
    if (isValidRegDefOf(MO, Reg, TRI))
      return &*Last;

  return Def < 0 ? nullptr : getInstFromId(MBB, Def);
}

static bool mayHaveSideEffects(const llvm::MachineInstr &MI) {
  return MI.mayLoadOrStore() || MI.mayRaiseFPException() ||
         MI.hasUnmodeledSideEffects() || MI.isTerminator() || MI.isCall() ||
         MI.isBarrier() || MI.isBranch() || MI.isReturn();
}

// Can we safely move 'From' to just before 'To'? To satisfy this, 'From' must
// not define a register that is used by any instructions, after and including,
// 'To'. These instructions also must not redefine any of Froms operands.
template <typename Iterator>
bool IPReachingDefInfo::isSafeToMove(const llvm::MachineInstr &From,
                                   const llvm::MachineInstr &To) const {
  if (&IPVecCFG->getPredMBB(From) != &IPVecCFG->getPredMBB(To) || &From == &To)
    return false;

  llvm::SmallSet<llvm::Register, 2> Defs;
  // First check that From would compute the same value if moved.
  for (auto &MO : From.operands()) {
    if (!isValidReg(MO))
      continue;
    if (MO.isDef())
      Defs.insert(MO.getReg());
    else if (!hasSameReachingDef(From, To, MO.getReg()))
      return false;
  }

  // Now walk checking that the rest of the instructions will compute the same
  // value and that we're not overwriting anything. Don't move the instruction
  // past any memory, control-flow or other ambiguous instructions.
  for (auto I = ++Iterator(From.getIterator()), E = Iterator(To.getIterator());
       I != E; ++I) {
    if (mayHaveSideEffects(*I))
      return false;
    for (auto &MO : I->operands())
      if (MO.isReg() && MO.getReg() && Defs.count(MO.getReg()))
        return false;
  }
  return true;
}

bool IPReachingDefInfo::isSafeToMoveForwards(const llvm::MachineInstr &From,
                                           const llvm::MachineInstr &To) const {
  using Iterator = PredicatedMachineBasicBlock::const_iterator;
  // Walk forwards until we find the instruction.
  for (auto I = Iterator(From), E = IPVecCFG->getPredMBB(From).end(); I != E;
       ++I)
    if (&*I == &To)
      return isSafeToMove<Iterator>(From, To);
  return false;
}

bool IPReachingDefInfo::isSafeToMoveBackwards(
    const llvm::MachineInstr &From, const llvm::MachineInstr &To) const {
  using Iterator = PredicatedMachineBasicBlock::const_reverse_iterator;
  // Walk backwards until we find the instruction.
  for (auto I = Iterator(From.getIterator()),
            E = IPVecCFG->getPredMBB(From).rend();
       I != E; ++I)
    if (&*I == &To)
      return isSafeToMove<Iterator>(From, To);
  return false;
}

bool IPReachingDefInfo::isSafeToRemove(const llvm::MachineInstr &MI,
                                     InstSet &ToRemove) const {
  llvm::SmallPtrSet<const llvm::MachineInstr *, 1> Ignore;
  llvm::SmallPtrSet<const llvm::MachineInstr *, 2> Visited;
  return isSafeToRemove(MI, Visited, ToRemove, Ignore);
}

bool IPReachingDefInfo::isSafeToRemove(const llvm::MachineInstr &MI,
                                     InstSet &ToRemove, InstSet &Ignore) const {
  llvm::SmallPtrSet<const llvm::MachineInstr *, 2> Visited;
  return isSafeToRemove(MI, Visited, ToRemove, Ignore);
}

bool IPReachingDefInfo::isSafeToRemove(const llvm::MachineInstr &MI,
                                     InstSet &Visited, InstSet &ToRemove,
                                     InstSet &Ignore) const {
  if (Visited.count(&MI) || Ignore.count(&MI))
    return true;
  else if (mayHaveSideEffects(MI)) {
    // Unless told to ignore the instruction, don't remove anything which has
    // side effects.
    return false;
  }

  Visited.insert(&MI);
  for (auto &MO : MI.operands()) {
    if (!isValidRegDef(MO))
      continue;

    llvm::SmallPtrSet<const llvm::MachineInstr *, 4> Uses;
    getGlobalUses(MI, MO.getReg(), Uses);

    for (auto *I : Uses) {
      if (Ignore.count(I) || ToRemove.count(I))
        continue;
      if (!isSafeToRemove(*I, Visited, ToRemove, Ignore))
        return false;
    }
  }
  ToRemove.insert(&MI);
  return true;
}

void IPReachingDefInfo::collectKilledOperands(const llvm::MachineInstr &MI,
                                            InstSet &Dead) const {
  Dead.insert(&MI);
  auto IsDead = [this, &Dead](const llvm::MachineInstr *Def,
                              llvm::Register Reg) {
    if (mayHaveSideEffects(*Def))
      return false;

    unsigned LiveDefs = 0;
    for (auto &MO : Def->operands()) {
      if (!isValidRegDef(MO))
        continue;
      if (!MO.isDead())
        ++LiveDefs;
    }

    if (LiveDefs > 1)
      return false;

    llvm::SmallPtrSet<const llvm::MachineInstr *, 4> Uses;
    getGlobalUses(*Def, Reg, Uses);
    return llvm::set_is_subset(Uses, Dead);
  };

  for (auto &MO : MI.operands()) {
    if (!isValidRegUse(MO))
      continue;
    if (const llvm::MachineInstr *Def = getMIOperand(MI, MO))
      if (IsDead(Def, MO.getReg()))
        collectKilledOperands(*Def, Dead);
  }
}

bool IPReachingDefInfo::isSafeToDefRegAt(const llvm::MachineInstr &MI,
                                       llvm::Register Reg) const {
  llvm::SmallPtrSet<const llvm::MachineInstr *, 1> Ignore;
  return isSafeToDefRegAt(MI, Reg, Ignore);
}

bool IPReachingDefInfo::isSafeToDefRegAt(const llvm::MachineInstr &MI,
                                       llvm::Register Reg,
                                       InstSet &Ignore) const {
  const llvm::TargetRegisterInfo *TRI =
      MI.getParent()->getParent()->getSubtarget().getRegisterInfo();
  // Check for any uses of the register after MI.
  if (isRegUsedAfter(MI, Reg)) {
    if (auto *Def = getReachingLocalMIDef(MI, Reg)) {
      llvm::SmallPtrSet<const llvm::MachineInstr *, 2> Uses;
      getGlobalUses(*Def, Reg, Uses);
      if (!llvm::set_is_subset(Uses, Ignore))
        return false;
    } else
      return false;
  }

  const PredicatedMachineBasicBlock &MBB = IPVecCFG->getPredMBB(MI);
  // Check for any defs after MI.
  if (isRegDefinedAfter(MI, Reg)) {
    auto I = llvm::MachineBasicBlock::const_iterator(MI);
    for (auto E = MBB.end(); I != E; ++I) {
      if (Ignore.count(&*I))
        continue;
      for (auto &MO : I->operands())
        if (isValidRegDefOf(MO, Reg, TRI))
          return false;
    }
  }
  return true;
}
} // namespace luthier