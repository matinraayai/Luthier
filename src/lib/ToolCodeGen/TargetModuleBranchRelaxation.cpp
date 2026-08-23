//===-- TargetModuleBranchRelaxation.cpp ----------------------------------===//
// Copyright @ Northeastern University Computer Architecture Lab
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
/// \file
/// Target-module branch relaxer — fork of
/// \c llvm/lib/CodeGen/BranchRelaxation.cpp. Top-level
/// \c run + the offset-tracking machinery (\c scanFunction,
/// \c computeBlockSize, \c adjustBlockOffsets, \c isBlockInRange,
/// \c splitBlockBeforeInstr, \c fixupConditionalBranch,
/// \c relaxBranchInstructions) are transposed verbatim; the sole substantive
/// change is in \c fixupUnconditionalBranch, which calls
/// \c emitLongBranch instead of \c TII->insertIndirectBranch.
/// That helper mirrors \c SIInstrInfo::insertIndirectBranch's body but
/// scavenges via \c TargetModuleScavenger (which sees the \c ReservedRegs
/// set and the SVA-lane \c SpillSink the caller installed).
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/TargetModuleBranchRelaxation.h"

#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/IPPredicatedLivenessPass.h"
#include "luthier/ToolCodeGen/PredicatedMachineBasicBlock.h"

#include <AMDGPU.h>
#include <GCNSubtarget.h>
#include <SIInstrInfo.h>
#include <SIMachineFunctionInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/DebugLoc.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCExpr.h>
#include <llvm/MC/MCSymbol.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-branch-relaxation"

namespace luthier {

namespace {

/// AMDGPU-specific long-branch emission. Forked from
/// \c SIInstrInfo::insertIndirectBranch (the pre-gfx12 path, which is
/// the only path Luthier currently needs — gfx12+ would use
/// \c S_ADD_PC_I64 and not require scavenging at all). Sole change vs.
/// stock: scavenges via \p RS (\c TargetModuleScavenger) instead of
/// the AMDGPU TII-supplied stock \c RegScavenger, so the SVA storage
/// reg is excluded and the SVA-lane spill sink fires when no free
/// SReg_64 is available.
void emitLongBranch(llvm::MachineBasicBlock &MBB,
                    llvm::MachineBasicBlock &DestBB,
                    llvm::MachineBasicBlock &RestoreBB,
                    const llvm::DebugLoc &DL, int64_t BrOffset,
                    TargetModuleScavenger &RS) {
  assert(MBB.empty() && "trampoline MBB must start empty");
  assert(MBB.pred_size() == 1 && "trampoline MBB must have exactly one pred");
  assert(RestoreBB.empty() && "restore MBB must start empty");

  auto *MF = MBB.getParent();
  auto &MRI = MF->getRegInfo();
  const auto &ST = MF->getSubtarget<llvm::GCNSubtarget>();
  const auto *TII = ST.getInstrInfo();
  const auto *MFI = MF->getInfo<llvm::SIMachineFunctionInfo>();
  auto &MCCtx = MF->getContext();
  auto I = MBB.end();

  // FIXME (carried from stock SIInstrInfo): RegScavenger doesn't like
  // running on an empty MBB, so we materialize PCReg as a vreg first
  // and patch it up after scavenging.
  llvm::Register PCReg =
      MRI.createVirtualRegister(&llvm::AMDGPU::SReg_64RegClass);

  const bool FlushSGPRWrites = (ST.isWave64() && ST.hasVALUMaskWriteHazard()) ||
                               ST.hasVALUReadSGPRHazard();
  auto ApplyHazardWorkarounds = [&]() {
    if (FlushSGPRWrites)
      llvm::BuildMI(MBB, I, DL, TII->get(llvm::AMDGPU::S_WAITCNT_DEPCTR))
          .addImm(llvm::AMDGPU::DepCtr::encodeFieldSaSdst(0, ST));
  };

  // Build the S_GETPC / S_ADD_U32 / S_ADDC_U32 / S_SETPC_B64 sequence,
  // tagging the points the AsmPrinter resolves the offset against.
  llvm::MachineInstr *GetPC =
      llvm::BuildMI(MBB, I, DL, TII->get(llvm::AMDGPU::S_GETPC_B64), PCReg);
  ApplyHazardWorkarounds();

  auto *PostGetPCLabel =
      MCCtx.createTempSymbol("luthier_post_getpc", /*AlwaysAddSuffix=*/true);
  GetPC->setPostInstrSymbol(*MF, PostGetPCLabel);

  auto *OffsetLo =
      MCCtx.createTempSymbol("luthier_offset_lo", /*AlwaysAddSuffix=*/true);
  auto *OffsetHi =
      MCCtx.createTempSymbol("luthier_offset_hi", /*AlwaysAddSuffix=*/true);
  llvm::BuildMI(MBB, I, DL, TII->get(llvm::AMDGPU::S_ADD_U32))
      .addReg(PCReg, llvm::RegState::Define, llvm::AMDGPU::sub0)
      .addReg(PCReg, llvm::RegState::NoFlags, llvm::AMDGPU::sub0)
      .addSym(OffsetLo, llvm::SIInstrInfo::MO_FAR_BRANCH_OFFSET);
  llvm::BuildMI(MBB, I, DL, TII->get(llvm::AMDGPU::S_ADDC_U32))
      .addReg(PCReg, llvm::RegState::Define, llvm::AMDGPU::sub1)
      .addReg(PCReg, llvm::RegState::NoFlags, llvm::AMDGPU::sub1)
      .addSym(OffsetHi, llvm::SIInstrInfo::MO_FAR_BRANCH_OFFSET);
  ApplyHazardWorkarounds();

  (void)llvm::BuildMI(&MBB, DL, TII->get(llvm::AMDGPU::S_SETPC_B64))
      .addReg(PCReg);

  // Scavenge an SReg_64 to replace PCReg.
  llvm::Register LongBranchReservedReg = MFI->getLongBranchReservedReg();
  llvm::Register Scav;
  bool ScavengerSpilled = false;
  if (LongBranchReservedReg) {
    RS.enterBasicBlock(MBB);
    Scav = LongBranchReservedReg;
  } else {
    RS.enterBasicBlockEnd(MBB);
    Scav = RS.scavengeRegisterBackwards(
        llvm::AMDGPU::SReg_64RegClass, llvm::MachineBasicBlock::iterator(GetPC),
        /*RestoreAfter=*/false, /*SPAdj=*/0, /*AllowSpill=*/false);
    if (!Scav) {
      // No globally-free reg. Invoke the SVA-lane sink with explicit
      // Spill (in MBB before GetPC) and Reload (in RestoreBB after
      // the long jump lands) insertion points. Pick a fixed pair
      // (SGPR0_SGPR1) — the sink emits explicit save/restore so any
      // pair works.
      Scav = llvm::AMDGPU::SGPR0_SGPR1;
      ScavengerSpilled = true;
      if (!RS.invokeSVASpillSink(MBB, GetPC->getIterator(), RestoreBB,
                                 RestoreBB.begin(), Scav,
                                 llvm::AMDGPU::SReg_64RegClass)) {
        llvm::report_fatal_error(
            "TargetModuleBranchRelaxation: no free SReg_64 found and SVA-lane "
            "spill sink unavailable; cannot relax long branch",
            /*GenCrashDiag=*/false);
      }
    }
  }

  RS.setRegUsed(Scav);
  MRI.replaceRegWith(PCReg, Scav);
  MRI.clearVirtRegs();

  auto *DestLabel =
      !ScavengerSpilled ? DestBB.getSymbol() : RestoreBB.getSymbol();
  auto *Offset = llvm::MCBinaryExpr::createSub(
      llvm::MCSymbolRefExpr::create(DestLabel, MCCtx),
      llvm::MCSymbolRefExpr::create(PostGetPCLabel, MCCtx), MCCtx);
  auto *Mask = llvm::MCConstantExpr::create(0xFFFFFFFFULL, MCCtx);
  OffsetLo->setVariableValue(
      llvm::MCBinaryExpr::createAnd(Offset, Mask, MCCtx));
  auto *ShAmt = llvm::MCConstantExpr::create(32, MCCtx);
  OffsetHi->setVariableValue(
      llvm::MCBinaryExpr::createAShr(Offset, ShAmt, MCCtx));
  (void)BrOffset;
}

/// Worker class — fork of \c llvm::BranchRelaxation (anonymous namespace
/// class) with one substantive change: \c fixupUnconditionalBranch's
/// long-branch emission calls \c emitLongBranch with the target
/// module reg scavenger.
class TargetModuleBranchRelaxationWorker {
  struct BasicBlockInfo {
    unsigned Offset = 0;
    unsigned Size = 0;
    BasicBlockInfo() = default;
    unsigned postOffset(const llvm::MachineBasicBlock &MBB) const {
      const unsigned PO = Offset + Size;
      const llvm::Align Alignment = MBB.getAlignment();
      const llvm::Align ParentAlign = MBB.getParent()->getAlignment();
      if (Alignment <= ParentAlign)
        return llvm::alignTo(PO, Alignment);
      return llvm::alignTo(PO, Alignment) + Alignment.value() -
             ParentAlign.value();
    }
  };

  llvm::SmallVector<BasicBlockInfo, 16> BlockInfo;
  llvm::MachineBasicBlock *TrampolineInsertionPoint = nullptr;
  llvm::SmallDenseSet<
      std::pair<llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>>
      RelaxedUnconditionals;
  TargetModuleScavenger RS;
  llvm::LivePhysRegs LiveRegs;

  llvm::MachineFunction *MF = nullptr;
  const llvm::TargetRegisterInfo *TRI = nullptr;
  const llvm::TargetInstrInfo *TII = nullptr;
  const llvm::TargetMachine *TM = nullptr;

  bool relaxBranchInstructions();
  void scanFunction();
  llvm::MachineBasicBlock *
  createNewBlockAfter(llvm::MachineBasicBlock &OrigMBB);
  llvm::MachineBasicBlock *createNewBlockAfter(llvm::MachineBasicBlock &OrigMBB,
                                               const llvm::BasicBlock *BB);
  llvm::MachineBasicBlock *
  splitBlockBeforeInstr(llvm::MachineInstr &MI,
                        llvm::MachineBasicBlock *DestBB);
  void adjustBlockOffsets(llvm::MachineBasicBlock &Start);
  void adjustBlockOffsets(llvm::MachineBasicBlock &Start,
                          llvm::MachineFunction::iterator End);
  bool isBlockInRange(const llvm::MachineInstr &MI,
                      const llvm::MachineBasicBlock &BB) const;
  bool fixupConditionalBranch(llvm::MachineInstr &MI);
  bool fixupUnconditionalBranch(llvm::MachineInstr &MI);
  uint64_t computeBlockSize(const llvm::MachineBasicBlock &MBB) const;
  unsigned getInstrOffset(const llvm::MachineInstr &MI) const;

  const IPPredicatedCFG &IPCFG;
  const IPPredicatedLiveness &IPLiveness;

public:
  TargetModuleBranchRelaxationWorker(
      const IPPredicatedCFG &IPCFG, const IPPredicatedLiveness &IPLiveness,
      const llvm::DenseSet<llvm::MCPhysReg> &ReservedRegs,
      TargetModuleScavenger::SVASpillCallback SpillSink)
      : IPCFG(IPCFG), IPLiveness(IPLiveness) {
    RS.setReservedRegs(ReservedRegs);
    if (SpillSink)
      RS.setSVASpillCallback(std::move(SpillSink));
  }

  bool run(llvm::MachineFunction &MF);
};

void TargetModuleBranchRelaxationWorker::scanFunction() {
  BlockInfo.clear();
  BlockInfo.resize(MF->getNumBlockIDs());
  TrampolineInsertionPoint = nullptr;
  RelaxedUnconditionals.clear();
  for (auto &MBB : *MF) {
    BlockInfo[MBB.getNumber()].Size = computeBlockSize(MBB);
    if (MBB.getSectionID() != llvm::MBBSectionID::ColdSectionID)
      TrampolineInsertionPoint = &MBB;
  }
  adjustBlockOffsets(*MF->begin());
}

uint64_t TargetModuleBranchRelaxationWorker::computeBlockSize(
    const llvm::MachineBasicBlock &MBB) const {
  uint64_t Size = 0;
  for (const auto &MI : MBB)
    Size += TII->getInstSizeInBytes(MI);
  return Size;
}

unsigned TargetModuleBranchRelaxationWorker::getInstrOffset(
    const llvm::MachineInstr &MI) const {
  const auto *MBB = MI.getParent();
  unsigned Offset = BlockInfo[MBB->getNumber()].Offset;
  for (auto I = MBB->begin(); &*I != &MI; ++I)
    Offset += TII->getInstSizeInBytes(*I);
  return Offset;
}

void TargetModuleBranchRelaxationWorker::adjustBlockOffsets(
    llvm::MachineBasicBlock &Start) {
  adjustBlockOffsets(Start, MF->end());
}

void TargetModuleBranchRelaxationWorker::adjustBlockOffsets(
    llvm::MachineBasicBlock &Start, llvm::MachineFunction::iterator End) {
  unsigned PrevNum = Start.getNumber();
  for (auto &MBB : llvm::make_range(
           std::next(llvm::MachineFunction::iterator(Start)), End)) {
    unsigned Num = MBB.getNumber();
    BlockInfo[Num].Offset = BlockInfo[PrevNum].postOffset(MBB);
    PrevNum = Num;
  }
}

llvm::MachineBasicBlock *TargetModuleBranchRelaxationWorker::createNewBlockAfter(
    llvm::MachineBasicBlock &OrigBB) {
  return createNewBlockAfter(OrigBB, OrigBB.getBasicBlock());
}

llvm::MachineBasicBlock *TargetModuleBranchRelaxationWorker::createNewBlockAfter(
    llvm::MachineBasicBlock &OrigMBB, const llvm::BasicBlock *BB) {
  auto *NewBB = MF->CreateMachineBasicBlock(BB);
  MF->insert(++OrigMBB.getIterator(), NewBB);
  NewBB->setSectionID(OrigMBB.getSectionID());
  NewBB->setIsEndSection(OrigMBB.isEndSection());
  OrigMBB.setIsEndSection(false);
  BlockInfo.insert(BlockInfo.begin() + NewBB->getNumber(), BasicBlockInfo());
  return NewBB;
}

llvm::MachineBasicBlock *TargetModuleBranchRelaxationWorker::splitBlockBeforeInstr(
    llvm::MachineInstr &MI, llvm::MachineBasicBlock *DestBB) {
  auto *OrigBB = MI.getParent();
  auto *NewBB = MF->CreateMachineBasicBlock(OrigBB->getBasicBlock());
  MF->insert(++OrigBB->getIterator(), NewBB);
  NewBB->setSectionID(OrigBB->getSectionID());
  NewBB->setIsEndSection(OrigBB->isEndSection());
  OrigBB->setIsEndSection(false);
  NewBB->splice(NewBB->end(), OrigBB, MI.getIterator(), OrigBB->end());
  TII->insertUnconditionalBranch(*OrigBB, NewBB, llvm::DebugLoc());
  BlockInfo.insert(BlockInfo.begin() + NewBB->getNumber(), BasicBlockInfo());
  NewBB->transferSuccessors(OrigBB);
  OrigBB->addSuccessor(NewBB);
  OrigBB->addSuccessor(DestBB);
  OrigBB->updateTerminator(NewBB);
  BlockInfo[OrigBB->getNumber()].Size = computeBlockSize(*OrigBB);
  BlockInfo[NewBB->getNumber()].Size = computeBlockSize(*NewBB);
  adjustBlockOffsets(*OrigBB, std::next(NewBB->getIterator()));
  if (TRI->trackLivenessAfterRegAlloc(*MF))
    computeAndAddLiveIns(LiveRegs, *NewBB);
  return NewBB;
}

bool TargetModuleBranchRelaxationWorker::isBlockInRange(
    const llvm::MachineInstr &MI, const llvm::MachineBasicBlock &DestBB) const {
  int64_t BrOffset = getInstrOffset(MI);
  int64_t DestOffset = BlockInfo[DestBB.getNumber()].Offset;
  const auto *SrcBB = MI.getParent();
  return TII->isBranchOffsetInRange(
      MI.getOpcode(), SrcBB->getSectionID() != DestBB.getSectionID()
                          ? TM->getMaxCodeSize()
                          : DestOffset - BrOffset);
}

bool TargetModuleBranchRelaxationWorker::fixupConditionalBranch(
    llvm::MachineInstr &MI) {
  // Verbatim port from stock BranchRelaxation::fixupConditionalBranch.
  llvm::DebugLoc DL = MI.getDebugLoc();
  auto *MBB = MI.getParent();
  llvm::MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  llvm::MachineBasicBlock *NewBB = nullptr;
  llvm::SmallVector<llvm::MachineOperand, 4> Cond;

  auto insertUncondBranch = [&](llvm::MachineBasicBlock *MBB,
                                llvm::MachineBasicBlock *DestBB) {
    unsigned &BBSize = BlockInfo[MBB->getNumber()].Size;
    int NewBrSize = 0;
    TII->insertUnconditionalBranch(*MBB, DestBB, DL, &NewBrSize);
    BBSize += NewBrSize;
  };
  auto insertBranch = [&](llvm::MachineBasicBlock *MBB,
                          llvm::MachineBasicBlock *TBB,
                          llvm::MachineBasicBlock *FBB,
                          llvm::SmallVectorImpl<llvm::MachineOperand> &Cond) {
    unsigned &BBSize = BlockInfo[MBB->getNumber()].Size;
    int NewBrSize = 0;
    TII->insertBranch(*MBB, TBB, FBB, Cond, DL, &NewBrSize);
    BBSize += NewBrSize;
  };
  auto removeBranch = [&](llvm::MachineBasicBlock *MBB) {
    unsigned &BBSize = BlockInfo[MBB->getNumber()].Size;
    int RemovedSize = 0;
    TII->removeBranch(*MBB, &RemovedSize);
    BBSize -= RemovedSize;
  };
  auto updateOffsetAndLiveness = [&](llvm::MachineBasicBlock *NewBB) {
    adjustBlockOffsets(*std::prev(NewBB->getIterator()),
                       std::next(NewBB->getIterator()));
    if (TRI->trackLivenessAfterRegAlloc(*MF))
      computeAndAddLiveIns(LiveRegs, *NewBB);
  };

  bool Fail = TII->analyzeBranch(*MBB, TBB, FBB, Cond);
  assert(!Fail && "branches to be relaxed must be analyzable");
  (void)Fail;

  if (MBB->getSectionID() != TBB->getSectionID() &&
      TBB->getSectionID() == llvm::MBBSectionID::ColdSectionID &&
      TrampolineInsertionPoint != nullptr) {
    NewBB =
        createNewBlockAfter(*TrampolineInsertionPoint, MBB->getBasicBlock());
    if (isBlockInRange(MI, *NewBB)) {
      insertUncondBranch(NewBB, TBB);
      MBB->replaceSuccessor(TBB, NewBB);
      NewBB->addSuccessor(TBB);
      removeBranch(MBB);
      insertBranch(MBB, NewBB, FBB, Cond);
      TrampolineInsertionPoint = NewBB;
      updateOffsetAndLiveness(NewBB);
      return true;
    }
    TrampolineInsertionPoint->setIsEndSection(NewBB->isEndSection());
    MF->erase(NewBB);
    NewBB = nullptr;
  }

  bool ReversedCond = !TII->reverseBranchCondition(Cond);
  if (ReversedCond) {
    // NOTE: stock LLVM takes a clean-reverse optimization here when
    // FBB is in range — `cond → FBB; uncond → TBB`. That leaves the
    // uncond out-of-range, which then gets a trampoline that pushes
    // FBB further, which then makes the new cond out-of-range, ad
    // infinitum under tight `--amdgpu-s-branch-bits`. We skip the
    // optimization and always take the split-block path, which lands
    // a layout-adjacent NewBB and breaks the feedback loop. Slightly
    // worse code in the in-range case, dramatically better
    // convergence in the out-of-range case.
    if (FBB) {
      if (TBB == FBB) {
        removeBranch(MBB);
        insertUncondBranch(MBB, TBB);
        return true;
      }
      NewBB = createNewBlockAfter(*MBB);
      insertUncondBranch(NewBB, FBB);
      MBB->replaceSuccessor(FBB, NewBB);
      NewBB->addSuccessor(FBB);
      updateOffsetAndLiveness(NewBB);
    }
    auto &NextBB = *std::next(llvm::MachineFunction::iterator(MBB));
    removeBranch(MBB);
    insertBranch(MBB, &NextBB, TBB, Cond);
    return true;
  }
  if (!FBB)
    FBB = &(*std::next(llvm::MachineFunction::iterator(MBB)));
  NewBB = createNewBlockAfter(*MBB);
  insertUncondBranch(NewBB, TBB);
  MBB->replaceSuccessor(TBB, NewBB);
  NewBB->addSuccessor(TBB);
  removeBranch(MBB);
  insertBranch(MBB, NewBB, FBB, Cond);
  updateOffsetAndLiveness(NewBB);
  return true;
}

bool TargetModuleBranchRelaxationWorker::fixupUnconditionalBranch(
    llvm::MachineInstr &MI) {
  auto *MBB = MI.getParent();
  unsigned OldBrSize = TII->getInstSizeInBytes(MI);
  auto *DestBB = TII->getBranchDestBlock(MI);
  int64_t DestOffset = BlockInfo[DestBB->getNumber()].Offset;
  int64_t SrcOffset = getInstrOffset(MI);
  assert(!TII->isBranchOffsetInRange(
      MI.getOpcode(), MBB->getSectionID() != DestBB->getSectionID()
                          ? TM->getMaxCodeSize()
                          : DestOffset - SrcOffset));
  BlockInfo[MBB->getNumber()].Size -= OldBrSize;

  llvm::MachineBasicBlock *BranchBB = MBB;
  if (!MBB->empty()) {
    BranchBB = createNewBlockAfter(*MBB);
    for (const auto *Succ : MBB->successors()) {
      for (const auto &LiveIn : Succ->liveins())
        BranchBB->addLiveIn(LiveIn);
    }
    BranchBB->sortUniqueLiveIns();
    BranchBB->addSuccessor(DestBB);
    MBB->replaceSuccessor(DestBB, BranchBB);
    if (TrampolineInsertionPoint == MBB)
      TrampolineInsertionPoint = BranchBB;
  }

  llvm::DebugLoc DL = MI.getDebugLoc();
  MI.eraseFromParent();

  auto *RestoreBB = createNewBlockAfter(MF->back(), DestBB->getBasicBlock());
  std::prev(RestoreBB->getIterator())
      ->setIsEndSection(RestoreBB->isEndSection());
  RestoreBB->setIsEndSection(false);

  emitLongBranch(*BranchBB, *DestBB, *RestoreBB, DL,
                        BranchBB->getSectionID() != DestBB->getSectionID()
                            ? TM->getMaxCodeSize()
                            : DestOffset - SrcOffset,
                        RS);

  BlockInfo[BranchBB->getNumber()].Size = computeBlockSize(*BranchBB);
  adjustBlockOffsets(*MBB, std::next(BranchBB->getIterator()));

  if (!RestoreBB->empty()) {
    if (MBB->getSectionID() == llvm::MBBSectionID::ColdSectionID &&
        DestBB->getSectionID() != llvm::MBBSectionID::ColdSectionID) {
      auto *NewBB = createNewBlockAfter(*TrampolineInsertionPoint);
      TII->insertUnconditionalBranch(*NewBB, DestBB, llvm::DebugLoc());
      BlockInfo[NewBB->getNumber()].Size = computeBlockSize(*NewBB);
      adjustBlockOffsets(*TrampolineInsertionPoint,
                         std::next(NewBB->getIterator()));
      TrampolineInsertionPoint = NewBB;
      BranchBB->replaceSuccessor(DestBB, NewBB);
      NewBB->addSuccessor(DestBB);
      DestBB = NewBB;
    }
    assert(!DestBB->isEntryBlock());
    auto *PrevBB = &*std::prev(DestBB->getIterator());
    if (auto *FT = PrevBB->getLogicalFallThrough()) {
      assert(FT == DestBB);
      (void)FT;
      TII->insertUnconditionalBranch(*PrevBB, DestBB, llvm::DebugLoc());
      BlockInfo[PrevBB->getNumber()].Size = computeBlockSize(*PrevBB);
    }
    MF->splice(DestBB->getIterator(), RestoreBB->getIterator());
    RestoreBB->addSuccessor(DestBB);
    BranchBB->replaceSuccessor(DestBB, RestoreBB);
    if (TRI->trackLivenessAfterRegAlloc(*MF))
      computeAndAddLiveIns(LiveRegs, *RestoreBB);
    BlockInfo[RestoreBB->getNumber()].Size = computeBlockSize(*RestoreBB);
    adjustBlockOffsets(*PrevBB, DestBB->getIterator());
    RestoreBB->setSectionID(DestBB->getSectionID());
    RestoreBB->setIsBeginSection(DestBB->isBeginSection());
    DestBB->setIsBeginSection(false);
    RelaxedUnconditionals.insert({BranchBB, RestoreBB});
  } else {
    MF->erase(RestoreBB);
    RelaxedUnconditionals.insert({BranchBB, DestBB});
  }
  return true;
}

bool TargetModuleBranchRelaxationWorker::relaxBranchInstructions() {
  bool Changed = false;
  for (auto &MBB : *MF) {
    auto Last = MBB.getLastNonDebugInstr();
    if (Last == MBB.end())
      continue;
    if (Last->isUnconditionalBranch()) {
      if (auto *DestBB = TII->getBranchDestBlock(*Last)) {
        if (!isBlockInRange(*Last, *DestBB) && !TII->isTailCall(*Last) &&
            !RelaxedUnconditionals.contains({&MBB, DestBB})) {
          fixupUnconditionalBranch(*Last);
          Changed = true;
        }
      }
    }
    llvm::MachineBasicBlock::iterator Next;
    for (auto J = MBB.getFirstTerminator(); J != MBB.end(); J = Next) {
      Next = std::next(J);
      auto &MI = *J;
      if (!MI.isConditionalBranch())
        continue;
      if (MI.getOpcode() == llvm::TargetOpcode::FAULTING_OP)
        continue;
      auto *DestBB = TII->getBranchDestBlock(MI);
      if (!isBlockInRange(MI, *DestBB)) {
        if (Next != MBB.end() && Next->isConditionalBranch())
          splitBlockBeforeInstr(*Next, DestBB);
        else
          fixupConditionalBranch(MI);
        Changed = true;
        Next = MBB.getFirstTerminator();
      }
    }
  }
  if (Changed)
    adjustBlockOffsets(MF->front());
  return Changed;
}

bool TargetModuleBranchRelaxationWorker::run(llvm::MachineFunction &mf) {
  MF = &mf;
  const auto &ST = MF->getSubtarget();
  TII = ST.getInstrInfo();
  TM = &MF->getTarget();
  TRI = ST.getRegisterInfo();
  MF->RenumberBlocks();

  // Seed per-MBB live-ins from the cached prototype-level predicated
  // liveness (per-PMBB live-in sets from `IPPredicatedLivenessAnalysis`,
  // indexed via `IPPredicatedCFG`) — one source of truth for liveness
  // across the whole target module, no second full backward dataflow.
  // `TracksLiveness` must be set before seeding: `MBB::livein_begin`
  // asserts on it, and the scavenger reads it downstream via
  // `enterBasicBlockEnd` → `LiveUnits.addLiveOuts`.
  MF->getProperties().setTracksLiveness();
  for (llvm::MachineBasicBlock &MBB : *MF) {
    if (MBB.empty())
      continue;
    const PredicatedMachineBasicBlock &PMBB =
        const_cast<IPPredicatedCFG &>(IPCFG).getPredMBB(MBB.front());
    const llvm::LivePhysRegs *LI = IPLiveness.getPMBBLiveIns(PMBB);
    if (!LI)
      continue;
    // Clear any stale live-ins from prior runs, then seed from the
    // PMBB live-in set. LiveInVector uses per-lane masks — we can't
    // recover finer information from LivePhysRegs, so grant every
    // seeded reg the full lane mask (MBB::addLiveIn's default).
    MBB.clearLiveIns();
    for (llvm::MCPhysReg R : *LI)
      MBB.addLiveIn(R);
    MBB.sortUniqueLiveIns();
  }

  scanFunction();
  bool MadeChange = false;
  // Bound the relaxer's outer fixed-point loop. Stock LLVM converges
  // naturally because each fixup tightens distance; under tight
  // `--amdgpu-s-branch-bits` and our SVA-aware scavenger insertions
  // we can fail to converge. Bail safely rather than spin.
  constexpr int kRelaxIterLimit = 64;
  for (int I = 0; I < kRelaxIterLimit; ++I) {
    if (!relaxBranchInstructions())
      break;
    MadeChange = true;
  }
  BlockInfo.clear();
  RelaxedUnconditionals.clear();
  return MadeChange;
}

} // namespace

bool TargetModuleBranchRelaxation::run(llvm::MachineFunction &MF) {
  TargetModuleBranchRelaxationWorker Worker(IPCFG, IPLiveness, ReservedRegs,
                                       SpillSink);
  return Worker.run(MF);
}

} // namespace luthier
