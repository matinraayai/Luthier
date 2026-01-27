//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a branch relaxation pass for the code we are
/// instrumenting
//===----------------------------------------------------------------------===//

#include "luthier/Tooling/SVStorageAndLoadLocations.h"
#include "luthier/Tooling/StateValueArraySpecs.h"
#include <cassert>
#include <cstdint>
#include <iterator>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/CodeGen/MachinePostDominators.h>
#include <llvm/CodeGen/RegisterScavenging.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/DebugLoc.h>
#include <llvm/InitializePasses.h>
#include <llvm/Pass.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <luthier/Tooling/BranchRelaxationPass.h>
#include <luthier/Tooling/ImmutableMachineInstr.h>
#include <luthier/Tooling/WrapperAnalysisPasses.h>
#include <memory>
#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-branch-relaxation"

namespace luthier {

char BranchRelaxationPass::ID = 0;

LUTHIER_INITIALIZE_LEGACY_PASS_BODY(BranchRelaxationPass, "branch-relaxation",
                                    "Branch Relaxation Pass", true,
                                    false); // Is it CFG only?

void BranchRelaxationPass::insertIndirectBranch(llvm::MachineBasicBlock &MBB,
                                       llvm::MachineBasicBlock &DestBB,
                                       llvm::MachineBasicBlock &RestoreBB,
                                       const llvm::DebugLoc &DL, int64_t BrOffset
                                       /*RegScavenger *RS*/) const { // Change to custom Register Scavenger
  assert(MBB.empty() &&
         "new block should be inserted for expanding unconditional branch");
  assert(MBB.pred_size() == 1);
  assert(RestoreBB.empty() &&
         "restore block should be inserted for restoring clobbered registers");

  llvm::MachineFunction *MF = MBB.getParent();
  llvm::MachineRegisterInfo &MRI = MF->getRegInfo();
  const llvm::SIMachineFunctionInfo *MFI = MF->getInfo<llvm::SIMachineFunctionInfo>();
  auto I = MBB.end();
  auto &MCCtx = MF->getContext();

  if (ST.hasAddPC64Inst()) {
    llvm::MCSymbol *Offset =
        MCCtx.createTempSymbol("offset", /*AlwaysAddSuffix=*/true);
    auto AddPC = llvm::BuildMI(MBB, I, DL, get(llvm::AMDGPU::S_ADD_PC_I64))
                     .addSym(Offset, llvm::MO_FAR_BRANCH_OFFSET);
    llvm::MCSymbol *PostAddPCLabel =
        MCCtx.createTempSymbol("post_addpc", /*AlwaysAddSuffix=*/true);
    AddPC->setPostInstrSymbol(*MF, PostAddPCLabel);
    auto *OffsetExpr = llvm::MCBinaryExpr::createSub(
        LLVM::MCSymbolRefExpr::create(DestBB.getSymbol(), MCCtx),
        LLVM::MCSymbolRefExpr::create(PostAddPCLabel, MCCtx), MCCtx);
    Offset->setVariableValue(OffsetExpr);
    return;
  }

  assert(RS && "RegScavenger required for long branching");

  // FIXME: Virtual register workaround for RegScavenger not working with empty
  // blocks.
  llvm::Register PCReg = MRI.createVirtualRegister(&llvm::AMDGPU::SReg_64RegClass);

  // Note: as this is used after hazard recognizer we need to apply some hazard
  // workarounds directly.
  const bool FlushSGPRWrites = (ST.isWave64() && ST.hasVALUMaskWriteHazard()) ||
                               ST.hasVALUReadSGPRHazard();
  auto ApplyHazardWorkarounds = [this, &MBB, &I, &DL, FlushSGPRWrites]() {
    if (FlushSGPRWrites)
      BuildMI(MBB, I, DL, get(llvm::AMDGPU::S_WAITCNT_DEPCTR))
          .addImm(llvm::AMDGPU::DepCtr::encodeFieldSaSdst(0, ST));
  };

  // We need to compute the offset relative to the instruction immediately after
  // s_getpc_b64. Insert pc arithmetic code before last terminator.
  llvm::MachineInstr *GetPC = BuildMI(MBB, I, DL, get(llvm::AMDGPU::S_GETPC_B64), PCReg);
  ApplyHazardWorkarounds();

  llvm::MCSymbol *PostGetPCLabel =
      MCCtx.createTempSymbol("post_getpc", /*AlwaysAddSuffix=*/true);
  GetPC->setPostInstrSymbol(*MF, PostGetPCLabel);

  llvm::MCSymbol *OffsetLo =
      MCCtx.createTempSymbol("offset_lo", /*AlwaysAddSuffix=*/true);
  llvm::MCSymbol *OffsetHi =
      MCCtx.createTempSymbol("offset_hi", /*AlwaysAddSuffix=*/true);
  llvm::BuildMI(MBB, I, DL, get(llvm::AMDGPU::S_ADD_U32))
      .addReg(PCReg, llvm::RegState::Define, llvm::AMDGPU::sub0)
      .addReg(PCReg, 0, llvm::AMDGPU::sub0)
      .addSym(OffsetLo, llvm::MO_FAR_BRANCH_OFFSET);
  llvm::BuildMI(MBB, I, DL, get(llvm::AMDGPU::S_ADDC_U32))
      .addReg(PCReg, llvm::RegState::Define, llvm::AMDGPU::sub1)
      .addReg(PCReg, 0, llvm::AMDGPU::sub1)
      .addSym(OffsetHi, llvm::MO_FAR_BRANCH_OFFSET);
  ApplyHazardWorkarounds();

  // Insert the indirect branch after the other terminator.
  llvm::BuildMI(&MBB, DL, get(llvm::AMDGPU::S_SETPC_B64)).addReg(PCReg);

  // If a spill is needed for the pc register pair, we need to insert a spill
  // restore block right before the destination block, and insert a short branch
  // into the old destination block's fallthrough predecessor.
  // e.g.:
  //
  // s_cbranch_scc0 skip_long_branch:
  //
  // long_branch_bb:
  //   spill s[8:9]
  //   s_getpc_b64 s[8:9]
  //   s_add_u32 s8, s8, restore_bb
  //   s_addc_u32 s9, s9, 0
  //   s_setpc_b64 s[8:9]
  //
  // skip_long_branch:
  //   foo;
  //
  // .....
  //
  // dest_bb_fallthrough_predecessor:
  // bar;
  // s_branch dest_bb
  //
  // restore_bb:
  //  restore s[8:9]
  //  fallthrough dest_bb
  ///
  // dest_bb:
  //   buzz;

  llvm::Register LongBranchReservedReg = MFI->getLongBranchReservedReg();
  llvm::Register Scav;

  // If we've previously reserved a register for long branches
  // avoid running the scavenger and just use those registers
  if (LongBranchReservedReg) {
    RS->enterBasicBlock(MBB);
    Scav = LongBranchReservedReg;
  } else {
    RS->enterBasicBlockEnd(MBB);
    Scav = RS->scavengeRegisterBackwards(
        llvm::AMDGPU::SReg_64RegClass, llvm::MachineBasicBlock::iterator(GetPC),
        /* RestoreAfter */ false, 0, /* AllowSpill */ false);
  }
  if (Scav) {
    RS->setRegUsed(Scav);
    MRI.replaceRegWith(PCReg, Scav);
    MRI.clearVirtRegs();
  } else {
    // As SGPR needs VGPR to be spilled, we reuse the slot of temporary VGPR for
    // SGPR spill.
    const llvm::GCNSubtarget &ST = MF->getSubtarget<llvm::GCNSubtarget>();
    TRI->spillEmergencySGPR(GetPC, RestoreBB, llvm::AMDGPU::SGPR0_SGPR1, RS);
    MRI.replaceRegWith(PCReg, llvm::AMDGPU::SGPR0_SGPR1);
    MRI.clearVirtRegs();
  }

  llvm::MCSymbol *DestLabel = Scav ? DestBB.getSymbol() : RestoreBB.getSymbol();
  // Now, the distance could be defined.
  auto *Offset = llvm::MCBinaryExpr::createSub(
      llvm::MCSymbolRefExpr::create(DestLabel, MCCtx),
      llvm::MCSymbolRefExpr::create(PostGetPCLabel, MCCtx), MCCtx);
  // Add offset assignments.
  auto *Mask = llvm::MCConstantExpr::create(0xFFFFFFFFULL, MCCtx);
  OffsetLo->setVariableValue(llvm::MCBinaryExpr::createAnd(Offset, Mask, MCCtx));
  auto *ShAmt = llvm::MCConstantExpr::create(32, MCCtx);
  OffsetHi->setVariableValue(llvm::MCBinaryExpr::createAShr(Offset, ShAmt, MCCtx));
}

void BranchRelaxationPass::scanFunction() {
  BlockInfo.clear();
  BlockInfo.resize(MF->getNumBlockIDs());

  TrampolineInsertionPoint = nullptr;
  RelaxedUnconditionals.clear();

  // Compute the size of all basic blocks, the last basic block of the function
  // is the trampoline insertion point
  for (llvm::MachineBasicBlock &MBB : *MF) {
    BlockInfo[MBB.getNumber()].Size = computeBlockSize(MBB);

    if (MBB.getSectionID() != MBBSectionID::ColdSectionID)
      TrampolineInsertionPoint = &MBB;
  }

  // Compute block offsets and known bits.
  adjustBlockOffsets(*MF->begin());

  if (TrampolineInsertionPoint == nullptr) {
    LLVM_DEBUG(llvm::dbgs()
               << "  No suitable trampoline insertion point found in "
               << MF->getName() << ".\n");
  }
}
/// Compute the size of the basic block in bytes
uint64_t BranchRelaxationPass::computeBlockSize(
    const llvm::MachineBasicBlock &MBB) const {
  uint64_t Size = 0;
  for (const llvm::MachineInstr &MI : MBB)
    Size += TII->getInstSizeInBytes(MI);
  return Size;
}

/// getInstrOffset - Return the current offset of the specified machine
/// instruction from the start of the function.  This offset changes as stuff is
/// moved around inside the function.
unsigned
BranchRelaxationPass::getInstrOffset(const llvm::MachineInstr &MI) const {
  const llvm::MachineBasicBlock *MBB = MI.getParent();

  // The offset is composed of two things: the sum of the sizes of all MBB's
  // before this instruction's block, and the offset from the start of the block
  // it is in.
  unsigned Offset = BlockInfo[MBB->getNumber()].Offset;

  // Sum instructions before MI in MBB.
  for (llvm::MachineBasicBlock::const_iterator I = MBB->begin(); &*I != &MI;
       ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    Offset += TII->getInstSizeInBytes(*I);
  }

  return Offset;
}
void BranchRelaxationPass::adjustBlockOffsets(llvm::MachineBasicBlock &Start) {
  adjustBlockOffsets(Start, MF->end());
}

void BranchRelaxationPass::adjustBlockOffsets(
    llvm::MachineBasicBlock &Start, llvm::MachineFunction::iterator End) {
  unsigned PrevNum = Start.getNumber();
  for (auto &MBB : llvm::make_range(
           std::next(llvm::MachineFunction::iterator(Start)), End)) {
    unsigned Num = MBB.getNumber();
    // Get the offset and known bits at the end of the layout predecessor.
    // Include the alignment of the current block.
    BlockInfo[Num].Offset = BlockInfo[PrevNum].postOffset(MBB);

    PrevNum = Num;
  }
}
/// Insert a new empty MachineBasicBlock and insert it after \p OrigMBB
llvm::MachineBasicBlock *
BranchRelaxationPass::createNewBlockAfter(llvm::MachineBasicBlock &OrigBB) {
  return createNewBlockAfter(OrigBB, OrigBB.getBasicBlock());
}

/// Insert a new empty MachineBasicBlock with \p BB as its BasicBlock
/// and insert it after \p OrigMBB
MachineBasicBlock *
BranchRelaxationPass::createNewBlockAfter(llvm::MachineBasicBlock &OrigMBB,
                                          const llvm::BasicBlock *BB) {
  // Create a new MBB for the code after the OrigBB.
  llvm::MachineBasicBlock *NewBB = MF->CreateMachineBasicBlock(BB);
  MF->insert(++OrigMBB.getIterator(), NewBB);

  // Place the new block in the same section as OrigBB
  NewBB->setSectionID(OrigMBB.getSectionID());
  NewBB->setIsEndSection(OrigMBB.isEndSection());
  OrigMBB.setIsEndSection(false);

  // Insert an entry into BlockInfo to align it properly with the block numbers.
  BlockInfo.insert(BlockInfo.begin() + NewBB->getNumber(), BasicBlockInfo());

  return NewBB;
}
/// Split the basic block containing MI into two blocks, which are joined by
/// an unconditional branch.  Update data structures and renumber blocks to
/// account for this change and returns the newly created block.
llvm::MachineBasicBlock *
BranchRelaxationPass::splitBlockBeforeInstr(llvm::MachineInstr &MI,
                                            llvm::MachineBasicBlock *DestBB) {
  llvm::MachineBasicBlock *OrigBB = MI.getParent();

  // Create a new MBB for the code after the OrigBB.
  llvm::MachineBasicBlock *NewBB =
      MF->CreateMachineBasicBlock(OrigBB->getBasicBlock());
  MF->insert(++OrigBB->getIterator(), NewBB);

  // Place the new block in the same section as OrigBB.
  NewBB->setSectionID(OrigBB->getSectionID());
  NewBB->setIsEndSection(OrigBB->isEndSection());
  OrigBB->setIsEndSection(false);

  // Splice the instructions starting with MI over to NewBB.
  NewBB->splice(NewBB->end(), OrigBB, MI.getIterator(), OrigBB->end());

  // Add an unconditional branch from OrigBB to NewBB.
  // Note the new unconditional branch is not being recorded.
  // There doesn't seem to be meaningful DebugInfo available; this doesn't
  // correspond to anything in the source.
  TII->insertUnconditionalBranch(*OrigBB, NewBB, DebugLoc());

  // Insert an entry into BlockInfo to align it properly with the block numbers.
  BlockInfo.insert(BlockInfo.begin() + NewBB->getNumber(), BasicBlockInfo());

  NewBB->transferSuccessors(OrigBB);
  OrigBB->addSuccessor(NewBB);
  OrigBB->addSuccessor(DestBB);

  // Cleanup potential unconditional branch to successor block.
  // Note that updateTerminator may change the size of the blocks.
  OrigBB->updateTerminator(NewBB);

  // Figure out how large the OrigBB is.  As the first half of the original
  // block, it cannot contain a tablejump.  The size includes
  // the new jump we added.  (It should be possible to do this without
  // recounting everything, but it's very confusing, and this is rarely
  // executed.)
  BlockInfo[OrigBB->getNumber()].Size = computeBlockSize(*OrigBB);

  // Figure out how large the NewMBB is. As the second half of the original
  // block, it may contain a tablejump.
  BlockInfo[NewBB->getNumber()].Size = computeBlockSize(*NewBB);

  // Update the offset of the new block.
  adjustBlockOffsets(*OrigBB, std::next(NewBB->getIterator()));

  // Need to fix live-in lists if we track liveness.
  if (TRI->trackLivenessAfterRegAlloc(*MF))
    llvm::computeAndAddLiveIns(LiveRegs, *NewBB);

  ++NumSplit;

  return NewBB;
}

/// isBlockInRange - Returns true if the distance between specific MI and
/// specific BB can fit in MI's displacement field.
bool BranchRelaxationPass::isBlockInRange(
    const llvm::MachineInstr &MI, const llvm::MachineBasicBlock &DestBB) const {
  int64_t BrOffset = getInstrOffset(MI);
  int64_t DestOffset = BlockInfo[DestBB.getNumber()].Offset;

  const llvm::MachineBasicBlock *SrcBB = MI.getParent();

  if (TII->isBranchOffsetInRange(MI.getOpcode(),
                                 SrcBB->getSectionID() != DestBB.getSectionID()
                                     ? TM->getMaxCodeSize()
                                     : DestOffset - BrOffset))
    return true;

  LLVM_DEBUG(llvm::dbgs() << "Out of range branch to destination "
                          << llvm::printMBBReference(DestBB) << " from "
                          << llvm::printMBBReference(*MI.getParent()) << " to "
                          << DestOffset << " offset " << DestOffset - BrOffset
                          << '\t' << MI);

  return false;
}

/// fixupConditionalBranch - Fix up a conditional branch whose destination is
/// too far away to fit in its displacement field. It is converted to an inverse
/// conditional branch + an unconditional branch to the destination.
bool BranchRelaxationPass::fixupConditionalBranch(llvm::MachineInstr &MI) {
  llvm::DebugLoc DL = MI.getDebugLoc();
  llvm::MachineBasicBlock *MBB = MI.getParent();
  llvm::MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  llvm::MachineBasicBlock *NewBB = nullptr;
  llvm::SmallVector<MachineOperand, 4> Cond;

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
                          llvm::SmallVectorImpl<MachineOperand> &Cond) {
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

  // Populate the block offset and live-ins for a new basic block.
  auto updateOffsetAndLiveness = [&](llvm::MachineBasicBlock *NewBB) {
    assert(NewBB != nullptr && "can't populate offset for nullptr");

    // Keep the block offsets approximately up to date. While they will be
    // slight underestimates, we will update them appropriately in the next
    // scan through the function.
    adjustBlockOffsets(*std::prev(NewBB->getIterator()),
                       std::next(NewBB->getIterator()));

    // Need to fix live-in lists if we track liveness.
    if (TRI->trackLivenessAfterRegAlloc(*MF))
      computeAndAddLiveIns(LiveRegs, *NewBB);
  };

  bool Fail = TII->analyzeBranch(*MBB, TBB, FBB, Cond);
  assert(!Fail && "branches to be relaxed must be analyzable");
  (void)Fail;

  // Since cross-section conditional branches to the cold section are rarely
  // taken, try to avoid inverting the condition. Instead, add a "trampoline
  // branch", which unconditionally branches to the branch destination. Place
  // the trampoline branch at the end of the function and retarget the
  // conditional branch to the trampoline.
  // tbz L1
  // =>
  // tbz L1Trampoline
  // ...
  // L1Trampoline: b  L1
  if (MBB->getSectionID() != TBB->getSectionID() &&
      TBB->getSectionID() == llvm::MBBSectionID::ColdSectionID &&
      TrampolineInsertionPoint != nullptr) {
    // If the insertion point is out of range, we can't put a trampoline there.
    NewBB =
        createNewBlockAfter(*TrampolineInsertionPoint, MBB->getBasicBlock());

    if (isBlockInRange(MI, *NewBB)) {
      LLVM_DEBUG(llvm::dbgs() << "  Retarget destination to trampoline at "
                              << NewBB->back());

      insertUncondBranch(NewBB, TBB);

      // Update the successor lists to include the trampoline.
      MBB->replaceSuccessor(TBB, NewBB);
      NewBB->addSuccessor(TBB);

      // Replace branch in the current (MBB) block.
      removeBranch(MBB);
      insertBranch(MBB, NewBB, FBB, Cond);

      TrampolineInsertionPoint = NewBB;
      llvm::updateOffsetAndLiveness(NewBB);
      return true;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "  Trampoline insertion point out of range for Bcc from "
               << llvm::printMBBReference(*MBB) << " to "
               << llvm::printMBBReference(*TBB) << ".\n");
    TrampolineInsertionPoint->setIsEndSection(NewBB->isEndSection());
    MF->erase(NewBB);
    NewBB = nullptr;
  }

  // Add an unconditional branch to the destination and invert the branch
  // condition to jump over it:
  // tbz L1
  // =>
  // tbnz L2
  // b   L1
  // L2:

  bool ReversedCond = !TII->reverseBranchCondition(Cond);
  if (ReversedCond) {
    if (FBB && isBlockInRange(MI, *FBB)) {
      // Last MI in the BB is an unconditional branch. We can simply invert the
      // condition and swap destinations:
      // beq L1
      // b   L2
      // =>
      // bne L2
      // b   L1
      LLVM_DEBUG(llvm::dbgs() << "  Invert condition and swap "
                                 "its destination with "
                              << MBB->back());

      removeBranch(MBB);
      insertBranch(MBB, FBB, TBB, Cond);
      return true;
    }
    if (FBB) {
      // If we get here with a MBB which ends like this:
      //
      // bb.1:
      // successors: %bb.2;
      // ...
      // BNE $x1, $x0, %bb.2
      // PseudoBR %bb.2
      //
      // Just remove conditional branch.
      if (TBB == FBB) {
        removeBranch(MBB);
        insertUncondBranch(MBB, TBB);
        return true;
      }
      // We need to split the basic block here to obtain two long-range
      // unconditional branches.
      NewBB = createNewBlockAfter(*MBB);

      insertUncondBranch(NewBB, FBB);
      // Update the succesor lists according to the transformation to follow.
      // Do it here since if there's no split, no update is needed.
      MBB->replaceSuccessor(FBB, NewBB);
      NewBB->addSuccessor(FBB);
      updateOffsetAndLiveness(NewBB);
    }

    // We now have an appropriate fall-through block in place (either naturally
    // or just created), so we can use the inverted the condition.
    llvm::MachineBasicBlock &NextBB =
        *std::next(llvm::MachineFunction::iterator(MBB));

    LLVM_DEBUG(llvm::dbgs() << "  Insert B to " << llvm::printMBBReference(*TBB)
                            << ", invert condition and change dest. to "
                            << llvm::printMBBReference(NextBB) << '\n');

    removeBranch(MBB);
    // Insert a new conditional branch and a new unconditional branch.
    insertBranch(MBB, &NextBB, TBB, Cond);
    return true;
  }
  // Branch cond can't be inverted.
  // In this case we always add a block after the MBB.
  LLVM_DEBUG(llvm::dbgs() << "  The branch condition can't be inverted. "
                          << "  Insert a new BB after " << MBB->back());

  if (!FBB)
    FBB = &(*std::next(llvm::MachineFunction::iterator(MBB)));

  // This is the block with cond. branch and the distance to TBB is too long.
  //    beq L1
  // L2:

  // We do the following transformation:
  //    beq NewBB
  //    b L2
  // NewBB:
  //    b L1
  // L2:

  NewBB = createNewBlockAfter(*MBB);
  insertUncondBranch(NewBB, TBB);

  LLVM_DEBUG(llvm::dbgs() << "  Insert cond B to the new BB "
                          << llvm::printMBBReference(*NewBB)
                          << "  Keep the exiting condition.\n"
                          << "  Insert B to " << llvm::printMBBReference(*FBB)
                          << ".\n"
                          << "  In the new BB: Insert B to "
                          << llvm::printMBBReference(*TBB) << ".\n");

  // Update the successor lists according to the transformation to follow.
  MBB->replaceSuccessor(TBB, NewBB);
  NewBB->addSuccessor(TBB);

  // Replace branch in the current (MBB) block.
  removeBranch(MBB);
  insertBranch(MBB, NewBB, FBB, Cond);

  updateOffsetAndLiveness(NewBB);
  return true;
}

bool BranchRelaxationPass::fixupUnconditionalBranch(llvm::MachineInstr &MI) {
  llvm::MachineBasicBlock *MBB = MI.getParent();
  unsigned OldBrSize = TII->getInstSizeInBytes(MI);
  llvm::MachineBasicBlock *DestBB = TII->getBranchDestBlock(MI);

  int64_t DestOffset = BlockInfo[DestBB->getNumber()].Offset;
  int64_t SrcOffset = getInstrOffset(MI);

  assert(!TII->isBranchOffsetInRange(
      MI.getOpcode(), MBB->getSectionID() != DestBB->getSectionID()
                          ? TM->getMaxCodeSize()
                          : DestOffset - SrcOffset));

  BlockInfo[MBB->getNumber()].Size -= OldBrSize;

  llvm::MachineBasicBlock *BranchBB = MBB;

  // If this was an expanded conditional branch, there is already a single
  // unconditional branch in a block.
  if (!MBB->empty()) {
    BranchBB = createNewBlockAfter(*MBB);

    // Add live outs.
    for (const llvm::MachineBasicBlock *Succ : MBB->successors()) {
      for (const llvm::MachineBasicBlock::RegisterMaskPair &LiveIn :
           Succ->liveins())
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

  // Create the optional restore block and, initially, place it at the end of
  // function. That block will be placed later if it's used; otherwise, it will
  // be erased.
  llvm::MachineBasicBlock *RestoreBB =
      createNewBlockAfter(MF->back(), DestBB->getBasicBlock());
  std::prev(RestoreBB->getIterator())
      ->setIsEndSection(RestoreBB->isEndSection());
  RestoreBB->setIsEndSection(false);

  TII->insertIndirectBranch(*BranchBB, *DestBB, *RestoreBB, DL,
                            BranchBB->getSectionID() != DestBB->getSectionID()
                                ? TM->getMaxCodeSize()
                                : DestOffset - SrcOffset,
                            RS.get());

  // Update the block size and offset for the BranchBB (which may be newly
  // created).
  BlockInfo[BranchBB->getNumber()].Size = computeBlockSize(*BranchBB);
  adjustBlockOffsets(*MBB, std::next(BranchBB->getIterator()));

  // If RestoreBB is required, place it appropriately.
  if (!RestoreBB->empty()) {
    // If the jump is Cold -> Hot, don't place the restore block (which is
    // cold) in the middle of the function. Place it at the end.
    if (MBB->getSectionID() == llvm::MBBSectionID::ColdSectionID &&
        DestBB->getSectionID() != llvm::MBBSectionID::ColdSectionID) {
      llvm::MachineBasicBlock *NewBB =
          createNewBlockAfter(*TrampolineInsertionPoint);
      TII->insertUnconditionalBranch(*NewBB, DestBB, DebugLoc());
      BlockInfo[NewBB->getNumber()].Size = computeBlockSize(*NewBB);
      adjustBlockOffsets(*TrampolineInsertionPoint,
                         std::next(NewBB->getIterator()));

      // New trampolines should be inserted after NewBB.
      TrampolineInsertionPoint = NewBB;

      // Retarget the unconditional branch to the trampoline block.
      BranchBB->replaceSuccessor(DestBB, NewBB);
      NewBB->addSuccessor(DestBB);

      DestBB = NewBB;
    }

    // In all other cases, try to place just before DestBB.

    // TODO: For multiple far branches to the same destination, there are
    // chances that some restore blocks could be shared if they clobber the
    // same registers and share the same restore sequence. So far, those
    // restore blocks are just duplicated for each far branch.
    assert(!DestBB->isEntryBlock());
    llvm::MachineBasicBlock *PrevBB = &*std::prev(DestBB->getIterator());
    // Fall through only if PrevBB has no unconditional branch as one of its
    // terminators.
    if (auto *FT = PrevBB->getLogicalFallThrough()) {
      assert(FT == DestBB);
      TII->insertUnconditionalBranch(*PrevBB, FT, DebugLoc());
      BlockInfo[PrevBB->getNumber()].Size = computeBlockSize(*PrevBB);
    }
    // Now, RestoreBB could be placed directly before DestBB.
    MF->splice(DestBB->getIterator(), RestoreBB->getIterator());
    // Update successors and predecessors.
    RestoreBB->addSuccessor(DestBB);
    BranchBB->replaceSuccessor(DestBB, RestoreBB);
    if (TRI->trackLivenessAfterRegAlloc(*MF))
      llvm::computeAndAddLiveIns(
          LiveRegs, *RestoreBB); // Should we use luthier one here????
    // Compute the restore block size.
    BlockInfo[RestoreBB->getNumber()].Size = computeBlockSize(*RestoreBB);
    // Update the estimated offset for the restore block.
    adjustBlockOffsets(*PrevBB, DestBB->getIterator());

    // Fix up section information for RestoreBB and DestBB
    RestoreBB->setSectionID(DestBB->getSectionID());
    RestoreBB->setIsBeginSection(DestBB->isBeginSection());
    DestBB->setIsBeginSection(false);
    RelaxedUnconditionals.insert({BranchBB, RestoreBB});
  } else {
    // Remove restore block if it's not required.
    MF->erase(RestoreBB);
    RelaxedUnconditionals.insert({BranchBB, DestBB});
  }

  return true;
}

bool BranchRelaxationPass::relaxBranchInstructions() {
  bool Changed = false;

  // Relaxing branches involves creating new basic blocks, so re-eval
  // end() for termination.
  for (llvm::MachineBasicBlock &MBB : *MF) {
    // Empty block?
    llvm::MachineBasicBlock::iterator Last = MBB.getLastNonDebugInstr();
    if (Last == MBB.end())
      continue;

    // Expand the unconditional branch first if necessary. If there is a
    // conditional branch, this will end up changing the branch destination of
    // it to be over the newly inserted indirect branch block, which may avoid
    // the need to try expanding the conditional branch first, saving an extra
    // jump.
    if (Last->isUnconditionalBranch()) {
      // Unconditional branch destination might be unanalyzable, assume these
      // are OK.
      if (llvm::MachineBasicBlock *DestBB = TII->getBranchDestBlock(*Last)) {
        if (!isBlockInRange(*Last, *DestBB) && !TII->isTailCall(*Last) &&
            !RelaxedUnconditionals.contains({&MBB, DestBB})) {
          fixupUnconditionalBranch(*Last);
          ++NumUnconditionalRelaxed;
          Changed = true;
        }
      }
    }

    // Loop over the conditional branches.
    llvm::MachineBasicBlock::iterator Next;
    for (llvm::MachineBasicBlock::iterator J = MBB.getFirstTerminator();
         J != MBB.end(); J = Next) {
      Next = std::next(J);
      llvm::MachineInstr &MI = *J;

      if (!MI.isConditionalBranch())
        continue;

      if (MI.getOpcode() == llvm::TargetOpcode::FAULTING_OP)
        // FAULTING_OP's destination is not encoded in the instruction stream
        // and thus never needs relaxed.
        continue;

      llvm::MachineBasicBlock *DestBB = TII->getBranchDestBlock(MI);
      if (!isBlockInRange(MI, *DestBB)) {
        if (Next != MBB.end() && Next->isConditionalBranch()) {
          // If there are multiple conditional branches, this isn't an
          // analyzable block. Split later terminators into a new block so
          // each one will be analyzable.

          splitBlockBeforeInstr(*Next, DestBB);
        } else {
          fixupConditionalBranch(MI);
          ++NumConditionalRelaxed;
        }

        Changed = true;

        // This may have modified all of the terminators, so start over.
        Next = MBB.getFirstTerminator();
      }
    }
  }

  // If we relaxed a branch, we must recompute offsets for *all* basic blocks.
  // Otherwise, we may underestimate branch distances and fail to relax a branch
  // that has been pushed out of range.
  if (Changed)
    adjustBlockOffsets(MF->front());

  return Changed;
}

bool BranchRelaxationPass::runOnMachineFunction(llvm::MachineFunction &IMF) {
  // auto *MDT =
  // TargetMFAM.getCachedResult<llvm::MachineDominatorTreeAnalysis>(MF); auto
  // *MPDT =
  // TargetMFAM.getCachedResult<llvm::MachinePostDominatorTreeAnalysis>(MF);

  bool MadeChanges{false};
  LLVM_DEBUG(llvm::dbgs() << "***** BranchRelaxation *****\n");

  auto &IModule = const_cast<llvm::Module &>(
      *getAnalysis<llvm::MachineModuleInfoWrapperPass>().getMMI().getModule());

  auto &IMAM = getAnalysis<IModuleMAMWrapperPass>().getMAM();

  const auto &IPIP =
      *IMAM.getCachedResult<InjectedPayloadAndInstPointAnalysis>(IModule);

  auto &TargetModule =
      IMAM.getCachedResult<TargetAppModuleAndMAMAnalysis>(IModule)
          ->getTargetAppModule();

  auto &TargetMAM = IMAM.getCachedResult<TargetAppModuleAndMAMAnalysis>(IModule)
                        ->getTargetAppMAM();

  MF = &(IPIP.at(IMF.getFunction())->getMF());
  const auto &StateValueLocations =
      *TargetMAM.getCachedResult<LRStateValueStorageAndLoadLocationsAnalysis>(
          TargetModule);

  auto *SVALoadPlan =
      StateValueLocations.getStateValueArrayLoadPlanForInstPoint(
          *IPIP.at(MF.getFunction()));
  if (!SVALoadPlan) {
    MF.getContext().reportError(
        {},
        llvm::formatv(
            "Could not find the state value load plan for Machine Instr {0}.",
            *IPIP.at(MF.getFunction())));
    return false;
  }

  SVA = StateValueLoadPlan->StateValueArrayLoadVGPR;

  const llvm::TargetSubtargetInfo &ST = TargetMF.getSubtarget();
  TII = ST.getInstrInfo();

  MF->RenumberBlocks();

  // We scan the machine function to compute information about the basic blocks
  scanFunction();

  while (relaxBranchInstructions())
    MadeChanges = true;

  BlockInfo.clear();
  RelaxedUnconditionals.clear();

  if (MadeChanges)
    return true;

  // if (MDT)
  //   MDT->updateBlockNumbers();
  // if (MPDT)
  //   MPDT->updateBlockNumbers();
  return false;
}

void BranchRelaxationPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<IModuleMAMWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

} // namespace luthier