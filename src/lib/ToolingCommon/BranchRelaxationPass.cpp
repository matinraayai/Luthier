//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a branch relaxation pass for the code we are
/// instrumenting
//===----------------------------------------------------------------------===//

#include <llvm/CodeGen/LivePhysRegs.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineInstr.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Target/TargetMachine.h>
#include <luthier/Tooling/BranchRelaxationPass.h>
#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-branch-relaxation"

namespace luthier {

bool BranchRelaxationPass::relaxBranchInstructions() {
  bool Changed = false;

  // Relaxing branches involves creating new basic blocks, so re-eval
  // end() for termination.
  for (llvm::MachineBasicBlock &MBB : *MF) {
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

  // If we relaxed a branch, we must recompute offsets for all basic blocks.
  // Otherwise, we may underestimate branch distances and fail to relax a branch
  // that has been pushed out of range.
  if (Changed)
    adjustBlockOffsets(MF->front());

  return Changed;
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
    TrampolineInsertionPoint = &MBB;
  }

  // Compute block offsets and known bits.
  adjustBlockOffsets(*MF->begin());

  if (TrampolineInsertionPoint == nullptr) {
    LLVM_DEBUG(dbgs() << "  No suitable trampoline insertion point found in "
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

llvm::PreservedAnalyses
BranchRelaxationPass::run(llvm::MachineFunction &TargetMF,
                          llvm::MachineFunctionAnalysisManager &TargetMFAM) {
  MF = &TargetMF;
  bool MadeChanges = false;
  LLVM_DEBUG(dbgs() << "***** BranchRelaxation *****\n");

  const llvm::TargetSubtargetInfo &ST = TargetMF.getSubtarget();
  TII = ST.getInstrInfo();

  MF->RenumberBlocks();

  // We scan the machine function to compute information about the basic blocks
  scanFunction();
    
  while(relaxBranchInstructions())
    MadeChanges = true;


  BlockInfo.clear();
  RelaxedUnconditionals.clear();

  if (MadeChanges)
    return llvm::PreservedAnalyses::none();
  return llvm::PreservedAnalyses::all();
}

} // namespace luthier