#ifndef LUTHIER_TOOLING_BRANCH_RELAXATION_PASS_H
#define LUTHIER_TOOLING_BRANCH_RELAXATION_PASS_H

#include "luthier/Tooling/LegacyPassSupport.h"
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/RegisterScavenging.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/ADT/DenseSet.h>

namespace luthier {
class BranchRelaxationPass;

LUTHIER_INITIALIZE_LEGACY_PASS_PROTOTYPE(BranchRelaxationPass);

class BranchRelaxationPass : public llvm::MachineFunctionPass {
  /// BasicBlockInfo - Information about the offset and size of a single
  /// basic block.
  struct BasicBlockInfo {
    /// Offset - Distance from the beginning of the function to the beginning
    /// of this basic block.
    ///
    /// The offset is always aligned as required by the basic block.
    unsigned Offset = 0;

    /// Size - Size of the basic block in bytes.  If the block contains
    /// inline assembly, this is a worst case estimate.
    ///
    /// The size does not include any alignment padding whether from the
    /// beginning of the block, or from an aligned jump table at the end.
    unsigned Size = 0;

    BasicBlockInfo() = default;

    /// Compute the offset immediately following this block. \p MBB is the next
    /// block.
    unsigned postOffset(const llvm::MachineBasicBlock &MBB) const {
      const unsigned PO = Offset + Size;
      const Align Alignment = MBB.getAlignment();
      const Align ParentAlign = MBB.getParent()->getAlignment();
      if (Alignment <= ParentAlign)
        return alignTo(PO, Alignment);

      // The alignment of this MBB is larger than the function's alignment, so
      // we can't tell whether or not it will insert nops. Assume that it will.
      return alignTo(PO, Alignment) + Alignment.value() - ParentAlign.value();
    }
  };

  llvm::SmallVector<BasicBlockInfo, 16> BlockInfo;
  // The basic block after which trampolines are inserted. This is the last
  // basic block that isn't in the cold section.
  llvm::MachineBasicBlock *TrampolineInsertionPoint = nullptr;
  llvm::SmallDenseSet<std::pair<llvm::MachineBasicBlock *, llvm::MachineBasicBlock *>> RelaxedUnconditionals;
  std::unique_ptr<llvm::RegScavenger> RS;
  llvm::LivePhysRegs LiveRegs;
  llvm::MachineFunction *MF = nullptr;
  llvm::MCRegister SVA{};
  const llvm::TargetRegisterInfo *TRI = nullptr;
  const llvm::TargetInstrInfo *TII = nullptr;
  const llvm::TargetMachine *TM = nullptr;
  
  MachineBasicBlock *createNewBlockAfter(llvm::MachineBasicBlock &OrigMBB);
  MachineBasicBlock *createNewBlockAfter(llvm::MachineBasicBlock &OrigMBB,
                                         const llvm::BasicBlock *BB);

  MachineBasicBlock *splitBlockBeforeInstr(llvm::MachineInstr &MI,
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

public:
  static char ID;
  BranchRelaxationPass : llvm::MachineFunctionPass(ID) {}
  bool runOnMachineFunction(llvm::MachineFunction &TargetMF) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
} // namespace luthier

#endif