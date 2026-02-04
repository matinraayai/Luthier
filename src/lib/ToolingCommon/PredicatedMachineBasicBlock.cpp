//===-- PredicatedMachineBasicBlock.cpp -----------------------------------===//
// Copyright 2026 @ Northeastern University Computer Architecture Lab
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
/// \file PredicatedMachineBasicBlock.cpp
/// Implements the \c PredicatedMachineBasicBlock class and its builder.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/PredicatedMachineBasicBlock.h"
#include "luthier/LLVM/CodeGenHelpers.h"
#include "luthier/Tooling/IPPredicatedCFG.h"
#include "luthier/Tooling/LinearMachineBasicBlock.h"
#include "luthier/Tooling/PredicatedMachineFunction.h"
#include <SIInstrInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier {

std::string PredicatedMachineBasicBlock::getName() const {
  const LinearMachineBasicBlock &SMBB = getParent();
  const llvm::MachineFunction &MF = SMBB.getParent().getMF();
  const llvm::MachineBasicBlock *MBB = getParent().getMBB();

  return llvm::formatv("{0}.{1}",
                       MBB ? MBB->getFullName() : MF.getName() + ".none",
                       getGlobalNumber())
      .str();
}

PredicatedMachineBasicBlock::iterator
PredicatedMachineBasicBlock::getFirstNonDebugInstr(bool SkipPseudoOp) {
  // Skip over begin-of-block dbg_value instructions.
  return skipDebugInstructionsForward(begin(), end(), SkipPseudoOp);
}

PredicatedMachineBasicBlock::iterator
PredicatedMachineBasicBlock::getLastNonDebugInstr(bool SkipPseudoOp) {
  // Skip over end-of-block dbg_value instructions.
  iterator B = begin(), I = end();
  while (I != B) {
    --I;
    // Return instruction that starts a bundle.
    if (I->isDebugInstr() || I->isInsideBundle())
      continue;
    if (SkipPseudoOp && I->isPseudoProbe())
      continue;
    return I;
  }
  // The block is all debug values.
  return end();
}

void PredicatedMachineBasicBlock::print(llvm::raw_ostream &OS,
                                        unsigned int Indent) const {
  const auto &ST = getParent().getParent().getMF().getSubtarget();
  const auto TII = ST.getInstrInfo();
  OS.indent(Indent) << "Predicated MBB " << getName() << "\n";

  OS.indent(Indent) << "Predecessors: [";
  llvm::interleave(
      Predecessors.begin(), Predecessors.end(),
      [&](const PredMBBBuilder &MBB) {
        OS << "MBB " << MBB.getPredMBB().getName();
      },
      [&]() { OS << ", "; });
  OS << "]\n";

  if (!this->empty()) {
    OS.indent(Indent) << "Instructions:\n";
    for (const auto &MI : *this) {
      MI.print(OS.indent(Indent + 2), true, false, false, true, TII);
    }
  }

  OS.indent(Indent) << "Successors: [";
  llvm::interleave(
      Successors.begin(), Successors.end(),
      [&](const PredMBBBuilder &MBB) {
        OS << "MBB " << MBB.getPredMBB().getName();
      },
      [&]() { OS << ", "; });
  OS << "]\n";
}

void PredicatedMachineBasicBlock::dump() const { print(llvm::dbgs(), 0); }

llvm::SmallVector<std::unique_ptr<PredMBBBuilder>, 6>
PredMBBBuilder::BreakDownToPredicatedMBBs(LinearMachineBasicBlock &Parent,
                                          llvm::MachineBasicBlock &MBB) {
  llvm::MachineFunction *MF = MBB.getParent();
  assert(MF && "MBB is not linked in a machine function");

  const llvm::TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();

  llvm::SmallVector<std::unique_ptr<PredMBBBuilder>, 6> Out;

  auto PredMBBAllocator =
      [&](PredicatedMachineBasicBlock::PredicateValue EMV) -> PredMBBBuilder & {
    Out.emplace_back(std::make_unique<PredMBBBuilder>(Parent, EMV));
    return *Out.back();
  };

  // The current block being processed in the MBB
  std::reference_wrapper<PredMBBBuilder> CurrentBlock =
      PredMBBAllocator(PredicatedMachineBasicBlock::One);
  // Set of VectorMBBs that are waiting to be connected to the next non-taken
  // block or joined into the next block that starts with a scalar instruction
  llvm::SmallDenseSet<std::reference_wrapper<PredMBBBuilder>>
      BlocksWithHangingEdges{
          PredMBBAllocator(PredicatedMachineBasicBlock::ZeroOrOne)};

  for (auto MI = MBB.instr_begin(), PrevMI = MBB.instr_end(),
            NextMI = ++MBB.instr_begin();
       MI != MBB.instr_end(); PrevMI = MI, ++MI,
            NextMI = NextMI != MBB.instr_end() ? ++NextMI : MBB.instr_end()) {
    // Include debug instruction in the current block regardless of the
    // predicate value
    if (MI->isDebugInstr())
      continue;
    // Check if the MI is vector (i.e. not scalar nor lane access),
    // whether it writes to the exec mask, and whether the last MI
    // (if exists) was a scalar instruction

    bool IsVector = isVector(*MI);
    bool WritesExecMask = MI->modifiesRegister(llvm::AMDGPU::EXEC, TRI);
    bool IsFormerMIScalar = PrevMI != MBB.instr_end() &&
                            (isScalar(*PrevMI) || isLaneAccess(*PrevMI));
    if (IsVector && (WritesExecMask || IsFormerMIScalar)) {
      // If the current instruction is a vector inst, and if it writes
      // to the exec mask or if the last instruction was a scalar inst,
      // then we need to do a "split": Create a new VectorMBB to
      // replace the current taken block and make the new block the successor
      // of the current block, and then put the just-replaced taken block
      // into the list of not-taken blocks that are yet to be connected
      PredMBBBuilder &NewCurrentBlock =
          PredMBBAllocator(PredicatedMachineBasicBlock::One);

      NewCurrentBlock.addPredecessorBlock(CurrentBlock);
      BlocksWithHangingEdges.insert(CurrentBlock);
      CurrentBlock = NewCurrentBlock;

      if (CurrentBlock.get().getPredMBB().empty())
        CurrentBlock.get().Out.Instructions = {MI, NextMI};
      else
        CurrentBlock.get().Out.Instructions = {
            CurrentBlock.get().Out.Instructions.begin(), MI};
    } else if (!IsVector && !IsFormerMIScalar) {
      // Otherwise, if we observe a scalar instruction, we have to do a "join"
      // operation: Create a new VectorMBB to replace the current taken block,
      // and make it the successor of the current taken block. Also make
      // all not taken blocks with hanging edges the predecessor of the new
      // taken block, and then clear the not-taken set
      PredMBBBuilder &NewCurrentBlock =
          PredMBBAllocator(PredicatedMachineBasicBlock::ZeroOrOne);

      NewCurrentBlock.addPredecessorBlock(CurrentBlock);
      for (auto Block : BlocksWithHangingEdges) {
        Block.get().addSuccessorBlock(NewCurrentBlock);
      }
      BlocksWithHangingEdges.clear();
      CurrentBlock = NewCurrentBlock;
    }
    // Add the current instruction to the current taken block
    if (CurrentBlock.get().getPredMBB().empty())
      CurrentBlock.get().Out.Instructions = {MI, NextMI};
    else
      CurrentBlock.get().Out.Instructions = {
          CurrentBlock.get().Out.Instructions.begin(), NextMI};

    /// If the current MI is a call, create a new block and set it to the
    /// current taken block
    /// Since calls are scalar, there aren't any not taken blocks with hanging
    /// edges we have to worry about here
    /// We let the IPVectorCFG take care of linking the successors of this
    /// block according to the MI's metadata
    if (MI->isCall() && NextMI != MBB.instr_end()) {
      CurrentBlock = PredMBBAllocator(
          isVector(*NextMI) ? PredicatedMachineBasicBlock::One
                            : PredicatedMachineBasicBlock::ZeroOrOne);
    }
  }

  PredMBBBuilder &ExitVectorBlock =
      PredMBBAllocator(PredicatedMachineBasicBlock::One);
  PredMBBBuilder &ExitScalarBlock =
      PredMBBAllocator(PredicatedMachineBasicBlock::ZeroOrOne);

  // Connect the current taken block to the exit taken block of the current
  // MBB
  CurrentBlock.get().addSuccessorBlock(ExitVectorBlock);
  // Connect all the non-taken blocks to the exit non-taken block of the
  // current MBB if there are any left; Otherwise, connect the current
  // taken block to the exit not taken block because it must have ended with
  // a join (scalar instruction)
  if (!BlocksWithHangingEdges.empty()) {
    for (auto Block : BlocksWithHangingEdges) {
      Block.get().addSuccessorBlock(ExitScalarBlock);
    }
  } else {
    CurrentBlock.get().addSuccessorBlock(ExitScalarBlock);
  }

  /// Now that all predicated blocks are created, populate their internal
  /// MI set to allow for fast lookup of their block
  for (std::unique_ptr<PredMBBBuilder> &PredMBB : Out) {
    for (const llvm::MachineInstr &MI : PredMBB->getPredMBB()) {
      PredMBB->Out.MIsSet.insert(&MI);
    }
  }
  return std::move(Out);
}

bool PredMBBBuilder::unlinkIfTrivialEmptyBlock() {
  if (getPredMBB().empty() &&
      (!getPredMBB().preds_empty() || getPredMBB().succs_size() == 1)) {
    for (auto &Succ : Out.Successors) {
      for (auto &Pred : Out.Predecessors) {
        Succ.get().addPredecessorBlock(Pred);
      }
    }
    for (auto &Succ : Out.Successors) {
      removeSuccessorBlock(Succ);
    }
    for (auto &Pred : Out.Predecessors) {
      removePredecessorBlock(Pred);
    }
    return true;
  }
  return false;
}

} // namespace luthier