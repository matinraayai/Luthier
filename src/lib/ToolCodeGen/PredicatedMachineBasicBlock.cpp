//===-- PredicatedMachineBasicBlock.cpp -----------------------------------===//
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
/// \file PredicatedMachineBasicBlock.cpp
/// Implements the \c PredicatedMachineBasicBlock class and its builder.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/PredicatedMachineBasicBlock.h"
#include "luthier/LLVM/CodeGenHelpers.h"
#include "luthier/LLVM/streams.h"
#include "luthier/ToolCodeGen/IPPredicatedCFG.h"
#include "luthier/ToolCodeGen/TargetMachineInstrMDNode.h"
#include <SIInstrInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/Support/FormatVariadic.h>

namespace luthier {

std::string PredicatedMachineBasicBlock::getName() const {
  return llvm::formatv("{0}.{1}", MBB.getFullName(), GlobalIdx).str();
}

PredicatedMachineBasicBlock::iterator
PredicatedMachineBasicBlock::getFirstNonDebugInstr(bool SkipPseudoOp) {
  return skipDebugInstructionsForward(begin(), end(), SkipPseudoOp);
}

PredicatedMachineBasicBlock::iterator
PredicatedMachineBasicBlock::getLastNonDebugInstr(bool SkipPseudoOp) {
  iterator B = begin(), I = end();
  while (I != B) {
    --I;
    if (I->isDebugInstr() || I->isInsideBundle())
      continue;
    if (SkipPseudoOp && I->isPseudoProbe())
      continue;
    return I;
  }
  return end();
}

bool PredicatedMachineBasicBlock::doesLastInstrModifyPredicate() const {
  const llvm::TargetRegisterInfo *TRI =
      MBB.getParent()->getSubtarget().getRegisterInfo();
  return back().modifiesRegister(llvm::AMDGPU::EXEC, TRI);
}

void PredicatedMachineBasicBlock::print(llvm::raw_ostream &OS,
                                        unsigned Indent) const {
  const auto &ST = MBB.getParent()->getSubtarget();
  const auto *TII = ST.getInstrInfo();

  OS.indent(Indent) << "PredMBB " << getName()
                    << " (function=" << MBB.getParent()->getName()
                    << ", MBB=" << llvm::printMBBReference(MBB) << ")\n";

  if (HasUnresolvedEdges)
    OS.indent(Indent) << "  (has unresolved edges)\n";

  OS.indent(Indent) << "Predecessors: [";
  llvm::interleave(
      Predecessors.begin(), Predecessors.end(),
      [&](const PredMBBBuilder &B) { OS << B.getPredMBB().getName(); },
      [&]() { OS << ", "; });
  OS << "]\n";

  if (!empty()) {
    OS.indent(Indent) << "Instructions:\n";
    for (const auto &MI : *this) {
      MI.print(OS.indent(Indent + 2), true, false, false, true, TII);
      // Tag any MI whose pcsections metadata lacks a trace-instr address for
      // tests and easier reading
      if (const auto *MD = TargetMachineInstrMDNode::getInstrMDNodeIfExists(MI))
        if (!MD->isTraceInstr())
          OS.indent(Indent + 2) << "    ; synthetic (no trace address)\n";
    }
  }

  OS.indent(Indent) << "Successors: [";
  llvm::interleave(
      Successors.begin(), Successors.end(),
      [&](const PredMBBBuilder &B) { OS << B.getPredMBB().getName(); },
      [&]() { OS << ", "; });
  OS << "]\n";
}

void PredicatedMachineBasicBlock::dump() const { print(luthier::dbgs(), 0); }

} // namespace luthier
