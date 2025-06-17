//===-- BranchTargetAnalysis.cpp ------------------------------------------===//
// Copyright 2022-2025 @ Northeastern University Computer Architecture Lab
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
/// Implements the \c BranchTargetOffsetsAnalysis class.
//===----------------------------------------------------------------------===//
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/MC/MCInst.h>
#include <llvm/MC/MCInstrAnalysis.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <luthier/Instrumentation/BranchTargetOffsetsAnalysis.h>
#include <luthier/Instrumentation/DisassembleAnalysisPass.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-target-offsets-analysis"

namespace luthier {

llvm::AnalysisKey BranchTargetOffsetsAnalysis::Key;

BranchTargetOffsetsAnalysis::Result
BranchTargetOffsetsAnalysis::run(llvm::MachineFunction &MF,
                                 llvm::MachineFunctionAnalysisManager &MFAM) {
  Result Out;
  llvm::ArrayRef<Instr> Instructions =
      MFAM.getResult<DisassemblerAnalysisPass>(MF).getInstructions();
  if (Instructions.empty())
    return Out;

  const llvm::TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const llvm::TargetMachine &TM = MF.getTarget();
  auto MIA = std::unique_ptr<llvm::MCInstrAnalysis>(
      TM.getTarget().createMCInstrAnalysis(TM.getMCInstrInfo()));

  uint64_t PrevInstOffset = Instructions[0].getOffset();

  for (unsigned int I = 0; I < Instructions.size(); ++I) {
    llvm::MCInst Inst = Instructions[I].getMCInst();
    uint64_t Offset = Instructions[I].getOffset();
    size_t Size = Instructions[I].getSize();
    if (TII.get(Inst.getOpcode()).isBranch()) {
      LLVM_DEBUG(

          llvm::dbgs() << "Instruction ";
          Inst.dump_pretty(llvm::dbgs(), nullptr, " ", TM.getMCRegisterInfo());
          llvm::dbgs() << llvm::formatv(
              " at idx {0}, offset {1:x}, size {2} is a branch; "
              "Evaluating its target.\n",
              I, Offset, Size);

      );
      if (uint64_t Target; MIA->evaluateBranch(Inst, Offset, 4, Target)) {
        LLVM_DEBUG(
            llvm::dbgs() << llvm::formatv(
                "Evaluated offset {0:x} as the branch target.\n", Target););
        Out.BranchTargetOffsets.insert(Target);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Failed to evaluate the branch target.\n");
      }
    }
    PrevInstOffset = Offset;
  }
  return Out;
}
} // namespace luthier