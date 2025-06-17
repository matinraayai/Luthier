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
#include <llvm/IR/Function.h>
#include <llvm/MC/MCInst.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/Target/TargetMachine.h>
#include <luthier/Instrumentation/BranchTargetOffsetsAnalysis.h>
#include <luthier/Instrumentation/DisassembleAnalysisPass.h>

#undef DEBUG_TYPE
#define DEBUG_TYPE "luthier-target-offsets-analysis"

namespace luthier {
static bool evaluateBranch(const llvm::MCInst &Inst, uint64_t Addr,
                           uint64_t Size, uint64_t &Target) {
  if (!Inst.getOperand(0).isImm())
    return false;
  int64_t Imm = Inst.getOperand(0).getImm();
  // Our branches take a simm16.
  Target = llvm::SignExtend64<16>(Imm) * 4 + Addr + 4;
  return true;
}

BranchTargetOffsetsAnalysis::Result
BranchTargetOffsetsAnalysis::run(llvm::Function &F,
                                 llvm::FunctionAnalysisManager &FAM) {
  Result Out;
  llvm::ArrayRef<Instr> Instructions =
      FAM.getResult<DisassemblerAnalysisPass>(F).getInstructions();
  if (Instructions.empty())
    return Out;
  auto &MAMProxy = FAM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  llvm::LLVMContext &Ctx = F.getContext();
  llvm::Module &M = *F.getParent();

  /// Get the MCInstrInfo of the current function
  LUTHIER_EMIT_ERROR_IN_CONTEXT(
      Ctx, LUTHIER_GENERIC_ERROR_CHECK(
               MAMProxy.cachedResultExists<llvm::MachineModuleAnalysis>(M),
               "MAM does not have a Machine Module Info analysis"));
  llvm::MachineModuleInfo &MMI =
      MAMProxy.getCachedResult<llvm::MachineModuleAnalysis>(M)->getMMI();
  const llvm::TargetMachine &TM = MMI.getTarget();
  const llvm::MCInstrInfo &MII = *TM.getMCInstrInfo();

  uint64_t PrevInstOffset = Instructions[0].getOffset();

  for (unsigned int I = 0; I < Instructions.size(); ++I) {
    llvm::MCInst Inst = Instructions[I].getMCInst();
    uint64_t Offset = Instructions[I].getOffset();
    size_t Size = Instructions[I].getSize();
    if (MII.get(Inst.getOpcode()).isBranch()) {
      LLVM_DEBUG(

          llvm::dbgs() << "Instruction ";
          Inst.dump_pretty(llvm::dbgs(), nullptr, " ", TM.getMCRegisterInfo());
          llvm::dbgs() << llvm::formatv(
              " at idx {0}, offset {1:x}, size {2} is a branch; "
              "Evaluating its target.\n",
              I, Offset, Size);

      );
      uint64_t Target;
      if (evaluateBranch(Inst, Offset, Size, Target)) {
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