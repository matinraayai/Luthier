//===-- InjectedPayloadAndInstPointAnalysis.cpp ---------------------------===//
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
/// Implements the \c InjectedPayloadAndInstPointAnalysis class.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGen/InjectedPayloadAndInstPointAnalysis.h"
#include "luthier/ToolCodeGen/FunctionAnnotations.h"
#include "luthier/ToolCodeGen/Prototype.h"
#include <llvm/ADT/StringRef.h>
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachineOperand.h>
#include <llvm/CodeGen/TargetOpcodes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-injected-payload-and-inst-points"

namespace luthier {

bool InjectedPayloadAndInstPoint::invalidate(
    Prototype &P, const llvm::PreservedAnalyses &PA,
    PrototypeAnalysisManager::Invalidator &Inv) {
  // Because this is read from the inner machine-passes pipeline via
  // PrototypeAnalysisManagerMachineFunctionProxy::getCachedResult,
  // Model this as a stateless outer analysis
  auto PAC = PA.getChecker<InjectedPayloadAndInstPointAnalysis>();
  return !PAC.preservedWhenStateless();
}

llvm::AnalysisKey InjectedPayloadAndInstPointAnalysis::Key;

InjectedPayloadAndInstPointAnalysis::Result
InjectedPayloadAndInstPointAnalysis::run(Prototype &P,
                                         PrototypeAnalysisManager &PAM) {
  InjectedPayloadAndInstPoint Result;

  llvm::Module &TargetModule = P.getTargetModule();
  llvm::Module &IModule = P.getInstrumentationModule();

  // Only the target module's MIR is walked below, so this is the target
  // module's FunctionAnalysisManager.
  llvm::FunctionAnalysisManager &FAM =
      PAM.getResult<TargetFunctionAnalysisManagerPrototypeProxy>(P)
          .getManager();

  // Index the instrumentation module's injected-payload definitions by name.
  llvm::DenseMap<llvm::StringRef, llvm::Function *> PayloadDefsByName;
  for (llvm::Function &F : IModule) {
    if (F.hasFnAttribute(InjectedPayloadAttribute))
      PayloadDefsByName[F.getName()] = &F;
  }

  // Walk the target module's MIR for PATCHPOINT markers
  for (llvm::Function &F : TargetModule) {
    llvm::MachineFunctionAnalysis::Result *MFRes =
        FAM.getCachedResult<llvm::MachineFunctionAnalysis>(F);
    if (!MFRes)
      continue;
    for (llvm::MachineBasicBlock &MBB : MFRes->getMF()) {
      for (llvm::MachineInstr &MI : MBB) {
        if (MI.getOpcode() != llvm::TargetOpcode::PATCHPOINT)
          continue;
        const llvm::MachineOperand &TargetOp = MI.getOperand(2);
        assert(TargetOp.isGlobal() &&
               "Second operand must be the injected payload function name");
        auto *ExternHandle = llvm::cast<llvm::Function>(
            const_cast<llvm::GlobalValue *>(TargetOp.getGlobal()));
        auto It = PayloadDefsByName.find(ExternHandle->getName());
        assert(It != PayloadDefsByName.end() &&
               "Payload extern decl doesn't have an associated definition");
        Result.addEntry(MI, *It->second, *ExternHandle);
      }
    }
  }

  return Result;
}

} // namespace luthier
