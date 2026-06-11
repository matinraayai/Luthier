//===-- TranslationStateTestPasses.cpp ------------------------------------===//
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
/// \file TranslationStateTestPasses.cpp
/// Implements the test-only \c TranslationState mark/flush passes.
//===----------------------------------------------------------------------===//
#include "luthier/ToolCodeGenTesting/TranslationStateTestPasses.h"
#include "luthier/ToolCodeGen/TraceIRTranslatorAnalysis.h"
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineFunction.h>
#include <llvm/CodeGen/MachineFunctionAnalysis.h>
#include <llvm/CodeGen/MachinePassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

namespace luthier {

namespace {

/// Runs \p Work over the lifted machine function of every defined function in
/// \p M with its TranslationState
void forEachTranslation(
    llvm::Module &M, llvm::ModuleAnalysisManager &MAM,
    llvm::function_ref<void(llvm::MachineFunction &, TranslationState &)>
        Work) {
  auto &FAM =
      MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto &MFAM = MAM.getResult<llvm::MachineFunctionAnalysisManagerModuleProxy>(M)
                   .getManager();
  for (llvm::Function &F : M) {
    if (F.isDeclaration())
      continue;
    llvm::MachineFunction &MF =
        FAM.getResult<llvm::MachineFunctionAnalysis>(F).getMF();
    if (MF.empty())
      continue;
    Work(MF, MFAM.getResult<TraceIRTranslatorAnalysis>(MF));
  }
}

/// PA shape shared by the flushing test passes: the lifted IR bodies are
/// rewritten wholesale; the MFs (and the pinned TranslationStates keyed on
/// them) must survive
llvm::PreservedAnalyses flushPreservedAnalyses() {
  llvm::PreservedAnalyses PA = llvm::PreservedAnalyses::none();
  PA.preserve<llvm::MachineFunctionAnalysisManagerModuleProxy>();
  PA.preserve<llvm::FunctionAnalysisManagerModuleProxy>();
  PA.preserve<llvm::MachineFunctionAnalysis>();
  return PA;
}

} // namespace

llvm::PreservedAnalyses
MarkRetranslateTestPass::run(llvm::Module &M,
                             llvm::ModuleAnalysisManager &MAM) {
  forEachTranslation(M, MAM,
                     [](llvm::MachineFunction &MF, TranslationState &TS) {
                       for (const llvm::MachineBasicBlock &MBB : MF)
                         TS.markDirty(MBB);
                     });
  return llvm::PreservedAnalyses::all();
}

llvm::PreservedAnalyses
FlushTranslationTestPass::run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
  forEachTranslation(
      M, MAM, [&](llvm::MachineFunction &, TranslationState &TS) {
        if (llvm::Error Err = TS.flush())
          M.getContext().emitError(llvm::toString(std::move(Err)));
      });
  return flushPreservedAnalyses();
}

llvm::PreservedAnalyses
WarmMarkFlushTestPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  forEachTranslation(
      M, MAM, [&](llvm::MachineFunction &MF, TranslationState &TS) {
        if (llvm::Error Err = TS.flush()) {
          M.getContext().emitError(llvm::toString(std::move(Err)));
          return;
        }
        for (const llvm::MachineBasicBlock &MBB : MF)
          TS.markDirty(MBB);
        if (llvm::Error Err = TS.flush())
          M.getContext().emitError(llvm::toString(std::move(Err)));
      });
  return flushPreservedAnalyses();
}

} // namespace luthier
