//===-- InstrumentationModuleIRGeneratorPass.cpp --------------------------===//
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
/// This file implements the Instrumentation Module IR Generator Pass.
//===----------------------------------------------------------------------===//
#include "tooling_common/IModuleIRGeneratorPass.hpp"
#include "luthier/common/ErrorCheck.h"
#include "luthier/common/LuthierError.h"
#include "luthier/consts.h"
#include "luthier/intrinsic/IntrinsicCalls.h"
#include "luthier/tooling/InstrumentationTask.h"
#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TimeProfiler.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-generate-inst-ir"

namespace luthier {

llvm::AnalysisKey InjectedPayloadAndInstPointAnalysis::Key;

InjectedPayloadAndInstPointAnalysis::Result
InjectedPayloadAndInstPointAnalysis::run(llvm::Module &M,
                                         llvm::ModuleAnalysisManager &) {
  return {};
}

static llvm::Expected<llvm::Function &> generateInjectedPayloadForApplicationMI(
    llvm::Module &IModule,
    llvm::ArrayRef<InstrumentationTask::hook_invocation_descriptor>
        HookInvocationSpecs,
    const llvm::MachineInstr &ApplicationMI) {
  auto &LLVMContext = IModule.getContext();
  // Create an empty function to house the code injected before the
  // target application MI
  llvm::FunctionType *FunctionType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(LLVMContext), {}, false);
  // Name of the injected payload function will contain the application MI
  // it will be injected before, as well as the number of the MI's MBB
  std::string IFuncName;
  llvm::raw_string_ostream NameOS(IFuncName);
  NameOS << "MI: ";
  // This is verbose to avoid adding a new line here.
  ApplicationMI.print(NameOS, true, false, false, false);
  NameOS << ", MMB ID: " << ApplicationMI.getParent()->getNumber();
  auto *InjectedPayload = llvm::Function::Create(
      FunctionType, llvm::GlobalValue::ExternalLinkage, IFuncName, IModule);
  // The instrumentation function has the C-Calling convention
  InjectedPayload->setCallingConv(llvm::CallingConv::C);
  // Prevent emission of the prologue/epilogue code, but still lower the stack
  // operands
  InjectedPayload->addFnAttr(llvm::Attribute::Naked);
  // Set an attribute indicating that this is the top-level function for an
  // injected payload
  InjectedPayload->addFnAttr(InjectedPayloadAttribute);

  LLVM_DEBUG(

      llvm::dbgs() << "Generating instrumentation function for MI: "
                   << ApplicationMI << ", MBB: "
                   << ApplicationMI.getParent()->getNumber() << "\n";
      llvm::dbgs()
      << "Number of hooks to be called in instrumentation function: "
      << HookInvocationSpecs.size() << "\n"

  );

  // Create an empty basic block to fill in with calls to hooks in the order
  // specified by the spec
  llvm::BasicBlock *BB =
      llvm::BasicBlock::Create(IModule.getContext(), "", InjectedPayload);
  llvm::IRBuilder<> Builder(BB);
  for (const auto &HookInvSpec : HookInvocationSpecs) {
    // Find the hook function inside the instrumentation module
    auto HookFunc = IModule.getFunction(HookInvSpec.HookName);
    LUTHIER_RETURN_ON_ERROR(LUTHIER_ERROR_CHECK(
        HookFunc != nullptr,
        "Failed to find hook {0} inside the instrumentation module.",
        HookInvSpec.HookName));
    // Construct the operands of the hook call
    llvm::SmallVector<llvm::Value *, 4> Operands;
    for (const auto &[Idx, Op] : llvm::enumerate(HookInvSpec.Args)) {
      if (holds_alternative<llvm::MCRegister>(Op)) {
        // Create a call to the read reg intrinsic to load the MC register
        // into a value, then pass it to the hook
        auto ReadRegVal = insertCallToIntrinsic(
            *HookFunc->getParent(), Builder, "luthier::readReg",
            *HookFunc->getArg(Idx)->getType(),
            std::get<llvm::MCRegister>(Op).id());
        Operands.push_back(ReadRegVal);
      } else {
        // Otherwise it's a constant, we can just pass it directly
        Operands.push_back(std::get<llvm::Constant *>(Op));
      }
    }
    // Finally, create a call to the hook
    (void)Builder.CreateCall(HookFunc, Operands);
  }
  // Put a ret void at the end of the instrumentation function to indicate
  // nothing is returned
  (void)Builder.CreateRetVoid();
  return *InjectedPayload;
}

llvm::PreservedAnalyses
IModuleIRGeneratorPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  auto &IPIP = MAM.getResult<InjectedPayloadAndInstPointAnalysis>(M);
  llvm::TimeTraceScope Scope("Instrumentation Module IR Generation");
  // Generate and populate the injected payload functions in the
  // instrumentation module and keep track of them inside the map
  for (const auto &[ApplicationMI, HookSpecs] : Task.getHookInsertionTasks()) {
    // Generate the Hooks for each MI
    auto HookFunc =
        generateInjectedPayloadForApplicationMI(M, HookSpecs, *ApplicationMI);
    if (auto Err = HookFunc.takeError()) {
      M.getContext().emitError(llvm::toString(std::move(Err)));
      return llvm::PreservedAnalyses::all();
    }
    IPIP.addEntry(*ApplicationMI, *HookFunc);
  }
  return llvm::PreservedAnalyses::all();
}
} // namespace luthier
