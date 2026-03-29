//===-- ProcessIntrinsicsAtIRLevelPass.cpp --------------------------------===//
// Copyright 2022-2026 @ Northeastern University Computer Architecture Lab
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
/// \file ProcessIntrinsicsAtIRLevelPass.cpp
/// Implements the \c ProcessIntrinsicsAtIRLevelPass class.
//===----------------------------------------------------------------------===//
#include "luthier/Tooling/ProcessIntrinsicsAtIRLevelPass.h"
#include "luthier/Common/ErrorCheck.h"
#include "luthier/Common/GenericLuthierError.h"
#include "luthier/Tooling/FunctionAnnotations.h"
#include "luthier/Tooling/IntrinsicProcessorsAnalysis.h"
#include "luthier/Tooling/WrapperAnalysisPasses.h"
#include <llvm/IR/MDBuilder.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/ScopedPrinter.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-process-intrinsics-at-ir-level-pass"

llvm::PreservedAnalyses luthier::ProcessIntrinsicsAtIRLevelPass::run(
    llvm::Module &IModule, llvm::ModuleAnalysisManager &IMAM) {

  const auto &IntrinsicsProcessors =
      IMAM.getResult<IntrinsicsProcessorsAnalysis>(IModule);

  llvm::LLVMContext &Ctx = IModule.getContext();
  llvm::MDBuilder MDB{Ctx};

  // Iterate over all functions and find the ones marked as a Luthier
  // intrinsic
  // Do an early increment on the range since we will remove the intrinsic
  // function once we have processed all its users
  for (auto &F : llvm::make_early_inc_range(IModule.functions())) {
    if (F.hasFnAttribute(IntrinsicAttribute)) {
      // Find the processor for this intrinsic
      auto IntrinsicName =
          F.getFnAttribute(IntrinsicAttribute).getValueAsString();
      // Ensure the processor is indeed registered with the Code Generator
      std::optional<IntrinsicProcessor> Processor =
          IntrinsicsProcessors.getProcessorIfRegistered(IntrinsicName);
      if (!Processor.has_value()) {
        LUTHIER_CTX_EMIT_ON_ERROR(
            Ctx, LUTHIER_MAKE_GENERIC_ERROR(llvm::formatv(
                     "Intrinsic {0} is not registered", IntrinsicName)));
      }

      LLVM_DEBUG(

          llvm::dbgs() << "Intrinsic being processed: " << F << "\n";
          llvm::dbgs() << "Base name of the intrinsic: " << IntrinsicName
                       << "\n";
          llvm::dbgs() << "Num uses of the intrinsic function : "
                       << F.getNumUses() << "\n"

      );

      // Iterate over all users of the intrinsic
      // Early increment the loop range since we will replace and delete the
      // user in the process
      for (auto *User : llvm::make_early_inc_range(F.users())) {
        LLVM_DEBUG(

            llvm::dbgs() << "User being processed: \n";
            User->print(llvm::dbgs());

        );

        // Ensure the user is a Call instruction; Anything other usage is
        // illegal
        auto *CallInst = llvm::dyn_cast<llvm::CallInst>(User);

        if (!CallInst) {
          IModule.getContext().emitError(
              llvm::formatv("Found a user of intrinsic {0} which is not a "
                            "call instruction: {1}.",
                            IntrinsicName, *User));
          return llvm::PreservedAnalyses::all();
        }
        // Call the IR processor of the intrinsic on the user
        llvm::Expected<IntrinsicIRLoweringInfo> IRLoweringInfoOrErr =
            Processor->IRProcessor(F, *CallInst, TM);
        if (auto Err = IRLoweringInfoOrErr.takeError()) {
          IModule.getContext().emitError(CallInst,
                                         llvm::toString(std::move(Err)));
        }

        // Set up the input/output value constraints

        // Add the output operand constraint, if the output is not void
        const auto &ReturnValInfo = IRLoweringInfoOrErr->getReturnValueInfo();
        std::stringstream ConstraintSS;

        if (!ReturnValInfo.Val->getType()->isVoidTy())
          ConstraintSS << "=" << ReturnValInfo.Constraint;
        // Construct argument type vector
        llvm::SmallVector<llvm::Type *, 4> ArgTypes;
        llvm::SmallVector<llvm::Value *, 4> ArgValues;
        ArgTypes.reserve(IRLoweringInfoOrErr->getArgsInfo().size());
        ArgValues.reserve(IRLoweringInfoOrErr->getArgsInfo().size());
        for (const auto &[I, ArgInfo] :
             llvm::enumerate(IRLoweringInfoOrErr->getArgsInfo())) {
          if (I != 0 || (I == 0 && !ReturnValInfo.Val->getType()->isVoidTy()))
            ConstraintSS << ",";
          ArgTypes.push_back(ArgInfo.Val->getType());
          ArgValues.push_back(const_cast<llvm::Value *>(ArgInfo.Val));
          ConstraintSS << ArgInfo.Constraint;
        }
        // Now that we have created the input/output argument constraints,
        // create a call to a placeholder inline assembly instruction in the
        // place of the user
        auto *PlaceHolderInlineAsm = llvm::InlineAsm::get(
            llvm::FunctionType::get(ReturnValInfo.Val->getType(), ArgTypes,
                                    false),
            IntrinsicName, ConstraintSS.str(), true);
        auto *InlineAsmPlaceholderCall =
            llvm::CallInst::Create(PlaceHolderInlineAsm, ArgValues);
        InlineAsmPlaceholderCall->insertBefore(*CallInst->getParent(),
                                               CallInst->getIterator());
        // Replace all occurrences of the user with the inline assembly
        // placeholder
        CallInst->replaceAllUsesWith(InlineAsmPlaceholderCall);
        // Transfer debug info of the original use to the inline assembly
        // placeholder
        InlineAsmPlaceholderCall->setDebugLoc(CallInst->getDebugLoc());
        // Remove the original user from its parent function
        CallInst->eraseFromParent();

        LLVM_DEBUG(

            llvm::dbgs() << "Use's inline assembly after IR processing: \n";
            InlineAsmPlaceholderCall->print(llvm::dbgs());

        );
        // Encode the extra info into the PC sections of the instruction
        // we just processed so that we can retrieve it later in the MIR
        // processing
        CallInst->setMetadata(
            llvm::LLVMContext::MD_pcsections,
            MDB.createPCSections(llvm::MDBuilder::PCSection{
                IntrinsicExtraInfoHeader,
                IRLoweringInfoOrErr->getExtraLoweringValues()}));
      }
      // Remove the intrinsic function once all its users has been processed
      F.dropAllReferences();
      F.eraseFromParent();
    }
  }

  return llvm::PreservedAnalyses::all();
}
