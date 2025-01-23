//===-- ProcessIntrinsicUsersAtIRLevelPass.cpp ----------------------------===//
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
/// This file implements the <tt>ProcessIntrinsicsAtIRLevelPass</tt>.
//===----------------------------------------------------------------------===//

#include "tooling_common/ProcessIntrinsicsAtIRLevelPass.hpp"
#include "tooling_common/WrapperAnalysisPasses.hpp"
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/ScopedPrinter.h>
#include <luthier/Consts.h>

#undef DEBUG_TYPE

#define DEBUG_TYPE "luthier-process-intrinsics-at-ir-level-pass"

llvm::PreservedAnalyses luthier::ProcessIntrinsicsAtIRLevelPass::run(
    llvm::Module &IModule, llvm::ModuleAnalysisManager &IMAM) {
  auto &IntrinsicLoweringInfoVec =
      IMAM.getResult<IntrinsicIRLoweringInfoMapAnalysis>(IModule)
          .getLoweringInfo();

  const auto &IntrinsicsProcessors =
      IMAM.getResult<IntrinsicsProcessorsAnalysis>(IModule).getProcessors();

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
      auto It = IntrinsicsProcessors.find(IntrinsicName);
      if (It == IntrinsicsProcessors.end()) {
        IModule.getContext().emitError(
            "Intrinsic " + llvm::Twine(IntrinsicName) +
            " is not registered with the code generator.");
        return llvm::PreservedAnalyses::all();
      }

      LLVM_DEBUG(

          llvm::dbgs() << "Intrinsic function being processed: " << F << "\n";
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

        if (CallInst == nullptr) {
          IModule.getContext().emitError(
              llvm::formatv("Found a user of intrinsic {0} which is not a "
                            "call instruction: {1}.",
                            IntrinsicName, *User));
          return llvm::PreservedAnalyses::all();
        }
        // Call the IR processor of the intrinsic on the user
        auto IRLoweringInfo = It->second.IRProcessor(F, *CallInst, TM);
        if (auto Err = IRLoweringInfo.takeError()) {
          IModule.getContext().emitError(toString(std::move(Err)));
        }

        // Set up the input/output value constraints

        // Add the output operand constraint, if the output is not void
        auto ReturnValInfo = IRLoweringInfo->getReturnValueInfo();
        std::string Constraint;
        if (!ReturnValInfo.Val->getType()->isVoidTy())
          Constraint += "=" + ReturnValInfo.Constraint;
        // Construct argument type vector
        llvm::SmallVector<llvm::Type *, 4> ArgTypes;
        llvm::SmallVector<llvm::Value *, 4> ArgValues;
        ArgTypes.reserve(IRLoweringInfo->getArgsInfo().size());
        ArgValues.reserve(IRLoweringInfo->getArgsInfo().size());
        //        llvm::interleave(IRLoweringInfo->getArgsInfo(), [&](const
        //        IntrinsicValueLoweringInfo & ArgInfo) {
        //          ArgTypes.push_back(ArgInfo.Val->getType());
        //          ArgValues.push_back(const_cast<llvm::Value *>(ArgInfo.Val));
        //          Constraint += ArgInfo.Constraint;
        //        }, [&]() {Constraint += ",";});
        for (const auto &[I, ArgInfo] :
             llvm::enumerate(IRLoweringInfo->getArgsInfo())) {
          if (I != 0 || (I == 0 && !ReturnValInfo.Val->getType()->isVoidTy()))
            Constraint += ",";
          ArgTypes.push_back(ArgInfo.Val->getType());
          ArgValues.push_back(const_cast<llvm::Value *>(ArgInfo.Val));
          Constraint += ArgInfo.Constraint;
        }
        // Now that we have created the input/output argument constraints,
        // create a call to a placeholder inline assembly instruction in the
        // place of the user
        auto *PlaceHolderInlineAsm = llvm::InlineAsm::get(
            llvm::FunctionType::get(ReturnValInfo.Val->getType(), ArgTypes,
                                    false),
            llvm::to_string(IntrinsicLoweringInfoVec.size()), Constraint, true);
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
        auto *ParentFunction =
            InlineAsmPlaceholderCall->getParent()->getParent();
        // If the function using the intrinsic is not an injected payload and a
        // hook (i.e a device function called from a hook), check if it's not
        // requesting access to a physical register or a kernel argument
        if (!ParentFunction->hasFnAttribute(HookAttribute) &&
            !ParentFunction->hasFnAttribute(
                InjectedPayloadAttribute)) {
          if (!IRLoweringInfo->accessed_phys_regs_empty()) {
            IModule.getContext().emitError(
                llvm::formatv("Intrinsic {0} used in function {1} requested "
                              "access to physical "
                              "registers, which is not allowed.",
                              IntrinsicName, ParentFunction->getName()));
            return llvm::PreservedAnalyses::all();
          }

          if (!IRLoweringInfo->accessed_kernargs_empty()) {
            IModule.getContext().emitError(
                llvm::formatv("Intrinsic {0} used in non-hook function {1} "
                              "requested access to kernel arguments"
                              " , which is not allowed.",
                              IntrinsicName, ParentFunction->getName()));
            return llvm::PreservedAnalyses::all();
          }
        }

        // Record the name of the intrinsic as well as the inline assembly
        // placeholder instruction used to keep track of it inside the
        // Module/MMI
        IRLoweringInfo->setIntrinsicName(IntrinsicName);
        IRLoweringInfo->setPlaceHolderInlineAsm(*PlaceHolderInlineAsm);

        LLVM_DEBUG(

            llvm::dbgs() << "Use's inline assembly after IR processing: \n";
            InlineAsmPlaceholderCall->print(llvm::dbgs());

        );
        // Finally, push back the IR lowering info of the intrinsic that
        // we just processed
        IntrinsicLoweringInfoVec.emplace_back(*IRLoweringInfo);
      }
      // Remove the intrinsic function once all its users has been processed
      F.dropAllReferences();
      F.eraseFromParent();
    }
  }

  return llvm::PreservedAnalyses::all();
}
